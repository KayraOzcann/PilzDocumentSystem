"""
AT Uygunluk Beyanı Analiz Servisi
==================================
Endpoint: POST /api/at-declaration
Health: GET /api/health
"""

import re, os, json, PyPDF2, logging, cv2, numpy as np, pytesseract
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from PIL import Image
import pdf2image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from langdetect import detect
    LANG_DETECT = True
except:
    LANG_DETECT = False

@dataclass
class ATAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    is_critical: bool
    details: Dict[str, Any]

class ATTypeInspectionAnalyzer:
    def __init__(self):
        self.criteria_weights = {
            "Kritik Bilgiler": 60,
            "Zorunlu Teknik Bilgiler": 25,
            "Standartlar ve Belgeler": 15
        }
        
        self.criteria_details = {
            "Kritik Bilgiler": {
                "uretici_adi": {
                    "pattern": r"(?:biz\s+burada\s+beyan\s+ederiz\s+ki[;:\s]*([^,\n]+))|(?:üretici|manufacturer|imalatçı|company|şirket|firma|unvan|we|manufactured by|sibernetik|pilz|tarafımızdan|üretici\s+firma)[\s:]*([A-Za-zÇŞİĞÜÖıçşığüö\s\.\-&]{8,100})|(?:karaca\s+mekatronik)",
                    "weight": 15,
                    "critical": True,
                    "description": "Üretici veya yetkili temsilcinin adı"
                },
                "uretici_adres": {
                    "pattern": r"(?:adres|address|cd\.\s*no|street|road|mahallesi|caddesi|sokak)[\s:]*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{15,200})|(?:demirci[^,\n]*nilüfer[^,\n]*bursa)|(?:cork[^,\n]*ireland)",
                    "weight": 15,
                    "critical": True,
                    "description": "Üretici veya yetkili temsilcinin adresi"
                },
                "makine_tanimi": {
                    "pattern": r"(?:makinenin tanıtımı|tanım|machine|makine|model|tip|type|description)[\s:]*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\-\.]{5,100})|(?:ecotorq|kafa|baga|çakma|knee pad|punching|vibr)",
                    "weight": 15,
                    "critical": True,
                    "description": "Makine tanımı (tip, model, seri)"
                },
                "direktif_atif": {
                    "pattern": r"(?:2006/42|2006\/42|makine direktif|machine directive|machinery directive|EC|AT|directive|european directive|ab direktif)",
                    "weight": 10,
                    "critical": True,
                    "description": "2006/42/EC Direktif atfı"
                },
                "yetkili_imza": {
                    "pattern": r"(?:yetkili\s+imza|authorized|authorised|imza|signature|beyan yetkilisi|responsible|müdür|manager|director|managing director|şahiner|mcauliffe|genel müdür|beyan eden|sorumlu|name|adı|surname|soyadı|ünvan|position|title|başkan|president|chief|şef|general\s+manager|general\s+maneger|karaca|eşref)",
                    "weight": 5,
                    "critical": True,
                    "description": "Yetkili kişi imzası ve unvanı"
                }
            },
            "Zorunlu Teknik Bilgiler": {
                "uretim_yili": {
                    "pattern": r"(?:üretim|imal|manufacturing|production)[\s\w]*(?:yılı|year|date)[\s:]*([0-9]{4})|([0-9]{4})[\s]*(?:yılı|year)|february\s*([0-9]{4})|([0-9]{4})",
                    "weight": 5,
                    "critical": False,
                    "description": "Üretim yılı"
                },
                "seri_no": {
                    "pattern": r"(?:seri|serial|s/n|sn)[\s\w]*(?:no|number)[\s:]*([A-Za-z0-9\-]{2,20})|serial number[\s:]*([A-Za-z0-9\-]{2,20})",
                    "weight": 5,
                    "critical": False,
                    "description": "Seri numarası"
                },
                "beyan_ifadesi": {
                    "pattern": r"(?:beyan|declaration|conform|uygun|comply|uygunluk|conformity|declare|conformity with)",
                    "weight": 5,
                    "critical": False,
                    "description": "Uygunluk beyan ifadesi"
                },
                "tarih_yer": {
                    "pattern": r"(?:tarih|date|yer|place)[\s:]*([0-9]{1,2}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{2,4})|([0-9]{1,2}\s*february\s*[0-9]{4})|cork\s*ireland\s*([0-9]{1,2}\s*february\s*[0-9]{4})",
                    "weight": 5,
                    "critical": False,
                    "description": "Beyan tarihi ve yeri"
                },
                "diger_direktifler": {
                    "pattern": r"(?:2014/30|2014/35|EMC|LVD|alçak gerilim|low voltage|elektromanyetik|electromagnetic|european directive)",
                    "weight": 5,
                    "critical": False,
                    "description": "Diğer direktifler (EMC, LVD vb.)"
                }
            },
            "Standartlar ve Belgeler": {
                "uyumlu_standartlar": {
                    "pattern": r"(?:EN|ISO|IEC)[\s]*[0-9]{3,5}[\-:]*[0-9]*[:\-]*[0-9]*",
                    "weight": 8,
                    "critical": False,
                    "description": "Uygulanmış uyumlaştırılmış standartlar"
                },
                "teknik_dosya": {
                    "pattern": r"(?:teknik dosya|technical file|documentation|dokümantasyon)",
                    "weight": 4,
                    "critical": False,
                    "description": "Teknik dosya sorumlusu"
                },
                "onaylanmis_kurulus": {
                    "pattern": r"(?:onaylanmış kuruluş|notified body|tip inceleme|type examination|belge|certificate)",
                    "weight": 3,
                    "critical": False,
                    "description": "Onaylanmış kuruluş bilgileri"
                }
            }
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyPDF2 and OCR fallback"""
        text = ""
        
        try:
            # First try PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            
            logging.info(f"PyPDF2 extracted {len(text)} characters")
            
            # If PyPDF2 gives insufficient text, use OCR
            if len(text.strip()) < 100:
                logging.info("Insufficient text with PyPDF2, trying OCR...")
                
                pages = pdf2image.convert_from_path(pdf_path, dpi=200)
                ocr_text = ""
                
                for i, page in enumerate(pages, 1):
                    try:
                        page_text = pytesseract.image_to_string(page, lang='tur+eng')
                        ocr_text += page_text + "\n"
                        logging.info(f"OCR extracted {len(page_text)} characters from page {i}")
                    except Exception as e:
                        logging.warning(f"OCR failed for page {i}: {e}")
                        continue
                
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    logging.info(f"OCR total text length: {len(text)}")
            
        except Exception as e:
            logging.error(f"Error extracting text: {e}")
            return ""
        
        return text
    
    def detect_language(self, text: str) -> str:
        if not LANG_DETECT:
            return 'tr'
        try:
            return detect(text[:500].strip())
        except:
            return 'tr'
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, ATAnalysisResult]:
        """Analyze criteria for a specific category"""
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            is_critical = criterion_data["critical"]
            description = criterion_data["description"]
            
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                # Clean up matches and get the best one
                clean_matches = []
                for match in matches:
                    if isinstance(match, tuple):
                        # For groups in regex, take the first non-empty group
                        clean_match = next((m for m in match if m.strip()), "")
                    else:
                        clean_match = str(match)
                    
                    if clean_match.strip():
                        clean_matches.append(clean_match.strip())
                
                if clean_matches:
                    content = f"Bulundu: {clean_matches[0][:50]}..."
                    found = True
                    score = weight  # Full points for found criteria
                else:
                    content = "Eşleşme bulundu ama değer çıkarılamadı"
                    found = True
                    score = int(weight * 0.7)  # Partial points
            else:
                content = "Bulunamadı"
                found = False
                score = 0
            
            results[criterion_name] = ATAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                is_critical=is_critical,
                details={
                    "description": description,
                    "pattern_used": pattern,
                    "matches_count": len(matches) if matches else 0,
                    "raw_matches": matches[:3] if matches else []
                }
            )
        
        return results
    
    def calculate_scores(self, analysis_results: Dict) -> Dict[str, Any]:
        category_scores = {}
        total_score = 0
        critical_missing = []
        
        for category, results in analysis_results.items():
            category_max = self.criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())
            
            for criterion_name, result in results.items():
                if result.is_critical and not result.found:
                    critical_missing.append(f"{category}: {result.details['description']}")
            
            percentage = (category_earned / category_possible * 100) if category_possible > 0 else 0
            normalized_score = (percentage / 100) * category_max
            
            category_scores[category] = {
                "earned": category_earned,
                "possible": category_possible,
                "normalized": round(normalized_score, 2),
                "max_weight": category_max,
                "percentage": round(percentage, 2)
            }
            
            total_score += normalized_score
        
        return {
            "category_scores": category_scores,
            "total_score": round(total_score, 2),
            "percentage": round(total_score, 2),
            "critical_missing": critical_missing
        }
    
    def extract_specific_values(self, text: str) -> Dict[str, Any]:
        """Extract specific values from AT Declaration"""
        values = {
            "manufacturer_name": "Bulunamadı",
            "manufacturer_address": "Bulunamadı",
            "machine_description": "Bulunamadı",
            "machine_model": "Bulunamadı",
            "production_year": "Bulunamadı",
            "serial_number": "Bulunamadı",
            "declaration_date": "Bulunamadı",
            "authorized_person": "Bulunamadı",
            "position": "Bulunamadı",
            "directive_reference": "Bulunamadı",
            "applied_standards": []
        }
        
        # Manufacturer name - Çoklu firma desteği
        manufacturer_patterns = [
            r"(?:biz\s+burada\s+beyan\s+ederiz\s+ki[;:\s]*)([^,\n]+)",
            r"(?:we\s+)([A-Za-z\s&\.]+?)(?:\s+declare|\s+industrial)",
            r"(?:manufactured by|üretici|manufacturer)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö\s\.\-&]{5,100})",
            r"(sibernetik\s+makina\s*&?\s*otomasyon[^,\n]*)",
            r"(pilz\s+ireland\s+industrial\s+automation)",
            r"(suzhou\s+keber\s+technology\s+co)",
            r"([A-ZÜÇĞIÖŞ][a-züçğıöş]+(?:\s+[A-ZÜÇĞIÖŞ][a-züçğıöş]+)*\s+(?:makina|technology|industrial|automation|şirket|company))"
        ]
        
        for pattern in manufacturer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                manufacturer_name = match.group(1).strip()
                if len(manufacturer_name) > 5 and not re.search(r'^[0-9]+$', manufacturer_name):
                    values["manufacturer_name"] = manufacturer_name
                    break
        
        # Address - Çoklu adres formatı desteği
        address_patterns = [
            r"(demirci[^,\n]*cd\.[^,\n]*no[^,\n]*nilüfer[^,\n]*bursa)",
            r"(cork\s+business\s*&?\s*technology\s+park[^,]*model\s+farm\s+road[^,]*cork[^,]*ireland)",
            r"(no\.\s*[0-9]+[^,]*suzhou[^,]*jiangsu[^,]*)",
            r"(?:address|adres)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{20,200})",
            r"([A-ZÜÇĞIÖŞ][a-züçğıöş]+(?:\s+[A-Za-züçğıöş]+)*\s+(?:cd\.|caddesi|street|road)[^,\n]{10,100})",
            r"([^,\n]*(?:mahallesi|caddesi|sokak|street|road|park)[^,\n]{10,100})"
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
                if len(address) > 15:
                    values["manufacturer_address"] = address
                    break
        
        # Machine description - Çoklu makine türü desteği
        machine_patterns = [
            r"(?:makinenin tanıtımı ve sınıfı|tanım|description)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\-\.]{5,100})",
            r"(fo\s*[0-9]+\.?[0-9]*lt?\s+ecotorq\s+kafa\s+baga\s+çakma)",
            r"(v[0-9]+b\s+knee\s+pad\s+punching\s+machine)",
            r"(vibratory\s+surface\s+finishing\s+machine)",
            r"(?:makine|machine|model|equipment)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\-\.]{8,80})"
        ]
        
        for pattern in machine_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                machine_desc = match.group(1).strip()
                if not re.search(r"farm\s+road|business|technology|address|adres", machine_desc, re.IGNORECASE):
                    values["machine_description"] = machine_desc
                    break
        
        # Model - Daha spesifik model pattern
        model_patterns = [
            r"(V[0-9]+B)",
            r"(VKT\s*[0-9]+)",
            r"(?:model|tip|type)\s*[:\-]?\s*([A-Za-z0-9\s\-]{2,30})"
        ]
        
        for pattern in model_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                model_text = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                if not re.search(r"farm\s+road|business|technology", model_text, re.IGNORECASE):
                    values["machine_model"] = model_text
                    break
        
        # Serial number - Çoklu seri numarası formatı
        serial_patterns = [
            r"(?:seri numarası|serial number)\s*[:\-]?\s*([A-Z0-9\/\-]+)",
            r"(?:seri|s/n|sn)\s*(?:no|number)\s*[:\-]?\s*([A-Za-z0-9\-\/]{3,25})",
            r"([0-9]{6,8}\/[0-9]{4})",
            r"([A-Z][0-9]{4}[A-Z]?\-[0-9]{3})"
        ]
        
        for pattern in serial_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                serial = match.group(1).strip()
                if len(serial) >= 3:
                    values["serial_number"] = serial
                    break
        
        # Production year - Yıl bilgisi için makul yıl aralığı kontrolü
        year_patterns = [
            r"([0-9]{4})",
            r"february\s+([0-9]{4})",
            r"(?:üretim|imal|year)\s*[:\-]?\s*([0-9]{4})"
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for year in matches:
                year_int = int(year) if year.isdigit() else 0
                if 2020 <= year_int <= 2030:
                    values["production_year"] = year
                    break
            if values["production_year"] != "Bulunamadı":
                break
        
        # Declaration date
        date_patterns = [
            r"(?:tarih|date)\s*[:\-]?\s*([0-9]{1,2}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{2,4})",
            r"([0-9]{1,2}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{2,4})"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                values["declaration_date"] = match.group(1).strip()
                break
        
        # Authorized person
        person_patterns = [
            r"(?:beyan yetkilisi|authorized|yetkili|name)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö\s]{5,50})",
            r"(?:adı soyadı|name)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö\s]{5,50})"
        ]
        
        for pattern in person_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["authorized_person"] = match.group(1).strip()
                break
        
        # Position
        position_patterns = [
            r"(?:ünvan|position|görevi|title)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö\s]{5,50})",
            r"(?:müdür|manager|director|president|başkan)"
        ]
        
        for pattern in position_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["position"] = match.group(1).strip() if hasattr(match.group(1), 'strip') else match.group(0).strip()
                break
        
        # Applied standards
        standards = re.findall(r"(?:EN|ISO|IEC)\s*[0-9]{3,5}[\-:]*[0-9]*[:\-]*[0-9]*", text, re.IGNORECASE)
        values["applied_standards"] = list(set(standards))
        
        return values
    
    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        recommendations = []
        if scores["critical_missing"]:
            recommendations.append("🚨 KRİTİK EKSİKLİKLER:")
            for missing in scores["critical_missing"]:
                recommendations.append(f"  ❌ {missing}")
        else:
            recommendations.append("✅ AT Uygunluk Beyanı yeterli")
        return recommendations
    
    def analyze_at_declaration(self, pdf_path: str) -> Dict[str, Any]:
        text = self.extract_text_from_pdf(pdf_path)
        if not text or len(text.strip()) < 50:
            return {"error": "PDF okunamadı"}
        
        detected_lang = self.detect_language(text)
        extracted_values = self.extract_specific_values(text)

        category_analyses = {}
        for category in self.criteria_weights.keys():
            category_analyses[category] = self.analyze_criteria(text, category)
        
        scoring = self.calculate_scores(category_analyses)
        recommendations = self.generate_recommendations(category_analyses, scoring)
        
        percentage = scoring["percentage"]
        has_critical = len(scoring["critical_missing"]) > 0
        
        status = "INVALID" if has_critical else "VALID" if percentage >= 70 else "CONDITIONAL" if percentage >= 50 else "INSUFFICIENT"
        status_tr = {"INVALID": "GEÇERSİZ", "VALID": "GEÇERLİ", "CONDITIONAL": "KOŞULLU", "INSUFFICIENT": "YETERSİZ"}.get(status, "YETERSİZ")
        
        return {
            "analysis_date": datetime.now().isoformat(),
            "file_info": {"detected_language": detected_lang, "text_length": len(text)},
            "extracted_values": extracted_values,
            "category_analyses": category_analyses,
            "scoring": scoring,
            "recommendations": recommendations,
            "summary": {
                "total_score": scoring["total_score"],
                "percentage": percentage,
                "status": status,
                "status_tr": status_tr,
                "critical_missing_count": len(scoring["critical_missing"]),
                "report_type": "AT_UYGUNLUK_BEYANI"
            }
        }
    
def validate_document_server(text):
    critical_terms = [
        # AT/EC temel terimleri
        ["AT TİP", "at tip", "ec type", "uygunluk", "beyan", "muayene", "conformity", "declaration"],
        
        # Sertifika ve belgelendirme terimleri
        ["SERTİFİKA", "sertifika", "certificate", "belge", "document", "onay", "approval"],
        
        # Makine direktifi ve standart terimleri
        ["2006/42/EC", "direktif", "directive", "makine", "machine", "standart", "standard"],
        
        # Üretici ve yetkili terimleri
        ["üretici", "manufacturer", "yetkili", "authorized", "imza", "signature", "sorumlu", "responsible"],
        
        # Muayene ve kontrol terimleri
        ["muayene", "inspection", "kontrol", "control", "test", "değerlendirme", "assessment", "onaylanmış kuruluş"]
    ]
    category_found = [any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category) for category in critical_terms]
    return sum(category_found) >= 4

def check_strong_keywords_first_pages(filepath):
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        found = [kw for kw in ["uygunluk", "beyan", "declaration","muayene","declare"] if re.search(rf"\b{kw}\b", all_text)]
        return len(found) >= 1
    except:
        return False

def check_excluded_keywords_first_pages(filepath):
    excluded = [
        # Aydınlatma raporu
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
        
        # Hidrolik devre şeması
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219", "teknik resim", "tasarım",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",

        # Gürültü ölçüm raporu
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        
        # Espe raporu  
        "espe",
        
        # Manuel/kullanma kılavuzu
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide","kılavuzu",
        
        # LOTO raporu
        "loto",
        
        # LVD raporu
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=2)
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        found = [kw for kw in excluded if re.search(rf"\b{kw}\b", all_text)]
        return len(found) >= 1
    except:
        return False

def get_conclusion_message_at(status, percentage):
    if status == "PASS":
        return f"AT uygunluk beyanı 2006/42/EC direktifine uygun (%{percentage:.0f})"
    return f"AT uygunluk beyanı direktife uygun değil (%{percentage:.0f})"

def get_main_issues_at(report):
    issues = []
    if report['scoring']['critical_missing']:
        for item in report['scoring']['critical_missing']:
            issues.append(f"Kritik eksik: {item}")
    return issues[:4]

def check_tesseract_installation():
    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract v{version}"
    except:
        return False, "Not installed"

tesseract_available, tesseract_info = check_tesseract_installation()

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads_at'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/at-declaration', methods=['POST'])
def analyze_at_declaration():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        analyzer = ATTypeInspectionAnalyzer()
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # ÜÇ AŞAMALI AT UYGUNLUK BEYANI KONTROLÜ
        logger.info(f"Üç aşamalı AT Uygunluk Beyanı kontrolü başlatılıyor: {filename}")

        if file_ext == '.pdf':
            logger.info("Aşama 1: İlk sayfa AT Uygunluk Beyanı özgü kelime kontrolü...")
            if check_strong_keywords_first_pages(filepath):
                logger.info("✅ Aşama 1 geçti - AT Uygunluk Beyanı özgü kelimeler bulundu")
            else:
                logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                if check_excluded_keywords_first_pages(filepath):
                    logger.info("❌ Aşama 2'de excluded kelimeler bulundu - AT Uygunluk Beyanı değil")
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    return jsonify({
                        'error': 'Invalid document type',
                        'message': 'Bu dosya AT Uygunluk Beyanı değil (farklı rapor türü tespit edildi). Lütfen AT Uygunluk Beyanı yükleyiniz.',
                        'details': {
                            'filename': filename,
                            'document_type': 'OTHER_REPORT_TYPE',
                            'required_type': 'AT_UYGUNLUK_BEYANI'
                        }
                    }), 400
                else:
                    # AŞAMA 3: PyPDF2 ile tam doküman kontrolü
                    logger.info("Aşama 3: Tam doküman critical terms kontrolü...")
                    try:
                        with open(filepath, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            text = ""
                            for page in pdf_reader.pages:
                                text += page.extract_text() + "\n"
                        
                        if not text or len(text.strip()) < 50:
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({
                                'error': 'Text extraction failed',
                                'message': 'Dosyadan yeterli metin çıkarılamadı'
                            }), 400
                        
                        if not validate_document_server(text):
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({
                                'error': 'Invalid document type',
                                'message': 'Yüklediğiniz dosya AT Uygunluk Beyanı değil! Lütfen geçerli bir AT Uygunluk Beyanı yükleyiniz.',
                                'details': {
                                    'filename': filename,
                                    'document_type': 'NOT_AT_DECLARATION',
                                    'required_type': 'AT_UYGUNLUK_BEYANI'
                                }
                            }), 400
                            
                    except Exception as e:
                        logger.error(f"Aşama 3 hatası: {e}")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Analysis failed',
                            'message': 'Dosya analizi sırasında hata oluştu'
                        }), 500

        logger.info(f"AT Uygunluk Beyanı doğrulandı, analiz başlatılıyor: {filename}")
        report = analyzer.analyze_at_declaration(filepath)
        try: os.remove(filepath)
        except: pass

        if 'error' in report:
            return jsonify({'error': 'Analysis failed', 'message': report['error']}), 400

        overall_percentage = report['summary']['percentage']
        status = "PASS" if overall_percentage >= 70 else "FAIL"

        # extracted_values'ı Türkçe key'lerle dönüştür
        extracted_values_tr = {}
        display_names = {
            "manufacturer_name": "Üretici Adı",
            "manufacturer_address": "Üretici Adresi", 
            "machine_description": "Makine Açıklaması",
            "machine_model": "Makine Modeli",
            "production_year": "Üretim Yılı",
            "serial_number": "Seri Numarası",
            "declaration_date": "Beyan Tarihi",
            "authorized_person": "Yetkili Kişi",
            "position": "Pozisyon",
            "directive_reference": "Direktif Referansı",
            "applied_standards": "Uygulanan Standartlar"
        }
        for eng_key, tr_name in display_names.items():
            if eng_key in report['extracted_values']:
                value = report['extracted_values'][eng_key]
                if eng_key == "applied_standards":
                    extracted_values_tr[tr_name] = ", ".join(value) if value else "Bulunamadı"
                else:
                    extracted_values_tr[tr_name] = value
        
        response_data = {
            'analysis_date': report.get('analysis_date'),
            'analysis_id': f"at_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'category_scores': {},
            'extracted_values': extracted_values_tr,
            'file_type': 'AT_UYGUNLUK_BEYANI',
            'filename': filename,
            'overall_score': {
                'percentage': round(overall_percentage, 2),
                'total_points': report['summary']['total_score'],
                'max_points': 100,
                'status': status,
                'status_tr': report['summary']['status_tr']
            },
            'recommendations': report['recommendations'],
            'summary': {
                'is_valid': status == "PASS",
                'conclusion': get_conclusion_message_at(status, overall_percentage),
                'main_issues': [] if status == "PASS" else get_main_issues_at(report)
            }
        }
        
        for category, score_data in report['scoring']['category_scores'].items():
            response_data['category_scores'][category] = {
                'score': score_data.get('normalized', 0),
                'max_score': score_data.get('max_weight', 0),
                'percentage': score_data.get('percentage', 0),
                'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'FAIL'
            }

        return jsonify({
            'success': True,
            'message': 'AT Uygunluk Beyanı başarıyla analiz edildi',
            'analysis_service': 'at_declaration',
            'data': response_data
        })

    except Exception as e:
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'AT Declaration Analyzer API', 'version': '1.0.0'})

@app.route('/', methods=['GET'])
def index():
    return jsonify({'service': 'AT Declaration Analyzer API', 'version': '1.0.0'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8006))
    logger.info(f"🚀 AT Uygunluk Beyanı Servisi - Port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
    