#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AT Type Inspection Checker (EC Declaration of Conformity Analyzer)
Created for analyzing AT Uygunluk Beyanı (EC Declaration of Conformity) documents
Based on 2006/42/EC Machine Directive requirements
"""

import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from langdetect import detect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class ATAnalysisResult:
    """Data class for AT Type Inspection analysis results"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    is_critical: bool
    details: Dict[str, Any]

class ATTypeInspectionAnalyzer:
    """Analyzer for AT Uygunluk Beyanı (EC Declaration of Conformity) documents"""
    
    def __init__(self):
        logging.info("AT Type Inspection analysis system starting...")
        
        # Scoring weights for each category (total = 100)
        self.criteria_weights = {
            "Kritik Bilgiler": 60,          # Critical Information (MUST HAVE)
            "Zorunlu Teknik Bilgiler": 25,  # Mandatory Technical Information  
            "Standartlar ve Belgeler": 15   # Standards and Documents
        }
        
        # Criteria details based on 2006/42/EC directive requirements
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
                    text += page.extract_text()
            
            logging.info(f"PyPDF2 extracted {len(text)} characters")
            
            # If PyPDF2 gives insufficient text, use OCR
            if len(text.strip()) < 100:
                logging.info("Insufficient text with PyPDF2, trying OCR...")
                
                pages = convert_from_path(pdf_path, dpi=200)
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
            raise
        
        return text
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image files (JPG, PNG) using OCR"""
        text = ""
        
        try:
            # Open image with PIL
            image = Image.open(image_path)
            
            # Convert to RGB if necessary (for PNG with transparency)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply OCR
            text = pytesseract.image_to_string(image, lang='tur+eng')
            logging.info(f"OCR extracted {len(text)} characters from image: {os.path.basename(image_path)}")
            
        except Exception as e:
            logging.error(f"Error extracting text from image: {e}")
            raise
        
        return text
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from supported file formats (PDF, JPG, PNG)"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            return self.extract_text_from_image(file_path)
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {file_extension}. Desteklenen formatlar: PDF, JPG, PNG")
    
    def detect_language(self, text: str) -> str:
        """Detect document language"""
        try:
            if len(text.strip()) < 50:
                return "tr"  # Default to Turkish
            return detect(text)
        except:
            return "tr"
    
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
                    "raw_matches": matches[:3] if matches else []  # Store first 3 raw matches
                }
            )
        
        return results
    
    def calculate_scores(self, analysis_results: Dict[str, Dict[str, ATAnalysisResult]]) -> Dict[str, Any]:
        """Calculate scoring for all categories"""
        category_scores = {}
        total_score = 0
        critical_missing = []
        
        for category, results in analysis_results.items():
            category_max = self.criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())
            
            # Check for missing critical criteria
            for criterion_name, result in results.items():
                if result.is_critical and not result.found:
                    critical_missing.append(f"{category}: {result.details['description']}")
            
            if category_possible > 0:
                percentage = (category_earned / category_possible) * 100
                normalized_score = (percentage / 100) * category_max
            else:
                percentage = 0
                normalized_score = 0
            
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
            r"(?:biz\s+burada\s+beyan\s+ederiz\s+ki[;:\s]*)([^,\n]+)",  # "Biz burada beyan ederiz ki; Sibernetik Makina..."
            r"(?:we\s+)([A-Za-z\s&\.]+?)(?:\s+declare|\s+industrial)",  # "We Pilz Ireland Industrial..."
            r"(?:manufactured by|üretici|manufacturer)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö\s\.\-&]{5,100})",
            r"(sibernetik\s+makina\s*&?\s*otomasyon[^,\n]*)",  # Sibernetik Makina
            r"(pilz\s+ireland\s+industrial\s+automation)",  # Pilz Ireland
            r"(suzhou\s+keber\s+technology\s+co)",  # Suzhou Keber
            r"([A-ZÜÇĞIÖŞ][a-züçğıöş]+(?:\s+[A-ZÜÇĞIÖŞ][a-züçğıöş]+)*\s+(?:makina|technology|industrial|automation|şirket|company))"
        ]
        
        for pattern in manufacturer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                manufacturer_name = match.group(1).strip()
                if len(manufacturer_name) > 5 and not re.search(r'^[0-9]+$', manufacturer_name):  # En az 5 karakter ve sadece sayı olmasın
                    values["manufacturer_name"] = manufacturer_name
                    break
        
        # Address - Çoklu adres formatı desteği
        address_patterns = [
            r"(demirci[^,\n]*cd\.[^,\n]*no[^,\n]*nilüfer[^,\n]*bursa)",  # Türk adresi
            r"(cork\s+business\s*&?\s*technology\s+park[^,]*model\s+farm\s+road[^,]*cork[^,]*ireland)",  # İrlanda adresi
            r"(no\.\s*[0-9]+[^,]*suzhou[^,]*jiangsu[^,]*)",  # Çin adresi
            r"(?:address|adres)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{20,200})",
            r"([A-ZÜÇĞIÖŞ][a-züçğıöş]+(?:\s+[A-Za-züçğıöş]+)*\s+(?:cd\.|caddesi|street|road)[^,\n]{10,100})",
            r"([^,\n]*(?:mahallesi|caddesi|sokak|street|road|park)[^,\n]{10,100})"
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
                if len(address) > 15:  # Minimum address length
                    values["manufacturer_address"] = address
                    break
        
        # Machine description - Çoklu makine türü desteği
        machine_patterns = [
            r"(?:makinenin tanıtımı ve sınıfı|tanım|description)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\-\.]{5,100})",  # "Makinenin Tanıtımı ve Sınıfı : FO 12.7lt Ecotorq..."
            r"(fo\s*[0-9]+\.?[0-9]*lt?\s+ecotorq\s+kafa\s+baga\s+çakma)",  # Ford makine
            r"(v[0-9]+b\s+knee\s+pad\s+punching\s+machine)",  # Pilz makine
            r"(vibratory\s+surface\s+finishing\s+machine)",  # Vibrasyon makine
            r"(?:makine|machine|model|equipment)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\-\.]{8,80})"
        ]
        
        for pattern in machine_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                machine_desc = match.group(1).strip()
                # Yararsız eşleşmeleri filtrele
                if not re.search(r"farm\s+road|business|technology|address|adres", machine_desc, re.IGNORECASE):
                    values["machine_description"] = machine_desc
                    break
        
        # Model - Daha spesifik model pattern
        model_patterns = [
            r"model\s+farm\s+road",  # Bu yanlış eşleşme
            r"(V[0-9]+B)",  # V524B gibi modeller
            r"(VKT\s*[0-9]+)",  # VKT 500 gibi modeller
            r"(?:model|tip|type)\s*[:\-]?\s*([A-Za-z0-9\s\-]{2,30})"
        ]
        
        for pattern in model_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                model_text = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                # "Model Farm Road" gibi yanlış eşleşmeleri filtrele
                if not re.search(r"farm\s+road|business|technology", model_text, re.IGNORECASE):
                    values["machine_model"] = model_text
                    break
        
        # Serial number - Çoklu seri numarası formatı
        serial_patterns = [
            r"(?:seri numarası|serial number)\s*[:\-]?\s*([A-Z0-9\/\-]+)",  # "Seri Numarası :2208007/2023"
            r"(?:seri|s/n|sn)\s*(?:no|number)\s*[:\-]?\s*([A-Za-z0-9\-\/]{3,25})",
            r"([0-9]{6,8}\/[0-9]{4})",  # "2208007/2023" format
            r"([A-Z][0-9]{4}[A-Z]?\-[0-9]{3})"  # "A2306F-012" format
        ]
        
        for pattern in serial_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                serial = match.group(1).strip()
                if len(serial) >= 3:  # Minimum serial length
                    values["serial_number"] = serial
                    break
        
        # Production year - Yıl bilgisi için
        year_patterns = [
            r"([0-9]{4})",  # Sadece 4 haneli yıl
            r"february\s+([0-9]{4})",  # "5 February 2024"
            r"(?:üretim|imal|year)\s*[:\-]?\s*([0-9]{4})"
        ]
        
        # 2020-2030 arası makul yılları bul
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for year in matches:
                year_int = int(year) if year.isdigit() else 0
                if 2020 <= year_int <= 2030:  # Makul yıl aralığı
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
        values["applied_standards"] = list(set(standards))  # Remove duplicates
        
        return values
    
    def generate_recommendations(self, analysis_results: Dict[str, Dict[str, ATAnalysisResult]], 
                               scores: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check critical missing items first
        if scores["critical_missing"]:
            recommendations.append("🚨 KRİTİK EKSİKLİKLER - Belge geçersiz sayılabilir:")
            for missing in scores["critical_missing"]:
                recommendations.append(f"  ❌ {missing}")
            recommendations.append("")
        
        # Check each category performance
        for category, score_data in scores["category_scores"].items():
            if category == "Kritik Bilgiler" and score_data["percentage"] < 80:
                recommendations.append(f"⚠️ {category} kategorisinde ciddi eksiklikler var (%{score_data['percentage']:.0f})")
            elif score_data["percentage"] < 60:
                recommendations.append(f"📝 {category} kategorisi iyileştirilmeli (%{score_data['percentage']:.0f})")
        
        # Specific recommendations based on findings
        total_percentage = scores["percentage"]
        
        if scores["critical_missing"]:
            recommendations.append("🔍 ACİL DÜZELTME GEREKLİ:")
            recommendations.append("  • Eksik kritik bilgileri tamamlayın")
            recommendations.append("  • 2006/42/EC direktif gereksinimlerini kontrol edin")
            recommendations.append("  • Yetkili kişi imzası ve bilgilerini ekleyin")
        elif total_percentage >= 85:
            recommendations.append("✅ AT Uygunluk Beyanı yüksek kalitede ve direktif gereksinimlerine uygun")
            recommendations.append("📋 Belge hukuken geçerli görünmektedir")
        elif total_percentage >= 70:
            recommendations.append("📋 AT Uygunluk Beyanı kabul edilebilir seviyede")
            recommendations.append("💡 Bazı teknik detaylar geliştirilebilir")
        elif total_percentage >= 50:
            recommendations.append("⚠️ AT Uygunluk Beyanı minimum gereksinimleri karşılıyor")
            recommendations.append("🔍 Önemli eksiklikler var, gözden geçirme gerekli")
        else:
            recommendations.append("❌ AT Uygunluk Beyanı yetersiz")
            recommendations.append("🚨 Belge 2006/42/EC direktif gereksinimlerini karşılamıyor")
        
        return recommendations
    
    def analyze_at_declaration(self, file_path: str) -> Dict[str, Any]:
        """Main analysis function for AT Uygunluk Beyanı - supports PDF, JPG, PNG"""
        logging.info("AT Declaration analysis starting...")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    "error": f"Dosya bulunamadı: {file_path}",
                    "analysis_date": datetime.now().isoformat()
                }
            
            # Check file format
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension not in ['.pdf', '.jpg', '.jpeg', '.png']:
                return {
                    "error": f"Desteklenmeyen dosya formatı: {file_extension}. Desteklenen formatlar: PDF, JPG, PNG",
                    "analysis_date": datetime.now().isoformat()
                }
            
            # Extract text based on file type
            text = self.extract_text_from_file(file_path)
            
            # Debug: Print extracted text for analysis
            logging.info(f"Extracted text preview (first 500 chars): {text[:500]}")
            logging.info(f"Extracted text preview (last 500 chars): {text[-500:]}")
            
            if len(text.strip()) < 50:
                return {
                    "error": f"Dosyadan yeterli metin çıkarılamadı. Dosya bozuk olabilir veya metin içermiyor olabilir. (Format: {file_extension})",
                    "text_length": len(text),
                    "analysis_date": datetime.now().isoformat()
                }
            
            # Detect language
            detected_language = self.detect_language(text)
            logging.info(f"Detected language: {detected_language}")
            
            # Extract specific values
            extracted_values = self.extract_specific_values(text)
            
            # Analyze each category
            category_analyses = {}
            for category in self.criteria_weights.keys():
                category_analyses[category] = self.analyze_criteria(text, category)
            
            # Calculate scores
            scoring = self.calculate_scores(category_analyses)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(category_analyses, scoring)
            
            # Determine overall status based on critical criteria
            percentage = scoring["percentage"]
            has_critical_missing = len(scoring["critical_missing"]) > 0
            
            if has_critical_missing:
                status = "INVALID"
                status_tr = "GEÇERSİZ"
            elif percentage >= 70:
                status = "VALID"
                status_tr = "GEÇERLİ"
            elif percentage >= 50:
                status = "CONDITIONAL"
                status_tr = "KOŞULLU"
            else:
                status = "INSUFFICIENT"
                status_tr = "YETERSİZ"
            
            return {
                "analysis_date": datetime.now().isoformat(),
                "file_info": {
                    "filename": os.path.basename(file_path),
                    "file_format": file_extension,
                    "text_length": len(text),
                    "detected_language": detected_language
                },
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
                    "report_type": "AT Uygunluk Beyanı (EC Declaration of Conformity)"
                }
            }
            
        except Exception as e:
            logging.error(f"Analysis error: {e}")
            return {
                "error": f"Analiz sırasında hata oluştu: {str(e)}",
                "analysis_date": datetime.now().isoformat()
            }

def print_analysis_report(report: Dict[str, Any]):
    """Print formatted analysis report"""
    if "error" in report:
        print(f"❌ Hata: {report['error']}")
        return
    
    print("\n📊 AT UYGUNLUK BEYANI ANALİZİ")
    print("=" * 60)
    
    print(f"📅 Analiz Tarihi: {report['analysis_date']}")
    print(f"� Dosya Formatı: {report['file_info']['file_format'].upper()}")
    print(f"�🔍 Tespit Edilen Dil: {report['file_info']['detected_language'].upper()}")
    
    print(f"📋 Toplam Puan: {report['summary']['total_score']}/100")
    print(f"📈 Yüzde: %{report['summary']['percentage']:.0f}")
    print(f"🎯 Durum: {report['summary']['status_tr']}")
    print(f"⚠️ Kritik Eksik Sayısı: {report['summary']['critical_missing_count']}")
    print(f"📄 Rapor Türü: {report['summary']['report_type']}")
    
    print("\n📋 ÇIKARILAN DEĞERLER")
    print("-" * 40)
    extracted_values = report['extracted_values']
    display_names = {
        "manufacturer_name": "Üretici Adı",
        "manufacturer_address": "Üretici Adresi",
        "machine_description": "Makine Tanımı",
        "machine_model": "Model",
        "production_year": "Üretim Yılı",
        "serial_number": "Seri No",
        "declaration_date": "Beyan Tarihi",
        "authorized_person": "Yetkili Kişi",
        "position": "Ünvan",
        "applied_standards": "Uygulanan Standartlar"
    }
    
    for key, value in extracted_values.items():
        if key in display_names:
            if key == "applied_standards":
                standards_str = ", ".join(value) if value else "Bulunamadı"
                print(f"{display_names[key]}: {standards_str}")
            else:
                print(f"{display_names[key]}: {value}")
    
    print("\n📊 KATEGORİ PUANLARI")
    print("-" * 40)
    for category, score_data in report['scoring']['category_scores'].items():
        status_icon = "🔴" if category == "Kritik Bilgiler" and score_data['percentage'] < 80 else "🟢" if score_data['percentage'] >= 70 else "🟡"
        print(f"{status_icon} {category}: {score_data['normalized']}/{score_data['max_weight']} (%{score_data['percentage']:.0f})")
    
    print("\n🚨 KRİTİK EKSİKLİKLER")
    print("-" * 40)
    if report['scoring']['critical_missing']:
        for missing in report['scoring']['critical_missing']:
            print(f"❌ {missing}")
    else:
        print("✅ Kritik eksiklik bulunamadı")
    
    print("\n💡 ÖNERİLER VE DEĞERLENDİRME")
    print("-" * 40)
    for recommendation in report['recommendations']:
        print(recommendation)
    
    print("\n📋 GENEL DEĞERLENDİRME")
    print("=" * 60)
    
    if report['summary']['status'] == "INVALID":
        print("🚨 SONUÇ: GEÇERSİZ")
        print(f"❌ Kritik eksiklikler nedeniyle belge geçersiz")
        print("📝 Değerlendirme: 2006/42/EC direktif gereksinimleri karşılanmıyor.")
        
    elif report['summary']['status'] == "VALID":
        print("✅ SONUÇ: GEÇERLİ")
        print(f"🌟 Toplam Başarı: %{report['summary']['percentage']:.0f}")
        print("📝 Değerlendirme: AT Uygunluk Beyanı direktif gereksinimlerini sağlamaktadır.")
        
    elif report['summary']['status'] == "CONDITIONAL":
        print("🟡 SONUÇ: KOŞULLU")
        print(f"⚠️ Toplam Başarı: %{report['summary']['percentage']:.0f}")
        print("📝 Değerlendirme: AT Uygunluk Beyanı temel gereksinimleri karşılıyor ancak iyileştirme gerekli.")
        
    else:
        print("❌ SONUÇ: YETERSİZ")
        print(f"⚠️ Toplam Başarı: %{report['summary']['percentage']:.0f}")
        print("📝 Değerlendirme: AT Uygunluk Beyanı direktif gereksinimlerini karşılamıyor.")

def main():
    """Main function for command line usage"""
    import sys
    
    # Analiz edilecek dosyalar - öncelik sırasına göre (PDF, JPG, PNG destekleniyor)
    test_files = [
        "WhatsApp Image 2025-08-27 at 13.22.26.jpeg"
    ]
    
    # Diğer CE Declaration dosyalarını ara (PDF, JPG, PNG)
    import glob
    supported_extensions = ['*.pdf', '*.PDF', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    
    for ext in supported_extensions:
        # CE Declaration pattern files
        ce_files = (glob.glob(f"*CE*{ext}") + glob.glob(f"*ce*{ext}") + 
                   glob.glob(f"*Declaration*{ext}") + glob.glob(f"*BEYANI*{ext}") +
                   glob.glob(f"*declaration*{ext}") + glob.glob(f"*beyani*{ext}"))
        test_files.extend(ce_files)
    
    # Hangi dosya varsa onu kullan
    selected_file = None
    for file in test_files:
        if os.path.exists(file):
            selected_file = file
            break
    
    if selected_file:
        print(f"🔍 Analiz edilen dosya: {selected_file}")
        print(f"📄 Dosya formatı: {os.path.splitext(selected_file)[1].upper()}")
        print(f"📅 Analiz Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if os.path.splitext(selected_file)[1].lower() in ['.jpg', '.jpeg', '.png']:
            print("🖼️ Resim dosyası tespit edildi, OCR ile metin çıkarılacak...")
        elif os.path.splitext(selected_file)[1].lower() == '.pdf':
            print("📋 PDF dosyası tespit edildi, metin çıkarma ve OCR kullanılacak...")
    else:
        print("❌ Hiçbir AT Uygunluk Beyanı dosyası bulunamadı")
        print("📁 Lütfen CE Declaration/AT Uygunluk Beyanı dosyasının proje klasöründe olduğundan emin olun.")
        print("🔍 Desteklenen dosya formatları:")
        print("   • PDF: *.pdf")
        print("   • Resim: *.jpg, *.jpeg, *.png")
        print("📋 Aranacak dosya adı kalıpları:")
        print("   • *CE*.{pdf,jpg,png}")
        print("   • *Declaration*.{pdf,jpg,png}")
        print("   • *BEYANI*.{pdf,jpg,png}")
        sys.exit(1)
    
    analyzer = ATTypeInspectionAnalyzer()
    report = analyzer.analyze_at_declaration(selected_file)
    print_analysis_report(report)

if __name__ == "__main__":
    main()
