# ============================================
# MONTAJ TALİMATLARI ANALİZ SERVİSİ
# Standalone Service - Azure App Service Ready
# Port: 8012
# ============================================

# ============================================
# IMPORTS
# ============================================
import os
import re
from datetime import datetime
from typing import Dict, List, Any
import PyPDF2
from dataclasses import dataclass
import logging
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pdf2image

# ============================================
# LOGGING CONFIGURATION
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# LANGUAGE DETECTION
# ============================================
try:
    from langdetect import detect
    LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False
    logger.warning("langdetect modülü bulunamadı - dil tespiti devre dışı")

# ============================================
# TESSERACT CHECK
# ============================================
def check_tesseract_installation():
    """Check if Tesseract is properly installed"""
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract OCR kurulu - Sürüm: {version}")
        return True, f"Tesseract v{version}"
    except Exception as e:
        logger.error(f"Tesseract OCR kurulu değil: {e}")
        return False, str(e)

tesseract_available, tesseract_info = check_tesseract_installation()

# ============================================
# ANALİZ SINIFI - DATA CLASSES
# ============================================
@dataclass
class ManualAnalysisResult:
    """Kullanma Kılavuzu analiz sonucu veri sınıfı"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

# ============================================
# ANALİZ SINIFI - MAIN ANALYZER
# ============================================
class ManualReportAnalyzer:
    """Kullanma Kılavuzu rapor analiz sınıfı"""
    
    def __init__(self):
        logger.info("Kullanma Kılavuzu analiz sistemi başlatılıyor...")
        
        self.criteria_weights = {
            "Genel Bilgiler": 10,
            "Giriş ve Amaç": 5,
            "Güvenlik Bilgileri": 15,
            "Ürün Tanıtımı": 10,
            "Kurulum ve Montaj Bilgileri": 15,
            "Kullanım Talimatları": 20,
            "Bakım ve Temizlik": 10,
            "Arıza Giderme": 15
        }
        
        self.criteria_details = {
            "Genel Bilgiler": {
                "kilavuz_adi_kod": {"pattern": r"(?:Kılavuz|Manual|Guide|Kullan[ıi]m\s*K[ıi]lavuzu|User\s*Manual|Operating\s*Manual)", "weight": 5},
                "urun_modeli": {"pattern": r"(?:Ürün|Product|Model|Seri\s*No|Serial\s*Number|Part\s*Number)", "weight": 3},
                "revizyon_bilgisi": {"pattern": r"(?:Revizyon|Revision|Rev\.?|Version|v)\s*[:=]?\s*(\d+|[A-Z])", "weight": 2}
            },
            "Giriş ve Amaç": {
                "kilavuz_amaci": {"pattern": r"(?:Amaç|Purpose|Objective|Bu\s*k[ıi]lavuz|This\s*manual|Introduction|Giriş)", "weight": 3},
                "kapsam": {"pattern": r"(?:Kapsam|Scope|Coverage|Bu\s*dokuman|This\s*document)", "weight": 2}
            },
            "Güvenlik Bilgileri": {
                "genel_guvenlik": {"pattern": r"(?:Güvenlik|Safety|Güvenlik\s*Uyar[ıi]s[ıi]|Safety\s*Warning|UYARI|WARNING|DİKKAT|CAUTION)", "weight": 4},
                "tehlikeler": {"pattern": r"(?:Tehlike|Hazard|Risk|Tehlikeli|Dangerous|Yaralanma|Injury)", "weight": 4},
                "guvenlik_prosedur": {"pattern": r"(?:Prosedür|Procedure|Güvenlik\s*Prosedür|Safety\s*Procedure|Uyulmas[ıi]\s*gereken)", "weight": 3},
                "kkd_gerekliligi": {"pattern": r"(?:KKD|PPE|Personal\s*Protective|Koruyucu\s*Donanım|Protective\s*Equipment|Eldiven|Glove|Gözlük|Goggle)", "weight": 4}
            },
            "Ürün Tanıtımı": {
                "urun_tanimi": {"pattern": r"(?:Ürün\s*Tan[ıi]m[ıi]|Product\s*Description|Genel\s*Tan[ıi]m|General\s*Description)", "weight": 3},
                "teknik_ozellikler": {"pattern": r"(?:Teknik\s*Özellik|Technical\s*Specification|Specification|Özellik|Feature)", "weight": 3},
                "bilesenler": {"pattern": r"(?:Bileşen|Component|Parça|Part|Liste|List|İçerik|Content)", "weight": 2},
                "gorseller": {"pattern": r"(?:Görsel|Image|Resim|Picture|Şekil|Figure|Fotoğraf|Photo)", "weight": 2}
            },
            "Kurulum ve Montaj Bilgileri": {
                "kurulum_oncesi": {"pattern": r"(?:Kurulum\s*Öncesi|Before\s*Installation|Hazırl[ıi]k|Preparation|Ön\s*hazırl[ıi]k)", "weight": 4},
                "montaj_talimatlari": {"pattern": r"(?:Montaj|Installation|Assembly|Ad[ıi]m|Step|Talimat|Instruction)", "weight": 4},
                "gerekli_aletler": {"pattern": r"(?:Alet|Tool|Malzeme|Material|Gerekli|Required|Equipment)", "weight": 3},
                "kurulum_kontrolu": {"pattern": r"(?:Kontrol|Check|Test|Doğrula|Verify|Kurulum\s*Sonras[ıi]|After\s*Installation)", "weight": 4}
            },
            "Kullanım Talimatları": {
                "calistirma": {"pattern": r"(?:Çal[ıi]şt[ıi]rma|Start|Operation|Açma|Turn\s*On|Power\s*On)", "weight": 5},
                "kullanim_kilavuzu": {"pattern": r"(?:Kullan[ıi]m|Usage|Use|Operating|Ad[ıi]m\s*ad[ıi]m|Step\s*by\s*step)", "weight": 5},
                "calisma_modlari": {"pattern": r"(?:Mod|Mode|Ayar|Setting|Çal[ıi]şma\s*Mod|Operating\s*Mode)", "weight": 5},
                "kullanim_ipuclari": {"pattern": r"(?:İpucu|Tip|Öneri|Recommendation|Doğru\s*kullan[ıi]m|Proper\s*use)", "weight": 5}
            },
            "Bakım ve Temizlik": {
                "duzenli_bakim": {"pattern": r"(?:Bak[ıi]m|Maintenance|Düzenli|Regular|Periyodik|Periodic)", "weight": 3},
                "temizlik_yontemleri": {"pattern": r"(?:Temizlik|Cleaning|Temizle|Clean|Hijyen|Hygiene)", "weight": 3},
                "parca_degisimi": {"pattern": r"(?:Parça\s*Değiş|Part\s*Replace|Yedek\s*Parça|Spare\s*Part|Değiştir|Replace)", "weight": 4}
            },
            "Arıza Giderme": {
                "sorun_cozumleri": {"pattern": r"(?:Sorun|Problem|Ar[ıi]za|Fault|Troubleshoot|Çözüm|Solution)", "weight": 5},
                "hata_kodlari": {"pattern": r"(?:Hata\s*Kod|Error\s*Code|Kod|Code|Alarm|Uyar[ıi]\s*Lambas[ıi]|Warning\s*Light)", "weight": 5},
                "teknik_destek": {"pattern": r"(?:Teknik\s*Destek|Technical\s*Support|Destek|Support|İletişim|Contact|Tel|Phone|E-?mail)", "weight": 3},
                "teknik_cizimler": {"pattern": r"(?:Çizim|Drawing|Şema|Scheme|Diyagram|Diagram|Plan)", "weight": 2}
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Metin dilini tespit et"""
        if not LANGUAGE_DETECTION_AVAILABLE:
            return 'tr'
        
        try:
            sample_text = text[:500].strip()
            if not sample_text:
                return 'tr'
                
            detected_lang = detect(sample_text)
            logger.info(f"Tespit edilen dil: {detected_lang}")
            return detected_lang
            
        except Exception as e:
            logger.warning(f"Dil tespiti başarısız: {e}")
            return 'tr'
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkarma"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    page_text = re.sub(r'\s+', ' ', page_text)
                    text += page_text + "\n"
                
                text = text.strip()
                
                if len(text) > 50:
                    logger.info("Metin PyPDF2 ile çıkarıldı")
                    return text
                
                logger.info("PyPDF2 ile yeterli metin bulunamadı, OCR deneniyor...")
                return self.extract_text_with_ocr(pdf_path)
                
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {e}")
            return self.extract_text_with_ocr(pdf_path)

    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """OCR ile metin çıkarma"""
        try:
            images = convert_from_path(pdf_path, dpi=300)
            
            all_text = ""
            for i, image in enumerate(images):
                try:
                    text = pytesseract.image_to_string(image, lang='tur+eng')
                    text = re.sub(r'\s+', ' ', text)
                    all_text += text + "\n"
                    logger.info(f"OCR ile sayfa {i+1}'den {len(text)} karakter çıkarıldı")
                except Exception as page_error:
                    logger.error(f"Sayfa {i+1} OCR hatası: {page_error}")
                    continue
            
            return all_text.strip()
            
        except Exception as e:
            logger.error(f"OCR metin çıkarma hatası: {e}")
            return ""

    def extract_text_from_docx(self, docx_path: str) -> str:
        """DOCX'den metin çıkar"""
        try:
            from docx import Document
            doc = Document(docx_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"DOCX metin çıkarma hatası: {e}")
            return ""

    def extract_text_from_txt(self, txt_path: str) -> str:
        """TXT'den metin çıkar"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"TXT metin çıkarma hatası: {e}")
            return ""
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, ManualAnalysisResult]:
        """Kriterleri analiz et"""
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                content = f"Found: {str(matches[:3])}"
                found = True
                score = weight
            else:
                content = "Not found"
                found = False
                score = 0
            
            results[criterion_name] = ManualAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_count": len(matches) if matches else 0}
            )
        
        return results
    
    def calculate_scores(self, analysis_results: Dict[str, Dict[str, ManualAnalysisResult]]) -> Dict[str, Any]:
        """Puanlama hesaplama"""
        category_scores = {}
        total_score = 0
        
        for category, results in analysis_results.items():
            category_max = self.criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())
            
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
            "percentage": round(total_score, 2)
        }
    
    def extract_specific_values(self, text: str) -> Dict[str, Any]:
        """Özel değerleri çıkar"""
        values = {
            "manual_name": "Bulunamadı",
            "product_model": "Bulunamadı",
            "safety_warnings_count": 0
        }
        
        manual_patterns = [
            r"(?:Kullan[ıi]m\s*K[ıi]lavuzu|User\s*Manual|Operating\s*Manual|Manual)",
        ]
        
        for pattern in manual_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["manual_name"] = match.group(0).strip()
                break
        
        model_patterns = [
            r"(?:Model|Product)\s*(?:No)?\s*[:\-]?\s*([A-Z0-9\-\.]{3,20})",
        ]
        
        for pattern in model_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["product_model"] = match.group(1).strip()
                break
        
        safety_patterns = [r"(?:UYARI|WARNING|DİKKAT|CAUTION|Güvenlik)"]
        safety_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in safety_patterns)
        values["safety_warnings_count"] = safety_count
        
        return values
    
    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        
        for category, score_data in scores["category_scores"].items():
            if score_data["percentage"] < 50:
                recommendations.append(f"⚠️ {category} kategorisinde ciddi eksiklikler var (%{score_data['percentage']:.0f})")
            elif score_data["percentage"] < 70:
                recommendations.append(f"📝 {category} kategorisi geliştirilebilir (%{score_data['percentage']:.0f})")
        
        return recommendations
    
    def analyze_manual(self, file_path: str) -> Dict[str, Any]:
        """Ana analiz fonksiyonu"""
        logger.info("Kullanım kılavuzu analizi başlıyor...")
        
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_ext in ['.docx', '.doc']:
                text = self.extract_text_from_docx(file_path)
            elif file_ext == '.txt':
                text = self.extract_text_from_txt(file_path)
            else:
                return {"error": f"Desteklenmeyen dosya formatı: {file_ext}"}
            
            if len(text.strip()) < 50:
                return {"error": "Yeterli metin çıkarılamadı", "text_length": len(text)}
            
            detected_language = self.detect_language(text)
            extracted_values = self.extract_specific_values(text)
            
            category_analyses = {}
            for category in self.criteria_weights.keys():
                category_analyses[category] = self.analyze_criteria(text, category)
            
            scoring = self.calculate_scores(category_analyses)
            recommendations = self.generate_recommendations(category_analyses, scoring)
            
            percentage = scoring["percentage"]
            status = "PASS" if percentage >= 70 else "FAIL"
            status_tr = "GEÇERLİ" if percentage >= 70 else "YETERSİZ"
            
            return {
                "analysis_date": datetime.now().isoformat(),
                "file_info": {
                    "filename": os.path.basename(file_path),
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
                    "report_type": "Kullanım Kılavuzu"
                }
            }
            
        except Exception as e:
            logger.error(f"Analiz hatası: {e}")
            return {"error": f"Analiz sırasında hata oluştu: {str(e)}"}

# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_document_server(text):
    """Server document validation"""
    critical_terms = [
        # Montaj temel terimleri (en az 1 tane olmalı)
        ["montaj", "assembly", "kurulum", "installation", "talimat", "instruction", "kılavuz", "manual"],
        
        # Adımlar/Prosedür terimleri (en az 1 tane olmalı)  
        ["adım", "step", "prosedür", "procedure", "sıralama", "sequence", "önce", "before", "sonra", "after"],
        
        # Araçlar/Malzemeler terimleri (mutlaka olmalı)
        ["araç", "tool", "malzeme", "material", "gerekli", "required", "parça", "part", "bileşen", "component"],
        
        # Güvenlik/Uyarı terimleri (en az 1 tane olmalı)
        ["güvenlik", "safety", "uyarı", "warning", "dikkat", "caution", "tehlike", "danger", "önlem", "precaution"]
    ]
    
    category_found = [
        any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category)
        for category in critical_terms
    ]
    
    valid_categories = sum(category_found)
    logger.info(f"Doküman validasyonu: {valid_categories}/4 kritik kategori bulundu")
    
    return valid_categories >= 4


def check_strong_keywords_first_pages(filepath):
    """Check strong keywords"""
    strong_keywords = [
        "montaj",
        "assembly",
        "kurulum",
        "installation",
        "talimat",
        "instruction",
        "kılavuz",
        "manual",
        "kılavuzu",
        "kılavuzun",
        "kullanma,"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=200, first_page=1, last_page=1)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        
        found_keywords = [kw for kw in strong_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa kontrol: {len(found_keywords)} özgü kelime")
        return len(found_keywords) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa kontrol hatası: {e}")
        return False


def check_excluded_keywords_first_pages(filepath):
    """Check excluded keywords"""
    excluded_keywords = [
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        
        # Espe raporu  
        "espe",
        
        # Hidrolik devre şeması
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219","teknik resim","tasarım",
        
        # Gürültü ölçüm raporu
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # LOTO raporu
        "loto",
        
        # LVD raporu
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        
        # AT tip muayene (AT uygunluk beyanı)
        "uygunluk", "beyan", "muayene", "conformity", "declaration",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service", "bakim", "MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",

        # Aydınlatma
        "aydınlatma", "lighting",  "illumination",  "lux",  "lümen",  "lumen",  "ts en 12464",  "en 12464", "ışık",  "ışık şiddeti",
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=200, first_page=1, last_page=2)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        
        found_excluded = [kw for kw in excluded_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa excluded kontrol: {len(found_excluded)} istenmeyen kelime")
        return len(found_excluded) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa excluded kontrol hatası: {e}")
        return False


def get_conclusion_message(status, percentage):
    """Get conclusion message"""
    if status == "PASS":
        return f"Montaj talimatları yüksek kalitede ve standartlara uygun (%{percentage:.0f})"
    else:
        return f"Montaj talimatları yetersiz, kapsamlı revizyon gerekli (%{percentage:.0f})"


def get_main_issues(analysis_result):
    """Get main issues"""
    issues = []
    
    for category, score_data in analysis_result['scoring']['category_scores'].items():
        if score_data['percentage'] < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    if not issues and analysis_result['scoring']['total_score'] < 50:
        issues = [
            "Güvenlik bilgileri yetersiz",
            "Montaj adımları eksik veya belirsiz",
            "Gerekli araçlar ve malzemeler belirtilmemiş"
        ]
    
    return issues[:4]


# ============================================
# FLASK SERVICE
# ============================================
app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads_montaj'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'docx', 'doc', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/assembly-instructions', methods=['POST'])
def analyze_manual():
    """Montaj Talimatları analiz endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        analyzer = ManualReportAnalyzer()
        analysis_result = analyzer.analyze_manual(filepath)
        
        try:
            os.remove(filepath)
        except:
            pass

        if 'error' in analysis_result:
            return jsonify({'error': 'Analysis failed', 'message': analysis_result['error']}), 400

        overall_percentage = analysis_result['summary']['percentage']
        status = "PASS" if overall_percentage >= 70 else "FAIL"
        
        response_data = {
            'analysis_date': analysis_result.get('analysis_date'),
            'analysis_id': f"montaj_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'category_scores': {},
            'extracted_values': analysis_result.get('extracted_values', {}),
            'file_type': 'MONTAJ_TALIMATLARI',
            'filename': filename,
            'overall_score': {
                'percentage': round(overall_percentage, 2),
                'total_points': analysis_result['scoring']['total_score'],
                'max_points': 100,
                'status': status,
                'status_tr': analysis_result['summary']['status_tr']
            },
            'recommendations': analysis_result.get('recommendations', []),
            'summary': {
                'is_valid': status == "PASS",
                'conclusion': get_conclusion_message(status, overall_percentage),
                'main_issues': [] if status == "PASS" else get_main_issues(analysis_result)
            }
        }
        
        for category, score_data in analysis_result['scoring']['category_scores'].items():
            response_data['category_scores'][category] = {
                'score': score_data['normalized'],
                'max_score': score_data['max_weight'],
                'percentage': score_data['percentage'],
                'status': 'PASS' if score_data['percentage'] >= 70 else 'FAIL'
            }

        return jsonify({
            'success': True,
            'message': 'Montaj Talimatları başarıyla analiz edildi',
            'analysis_service': 'montaj_talimatları',
            'data': response_data
        })

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': 'Server error', 'message': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'Montaj Talimatları Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'report_type': 'MONTAJ_TALIMATLARI'
    })


@app.route('/', methods=['GET'])
def index():
    """Index page"""
    return jsonify({
        'service': 'Montaj Talimatları Analyzer API',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/assembly-instructions': 'Montaj talimatları analizi',
            'GET /api/health': 'Health check',
            'GET /': 'API info'
        }
    })


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Montaj Talimatları Analiz Servisi")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8012))
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)