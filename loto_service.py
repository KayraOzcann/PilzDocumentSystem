"""
LOTO Prosedürü Analiz Servisi
==============================
Azure App Service için optimize edilmiş standalone servis

Endpoint: POST /api/loto-report
Health Check: GET /api/health
"""

# ============================================
# IMPORTS
# ============================================
import re
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import PyPDF2
from dataclasses import dataclass, asdict
import logging
import cv2
import numpy as np
from PIL import Image
import pdf2image
import pytesseract

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# ============================================
# LOGGING
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
    logger.warning("langdetect not available")

# ============================================
# DATA CLASSES
# ============================================
@dataclass
class LOTOAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

# ============================================
# ANALYZER CLASS
# ============================================
class LOTOReportAnalyzer:
    def __init__(self):
        self.criteria_weights = {
            "Genel Rapor Bilgileri": 10,
            "Tesis ve Makine Tanımı": 10,
            "LOTO Politikası Değerlendirmesi": 10,
            "Enerji Kaynakları Analizi": 25,
            "İzolasyon Noktaları ve Prosedürler": 25,
            "Teknik Değerlendirme ve Sonuçlar": 15,
            "Dokümantasyon ve Referanslar": 5
        }
        
        self.criteria_details = {
            "Genel Rapor Bilgileri": {
                "proje_adi_belge_no": {"pattern": r"(?:Proje\s*Ad[ıi]|LOTO|Lockout|Tagout)", "weight": 2},
                "rapor_tarihi_versiyon": {"pattern": r"(?:Rapor\s*Tarihi|Report\s*Date|Tarih|Versiyon)", "weight": 2},
                "hazirlayan_firma": {"pattern": r"(?:Hazırlayan|Prepared\s*by|Company|Firma)", "weight": 2},
                "musteri_bilgileri": {"pattern": r"(?:Müşteri|Customer|Client|Tesis)", "weight": 2},
                "imza_onay": {"pattern": r"(?:İmza|Signature|Onay|Approval)", "weight": 2}
            },
            "Tesis ve Makine Tanımı": {
                "tesis_bilgileri": {"pattern": r"(?:Tesis|Facility|Plant|Factory)", "weight": 2},
                "makine_tanimi": {"pattern": r"(?:Makine|Machine|Equipment)", "weight": 2},
                "makine_teknik_bilgi": {"pattern": r"(?:Üretici|Manufacturer|Seri\s*No|Model)", "weight": 2},
                "makine_fotograflari": {"pattern": r"(?:Fotoğraf|Photo|Image|Görsel)", "weight": 2},
                "lokasyon_konumu": {"pattern": r"(?:Lokasyon|Location|Konum|Position)", "weight": 2}
            },
            "LOTO Politikası Değerlendirmesi": {
                "mevcut_politika": {"pattern": r"(?:Politika|Policy|LOTO\s*Policy|Prosedür)", "weight": 2},
                "politika_uygunluk": {"pattern": r"(?:Kontrol\s*Listesi|Checklist|Evet|Hayır)", "weight": 3},
                "prosedur_degerlendirme": {"pattern": r"(?:Prosedür|Procedure|Değerlendirme|Assessment)", "weight": 2},
                "personel_gorusme": {"pattern": r"(?:Personel|Personnel|Görüşme|Interview)", "weight": 2},
                "egitim_durumu": {"pattern": r"(?:Eğitim|Training|Education|Kurs)", "weight": 1}
            },
            "Enerji Kaynakları Analizi": {
                "enerji_kaynagi_tanimlama": {"pattern": r"(?:Enerji\s*Kaynağ[ıi]|Energy\s*Source|Elektrik|Pn[öo]matik|Hidrolik)", "weight": 6},
                "izolasyon_cihazi_bilgi": {"pattern": r"(?:İzolasyon\s*Cihaz[ıi]|Isolation.*?Device|Switch|Valve)", "weight": 6},
                "cihaz_durumu_kontrol": {"pattern": r"(?:Çalış[ıt][ıa]rılabilirlik|Kilitlenebilirlik|Lockable)", "weight": 6},
                "kilitleme_ekipman": {"pattern": r"(?:Kilit|Lock|Padlock|Etiket|Tag)", "weight": 4},
                "uygunsuz_enerji_tablosu": {"pattern": r"(?:Uygunsuz\s*Enerji|Hazardous.*?Energy)", "weight": 3}
            },
            "İzolasyon Noktaları ve Prosedürler": {
                "izolasyon_noktalari_tablo": {"pattern": r"(?:İzolasyon\s*Nokta|Isolation.*?Point|Layout|Şema)", "weight": 6},
                "prosedur_detaylari": {"pattern": r"(?:Prosedür\s*Detay|Procedure.*?Detail|Step.*?by.*?step)", "weight": 6},
                "mevcut_prosedur_analiz": {"pattern": r"(?:Mevcut\s*Prosedür|Current.*?Procedure)", "weight": 4},
                "tavsiyeler": {"pattern": r"(?:Tavsiye|Recommendation|Suggest|İyileştirme)", "weight": 5},
                "izolasyon_fotograflari": {"pattern": r"(?:İzolasyon.*?Fotoğraf|Lock.*?Tag)", "weight": 4}
            },
            "Teknik Değerlendirme ve Sonuçlar": {
                "kabul_edilebilirlik": {"pattern": r"(?:Kabul\s*Edilebilir|Acceptable|LOTO\s*Uygun|Evet|Hayır)", "weight": 4},
                "bulgular_yorumlar": {"pattern": r"(?:BULGULAR|FINDINGS|YORUMLAR|COMMENTS|Bulgu)", "weight": 3},
                "sonuc_tablolari": {"pattern": r"(?:Sonuç\s*Tablo|Result.*?Table|Summary)", "weight": 3},
                "oneriler": {"pattern": r"(?:Öneri|Recommendation|İyileştirme|Improvement)", "weight": 3},
                "mevzuat_uygunlugu": {"pattern": r"(?:2006/42/EC|2009/104/EC|Direktif|Directive)", "weight": 2}
            },
            "Dokümantasyon ve Referanslar": {
                "mevzuat_referanslari": {"pattern": r"(?:2006/42/EC|2009/104/EC|AB\s*Direktif|EU.*?Directive)", "weight": 3},
                "normatif_referanslar": {"pattern": r"(?:EN\s*ISO|ISO|12100|60204|Standard)", "weight": 2}
            }
        }
    
    def detect_language(self, text: str) -> str:
        if not LANGUAGE_DETECTION_AVAILABLE:
            return 'tr'
        try:
            return detect(text[:500].strip())
        except:
            return 'tr'
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"PDF okuma hatası: {e}")
            return ""
    
    def detect_document_type(self, text: str) -> str:
        analysis_count = sum(1 for pattern in [r"analiz", r"bulgular", r"sonuç", r"değerlendirme"] 
                           if re.search(pattern, text, re.IGNORECASE))
        procedure_count = sum(1 for pattern in [r"prosedür", r"talimat", r"adım", r"zone"] 
                            if re.search(pattern, text, re.IGNORECASE))
        return "procedure_document" if procedure_count > analysis_count else "analysis_report"
    
    def analyze_criteria(self, text: str, category: str, document_type: str = "analysis_report") -> Dict[str, LOTOAnalysisResult]:
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                content = f"Bulunan: {str(matches[:3])}"
                found = True
                score = weight
            else:
                content = "Bulunamadı"
                found = False
                score = 0
            
            results[criterion_name] = LOTOAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_count": len(matches) if matches else 0}
            )
        
        return results
    
    def calculate_scores(self, analysis_results: Dict) -> Dict[str, Any]:
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
        return {
            "proje_adi": "Bulunamadı",
            "rapor_tarihi": "Bulunamadı",
            "hazirlayan_firma": "Bulunamadı",
            "kabul_durumu": "Bulunamadı"
        }
    
    def generate_recommendations(self, analysis_results: Dict, scores: Dict, document_type: str = "analysis_report") -> List[str]:
        recommendations = []
        pass_threshold = 50 if document_type == "procedure_document" else 70
        total_percentage = scores["percentage"]
        
        if total_percentage >= pass_threshold:
            recommendations.append(f"✅ LOTO {'Prosedürü' if document_type == 'procedure_document' else 'Raporu'} GEÇERLİ (Toplam: %{total_percentage:.1f})")
        else:
            recommendations.append(f"❌ LOTO {'Prosedürü' if document_type == 'procedure_document' else 'Raporu'} EKSİK (Toplam: %{total_percentage:.1f})")
        
        return recommendations
    
    def analyze_loto_report(self, pdf_path: str) -> Dict[str, Any]:
        logger.info("LOTO rapor analizi başlatılıyor...")
        
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "PDF okunamadı"}
        
        detected_lang = self.detect_language(text)
        document_type = self.detect_document_type(text)
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category, document_type)
        
        scores = self.calculate_scores(analysis_results)
        extracted_values = self.extract_specific_values(text)
        recommendations = self.generate_recommendations(analysis_results, scores, document_type)
        
        pass_threshold = 50 if document_type == "procedure_document" else 70
        final_status = "PASS" if scores["percentage"] >= pass_threshold else "FAIL"
        
        return {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgisi": {
                "detected_language": detected_lang,
                "document_type": document_type,
                "pass_threshold": pass_threshold
            },
            "cikarilan_degerler": extracted_values,
            "kategori_analizleri": analysis_results,
            "puanlama": scores,
            "oneriler": recommendations,
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["percentage"],
                "durum": final_status,
                "rapor_tipi": "LOTO",
                "belge_turu": document_type
            }
        }

# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_document_server(text):
    critical_terms = [
        # LOTO temel terimleri
        ["loto", "lockout", "tagout", "kilitleme", "etiketleme", "lockout tagout"],
        
        # Enerji kaynakları ve izolasyon terimleri
        ["enerji", "energy", "izolasyon", "isolation", "kaynaklar", "sources", "elektrik", "mekanik"],
        
        # Güvenlik prosedürü terimleri
        ["prosedür", "procedure", "güvenlik", "safety", "iş güvenliği", "work safety", "önlem", "precaution"],
        
        # Makine ve ekipman terimleri
        ["makine", "machine", "ekipman", "equipment", "sistem", "system", "tesis", "facility"],
        
        # Kontrol ve değerlendirme terimleri
        ["kontrol", "control", "değerlendirme", "evaluation", "analiz", "analysis", "risk", "hazard"]
    ]
    
    category_found = [any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category) for category in critical_terms]
    valid = sum(category_found)
    logger.info(f"LOTO validasyon: {valid}/5 kategori")
    return valid >= 4

def check_strong_keywords_first_pages(filepath):
    strong_keywords = ["loto"]
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        found = [kw for kw in strong_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa: {len(found)} LOTO kelime bulundu")
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"OCR hatası: {e}")
        return False

def check_excluded_keywords_first_pages(filepath):
    excluded = [
        # Aydınlatma raporu
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
        
        # Hidrolik devre şeması
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219","teknik resim","tasarım",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # AT Uygunluk Beyanı
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        
        # LVD raporu (eski strong_keywords LVD'den)
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri"
        
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        
        # Espe raporu  
        "espe",
        
        # Gürültü ölçüm raporu
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # Manuel/kullanma kılavuzu
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide","kılavuzu",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama", "TOPRAKLAMA DİRENCİ",
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
        found = [kw for kw in excluded if re.search(rf"\b{kw.lower()}\b", all_text)]
        return len(found) >= 2
    except Exception as e:
        logger.warning(f"Excluded OCR hatası: {e}")
        return False

def get_conclusion_message_loto(status, percentage):
    if status == "PASS":
        return f"LOTO prosedürü OSHA standartlarına uygun (%{percentage:.0f})"
    return f"LOTO prosedürü standartlara uygun değil (%{percentage:.0f})"

def get_main_issues_loto(report):
    issues = []
    if 'puanlama' in report and 'category_scores' in report['puanlama']:
        for category, score_data in report['puanlama']['category_scores'].items():
            if isinstance(score_data, dict) and score_data.get('percentage', 0) < 50:
                issues.append(f"{category} kategorisinde ciddi eksiklikler")
    return issues[:4]

def check_tesseract_installation():
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract OCR kurulu - Sürüm: {version}")
        return True, f"Tesseract v{version}"
    except Exception as e:
        logger.error(f"Tesseract OCR kurulu değil: {e}")
        return False, str(e)

tesseract_available, tesseract_info = check_tesseract_installation()

# ============================================
# FLASK APP
# ============================================
app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads_loto'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/loto-report', methods=['POST'])
def analyze_loto_report():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"LOTO analizi başlatılıyor: {filename}")

            analyzer = LOTOReportAnalyzer()
            file_ext = os.path.splitext(filepath)[1].lower()
            
            # 3 AŞAMALI KONTROL
            if file_ext == '.pdf':
                logger.info("Aşama 1: LOTO özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath):
                    logger.info("✅ Aşama 1 geçti")
                else:
                    logger.info("Aşama 2: Excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath):
                        logger.info("❌ Excluded kelimeler bulundu")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya LOTO prosedürü değil'
                        }), 400
                    else:
                        # AŞAMA 3
                        logger.info("Aşama 3: Tam doküman kontrolü...")
                        try:
                            with open(filepath, 'rb') as f:
                                pdf_reader = PyPDF2.PdfReader(f)
                                text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
                            
                            if not text or len(text.strip()) < 50 or not validate_document_server(text):
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                                return jsonify({
                                    'error': 'Invalid document type',
                                    'message': 'Yüklediğiniz dosya LOTO prosedürü değil!'
                                }), 400
                        except Exception as e:
                            logger.error(f"Aşama 3 hatası: {e}")
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({'error': 'Analysis failed'}), 500

            # Analizi yap
            logger.info(f"LOTO analizi yapılıyor: {filename}")
            report = analyzer.analyze_loto_report(filepath)
            
            try:
                os.remove(filepath)
            except:
                pass

            if 'error' in report:
                return jsonify({'error': 'Analysis failed', 'message': report['error']}), 400

            overall_percentage = report.get('ozet', {}).get('yuzde', 0)
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': report.get('analiz_tarihi'),
                'analysis_details': {'found_criteria': len([c for c in report.get('puanlama', {}).get('category_scores', {}).values() if isinstance(c, dict) and c.get('score', 0) > 0])},
                'analysis_id': f"loto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': {'is_valid': True, 'message': 'LOTO için tarih kontrolü uygulanmaz'},
                'extracted_values': report.get('cikarilan_degerler', {}),
                'file_type': 'LOTO_PROSEDURU',
                'filename': filename,
                'language_info': {'detected_language': report.get('dosya_bilgisi', {}).get('detected_language', 'turkish')},
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': report.get('ozet', {}).get('toplam_puan', 0),
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'quality_level': "MÜKEMMEL" if overall_percentage >= 90 else "İYİ" if overall_percentage >= 70 else "KÖTÜ"
                },
                'recommendations': report.get('oneriler', []),
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_loto(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_loto(report)
                }
            }
            
            if 'puanlama' in report and 'category_scores' in report['puanlama']:
                for category, score_data in report['puanlama']['category_scores'].items():
                    if isinstance(score_data, dict):
                        response_data['category_scores'][category] = {
                            'score': score_data.get('normalized', 0),
                            'max_score': score_data.get('max_weight', 0),
                            'percentage': score_data.get('percentage', 0),
                            'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'FAIL'
                        }

            return jsonify({
                'success': True,
                'message': 'LOTO Prosedürü başarıyla analiz edildi',
                'analysis_service': 'loto_report',
                'service_description': 'LOTO Prosedürü Analizi',
                'data': response_data
            })

        except Exception as analysis_error:
            try:
                if 'filepath' in locals():
                    os.remove(filepath)
            except:
                pass
            logger.error(f"Analiz hatası: {str(analysis_error)}")
            return jsonify({'error': 'Analysis failed', 'message': str(analysis_error)}), 500

    except Exception as e:
        logger.error(f"API hatası: {str(e)}")
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'LOTO Report Analyzer API',
        'version': '1.0.0',
        'report_type': 'LOTO_PROSEDURU'
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'LOTO Report Analyzer API',
        'version': '1.0.0',
        'description': 'LOTO prosedürlerini analiz eden REST API servisi',
        'endpoints': {
            'POST /api/loto-report': 'LOTO prosedür analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /': 'Bu bilgi sayfası'
        }
    })

# ============================================
# APPLICATION ENTRY POINT
# ============================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("LOTO Prosedürü Analiz Servisi")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8005))
    
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)