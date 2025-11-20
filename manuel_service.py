
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
class ManualAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

# ============================================
# ANALYZER CLASS
# ============================================
class ManualReportAnalyzer:
    def __init__(self):
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
                "kilavuz_adi_kod": {"pattern": r"(?:Kılavuz|Manual|Guide|Kullan[ıi]m\s*K[ıi]lavuzu)", "weight": 5},
                "urun_modeli": {"pattern": r"(?:Ürün|Product|Model|Seri\s*No)", "weight": 3},
                "revizyon_bilgisi": {"pattern": r"(?:Revizyon|Revision|Rev\.?|Version)", "weight": 2}
            },
            "Giriş ve Amaç": {
                "kilavuz_amaci": {"pattern": r"(?:Amaç|Purpose|Objective|Bu\s*k[ıi]lavuz)", "weight": 3},
                "kapsam": {"pattern": r"(?:Kapsam|Scope|Coverage)", "weight": 2}
            },
            "Güvenlik Bilgileri": {
                "genel_guvenlik": {"pattern": r"(?:Güvenlik|Safety|UYARI|WARNING|DİKKAT|CAUTION)", "weight": 4},
                "tehlikeler": {"pattern": r"(?:Tehlike|Hazard|Risk|Dangerous)", "weight": 4},
                "guvenlik_prosedur": {"pattern": r"(?:Prosedür|Procedure|Güvenlik\s*Prosedür)", "weight": 3},
                "kkd_gerekliligi": {"pattern": r"(?:KKD|PPE|Personal\s*Protective|Koruyucu\s*Donanım)", "weight": 4}
            },
            "Ürün Tanıtımı": {
                "urun_tanimi": {"pattern": r"(?:Ürün\s*Tan[ıi]m[ıi]|Product\s*Description)", "weight": 3},
                "teknik_ozellikler": {"pattern": r"(?:Teknik\s*Özellik|Technical\s*Specification)", "weight": 3},
                "bilesenler": {"pattern": r"(?:Bileşen|Component|Parça|Part)", "weight": 2},
                "gorseller": {"pattern": r"(?:Görsel|Image|Resim|Picture|Şekil|Figure)", "weight": 2}
            },
            "Kurulum ve Montaj Bilgileri": {
                "kurulum_oncesi": {"pattern": r"(?:Kurulum\s*Öncesi|Before\s*Installation)", "weight": 4},
                "montaj_talimatlari": {"pattern": r"(?:Montaj|Installation|Assembly|Ad[ıi]m|Step)", "weight": 4},
                "gerekli_aletler": {"pattern": r"(?:Alet|Tool|Malzeme|Material)", "weight": 3},
                "kurulum_kontrolu": {"pattern": r"(?:Kontrol|Check|Test|Doğrula)", "weight": 4}
            },
            "Kullanım Talimatları": {
                "calistirma": {"pattern": r"(?:Çal[ıi]şt[ıi]rma|Start|Operation)", "weight": 5},
                "kullanim_kilavuzu": {"pattern": r"(?:Kullan[ıi]m|Usage|Use|Operating)", "weight": 5},
                "calisma_modlari": {"pattern": r"(?:Mod|Mode|Ayar|Setting)", "weight": 5},
                "kullanim_ipuclari": {"pattern": r"(?:İpucu|Tip|Öneri|Recommendation)", "weight": 5}
            },
            "Bakım ve Temizlik": {
                "duzenli_bakim": {"pattern": r"(?:Bak[ıi]m|Maintenance|Düzenli|Regular)", "weight": 3},
                "temizlik_yontemleri": {"pattern": r"(?:Temizlik|Cleaning|Temizle|Clean)", "weight": 3},
                "parca_degisimi": {"pattern": r"(?:Parça\s*Değiş|Part\s*Replace|Yedek\s*Parça)", "weight": 4}
            },
            "Arıza Giderme": {
                "sorun_cozumleri": {"pattern": r"(?:Sorun|Problem|Ar[ıi]za|Fault|Troubleshoot)", "weight": 5},
                "hata_kodlari": {"pattern": r"(?:Hata\s*Kod|Error\s*Code|Alarm)", "weight": 5},
                "teknik_destek": {"pattern": r"(?:Teknik\s*Destek|Technical\s*Support|İletişim|Contact)", "weight": 3},
                "teknik_cizimler": {"pattern": r"(?:Çizim|Drawing|Şema|Scheme|Diyagram)", "weight": 2}
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
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, ManualAnalysisResult]:
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                content = f"Bulunan: {str(matches[:3])}"
                found = True
                score = min(weight, len(matches) * (weight // 2))
            else:
                content = "Bulunamadı"
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
            "kilavuz_adi": "Bulunamadı",
            "urun_modeli": "Bulunamadı",
            "hazırlama_tarihi": "Bulunamadı",
            "hazirlayan": "Bulunamadı"
        }

    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        recommendations = []
        total_percentage = scores["percentage"]
        
        if total_percentage >= 70:
            recommendations.append(f"✅ Kullanma Kılavuzu GEÇERLİ (Toplam: %{total_percentage:.1f})")
        else:
            recommendations.append(f"❌ Kullanma Kılavuzu GEÇERSİZ (Toplam: %{total_percentage:.1f})")
        
        return recommendations

    def analyze_manual_report(self, pdf_path: str) -> Dict[str, Any]:
        logger.info("Manuel analizi başlatılıyor...")
        
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "PDF okunamadı"}
        
        detected_lang = self.detect_language(text)
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category)
        
        scores = self.calculate_scores(analysis_results)
        extracted_values = self.extract_specific_values(text)
        recommendations = self.generate_recommendations(analysis_results, scores)
        
        return {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgisi": {"detected_language": detected_lang},
            "cikarilan_degerler": extracted_values,
            "kategori_analizleri": analysis_results,
            "puanlama": scores,
            "oneriler": recommendations,
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["percentage"],
                "durum": "PASS" if scores["percentage"] >= 70 else "FAIL",
                "rapor_tipi": "KULLANIM_KILAVUZU"
            }
        }

# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_document_server(text):
    critical_terms = [
        # Manuel/kılavuz temel terimleri
        ["kullanma", "kılavuz", "manual", "instruction", "talimat", "guide", "kılavuzu", "user manual"],
        
        # Güvenlik ve uyarı terimleri
        ["güvenlik", "safety", "uyarı", "warning", "dikkat", "attention", "tehlike", "danger"],
        
        # Kurulum ve montaj terimleri
        ["kurulum", "installation", "montaj", "assembly", "setup", "kurma", "takma", "yerleştirme"],
        
        # Kullanım ve işletim terimleri
        ["kullanım", "operation", "işletim", "çalıştırma", "kullanma", "nasıl kullanılır", "how to use"],
        
        # Bakım ve arıza terimleri
        ["bakım", "maintenance", "temizlik", "cleaning", "arıza", "troubleshooting", "onarım", "repair"]
    ]
    
    category_found = [any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category) for category in critical_terms]
    valid = sum(category_found)
    logger.info(f"Manuel validasyon: {valid}/5 kategori")
    return valid >= 4

def check_strong_keywords_first_pages(filepath):
    strong_keywords = [
        "kullanma",
        "kılavuz",
        "manual",
        "instruction",
        "talimat",
        "guide",
        "kılavuzu"
    ]
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
        logger.info(f"İlk sayfa: {len(found)} manuel kelime bulundu")
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"OCR hatası: {e}")
        return False

def check_excluded_keywords_first_pages(filepath):
    excluded = [
        # Aydınlatma raporu
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
        
        # Hidrolik devre şeması
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # AT Uygunluk Beyanı
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        
        # LVD raporu
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        
        # LOTO raporu (eski strong_keywords LOTO'dan)
        "loto",
        
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        
        # Espe raporu  
        "espe",
        
        # Gürültü ölçüm raporu
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204","topraklama", "TOPRAKLAMA DİRENCİ",
    ]
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        found = [kw for kw in excluded if re.search(rf"\b{kw.lower()}\b", all_text)]
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"Excluded OCR hatası: {e}")
        return False

def get_conclusion_message_manuel(status, percentage):
    if status == "PASS":
        return f"Kullanım kılavuzu yeterli kriterleri sağlamaktadır (%{percentage:.0f})"
    return f"Kullanım kılavuzu yetersiz kriterlere sahip (%{percentage:.0f})"

def get_main_issues_manuel(report):
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

UPLOAD_FOLDER = 'temp_uploads_manuel'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/manuel-report', methods=['POST'])
def analyze_manuel_report():
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
            logger.info(f"Manuel analizi başlatılıyor: {filename}")

            analyzer = ManualReportAnalyzer()
            file_ext = os.path.splitext(filepath)[1].lower()
            
            # 3 AŞAMALI KONTROL
            if file_ext == '.pdf':
                logger.info("Aşama 1: Manuel özgü kelime kontrolü...")
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
                            'message': 'Bu dosya kullanım kılavuzu değil'
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
                                    'message': 'Yüklediğiniz dosya kullanım kılavuzu değil!'
                                }), 400
                        except Exception as e:
                            logger.error(f"Aşama 3 hatası: {e}")
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({'error': 'Analysis failed'}), 500

            # Analizi yap
            logger.info(f"Manuel analizi yapılıyor: {filename}")
            report = analyzer.analyze_manual_report(filepath)
            
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
                'analysis_id': f"manuel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': {'is_valid': True, 'message': 'Manuel için tarih kontrolü uygulanmaz'},
                'extracted_values': report.get('cikarilan_degerler', {}),
                'file_type': 'KULLANIM_KILAVUZU',
                'filename': filename,
                'language_info': {'detected_language': report.get('dosya_bilgisi', {}).get('detected_language', 'turkish')},
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': report.get('ozet', {}).get('toplam_puan', 0),
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'text_quality': 'good'
                },
                'recommendations': report.get('oneriler', []),
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_manuel(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_manuel(report)
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
                'message': 'Manuel/Kullanım Kılavuzu başarıyla analiz edildi',
                'analysis_service': 'manuel_report',
                'service_description': 'Manuel/Kullanım Kılavuzu Analizi',
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
        'service': 'Manual Report Analyzer API',
        'version': '1.0.0',
        'report_type': 'KULLANIM_KILAVUZU'
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'Manual Report Analyzer API',
        'version': '1.0.0',
        'description': 'Kullanım kılavuzlarını analiz eden REST API servisi',
        'endpoints': {
            'POST /api/manuel-report': 'Manuel analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /': 'Bu bilgi sayfası'
        }
    })

# ============================================
# APPLICATION ENTRY POINT
# ============================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Manuel/Kullanım Kılavuzu Analiz Servisi")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8004))
    
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)