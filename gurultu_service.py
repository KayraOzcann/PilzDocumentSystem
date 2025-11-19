"""
Gürültü Ölçüm Raporu Analiz Servisi
====================================
Azure App Service için optimize edilmiş standalone servis

Bu dosya şunları içerir:
1. NoiseReportAnalyzer sınıfı (noise_report_checker.py)
2. Flask API servisi (server4.py)
3. Azure-friendly konfigürasyon

Endpoint: POST /api/noise-report
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
from docx import Document
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
# LOGGING CONFIGURATION
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================
# ANALİZ SINIFI - DATA CLASSES
# ============================================
@dataclass
class NoiseCriteria:
    rapor_kimlik_bilgileri: Dict[str, Any]
    olcum_ortam_ekipman: Dict[str, Any]
    olcum_cihazi_bilgileri: Dict[str, Any]
    olcum_metodolojisi: Dict[str, Any]
    olcum_sonuclari: Dict[str, Any]
    degerlendirme_yorum: Dict[str, Any]
    ekler_gorseller: Dict[str, Any]

@dataclass
class NoiseAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]


# ============================================
# ANALİZ SINIFI - MAIN ANALYZER
# ============================================
class NoiseReportAnalyzer:
    
    def __init__(self):
        self.criteria_weights = {
            "Rapor Kimlik Bilgileri": 15,
            "Ölçüm Yapılan Ortam ve Ekipman Bilgileri": 15,
            "Ölçüm Cihazı Bilgileri": 15,
            "Ölçüm Metodolojisi": 20,
            "Ölçüm Sonuçları": 20,
            "Değerlendirme ve Yorum": 10,
            "Ekler ve Görseller": 5
        }
        
        self.criteria_details = {
            "Rapor Kimlik Bilgileri": {
                "rapor_numarasi": {"pattern": r"(?:Rapor\s*No\s*[:=]\s*|Belge\s*Numarası\s*[:=]\s*|C\d{2}\.\d{3})", "weight": 3},
                "rapor_tarihi": {"pattern": r"(?:Rapor\s*)?Tarihi?\s*[:=]\s*(\d{2}[./]\d{2}[./]\d{4})", "weight": 2},
                "olcum_tarihi": {"pattern": r"(?:Ölçüm\s*Tarihi|İnceleme\s*Tarihi)\s*[:=]?\s*(\d{2}[./]\d{2}[./]\d{4})", "weight": 3},
                "hazirlayan_kurulus": {"pattern": r"(?:Pilz\s*Servisleri|Pilz\s*Emniyet|Hazırlayan)", "weight": 2},
                "olcum_yapan_uzman": {"pattern": r"(?:Yapan\s*[:=]\s*|Kaan\s*Karabağ|Savaş\s*Şahan)", "weight": 3},
                "uzman_imza": {"pattern": r"(?:İmza|Yetkilisi)", "weight": 2}
            },
            "Ölçüm Yapılan Ortam ve Ekipman Bilgileri": {
                "firma_adi": {"pattern": r"(?:FORD\s*OTOSAN|Ford|Otosan)", "weight": 3},
                "firma_adresi": {"pattern": r"(?:Denizevler\s*Mah|Gölcük/Kocaeli|Ali\s*Uçar\s*Cad)", "weight": 2},
                "ortam_tanimi": {"pattern": r"(?:Otomatik\s*Robotlu\s*Kaynak|Kaynak\s*Hattı|fabrika|atölye)", "weight": 3},
                "makine_adi": {"pattern": r"(?:8X9J\s*Otomatik|Robotlu\s*Kaynak|Kaynak\s*Hattı)", "weight": 3},
                "makine_konumu": {"pattern": r"(?:8X\d{2}\s*LH|8X\d{2}\s*RH|BÖLGESİ)", "weight": 2},
                "cevresel_kosullar": {"pattern": r"(?:Sıcaklık|Nem|Rüzgar|kapalı\s*ortam)", "weight": 2}
            },
            "Ölçüm Cihazı Bilgileri": {
                "cihaz_marka": {"pattern": r"(?:PCE\s*Gürültü|PCE)", "weight": 3},
                "cihaz_model": {"pattern": r"(?:PCE-322A|322A)", "weight": 3},
                "seri_numarasi": {"pattern": r"(?:Seri\s*Numarası\s*[:=]\s*|180914367)", "weight": 3},
                "kalibrasyon_tarihi": {"pattern": r"(?:Kalibrasyon\s*Tarihi\s*[:=]\s*|4\.10\.2020)", "weight": 3},
                "mikrofon_bilgileri": {"pattern": r"(?:mikrofon|aksesuar)", "weight": 2},
                "cihaz_ayarlari": {"pattern": r"(?:Hızlı|Yavaş|Sample\s*Rate|50ms|100ms)", "weight": 1}
            },
            "Ölçüm Metodolojisi": {
                "uygulanan_standart": {"pattern": r"(?:ISO\s*11201|ISO\s*9612|ISO\s*3744|EN\s*ISO\s*4871|EN\s*ISO\s*11200)", "weight": 5},
                "olcum_turu": {"pattern": r"(?:emission\s*sound\s*pressure|Time-averaged|LpA|LpC)", "weight": 3},
                "olcum_yukseklik": {"pattern": r"(?:yükseklik|height)", "weight": 2},
                "olcum_noktalari": {"pattern": r"(?:8X\d{2}\s*LH|8X\d{2}\s*RH|Ölçüm\s*Noktası)", "weight": 5},
                "olcum_suresi": {"pattern": r"(?:1\s*dakika|Ölçüm\s*Süresi)", "weight": 3},
                "arka_plan_gurultu": {"pattern": r"(?:arka\s*plan|background)", "weight": 2}
            },
            "Ölçüm Sonuçları": {
                "ses_basinc_seviyesi": {"pattern": r"(?:LpA\s*\(dBA\)|LpA\s*\(dBC\)|dB\(A\)|dB\(C\))", "weight": 5},
                "laeeq_degeri": {"pattern": r"(?:LAeq|L\s*peqT|Time-averaged)", "weight": 4},
                "lmax_lmin": {"pattern": r"(?:En\s*düşük\s*Değer|En\s*yüksek\s*Değer|Lmax|Lmin)", "weight": 3},
                "lcpeak_degeri": {"pattern": r"(?:LCpeak|LpC\s*peak|Peak\s*sound)", "weight": 3},
                "nokta_degerleri": {"pattern": r"(?:7[0-9],\d|9[0-9],\d)", "weight": 3},
                "maruziyet_suresi": {"pattern": r"(?:T\s*=|çalışma\s*süresi|8\s*saat)", "weight": 2}
            },
            "Değerlendirme ve Yorum": {
                "yasal_sinirlar": {"pattern": r"(?:85\s*dB|87\s*dB|yasal\s*sınır)", "weight": 3},
                "risk_degerlendirme": {"pattern": r"(?:risk\s*değerlendirme|maruziyet\s*risk)", "weight": 2},
                "onlemler": {"pattern": r"(?:kulaklık|izolasyon|perdeleme|önlem)", "weight": 3},
                "lex_8h": {"pattern": r"(?:LEX,8h|günlük\s*gürültü|8\s*saatlik)", "weight": 2}
            },
            "Ekler ve Görseller": {
                "ortam_krokisi": {"pattern": r"(?:kroki|çizim|plan)", "weight": 2},
                "fotograflar": {"pattern": r"(?:fotoğraf|görsel|resim)", "weight": 2},
                "kalibrasyon_sertifika": {"pattern": r"(?:kalibrasyon\s*sertifika|sertifika)", "weight": 1}
            }
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"PDF okuma hatası: {e}")
            return ""
    
    def check_report_date_validity(self, text: str) -> Tuple[bool, str, str]:
        date_patterns = [
            r"(?:Ölçüm\s*Tarihi|İnceleme)\s*[:=]?\s*(\d{2}[./]\d{2}[./]\d{4})",
            r"(\d{2}[./]\d{2}[./]\d{4})"
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                date_str = matches[0]
                try:
                    date_str = date_str.replace('.', '/').replace('-', '/')
                    report_date = datetime.strptime(date_str, '%d/%m/%Y')
                    one_year_ago = datetime.now() - timedelta(days=365)
                    is_valid = report_date >= one_year_ago
                    return is_valid, date_str, f"Rapor tarihi: {date_str} {'(GEÇERLİ)' if is_valid else '(GEÇERSİZ - 1 yıldan eski)'}"
                except ValueError:
                    continue
        return False, "", "Rapor tarihi bulunamadı"
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, NoiseAnalysisResult]:
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                content = str(matches[0]) if len(matches) == 1 else str(matches)
                found = True
                score = weight
            else:
                content = "Bulunamadı"
                found = False
                score = 0
            
            results[criterion_name] = NoiseAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_found": len(matches) if matches else 0}
            )
        
        return results
    
    def extract_specific_values(self, text: str) -> Dict[str, Any]:
        values = {}
        value_patterns = {
            "rapor_no": r"(?:Belge\s*Numarası\s*[:=]\s*|C\d{2}\.\d{3})",
            "olcum_tarihi": r"(?:İnceleme\s*Tarihi\s*[:=]?\s*)?(\d{2}[./]\d{2}[./]\d{4})",
            "firma_adi": r"(?:FORD\s*OTOSAN|Ford)",
            "makine_adi": r"(?:8X9J\s*Otomatik|Robotlu\s*Kaynak|Kaynak\s*Hattı)",
            "cihaz_marka": r"(?:PCE\s*Gürültü|PCE)",
            "olcum_yapan": r"(?:Yapan\s*[:=]\s*|Kaan\s*Karabağ)"
        }
        
        for key, pattern in value_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                values[key] = matches[0].strip()
            else:
                values[key] = "Bulunamadı"
        
        return values
    
    def calculate_scores(self, analysis_results: Dict[str, Dict[str, NoiseAnalysisResult]]) -> Dict[str, Any]:
        category_scores = {}
        total_score = 0
        
        for category, results in analysis_results.items():
            category_max = self.criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())
            
            normalized_score = (category_earned / category_possible * category_max) if category_possible > 0 else 0
            
            category_scores[category] = {
                "earned": category_earned,
                "possible": category_possible,
                "normalized": round(normalized_score, 2),
                "max_weight": category_max,
                "percentage": round((category_earned / category_possible * 100), 2) if category_possible > 0 else 0
            }
            
            total_score += normalized_score
        
        return {
            "category_scores": category_scores,
            "total_score": round(total_score, 2),
            "total_max_score": 100,
            "overall_percentage": round((total_score / 100 * 100), 2)
        }
    
    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        recommendations = []
        
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 50:
                recommendations.append(f"❌ {category} bölümü yetersiz (%{category_score:.1f})")
            elif category_score < 80:
                recommendations.append(f"⚠️ {category} bölümü geliştirilmeli (%{category_score:.1f})")
            else:
                recommendations.append(f"✅ {category} bölümü yeterli (%{category_score:.1f})")
        
        if scores["overall_percentage"] < 70:
            recommendations.append("\n🚨 GENEL ÖNERİLER:")
            recommendations.append("- Rapor ISO 11201, ISO 9612 standartlarına tam uyumlu hale getirilmelidir")
        
        return recommendations
    
    def generate_detailed_report(self, pdf_path: str, docx_path: str = None) -> Dict[str, Any]:
        logger.info("Gürültü ölçüm raporu analizi başlatılıyor...")
        
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return {"error": "PDF okunamadı"}
        
        date_valid, date_str, date_message = self.check_report_date_validity(pdf_text)
        extracted_values = self.extract_specific_values(pdf_text)
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(pdf_text, category)
        
        scores = self.calculate_scores(analysis_results)
        recommendations = self.generate_recommendations(analysis_results, scores)
        
        report = {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tarih_gecerliligi": {
                "gecerli": date_valid,
                "tarih": date_str,
                "mesaj": date_message
            },
            "cikarilan_degerler": extracted_values,
            "kategori_analizleri": analysis_results,
            "puanlama": scores,
            "oneriler": recommendations,
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["overall_percentage"],
                "durum": "GEÇERLİ" if scores["overall_percentage"] >= 70 else "YETERSİZ",
                "tarih_durumu": "GEÇERLİ" if date_valid else "GEÇERSİZ"
            }
        }
        
        return report


# ============================================
# HELPER FUNCTIONS (Server Validasyon)
# ============================================
def validate_document_server(text):
    """Gürültü doküman validasyonu"""
    critical_terms = [
        ["gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic"],
        ["ölçüm", "measurement", "test", "analiz", "analysis"],
        ["dba", "dbc", "leq", "lmax", "lmin"],
        ["iso 11201", "iso 9612", "iso 3744"],
        ["maruziyet", "exposure", "yasal sınır", "limit"]
    ]
    
    category_found = [any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category) for category in critical_terms]
    valid = sum(category_found)
    logger.info(f"Gürültü validasyon: {valid}/5 kategori")
    return valid >= 4


def check_strong_keywords_first_pages(filepath):
    """İlk sayfada gürültü özgü kelime kontrolü - OCR"""
    strong_keywords = ["gürültü", "noise", "ses", "sound", "decibel", "akustik", "acoustic"]
    
    try:
        Image.MAX_IMAGE_PIXELS = None
        pages = pdf2image.convert_from_path(filepath, dpi=150, first_page=1, last_page=1)
        
        all_text = ""
        for page in pages:
            if page.size[0] > 2000 or page.size[1] > 2000:
                page = page.resize((1500, int(1500 * page.size[1] / page.size[0])), Image.Resampling.LANCZOS)
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        
        found = [kw for kw in strong_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa: {len(found)} gürültü kelime bulundu")
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"OCR hatası: {e}")
        return False


def check_excluded_keywords_first_pages(filepath):
    """İlk sayfada excluded keyword kontrolü - OCR"""
    excluded = ["aydınlatma", "hidrolik", "pnömatik", "isg", "uygunluk", "lvd", "loto", 
                "kullanma", "hrc", "elektrik", "espe", "montaj", "bakım", "titreşim", 
                "at tip", "topraklama"]
    
    try:
        Image.MAX_IMAGE_PIXELS = None
        pages = pdf2image.convert_from_path(filepath, dpi=400, first_page=1, last_page=2)
        
        all_text = ""
        for page in pages:
            if page.size[0] > 2000 or page.size[1] > 2000:
                page = page.resize((1500, int(1500 * page.size[1] / page.size[0])), Image.Resampling.LANCZOS)
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        
        found = [kw for kw in excluded if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"Excluded: {len(found)} kelime bulundu")
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"Excluded OCR hatası: {e}")
        return False


def get_conclusion_message_noise(status, percentage):
    """Sonuç mesajı - Gürültü"""
    if status == "PASS":
        return f"Gürültü ölçüm raporu standartlara uygun (%{percentage:.0f})"
    return f"Gürültü ölçüm raporu standartlara uygun değil (%{percentage:.0f})"


def get_main_issues_noise(report):
    """Ana sorunlar - Gürültü"""
    issues = []
    for category, score_data in report["puanlama"]["category_scores"].items():
        if score_data.get('percentage', 0) < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    return issues[:4]


# ============================================
# TESSERACT CHECK
# ============================================
def check_tesseract_installation():
    """Tesseract kontrolü"""
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract OCR kurulu - Sürüm: {version}")
        return True, f"Tesseract v{version}"
    except Exception as e:
        logger.error(f"Tesseract OCR kurulu değil: {e}")
        return False, str(e)

tesseract_available, tesseract_info = check_tesseract_installation()


# ============================================
# FLASK SERVİS KATMANI - CONFIGURATION
# ============================================
app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads_noise'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================
# FLASK SERVİS KATMANI - API ENDPOINTS
# ============================================
@app.route('/api/noise-report', methods=['POST'])
def analyze_noise_report():
    """Gürültü Ölçüm Raporu analiz endpoint'i"""
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
            logger.info(f"Gürültü analizi başlatılıyor: {filename}")

            analyzer = NoiseReportAnalyzer()
            file_ext = os.path.splitext(filepath)[1].lower()
            
            # 3 AŞAMALI KONTROL
            if file_ext == '.pdf':
                logger.info("Aşama 1: Gürültü özgü kelime kontrolü...")
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
                            'message': 'Bu dosya gürültü ölçüm raporu değil'
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
                                    'message': 'Yüklediğiniz dosya gürültü ölçüm raporu değil!'
                                }), 400
                        except Exception as e:
                            logger.error(f"Aşama 3 hatası: {e}")
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({'error': 'Analysis failed'}), 500

            # Analizi yap
            logger.info(f"Gürültü analizi yapılıyor: {filename}")
            full_report = analyzer.generate_detailed_report(filepath, None)
            
            try:
                os.remove(filepath)
            except:
                pass

            if 'error' in full_report:
                return jsonify({'error': 'Analysis failed', 'message': full_report['error']}), 400

            overall_percentage = full_report["ozet"]["yuzde"]
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': full_report["analiz_tarihi"],
                'analysis_details': {'found_criteria': len([c for c in full_report["puanlama"]["category_scores"].values() if c.get('percentage', 0) > 50])},
                'analysis_id': f"noise_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': {
                    'is_valid': full_report["tarih_gecerliligi"]["gecerli"],
                    'message': full_report["tarih_gecerliligi"]["mesaj"]
                },
                'extracted_values': full_report["cikarilan_degerler"],
                'file_type': 'GURULTU_OLCUM_RAPORU',
                'filename': filename,
                'language_info': {'detected_language': 'turkish', 'text_length': 0},
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': full_report["ozet"]["toplam_puan"],
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'text_quality': 'good'
                },
                'recommendations': full_report["oneriler"],
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_noise(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_noise(full_report)
                }
            }
            
            for kategori, score_data in full_report["puanlama"]["category_scores"].items():
                response_data['category_scores'][kategori] = {
                    "score": score_data["normalized"],
                    "max_score": score_data["max_weight"],
                    "percentage": round(score_data["percentage"], 1),
                    'status': 'PASS' if score_data["percentage"] >= 70 else 'FAIL'
                }

            return jsonify({
                'success': True,
                'message': 'Gürültü Ölçüm Raporu başarıyla analiz edildi',
                'analysis_service': 'noise_report',
                'service_description': 'Gürültü Ölçüm Rapor Analizi',
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
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Noise Report Analyzer API',
        'version': '1.0.0',
        'report_type': 'GURULTU_OLCUM_RAPORU'
    })


@app.route('/', methods=['GET'])
def index():
    """API bilgileri"""
    return jsonify({
        'service': 'Noise Report Analyzer API',
        'version': '1.0.0',
        'description': 'Gürültü ölçüm raporlarını analiz eden REST API servisi',
        'endpoints': {
            'POST /api/noise-report': 'Gürültü ölçüm raporu analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /': 'Bu bilgi sayfası'
        }
    })


# ============================================
# APPLICATION ENTRY POINT (Azure-Friendly)
# ============================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Gürültü Ölçüm Raporu Analiz Servisi")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8003))
    
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)