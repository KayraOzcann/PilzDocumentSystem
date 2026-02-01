# ============================================
# IMPORTS
# ============================================
import os
import re
from datetime import datetime
from typing import Dict, Any, List
import PyPDF2
from docx import Document
import logging
import pdf2image
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from dataclasses import dataclass
import cv2
import numpy as np

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# ============================================
# DATABASE IMPORTS (YENİ)
# ============================================
from flask import current_app
from database import db, init_db
from db_loader import load_service_config

# ============================================
# CONFIG IMPORT (YENİ)
# ============================================
from config import Config

# ============================================
# LOGGING CONFIGURATION
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================
# DATA CLASSES
# ============================================
@dataclass
class TopraklamaAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]


# ============================================
# ANALYZER CLASS
# ============================================
class EN60204TopraklamaRaporuAnaliz:
    def __init__(self, app=None):
        logger.info("EN 60204-1 Topraklama Ölçüm Raporu Analiz Sistemi başlatılıyor...")

        if not app:
            raise ValueError("❌ Flask app context gerekli!")

        with app.app_context():
            config = load_service_config('grounding_report')
        
            self.criteria_weights = config.get('criteria_weights', {})
            self.criteria_details = config.get('criteria_details', {})
            self.pattern_definitions = config.get('pattern_definitions', {})
            self.validation_keywords = config.get('validation_keywords', {})
            self.category_actions = config.get('category_actions', {})
        
            if not self.criteria_weights:
                raise ValueError("❌ DB'den criteria_weights yüklenemedi!")
        

    def extract_text_with_ocr(self, file_path: str) -> str:
        """OCR ile metin çıkarımı (PDF veya görsel)"""
        try:
            images = convert_from_path(file_path, dpi=300)
            text = ""
            for i, image in enumerate(images):
                logger.info(f"OCR işleniyor: Sayfa {i+1}")
                text += pytesseract.image_to_string(image, lang='tur+eng')
            return text
        except Exception as e:
            logger.error(f"OCR hatası: {e}")
            return ""

    def extract_text_from_file(self, file_path: str) -> str:
        """Dosya türüne göre metin çıkarımı (OCR dahil)"""
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.pdf':
            # Önce normal PDF metin çıkarımı dene
            text = self.extract_text_from_pdf(file_path)
            if not text.strip():
                logger.info("PDF'de metin bulunamadı, OCR ile devam ediliyor...")
                text = self.extract_text_with_ocr(file_path)
            return text
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            return pytesseract.image_to_string(Image.open(file_path), lang='tur+eng')
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            logger.error(f"Desteklenmeyen dosya formatı: {file_ext}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkarımı"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {e}")
            return ""

    def extract_text_from_docx(self, docx_path: str) -> str:
        """DOCX'den metin çıkarımı"""
        try:
            doc = Document(docx_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            logger.error(f"DOCX hatası: {e}")
            return ""

    def extract_text_from_txt(self, txt_path: str) -> str:
        """TXT'den metin çıkarımı"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""

    def analyze_criteria(self, text: str, category: str) -> Dict[str, TopraklamaAnalysisResult]:
        """Kriterleri analiz et"""
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)

            if matches:
                content = f"Bulunan: {matches[:3]}"
                found = True
                score = min(weight, len(matches) * (weight // 2)) if weight > 2 else min(weight, len(matches))
            else:
                content = "Bulunamadı"
                found = False
                score = 0

            results[criterion_name] = TopraklamaAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_count": len(matches) if matches else 0}
            )
        return results

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, TopraklamaAnalysisResult]]) -> Dict[str, Any]:
        """Puanları hesapla"""
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
        """Spesifik değerleri çıkar"""

         # DB'den extract_values pattern'lerini al
        extract_patterns = self.pattern_definitions.get('extract_values', {})

        values = {
            "sebeke_tipi": "Bulunamadı",
            "olcum_tarihi": "Bulunamadı",
            "topraklama_direnci": "Bulunamadı",
            "zs_degeri": "Bulunamadı",
            "olcum_cihazi": "Bulunamadı",
            "kalibrasyon_tarihi": "Bulunamadı",
            "sonuc": "Bulunamadı"
        }

         # DB'den gelen pattern'leri kullan
        for key, patterns_list in extract_patterns.items():
            if key in values:
                for pattern in patterns_list:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        if match.lastindex and match.lastindex > 0:
                            values[key] = match.group(1).strip()
                        else:
                            values[key] = match.group(0).strip()
                        break

        # Sonuç
        if re.search(r"uygun\s*değil|geçersiz|kabul\s*edilmez", text, re.IGNORECASE):
            values["sonuc"] = "Uygun Değil"
        elif re.search(r"uygun|geçerli|kabul\s*edilebilir", text, re.IGNORECASE):
            values["sonuc"] = "Uygun"

        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        total_percentage = scores["percentage"]

        if total_percentage >= 70:
            recommendations.append(f"✅ EN 60204-1 Topraklama Raporu GEÇERLİ (Toplam: %{total_percentage:.1f})")
        else:
            recommendations.append(f"❌ EN 60204-1 Topraklama Raporu GEÇERSİZ (Toplam: %{total_percentage:.1f})")

        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]

            if category_score < 40:
                recommendations.append(f"🔴 {category} bölümü yetersiz (%{category_score:.1f})")
                missing_items = [name for name, result in results.items() if not result.found]
                if missing_items:
                    recommendations.append(f"   Eksik: {', '.join(missing_items[:3])}")
            elif category_score < 70:
                recommendations.append(f"🟡 {category} bölümü geliştirilmeli (%{category_score:.1f})")
            else:
                recommendations.append(f"🟢 {category} bölümü yeterli (%{category_score:.1f})")

        return recommendations

    def generate_improvement_actions(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """İyileştirme önerileri oluştur - DB'den"""
        actions = []
        sorted_categories = sorted(scores["category_scores"].items(), key=lambda x: x[1]["percentage"])

        # DB'den category_actions al
        category_actions = self.category_actions

        for category, score_data in sorted_categories[:5]:
            if score_data["percentage"] < 70:
                actions.extend(category_actions.get(category, []))

        if scores["percentage"] < 50:
            actions.insert(0, "ÖNCE: Rapor formatını EN 60204-1 standardına uygun olarak yeniden düzenleyiniz")

        return actions

    def validate_document(self, text: str) -> bool:
        """Doküman validasyonu"""
        keywords = ["topraklama", "toprak", "earth", "ground", "EN 60204", "ölçüm", "measurement", "Zs", "TN", "TT", "IT"]
        found = sum(1 for kw in keywords if re.search(kw, text, re.IGNORECASE))
        return found >= 4

    def analyze_report(self, file_path: str) -> Dict[str, Any]:
        """Ana analiz fonksiyonu"""
        logger.info("EN 60204-1 Topraklama Ölçüm Raporu analizi başlatılıyor...")

        if not os.path.exists(file_path):
            return {"error": f"Dosya bulunamadı: {file_path}"}

        text = self.extract_text_from_file(file_path)
        if not text.strip():
            return {"error": "Dosyadan metin çıkarılamadı (OCR dahil)"}

        if not self.validate_document(text):
            return {
                "error": "YANLIŞ DOKÜMAN: Bu dosya EN 60204-1 topraklama ölçüm raporu değil!",
                "document_type": "UNKNOWN",
                "suggestion": "Lütfen geçerli bir topraklama ölçüm raporu yükleyiniz."
            }

        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category)

        scores = self.calculate_scores(analysis_results)
        extracted_values = self.extract_specific_values(text)
        recommendations = self.generate_recommendations(analysis_results, scores)
        improvement_actions = self.generate_improvement_actions(analysis_results, scores)

        final_status = "PASS" if scores["percentage"] >= 70 else "FAIL"

        return {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgisi": {
                "file_path": file_path,
                "file_type": os.path.splitext(file_path)[1].lower(),
            },
            "cikarilan_degerler": extracted_values,
            "kategori_analizleri": analysis_results,
            "puanlama": scores,
            "oneriler": recommendations,
            "iyilestirme_eylemleri": improvement_actions,
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["percentage"],
                "durum": final_status,
                "rapor_tipi": "EN60204_TOPRAKLAMA_OLCUM_RAPORU"
            }
        }


# ============================================
# HELPER FUNCTIONS (Server Validasyon)
# ============================================
def validate_document_server(text, validation_keywords):
    """Server kodunda doküman validasyonu - DB'den keywords"""
    
    # DB'den critical_terms al
    critical_terms_data = validation_keywords.get('critical_terms', [])
    
    if not critical_terms_data:
        logger.error("❌ DB'den critical_terms yüklenemedi!")
        raise ValueError("Critical terms bulunamadı")
    
    # Liste formatına dönüştür
    critical_terms = []
    for item in critical_terms_data:
        if isinstance(item, dict) and 'keywords' in item:
            critical_terms.append(item['keywords'])
        elif isinstance(item, list):
            critical_terms.append(item)
    
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"Topraklama Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    valid_categories = sum(category_found)
    
    logger.info(f"Doküman validasyonu: {valid_categories}/4 kritik kategori bulundu")
    
    return valid_categories >= 3


def check_strong_keywords_first_pages(filepath, validation_keywords):
    """İlk sayfa kontrol - DB'den keywords"""
    strong_keywords = validation_keywords.get('strong_keywords', [])

    if not strong_keywords:
        logger.error("❌ DB'den strong_keywords yüklenemedi!")
        raise ValueError("Strong keywords bulunamadı")
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        
        all_text = ""
        for i, page in enumerate(pages):
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng')
            all_text += text.lower() + " "
        
        found_keywords = []
        for keyword in strong_keywords:
            if re.search(rf"\b{keyword.lower()}\b", all_text):
                found_keywords.append(keyword)
        
        logger.info(f"İlk sayfa kontrol: {len(found_keywords)} özgü kelime: {found_keywords}")
        return len(found_keywords) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa kontrol hatası: {e}")
        return False


def check_excluded_keywords_first_pages(filepath, validation_keywords):
    """İlk sayfa excluded kontrol - DB'den keywords"""
    excluded_keywords = validation_keywords.get('excluded_keywords', [])
    
    if not excluded_keywords:
        logger.error("❌ DB'den excluded_keywords yüklenemedi!")
        raise ValueError("Excluded keywords bulunamadı")
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=200, first_page=1, last_page=2)
        
        all_text = ""
        for i, page in enumerate(pages):
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng')
            all_text += text.lower() + " "
        
        logger.info(f"OCR okunan text: {all_text[:500]}...")
        
        found_excluded = []
        for keyword in excluded_keywords:
            if re.search(rf"\b{keyword.lower()}\b", all_text):
                found_excluded.append(keyword)
        
        logger.info(f"İlk sayfa excluded kontrol: {len(found_excluded)} istenmeyen kelime: {found_excluded}")
        return len(found_excluded) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa excluded kontrol hatası: {e}")
        return False


def get_conclusion_message_topraklama(status, percentage):
    """Sonuç mesajını döndür - Topraklama için"""
    if status == "PASS":
        return f"Topraklama ölçüm raporu EN 60204-1 standardına uygundur (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"Topraklama raporu kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"Topraklama raporu EN 60204-1 standardına uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"


def get_main_issues_topraklama(analysis_result):
    """Ana sorunları listele - Topraklama için"""
    issues = []
    
    for category, score_data in analysis_result['puanlama']['category_scores'].items():
        if score_data['percentage'] < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    if not issues:
        if analysis_result['puanlama']['total_score'] < 50:
            issues = [
                "Şebeke tipi (TN/TT/IT) belirtilmemiş",
                "Topraklama ölçüm sonuçları eksik",
                "Zs değerlendirmesi yapılmamış",
                "Ölçüm cihazı kalibrasyon bilgileri eksik",
                "EN 60204-1 standard kontrolü eksik"
            ]
    
    return issues[:4]


# ============================================
# FLASK SERVICE LAYER - CONFIGURATION
# ============================================
app = Flask(__name__)

# Database configuration (YENİ)
app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
app.config['SQLALCHEMY_ECHO'] = Config.SQLALCHEMY_ECHO

UPLOAD_FOLDER = 'temp_uploads_topraklama'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================
# FLASK SERVICE LAYER - API ENDPOINTS
# ============================================
@app.route('/api/topraklama-report', methods=['POST'])
def analyze_topraklama_report():
    """Topraklama Ölçüm Raporu analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided', 'message': 'Lütfen analiz edilmek üzere bir topraklama raporu sağlayın'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected', 'message': 'Lütfen bir dosya seçin'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type', 'message': 'Sadece PDF, DOCX, DOC ve TXT dosyaları kabul edilir'}), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"Topraklama Ölçüm Raporu kontrol ediliyor: {filename}")

            analyzer = EN60204TopraklamaRaporuAnaliz(app=app)  # app parametresi ekle
            
            # ÜÇ AŞAMALI TOPRAKLAMA KONTROLÜ
            logger.info(f"Üç aşamalı topraklama kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                logger.info("Aşama 1: İlk sayfa topraklama özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath, analyzer.validation_keywords):
                    logger.info("✅ Aşama 1 geçti - Topraklama özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath, analyzer.validation_keywords):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - Topraklama değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya topraklama raporu değil (farklı rapor türü tespit edildi). Lütfen topraklama ölçüm raporu yükleyiniz.',
                            'details': {'filename': filename, 'document_type': 'OTHER_REPORT_TYPE', 'required_type': 'TOPRAKLAMA_OLCUM_RAPORU'}
                        }), 400
                    else:
                        logger.info("Aşama 3: Tam doküman critical terms kontrolü...")
                        try:
                            with open(filepath, 'rb') as file:
                                pdf_reader = PyPDF2.PdfReader(file)
                                text = ""
                                for page in pdf_reader.pages:
                                    text += page.extract_text() + "\n"
                            
                            if not text or len(text.strip()) < 50:
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                                return jsonify({'error': 'Text extraction failed', 'message': 'Dosyadan yeterli metin çıkarılamadı'}), 400
                            
                            if not validate_document_server(text, analyzer.validation_keywords):
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                                return jsonify({
                                    'error': 'Invalid document type',
                                    'message': 'Yüklediğiniz dosya topraklama ölçüm raporu değil! Lütfen geçerli bir topraklama raporu yükleyiniz.',
                                    'details': {'filename': filename, 'document_type': 'NOT_TOPRAKLAMA_REPORT', 'required_type': 'TOPRAKLAMA_OLCUM_RAPORU'}
                                }), 400
                                
                        except Exception as e:
                            logger.error(f"Aşama 3 hatası: {e}")
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({'error': 'Analysis failed', 'message': 'Dosya analizi sırasında hata oluştu'}), 500

            elif file_ext in ['.docx', '.doc', '.txt']:
                logger.info(f"DOCX/TXT dosyası için tam doküman kontrolü: {file_ext}")
                text = ""
                if file_ext in ['.docx', '.doc']:
                    text = analyzer.extract_text_from_docx(filepath)
                elif file_ext == '.txt':
                    text = analyzer.extract_text_from_txt(filepath)
                
                if not text or len(text.strip()) == 0:
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    return jsonify({'error': 'Text extraction failed', 'message': 'Dosyadan yeterli metin çıkarılamadı'}), 400
                
                if not validate_document_server(text, analyzer.validation_keywords):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    return jsonify({
                        'error': 'Invalid document type',
                        'message': 'Yüklediğiniz dosya topraklama ölçüm raporu değil! Lütfen geçerli bir topraklama raporu yükleyiniz.',
                        'details': {'filename': filename, 'document_type': 'NOT_TOPRAKLAMA_REPORT', 'required_type': 'TOPRAKLAMA_OLCUM_RAPORU'}
                    }), 400

            logger.info(f"Topraklama raporu doğrulandı, analiz başlatılıyor: {filename}")
            analysis_result = analyzer.analyze_report(filepath)
            
            try:
                os.remove(filepath)
                logger.info(f"Uploaded file {filename} cleaned up")
            except Exception as e:
                logger.warning(f"Could not remove uploaded file {filename}: {e}")

            if 'error' in analysis_result:
                return jsonify({'error': 'Analysis failed', 'message': analysis_result['error'], 'details': {'filename': filename}}), 400

            overall_percentage = analysis_result['puanlama']['percentage']
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': analysis_result.get('analiz_tarihi', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'analysis_id': f"topraklama_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'filename': filename,
                'file_type': 'TOPRAKLAMA_OLCUM_RAPORU',
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': analysis_result['puanlama']['total_score'],
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ'
                },
                'category_scores': {},
                'extracted_values': analysis_result.get('cikarilan_degerler', {}),
                'recommendations': analysis_result.get('oneriler', []),
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_topraklama(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_topraklama(analysis_result)
                }
            }
            
            for category, score_data in analysis_result['puanlama']['category_scores'].items():
                response_data['category_scores'][category] = {
                    'score': score_data['normalized'],
                    'max_score': score_data['max_weight'],
                    'percentage': score_data['percentage'],
                    'status': 'PASS' if score_data['percentage'] >= 70 else 'CONDITIONAL' if score_data['percentage'] >= 50 else 'FAIL'
                }

            return jsonify({
                'success': True,
                'message': 'Topraklama Ölçüm Raporu başarıyla analiz edildi',
                'analysis_service': 'topraklama',
                'service_description': 'Topraklama Ölçüm Raporu Analizi',
                'data': response_data
            })

        except Exception as analysis_error:
            try:
                if 'filepath' in locals():
                    os.remove(filepath)
            except:
                pass
            
            logger.error(f"Analiz hatası: {str(analysis_error)}")
            return jsonify({
                'error': 'Analysis failed',
                'message': f'Topraklama raporu analizi sırasında hata oluştu: {str(analysis_error)}',
                'details': {'error_type': type(analysis_error).__name__, 'file_processed': filename if 'filename' in locals() else 'unknown'}
            }), 500

    except Exception as e:
        logger.error(f"API endpoint hatası: {str(e)}")
        return jsonify({'error': 'Server error', 'message': f'Sunucu hatası: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Grounding Report Analyzer API',
        'version': '1.0.0',
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': True,
        'report_type': 'TOPRAKLAMA_OLCUM_RAPORU'
    })


@app.route('/', methods=['GET'])
def index():
    """Ana sayfa - API bilgileri"""
    return jsonify({
        'service': 'Grounding Report Analyzer API',
        'version': '1.0.0',
        'description': 'Topraklama Ölçüm Raporlarını analiz eden REST API servisi',
        'endpoints': {
            'POST /api/topraklama-report': 'Topraklama raporu analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /': 'Bu bilgi sayfası'
        }
    })

# ============================================
# DATABASE INITIALIZATION
# ============================================
with app.app_context():
    db.init_app(app)
    logger.info("✅ Database bağlantısı kuruldu")


# ============================================
# APPLICATION ENTRY POINT (Azure-Friendly)
# ============================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Topraklama Ölçüm Raporu Analiz Servisi")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8016))
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)