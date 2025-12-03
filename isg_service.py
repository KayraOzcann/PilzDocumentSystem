# ============================================
# İSG PERİYODİK KONTROL ANALİZ SERVİSİ
# Standalone Service - Azure App Service Ready
# Database-driven configuration ile dinamik pattern yönetimi
# Port: 8009
# ============================================

# ============================================
# IMPORTS
# ============================================
import os
import json
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Any
import PyPDF2
from docx import Document
from dataclasses import dataclass
import logging
import cv2
import numpy as np
from PIL import Image
import pdf2image
import pytesseract

from flask import Flask, request, jsonify, current_app
from werkzeug.utils import secure_filename

# ============================================
# DATABASE IMPORTS (YENİ)
# ============================================
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
# LANGUAGE DETECTION (Optional)
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
class ISGKontrolAnalysisResult:
    """İSG Kontrol analiz sonucu veri sınıfı"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

# ============================================
# ANALİZ SINIFI - MAIN ANALYZER
# ============================================
class ISGPeriyodikKontrolAnalyzer:
    """İSG Periyodik Kontrol Formu analiz sınıfı"""
    
    def __init__(self, app=None):
        logger.info("İSG Periyodik Kontrol analysis system starting...")
        
        # Flask app context varsa DB'den yükle, yoksa boş başlat
        if app:
            with app.app_context():
                try:
                    config = load_service_config('isg_periodic_control')
                    
                    # DB'den yüklenen veriler
                    self.criteria_weights = config.get('criteria_weights', {})
                    self.criteria_details = config.get('criteria_details', {})
                    self.pattern_definitions = config.get('pattern_definitions', {})
                    self.validation_keywords = config.get('validation_keywords', {})
                    self.category_actions = config.get('category_actions', {})
                    
                    # approval_patterns da pattern_definitions içinde
                    self.approval_patterns = self.pattern_definitions.get('approval_patterns', {})
                    
                    logger.info(f"✅ Veritabanından yüklendi: {len(self.criteria_weights)} kategori")
                    
                except Exception as e:
                    logger.error(f"⚠️ Veritabanından yükleme başarısız: {e}")
                    logger.warning("⚠️ Fallback: Boş config kullanılıyor")
                    self.criteria_weights = {}
                    self.criteria_details = {}
                    self.pattern_definitions = {}
                    self.validation_keywords = {}
                    self.category_actions = {}
                    self.approval_patterns = {}
        else:
            # Flask app yoksa boş başlat
            logger.warning("⚠️ Flask app context yok, boş config kullanılıyor")
            self.criteria_weights = {}
            self.criteria_details = {}
            self.pattern_definitions = {}
            self.validation_keywords = {}
            self.category_actions = {}
            self.approval_patterns = {}
    
    def is_isg_periyodik_kontrol_report(self, text: str) -> bool:
        """Metnin İSG periyodik kontrol raporu olup olmadığını kontrol et - DB'den"""
        
        # ÜÇ AŞAMALI KONTROL SİSTEMİ KULLANILDIĞI İÇİN BU FONKSİYON DEVREDİŞI
        # Fonksiyon kodda duruyor ama her zaman True döndürüyor
        return True
        
        # DB'den pattern'leri al
        extract_values = self.pattern_definitions.get('extract_values', {})
        
        isg_indicators = extract_values.get('isg_indicators', [])
        elektrik_indicators = extract_values.get('elektrik_indicators', [])
        
        isg_score = sum(len(re.findall(p, text, re.IGNORECASE)) for p in isg_indicators)
        elektrik_score = sum(len(re.findall(p, text, re.IGNORECASE)) for p in elektrik_indicators)
        
        logger.info(f"İSG indicators: {isg_score}, Elektrik indicators: {elektrik_score}")
        
        if elektrik_score > 5 and isg_score < 3:
            return False
        
        return isg_score >= 2
    
    def check_report_date_validity(self, olcum_tarihi: str) -> Dict[str, Any]:
        """Rapor tarihinin geçerliliğini kontrol et - 1 yıldan eski ise geçersiz"""
        result = {
            "is_valid": True,
            "days_old": 0,
            "message": "Tarih geçerli",
            "formatted_date": olcum_tarihi
        }
        
        if olcum_tarihi == "Bulunamadı":
            result["is_valid"] = False
            result["message"] = "Ölçüm tarihi bulunamadı"
            return result
        
        try:
            date_formats = ["%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"]
            
            parsed_date = None
            for format_str in date_formats:
                try:
                    parsed_date = datetime.strptime(olcum_tarihi, format_str)
                    break
                except ValueError:
                    continue
            
            if parsed_date is None:
                result["is_valid"] = False
                result["message"] = f"Tarih formatı tanınmıyor: {olcum_tarihi}"
                return result
            
            today = datetime.now()
            days_difference = (today - parsed_date).days
            
            result["days_old"] = days_difference
            result["formatted_date"] = parsed_date.strftime("%d.%m.%Y")
            
            if days_difference > 365:
                result["is_valid"] = False
                result["message"] = f"Rapor tarihi 1 yıldan eski ({days_difference} gün önce)"
            elif days_difference < 0:
                result["is_valid"] = False
                result["message"] = "Rapor tarihi gelecekte - hatalı tarih"
            else:
                result["message"] = f"Rapor tarihi geçerli ({days_difference} gün önce)"
                
        except Exception as e:
            result["is_valid"] = False
            result["message"] = f"Tarih kontrolü hatası: {str(e)}"
        
        return result
            
    def detect_language(self, text: str) -> str:
        """Metin dilini tespit et"""
        if not LANGUAGE_DETECTION_AVAILABLE:
            return 'tr'
        
        try:
            sample_text = text[:500].strip()
            if not sample_text:
                return 'tr'
                
            detected_lang = detect(sample_text)
            logger.info(f"Detected language: {detected_lang}")
            return detected_lang if detected_lang in ['tr', 'en'] else 'tr'
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'tr'
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkar - PyPDF2 ve OCR ile"""
        pypdf_text = ""
        ocr_text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pypdf_text += page_text + "\n"
                
                if len(pypdf_text.strip()) > 50:
                    logger.info("Text extracted using PyPDF2")
                    return pypdf_text.strip()
                
        except Exception as e:
            logger.error(f"PDF text extraction error: {e}")
        
        try:
            logger.info("Insufficient text with PyPDF2, trying OCR...")
            images = pdf2image.convert_from_path(pdf_path, dpi=300)
            
            for i, image in enumerate(images):
                try:
                    text = pytesseract.image_to_string(image, lang='tur+eng')
                    text = re.sub(r'\s+', ' ', text)
                    ocr_text += text + "\n"
                    logger.info(f"OCR extracted {len(text)} characters from page {i+1}")
                except Exception as page_error:
                    logger.error(f"Page {i+1} OCR error: {page_error}")
                    continue
            
            logger.info(f"OCR total text length: {len(ocr_text)}")
            return ocr_text.strip() if ocr_text.strip() else pypdf_text.strip()
            
        except Exception as e:
            logger.error(f"OCR text extraction error: {e}")
            return pypdf_text.strip()
    
    def extract_approval_status(self, text: str, criteria_text: str) -> Dict[str, Any]:
        """Onay durumunu tespit et - DB'den gelen approval patterns ile"""
        status = {"uygun": False, "uygun_degil": False, "not_var": False, "confidence": 0.0}
        
        # self.approval_patterns zaten __init__'te yüklendi
        approval_patterns = self.approval_patterns
        
        criteria_lower = criteria_text.lower()
        text_lower = text.lower()
        
        for keyword in criteria_lower.split():
            if keyword in text_lower:
                pos = text_lower.find(keyword)
                if pos != -1:
                    start = max(0, pos - 100)
                    end = min(len(text), pos + 200)
                    context = text[start:end]
                    
                    for pattern_name, pattern_list in approval_patterns.items():
                        for pattern in pattern_list:
                            if re.search(pattern, context, re.IGNORECASE):
                                status[pattern_name] = True
                                status["confidence"] = 0.8
                                break
                    break
        
        return status
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, ISGKontrolAnalysisResult]:
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
                
                approval_status = self.extract_approval_status(text, str(matches[0]) if matches else "")
                
                if approval_status["uygun_degil"]:
                    score = 0
                elif approval_status["uygun"]:
                    score = weight
                else:
                    score = int(weight * 0.8)
                    
            else:
                content = "Not found"
                found = False
                score = 0
                approval_status = {"uygun": False, "uygun_degil": False, "not_var": False, "confidence": 0.0}
            
            results[criterion_name] = ISGKontrolAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_count": len(matches) if matches else 0, "approval_status": approval_status}
            )
        
        return results

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, ISGKontrolAnalysisResult]]) -> Dict[str, Any]:
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
        """İSG Kontrol formundan özel değerleri çıkar - DB'den gelen pattern'ler ile"""
        values = {
            "firma_adi": "Bulunamadı",
            "olcum_tarihi": "Bulunamadı",
            "rapor_numarasi": "Bulunamadı",
            "laboratuvar": "Bulunamadı",
            "adres": "Bulunamadı",
            "gurultu_seviye": "Bulunamadı",
            "aydinlatma_seviye": "Bulunamadı",
            "genel_degerlendirme": "Bulunamadı"
        }
        
        # DB'den pattern'leri al
        extract_values = self.pattern_definitions.get('extract_values', {})
        
        firma_patterns = extract_values.get('firma_adi', [])
        date_patterns = extract_values.get('olcum_tarihi', [])
        rapor_patterns = extract_values.get('rapor_numarasi', [])
        lab_patterns = extract_values.get('laboratuvar', [])
        gurultu_patterns = extract_values.get('gurultu_seviye', [])
        aydinlatma_patterns = extract_values.get('aydinlatma_seviye', [])
        degerlendirme_patterns = extract_values.get('genel_degerlendirme', [])
        
        # Firma adı
        for pattern in firma_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                firma_name = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                if len(firma_name) > 3:
                    values["firma_adi"] = firma_name
                    break
        
        # Ölçüm tarihi
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                values["olcum_tarihi"] = match.group(1).strip()
                break
        
        # Rapor numarası
        for pattern in rapor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["rapor_numarasi"] = match.group(1).strip()
                break

        # Laboratuvar
        for pattern in lab_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["laboratuvar"] = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                break
        
        # Gürültü seviyesi
        for pattern in gurultu_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["gurultu_seviye"] = f"{match.group(1)} dB(A)"
                break
        
        # Aydınlatma seviyesi
        for pattern in aydinlatma_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["aydinlatma_seviye"] = f"{match.group(1)} lux"
                break

        # Genel değerlendirme
        for pattern in degerlendirme_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) > 0:
                    evaluation = match.group(1).strip()
                    if len(evaluation) > 2:
                        values["genel_degerlendirme"] = evaluation
                else:
                    values["genel_degerlendirme"] = match.group(0).strip()
                break
        
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """İSG Kontrol için öneriler oluştur"""
        recommendations = []
        total_percentage = scores["percentage"]
        
        if total_percentage >= 70:
            recommendations.append(f"✅ İSG Periyodik Kontrol GEÇERLİ (Toplam: %{total_percentage:.0f})")
        elif total_percentage >= 50:
            recommendations.append(f"🟡 İSG Periyodik Kontrol KOŞULLU (Toplam: %{total_percentage:.0f})")
        else:
            recommendations.append(f"❌ İSG Periyodik Kontrol YETERSİZ (Toplam: %{total_percentage:.0f})")
        
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 50:
                recommendations.append(f"🔴 {category} bölümü yetersiz (%{category_score:.0f})")
            elif category_score < 70:
                recommendations.append(f"🟡 {category} bölümü geliştirilmeli (%{category_score:.0f})")
            else:
                recommendations.append(f"🟢 {category} bölümü yeterli (%{category_score:.0f})")
        
        return recommendations

    def analyze_isg_kontrol(self, pdf_path: str) -> Dict[str, Any]:
        """Ana İSG Kontrol analiz fonksiyonu"""
        logger.info("İSG Periyodik Kontrol analysis starting...")
        
        if not os.path.exists(pdf_path):
            return {"error": f"PDF dosyası bulunamadı: {pdf_path}"}
        
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "PDF'den metin çıkarılamadı"}
        
        if not self.is_isg_periyodik_kontrol_report(text):
            return {
                "error": "Bu dosya İSG Periyodik Kontrol raporu değil",
                "suggestion": "Bu dosya elektrik/YG raporu veya başka bir rapor türü olabilir",
                "detected_type": "NON_ISG_REPORT"
            }
        
        detected_lang = self.detect_language(text)
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category)
        
        scores = self.calculate_scores(analysis_results)
        extracted_values = self.extract_specific_values(text)
        
        date_check = self.check_report_date_validity(extracted_values.get("olcum_tarihi", "Bulunamadı"))
        
        normal_recommendations = self.generate_recommendations(analysis_results, scores)
        normal_status = "PASS" if scores["percentage"] >= 70 else ("CONDITIONAL" if scores["percentage"] >= 50 else "FAIL")
        final_score = scores["total_score"]
        final_percentage = scores["percentage"]
        
        if not date_check["is_valid"]:
            final_status = "FAIL"
            recommendations = [
                f"❌ RAPOR GEÇERSİZ: {date_check['message']}",
                "🔴 İSG raporları en fazla 1 yıl geçerlidir",
                "📅 Yeni bir İSG ölçüm raporu temin edilmelidir",
                "",
                "📊 PUANLAMA (Referans için):"
            ] + normal_recommendations
        else:
            recommendations = normal_recommendations
            final_status = normal_status
        
        report = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_info": {"pdf_path": pdf_path, "detected_language": detected_lang},
            "extracted_values": extracted_values,
            "date_validity": date_check,
            "category_analyses": analysis_results,
            "scoring": scores,
            "recommendations": recommendations,
            "summary": {
                "total_score": final_score,
                "percentage": final_percentage,
                "status": final_status,
                "report_type": "ISG_PERIYODIK_KONTROL"
            }
        }
        
        return report

# ============================================
# HELPER FUNCTIONS (Server Validasyon)
# ============================================
def map_language_code(lang_code):
    """Dil kodunu tam isme çevir"""
    lang_mapping = {
        'tr': 'turkish',
        'en': 'english', 
        'de': 'german',
        'fr': 'french',
        'es': 'spanish',
        'it': 'italian'
    }
    return lang_mapping.get(lang_code, 'turkish')


def validate_document_server(text, validation_keywords):
    """Server kodunda doküman validasyonu - İSG için - DB'den"""
    
    # DB'den gelen critical_terms
    critical_terms_data = validation_keywords.get('critical_terms', {})
    
    # Liste formatına dönüştür
    critical_terms = []
    for key, value in critical_terms_data.items():
        if key.startswith('category_') and isinstance(value, list):
            critical_terms.append(value)
    
    if not critical_terms:
        logger.warning("⚠️ Critical terms bulunamadı, validasyon atlanıyor")
        return True
    
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"İSG Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    valid_categories = sum(category_found)
    logger.info(f"Doküman validasyonu: {valid_categories}/{len(critical_terms)} kritik kategori bulundu")
    
    return valid_categories >= 4


def check_strong_keywords_first_pages(filepath, validation_keywords):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - İSG için - DB'den"""
    
    # DB'den strong keywords al
    strong_keywords = validation_keywords.get('strong_keywords', [])
    
    if not strong_keywords:
        logger.warning("⚠️ Strong keywords bulunamadı, validasyon atlanıyor")
        return True
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng')
            all_text += text.lower() + " "
        
        found_keywords = [kw for kw in strong_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa kontrol: {len(found_keywords)} özgü kelime: {found_keywords}")
        return len(found_keywords) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa kontrol hatası: {e}")
        return False


def check_excluded_keywords_first_pages(filepath, validation_keywords):
    """İlk 1-2 sayfada istenmeyen rapor türlerinin kelimelerini ara - DB'den"""
    
    # DB'den excluded keywords al
    excluded_keywords = validation_keywords.get('excluded_keywords', [])
    
    if not excluded_keywords:
        logger.warning("⚠️ Excluded keywords bulunamadı, validasyon atlanıyor")
        return False
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=2)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng')
            all_text += text.lower() + " "
        
        found_excluded = [kw for kw in excluded_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa excluded kontrol: {len(found_excluded)} istenmeyen kelime: {found_excluded}")
        return len(found_excluded) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa excluded kontrol hatası: {e}")
        return False


def get_conclusion_message_isg(status, percentage):
    """Sonuç mesajını döndür - İSG için"""
    if status == "PASS":
        return f"İSG periyodik kontrol raporu İş Sağlığı ve Güvenliği Yönetmeliği gereksinimlerine uygundur (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"İSG raporu kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"İSG raporu yönetmelik gereksinimlerine uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"


def get_main_issues_isg(report):
    """Ana sorunları listele - İSG için"""
    issues = []
    
    for category, score_data in report['scoring']['category_scores'].items():
        if isinstance(score_data, dict) and score_data.get('percentage', 0) < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    if not issues and report['summary']['total_score'] < 50:
        issues = [
            "Gürültü ölçüm sonuçları eksik veya yetersiz",
            "Aydınlatma seviyesi kontrolü yapılmamış",
            "Laboratuvar akreditasyon bilgileri eksik",
            "Ölçüm tarihi geçerlilik süresi aşmış",
            "Yasal limit değerlerle karşılaştırma eksik"
        ]
    
    return issues[:4]


# ============================================
# FLASK SERVİS KATMANI - CONFIGURATION
# ============================================
app = Flask(__name__)

# Database configuration (YENİ)
app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
app.config['SQLALCHEMY_ECHO'] = Config.SQLALCHEMY_ECHO

# Upload configuration
UPLOAD_FOLDER = 'temp_uploads_isg'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================
# FLASK SERVİS KATMANI - API ENDPOINTS
# ============================================
@app.route('/api/isg-control', methods=['POST'])
def analyze_isg_control():
    """İSG Periyodik Kontrol analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere bir İSG periyodik kontrol raporu sağlayın'
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected', 'message': 'Lütfen bir dosya seçin'}), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Sadece PDF, JPG, JPEG ve PNG dosyaları kabul edilir'
            }), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"İSG Periyodik Kontrol raporu kontrol ediliyor: {filename}")

            # Create analyzer instance with app context
            analyzer = ISGPeriyodikKontrolAnalyzer(app=app)
            
            logger.info(f"Üç aşamalı İSG kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                logger.info("Aşama 1: İlk sayfa İSG özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath, analyzer.validation_keywords):
                    logger.info("✅ Aşama 1 geçti - İSG özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath, analyzer.validation_keywords):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - İSG değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya İSG periyodik kontrol raporu değil (farklı rapor türü tespit edildi).',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'ISG_PERIYODIK_KONTROL_RAPORU'
                            }
                        }), 400
                    else:
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
                            
                            if not validate_document_server(text, analyzer.validation_keywords):
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                                return jsonify({
                                    'error': 'Invalid document type',
                                    'message': 'Yüklediğiniz dosya İSG periyodik kontrol raporu değil!',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_ISG_REPORT',
                                        'required_type': 'ISG_PERIYODIK_KONTROL_RAPORU'
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

            elif file_ext in ['.jpg', '.jpeg', '.png']:
                if not tesseract_available:
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    
                    return jsonify({
                        'error': 'OCR not available',
                        'message': 'Tesseract OCR kurulu değil. Resim dosyalarını analiz edebilmek için Tesseract kurulumu gereklidir.',
                        'details': {'tesseract_error': tesseract_info, 'file_type': file_ext, 'requires_ocr': True}
                    }), 500

            logger.info(f"İSG periyodik kontrol raporu doğrulandı, analiz başlatılıyor: {filename}")
            report = analyzer.analyze_isg_kontrol(filepath)
            
            try:
                os.remove(filepath)
                logger.info(f"Uploaded file {filename} cleaned up")
            except Exception as e:
                logger.warning(f"Could not remove uploaded file {filename}: {e}")

            if 'error' in report:
                return jsonify({
                    'error': 'Analysis failed',
                    'message': report['error'],
                    'details': {'filename': filename, 'analysis_details': report.get('details', {})}
                }), 400

            overall_percentage = report['summary']['percentage']
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': report.get('analysis_date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'analysis_details': {
                    'found_criteria': len([c for c in report['scoring']['category_scores'].values() if isinstance(c, dict) and c.get('score', 0) > 0]),
                    'total_criteria': len(report['scoring']['category_scores']),
                    'percentage': round(overall_percentage, 1)
                },
                'analysis_id': f"isg_kontrol_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': {
                    'is_valid': report['date_validity']['is_valid'],
                    'message': report['date_validity']['message'],
                    'days_old': report['date_validity']['days_old'],
                    'formatted_date': report['date_validity']['formatted_date']
                },
                'extracted_values': report['extracted_values'],
                'file_type': 'ISG_PERIYODIK_KONTROL_RAPORU',
                'filename': filename,
                'language_info': {
                    'detected_language': report['file_info']['detected_language'].upper(),
                    'file_type': file_ext.replace('.', '')
                },
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': report['summary']['total_score'],
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'report_type': report['summary']['report_type']
                },
                'recommendations': report['recommendations'],
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_isg(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_isg(report)
                }
            }
            
            for category, score_data in report['scoring']['category_scores'].items():
                if isinstance(score_data, dict):
                    response_data['category_scores'][category] = {
                        'score': score_data.get('score', 0),
                        'max_score': score_data.get('max_score', score_data.get('weight', 0)),
                        'percentage': score_data.get('percentage', 0),
                        'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'CONDITIONAL' if score_data.get('percentage', 0) >= 50 else 'FAIL'
                    }

            return jsonify({
                'success': True,
                'message': 'İSG Periyodik Kontrol Raporu başarıyla analiz edildi',
                'analysis_service': 'isg_periyodik_kontrol',
                'service_description': 'İSG Periyodik Kontrol Analizi',
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
                'message': f'İSG periyodik kontrol raporu analizi sırasında hata oluştu: {str(analysis_error)}',
                'details': {
                    'error_type': type(analysis_error).__name__,
                    'file_processed': filename if 'filename' in locals() else 'unknown'
                }
            }), 500

    except Exception as e:
        logger.error(f"API endpoint hatası: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'message': f'Sunucu hatası: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint - İSG için"""
    return jsonify({
        'status': 'healthy',
        'service': 'ISG Periodic Control Report Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'tesseract_info': tesseract_info,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'report_type': 'ISG_PERIYODIK_KONTROL_RAPORU',
        'regulations': 'İş Sağlığı ve Güvenliği Yönetmeliği'
    })


@app.route('/', methods=['GET'])
def index():
    """Ana sayfa - API bilgileri - İSG için"""
    return jsonify({
        'service': 'ISG Periodic Control Report Analyzer API',
        'version': '1.0.0',
        'description': 'İSG Periyodik Kontrol Raporlarını analiz eden REST API servisi',
        'endpoints': {
            'POST /api/isg-control': 'İSG Periyodik Kontrol raporu analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /': 'Bu bilgi sayfası'
        },
        'usage': {
            'upload_format': 'multipart/form-data',
            'file_field': 'file',
            'supported_types': list(ALLOWED_EXTENSIONS),
            'max_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        },
        'scoring': {
            'PASS': '≥70% - Yönetmelik gereksinimlerine uygun',
            'CONDITIONAL': '50-69% - Kabul edilebilir',
            'FAIL': '<50% - Yönetmelik gereksinimlerine uygun değil'
        }
    })


# ============================================
# DATABASE INITIALIZATION
# ============================================
with app.app_context():
    db.init_app(app)


# ============================================
# APPLICATION ENTRY POINT (Azure-Friendly)
# ============================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("İSG Periyodik Kontrol Analiz Servisi")
    logger.info("=" * 60)
    logger.info(f"📁 Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"📊 Desteklenen formatlar: {', '.join(ALLOWED_EXTENSIONS)}")
    logger.info(f"📏 Maksimum dosya boyutu: {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB")
    logger.info(f"🔍 Tesseract OCR: {'Kurulu' if tesseract_available else 'Kurulu değil'}")
    logger.info("")
    logger.info("🔗 API Endpoints:")
    logger.info("  POST /api/isg-control - İSG Periyodik Kontrol raporu analizi")
    logger.info("  GET /api/health - Servis sağlık kontrolü")
    logger.info("  GET / - API bilgileri")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8009))
    
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )