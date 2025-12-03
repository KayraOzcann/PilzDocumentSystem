#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Report Checker (Kullanım Kılavuzu Analiz Sistemi)
Created for analyzing operating manuals from various companies
Supports both Turkish and English with OCR capabilities
Azure App Service için optimize edilmiş standalone servis
Database-driven configuration ile dinamik pattern yönetimi
"""

import re
import os
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

try:
    from langdetect import detect
    LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ManualAnalysisResult:
    """Kullanma Kılavuzu analiz sonucu veri sınıfı"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

class ManualReportAnalyzer:
    """Kullanma Kılavuzu rapor analiz sınıfı"""
    
    def __init__(self, app=None):
        logger.info("Kullanma Kılavuzu analiz sistemi başlatılıyor...")
        
        # Flask app context varsa DB'den yükle, yoksa boş başlat
        if app:
            with app.app_context():
                try:
                    config = load_service_config('assembly_instructions')
                    
                    # DB'den yüklenen veriler
                    self.criteria_weights = config.get('criteria_weights', {})
                    self.criteria_details = config.get('criteria_details', {})
                    self.pattern_definitions = config.get('pattern_definitions', {})
                    self.validation_keywords = config.get('validation_keywords', {})
                    self.category_actions = config.get('category_actions', {})
                    
                    logger.info(f"✅ Veritabanından yüklendi: {len(self.criteria_weights)} kategori")
                    
                except Exception as e:
                    logger.error(f"⚠️ Veritabanından yükleme başarısız: {e}")
                    logger.warning("⚠️ Fallback: Boş config kullanılıyor")
                    self.criteria_weights = {}
                    self.criteria_details = {}
                    self.pattern_definitions = {}
                    self.validation_keywords = {}
                    self.category_actions = {}
        else:
            # Flask app yoksa boş başlat
            logger.warning("⚠️ Flask app context yok, boş config kullanılıyor")
            self.criteria_weights = {}
            self.criteria_details = {}
            self.pattern_definitions = {}
            self.validation_keywords = {}
            self.category_actions = {}
    
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
        """PDF'den metin çıkarma - PyPDF2 ve OCR ile"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    page_text = re.sub(r'\s+', ' ', page_text)
                    page_text = page_text.replace('|', ' ')
                    text += page_text + "\n"
                
                text = text.replace('—', '-')
                text = text.replace('"', '"').replace('"', '"')
                text = text.replace('´', "'")
                text = re.sub(r'[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]+', ' ', text)
                text = text.strip()
                
                if len(text) > 50:
                    logger.info("Metin PyPDF2 ile çıkarıldı")
                    return text
                
                logger.info("PyPDF2 ile yeterli metin bulunamadı, OCR deneniyor...")
                return self.extract_text_with_ocr(pdf_path)
                
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {e}")
            logger.info("OCR'a geçiliyor...")
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
                    text = text.replace('|', ' ')
                    all_text += text + "\n"
                    
                    logger.info(f"OCR ile sayfa {i+1}'den {len(text)} karakter çıkarıldı")
                    
                except Exception as page_error:
                    logger.error(f"Sayfa {i+1} OCR hatası: {page_error}")
                    continue
            
            all_text = all_text.replace('—', '-')
            all_text = all_text.replace('"', '"').replace('"', '"')
            all_text = all_text.replace('´', "'")
            all_text = re.sub(r'[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]+', ' ', all_text)
            all_text = all_text.strip()
            
            logger.info(f"OCR toplam metin uzunluğu: {len(all_text)}")
            return all_text
            
        except Exception as e:
            logger.error(f"OCR metin çıkarma hatası: {e}")
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
                details={
                    "pattern_used": pattern,
                    "matches_count": len(matches) if matches else 0
                }
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
        """Özel değerleri çıkar - DB'den pattern'ler ile"""
        values = {
            "manual_name": "Bulunamadı",
            "product_model": "Bulunamadı",
            "revision_info": "Bulunamadı",
            "manufacturer": "Bulunamadı",
            "contact_info": "Bulunamadı",
            "safety_warnings_count": 0
        }
        
        # DB'den extract_values pattern'lerini al
        extract_values = self.pattern_definitions.get('extract_values', {})
        
        # Manual name extraction
        manual_patterns = extract_values.get('manual_namei', [])
        for pattern in manual_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["manual_name"] = match.group(0).strip()
                break
        
        # Product model
        model_patterns = extract_values.get('product_model', [])
        for pattern in model_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if match.groups():
                    values["product_model"] = match.group(1).strip()
                else:
                    values["product_model"] = match.group(0).strip()
                break
        
        # Safety warnings count
        safety_patterns = extract_values.get('safety_warnings_count', [])
        safety_count = 0
        for pattern in safety_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            safety_count += len(matches)
        
        values["safety_warnings_count"] = safety_count
        
        return values
    
    def generate_recommendations(self, analysis_results: Dict[str, Dict[str, ManualAnalysisResult]], scores: Dict[str, Any]) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        
        for category, score_data in scores["category_scores"].items():
            if score_data["percentage"] < 50:
                recommendations.append(f"⚠️ {category} kategorisinde ciddi eksiklikler var (%{score_data['percentage']:.0f})")
            elif score_data["percentage"] < 70:
                recommendations.append(f"📝 {category} kategorisi geliştirilebilir (%{score_data['percentage']:.0f})")
        
        missing_critical = []
        for category, results in analysis_results.items():
            for criterion_name, result in results.items():
                if not result.found and result.max_score >= 4:
                    missing_critical.append(f"{category}: {criterion_name}")
        
        if missing_critical:
            recommendations.append("🔍 Eksik kritik kriterler:")
            for item in missing_critical[:5]:
                recommendations.append(f"  • {item}")
        
        total_percentage = scores["percentage"]
        if total_percentage >= 80:
            recommendations.append("✅ Kullanım kılavuzu yüksek kalitede ve standartlara uygun")
        elif total_percentage >= 70:
            recommendations.append("📋 Kullanım kılavuzu kabul edilebilir seviyede")
        else:
            recommendations.append("❌ Kullanım kılavuzu yetersiz, kapsamlı revizyon gerekli")
        
        return recommendations
    
    def analyze_manual(self, pdf_path: str) -> Dict[str, Any]:
        """Ana analiz fonksiyonu"""
        logger.info("Kullanım kılavuzu analizi başlıyor...")
        
        try:
            text = self.extract_text_from_pdf(pdf_path)
            
            if len(text.strip()) < 50:
                return {
                    "error": "PDF'den yeterli metin çıkarılamadı. Dosya bozuk olabilir veya sadece resim içeriyor olabilir.",
                    "text_length": len(text)
                }
            
            detected_language = self.detect_language(text)
            logger.info(f"Tespit edilen dil: {detected_language}")
            
            extracted_values = self.extract_specific_values(text)
            
            category_analyses = {}
            for category in self.criteria_weights.keys():
                category_analyses[category] = self.analyze_criteria(text, category)
            
            scoring = self.calculate_scores(category_analyses)
            recommendations = self.generate_recommendations(category_analyses, scoring)
            
            percentage = scoring["percentage"]
            if percentage >= 70:
                status = "PASS"
                status_tr = "GEÇERLİ"
            else:
                status = "FAIL"
                status_tr = "YETERSİZ"
            
            return {
                "analysis_date": datetime.now().isoformat(),
                "file_info": {
                    "filename": os.path.basename(pdf_path),
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
                    "report_type": "Montaj Talimatları"
                }
            }
            
        except Exception as e:
            logger.error(f"Analiz hatası: {e}")
            return {
                "error": f"Analiz sırasında hata oluştu: {str(e)}",
                "analysis_date": datetime.now().isoformat()
            }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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

def validate_document_server(text, validation_keywords):
    """Server kodunda doküman validasyonu - Montaj Talimatları için - DB'den"""
    
    # DB'den critical_terms al
    critical_terms_data = validation_keywords.get('critical_terms', {})
    
    # Liste formatına dönüştür
    critical_terms = []
    for key, value in critical_terms_data.items():
        if key.startswith('category_') and isinstance(value, list):
            critical_terms.append(value)
    
    if not critical_terms:
        logger.warning("⚠️ Critical terms bulunamadı, validasyon atlanıyor")
        return True
    
    # Her kategori için kontrol
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"Montaj Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    # Tüm kategorilerden en az bir terim bulunmalı
    valid_categories = sum(category_found)
    
    logger.info(f"Doküman validasyonu: {valid_categories}/{len(critical_terms)} kritik kategori bulundu")
    
    # Tüm kategorilerde terim bulunmalı (n-1 kuralı yerine hepsi)
    return valid_categories >= len(critical_terms)

def check_strong_keywords_first_pages(filepath, validation_keywords):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - Montaj Talimatları için - DB'den"""
    
    # DB'den strong keywords al
    strong_keywords = validation_keywords.get('strong_keywords', [])
    
    if not strong_keywords:
        logger.warning("⚠️ Strong keywords bulunamadı, validasyon atlanıyor")
        return True
    
    try:
        pages = convert_from_path(filepath, dpi=200, first_page=1, last_page=1)
        
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
    """İlk 1-2 sayfada istenmeyen rapor türlerinin kelimelerini ara - DB'den"""
    
    # DB'den excluded keywords al
    excluded_keywords = validation_keywords.get('excluded_keywords', [])
    
    if not excluded_keywords:
        logger.warning("⚠️ Excluded keywords bulunamadı, validasyon atlanıyor")
        return False
    
    try:
        pages = convert_from_path(filepath, dpi=200, first_page=1, last_page=2)
        
        all_text = ""
        for i, page in enumerate(pages):
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng')
            all_text += text.lower() + " "
        
        # OCR text'ini logla
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

def get_conclusion_message(status, percentage):
    """Sonuç mesajını döndür - Montaj için"""
    if status == "PASS":
        return f"Montaj talimatları yüksek kalitede ve standartlara uygun (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"Montaj talimatları kabul edilebilir ancak iyileştirme gerekli (%{percentage:.0f})"
    else:
        return f"Montaj talimatları yetersiz, kapsamlı revizyon gerekli (%{percentage:.0f})"

def get_main_issues(analysis_result):
    """Ana sorunları listele - Montaj için"""
    issues = []
    
    for category, score_data in analysis_result['scoring']['category_scores'].items():
        if score_data['percentage'] < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    # Eğer hiç kritik sorun yoksa genel sorunları ekle
    if not issues:
        if analysis_result['scoring']['total_score'] < 50:
            issues = [
                "Güvenlik bilgileri yetersiz",
                "Montaj adımları eksik veya belirsiz",
                "Gerekli araçlar ve malzemeler belirtilmemiş",
                "Teknik detaylar yetersiz"
            ]
    
    return issues[:4]  # En fazla 4 ana sorun göster

# ============================================================================
# FLASK APP - Azure App Service için
# ============================================================================

app = Flask(__name__)

# Database configuration (YENİ)
app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
app.config['SQLALCHEMY_ECHO'] = Config.SQLALCHEMY_ECHO

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads_montaj'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'docx', 'doc', 'txt'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/assembly-instructions', methods=['POST'])
def analyze_manual():
    """Montaj Talimatları analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere Montaj Talimatları sağlayın'
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Lütfen bir dosya seçin'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Sadece PDF, JPG, JPEG, PNG, DOCX, DOC ve TXT dosyaları kabul edilir'
            }), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"Montaj Talimatları kontrol ediliyor: {filename}")

            # Create analyzer instance with app context
            analyzer = ManualReportAnalyzer(app=app)
            
            # ÜÇ AŞAMALI MONTAJ KONTROLÜ
            logger.info(f"Üç aşamalı montaj kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()
            requires_ocr = file_ext in ['.jpg', '.jpeg', '.png']

            if file_ext == '.pdf':
                # PDF için üç aşamalı kontrol (OCR dahil)
                logger.info("Aşama 1: İlk sayfa montaj özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath, analyzer.validation_keywords):
                    logger.info("✅ Aşama 1 geçti - Montaj özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath, analyzer.validation_keywords):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - Montaj değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya montaj talimatları değil (farklı rapor türü tespit edildi). Lütfen montaj talimatları yükleyiniz.',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'MONTAJ_TALIMATLARI'
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
                            
                            if not validate_document_server(text, analyzer.validation_keywords):
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                                return jsonify({
                                    'error': 'Invalid document type',
                                    'message': 'Yüklediğiniz dosya montaj talimatları değil! Lütfen geçerli montaj talimatları yükleyiniz.',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_MONTAJ_DOCUMENT',
                                        'required_type': 'MONTAJ_TALIMATLARI'
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

            elif requires_ocr:
                # Resim dosyaları için OCR kontrolü
                if not tesseract_available:
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    
                    return jsonify({
                        'error': 'OCR not available',
                        'message': 'Tesseract OCR kurulu değil. Resim dosyalarını analiz edebilmek için Tesseract kurulumu gereklidir.',
                        'details': {
                            'tesseract_error': tesseract_info,
                            'file_type': file_ext,
                            'requires_ocr': True
                        }
                    }), 500

            # Buraya kadar geldiyse montaj talimatları, şimdi analizi yap
            logger.info(f"Montaj talimatları doğrulandı, analiz başlatılıyor: {filename}")
            analysis_result = analyzer.analyze_manual(filepath)
            
            # Clean up the uploaded file
            try:
                os.remove(filepath)
                logger.info(f"Uploaded file {filename} cleaned up")
            except Exception as e:
                logger.warning(f"Could not remove uploaded file {filename}: {e}")

            # Check if analysis was successful
            if 'error' in analysis_result:
                return jsonify({
                    'error': 'Analysis failed',
                    'message': analysis_result['error'],
                    'details': {
                        'filename': filename,
                        'text_length': analysis_result.get('text_length', 0)
                    }
                }), 400

            # Extract key results for API response
            overall_percentage = analysis_result['summary']['percentage']
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': analysis_result.get('analysis_date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'analysis_details': {
                    'found_criteria': len([r for results in analysis_result.get('category_analyses', {}).values() 
                                         for r in results.values() if r.found]),
                    'total_criteria': len([r for results in analysis_result.get('category_analyses', {}).values() 
                                         for r in results.values()]),
                    'percentage': round((len([r for results in analysis_result.get('category_analyses', {}).values() 
                                            for r in results.values() if r.found]) / 
                                       max(1, len([r for results in analysis_result.get('category_analyses', {}).values() 
                                                 for r in results.values()])) * 100), 1)
                },
                'analysis_id': f"montaj_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': 'N/A',
                'extracted_values': analysis_result.get('extracted_values', {}),
                'file_type': 'MONTAJ_TALIMATLARI',
                'filename': filename,
                'language_info': {
                    'detected_language': analysis_result['file_info']['detected_language'],
                    'text_length': analysis_result['file_info']['text_length']
                },
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': analysis_result['scoring']['total_score'],
                    'max_points': 100,
                    'status': status,
                    'status_tr': analysis_result['summary']['status_tr'],
                    'text_quality': 'good' if analysis_result['file_info']['text_length'] > 1000 else 'fair'
                },
                'recommendations': analysis_result.get('recommendations', []),
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues(analysis_result)
                }
            }
            
            # Add category scores
            for category, score_data in analysis_result['scoring']['category_scores'].items():
                response_data['category_scores'][category] = {
                    'score': score_data['normalized'],
                    'max_score': score_data['max_weight'],
                    'percentage': score_data['percentage'],
                    'status': 'PASS' if score_data['percentage'] >= 70 else 'CONDITIONAL' if score_data['percentage'] >= 50 else 'FAIL'
                }

            return jsonify({
                'success': True,
                'message': 'Montaj Talimatları başarıyla analiz edildi',
                'analysis_service': 'montaj_talimatları',
                'service_description': 'Montaj Talimatları Analizi',
                'data': response_data
            })

        except Exception as analysis_error:
            # Clean up file on error
            try:
                if 'filepath' in locals():
                    os.remove(filepath)
            except:
                pass
            
            logger.error(f"Analiz hatası: {str(analysis_error)}")
            return jsonify({
                'error': 'Analysis failed',
                'message': f'Montaj talimatları analizi sırasında hata oluştu: {str(analysis_error)}',
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
    """Health check endpoint - Montaj için"""
    return jsonify({
        'status': 'healthy',
        'service': 'Montaj Talimatları Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'tesseract_info': tesseract_info,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': True,
        'report_type': 'MONTAJ_TALIMATLARI'
    })

@app.route('/api/test-analysis', methods=['GET'])
def test_analysis():
    """Test endpoint for debugging - Montaj için"""
    try:
        analyzer = ManualReportAnalyzer(app=app)
        test_info = {
            'analyzer_initialized': True,
            'available_methods': [method for method in dir(analyzer) if not method.startswith('_')],
            'criteria_categories': list(analyzer.criteria_weights.keys()) if hasattr(analyzer, 'criteria_weights') else [],
            'criteria_weights': analyzer.criteria_weights if hasattr(analyzer, 'criteria_weights') else {},
            'tesseract_status': tesseract_available,
            'ocr_support': True
        }
        
        return jsonify({
            'success': True,
            'message': 'Test başarılı',
            'data': test_info
        })
    except Exception as e:
        return jsonify({
            'error': 'Test failed',
            'message': str(e)
        }), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Montaj analiz kategorilerini döndür"""
    try:
        analyzer = ManualReportAnalyzer(app=app)
        return jsonify({
            'success': True,
            'data': {
                'categories': analyzer.criteria_weights,
                'total_weight': sum(analyzer.criteria_weights.values()),
                'criteria_details': {
                    category: list(criteria.keys()) 
                    for category, criteria in analyzer.criteria_details.items()
                },
                'standard_reference': 'Montaj ve Kurulum Standartları'
            }
        })
    except Exception as e:
        return jsonify({
            'error': 'Failed to get categories',
            'message': str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """API bilgileri"""
    return jsonify({
        'service': 'Montaj Talimatları Analyzer API',
        'version': '1.0.0',
        'description': 'Montaj Talimatlarını analiz eden REST API servisi',
        'endpoints': {
            'POST /api/assembly-instructions': 'Montaj talimatları analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /api/test-analysis': 'Test analizi',
            'GET /api/categories': 'Analiz kategorileri',
            'GET /': 'Bu bilgi sayfası'
        },
        'usage': {
            'upload_format': 'multipart/form-data',
            'file_field': 'file',
            'supported_types': list(ALLOWED_EXTENSIONS),
            'max_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        }
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large',
        'message': 'Dosya boyutu 50MB limitini aşıyor'
    }), 413

@app.errorhandler(400)
def bad_request(e):
    return jsonify({
        'error': 'Bad request',
        'message': 'Geçersiz istek'
    }), 400

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': 'Sunucu hatası oluştu'
    }), 500

# ============================================
# DATABASE INITIALIZATION
# ============================================
with app.app_context():
    db.init_app(app)

# ============================================
# APPLICATION ENTRY POINT (Azure-Friendly)
# ============================================
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Montaj Talimatları Analyzer API")
    logger.info("=" * 60)
    logger.info(f"📁 Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"📊 Desteklenen formatlar: {', '.join(ALLOWED_EXTENSIONS)}")
    logger.info(f"📏 Maksimum dosya boyutu: {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB")
    logger.info(f"🔧 Tesseract OCR: {'Kurulu' if tesseract_available else 'Kurulu değil'}")
    logger.info("")
    logger.info("🔗 API Endpoints:")
    logger.info("  POST /api/assembly-instructions - Montaj talimatları analizi")
    logger.info("  GET  /api/health                - Sağlık kontrolü")
    logger.info("  GET  /api/test-analysis         - Test analizi")
    logger.info("  GET  /api/categories            - Analiz kategorileri")
    logger.info("  GET  /                          - API bilgileri")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8012))
    
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)