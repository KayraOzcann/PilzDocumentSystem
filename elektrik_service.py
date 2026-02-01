"""
Elektrik Devre Şeması Analiz Servisi
====================================
Azure App Service için optimize edilmiş standalone servis
Database-driven configuration ile dinamik pattern yönetimi

Endpoint: POST /api/elektrik-report
Health: GET /api/health
"""

import re
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import PyPDF2
from docx import Document
import pandas as pd
from dataclasses import dataclass, asdict
import logging
import math
from collections import Counter
from PIL import Image
import cv2
import numpy as np
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ComponentDetection:
    component_type: str
    label: str
    position: Tuple[int, int]
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    
@dataclass
class CircuitAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: float
    max_score: float
    details: Dict[str, Any]
    visual_evidence: List[ComponentDetection]

class AdvancedElectricCircuitAnalyzer:
    
    def __init__(self, app=None):
        logger.info("Elektrik Devre Şeması Analiz Sistemi başlatılıyor...")
        
        # Flask app context varsa DB'den yükle, yoksa boş başlat
        if app:
            with app.app_context():
                try:
                    config = load_service_config('electric_circuit')
                    
                    # DB'den yüklenen veriler
                    self.electric_criteria_weights = config.get('criteria_weights', {})
                    self.electric_criteria_details = config.get('criteria_details', {})
                    self.pattern_definitions = config.get('pattern_definitions', {})
                    self.validation_keywords = config.get('validation_keywords', {})
                    self.category_actions = config.get('category_actions', {})
                    
                    # component_templates pattern_definitions içinde
                    self.component_templates = self.pattern_definitions.get('electric', {})
                    
                    logger.info(f"✅ Veritabanından yüklendi: {len(self.electric_criteria_weights)} kategori")
                    
                except Exception as e:
                    logger.error(f"⚠️ Veritabanından yükleme başarısız: {e}")
                    logger.warning("⚠️ Fallback: Boş config kullanılıyor")
                    self.electric_criteria_weights = {}
                    self.electric_criteria_details = {}
                    self.pattern_definitions = {}
                    self.validation_keywords = {}
                    self.category_actions = {}
                    self.component_templates = {}
        else:
            # Flask app yoksa boş başlat
            logger.warning("⚠️ Flask app context yok, boş config kullanılıyor")
            self.electric_criteria_weights = {}
            self.electric_criteria_details = {}
            self.pattern_definitions = {}
            self.validation_keywords = {}
            self.category_actions = {}
            self.component_templates = {}

    def _preprocess_image_for_ocr(self, img: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrasted = clahe.apply(denoised)
            binary = cv2.adaptiveThreshold(
                contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 8
            )
            kernel = np.ones((2,2), np.uint8)
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            edges = cv2.Canny(morph, 50, 150)
            enhanced = cv2.addWeighted(morph, 0.7, edges, 0.3, 0)
            return Image.fromarray(enhanced)
        except Exception as e:
            logger.warning(f"Advanced image preprocessing failed: {e}")
            return img
            
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text and symbols from PDF using PyMuPDF and OCR"""
        try:
            import fitz  # PyMuPDF
            import io
            
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file does not exist: {pdf_path}")
                return ""
                
            if not os.access(pdf_path, os.R_OK):
                logger.error(f"PDF file is not readable: {pdf_path}")
                return ""
            
            text = ""
            try:
                pdf_document = fitz.open(pdf_path)
                
                # Configure OCR for better symbol recognition
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,-_/\\()[]{}+=<>~!@#$%^&*⏚⊥Ω∆±→←↑↓~≈⎓⌁∿⚡⤾⟲⥀▶►⊳⊲△↧●○◯⊙⟶⇾▭□⊏⊐|\\'
                
                for page_num in range(len(pdf_document)):
                    try:
                        page = pdf_document[page_num]
                        
                        # First try direct text extraction
                        try:
                            page_text = page.get_text("text", flags=2)
                        except:
                            page_text = ""
                        
                        # If no text or minimal text, use OCR
                        if not page_text.strip() or len(page_text.strip()) < 50:
                            zoom = 4.0
                            mat = fitz.Matrix(zoom, zoom)
                            try:
                                pix = page.get_pixmap(matrix=mat, alpha=False)
                            except:
                                try:
                                    pix = page.get_pixmap(zoom=zoom, alpha=False)
                                except:
                                    logger.warning(f"Could not get pixmap for page {page_num + 1}")
                                    continue
                            
                            img_data = pix.tobytes("png")
                            img = Image.open(io.BytesIO(img_data))
                            processed_img = self._preprocess_image_for_ocr(img)
                            page_text = pytesseract.image_to_string(processed_img, config=custom_config)
                        
                        page_text = self._normalize_electrical_text(page_text)
                        text += page_text + "\n"
                        
                        logger.info(f"Successfully extracted text from page {page_num + 1}")
                        
                    except Exception as page_error:
                        logger.warning(f"Failed to process page {page_num + 1}: {str(page_error)}")
                        continue
                
                pdf_document.close()
                
                if not text.strip():
                    logger.warning(f"No text extracted from PDF: {pdf_path}")
                    return ""
                
                logger.info(f"Successfully extracted text from PDF: {pdf_path}")
                logger.info(f"Extracted text length: {len(text)} characters")
                    
                return text
                
            except Exception as doc_error:
                logger.error(f"Failed to process PDF document: {str(doc_error)}")
                return ""
                
        except ImportError as imp_error:
            logger.error(f"Required library not found: {str(imp_error)}")
            logger.error("Please install: pip install PyMuPDF opencv-python Pillow pytesseract")
            return ""
        except Exception as e:
            logger.error(f"PDF text extraction error: {str(e)}")
            return ""

    def _process_electrical_symbols(self, text: str) -> str:
        """Process and normalize electrical symbols - DB'den"""
        # DB'den symbol_map al
        symbol_map = self.pattern_definitions.get('symbol_map', {})
        
        for symbol, replacement in symbol_map.items():
            text = text.replace(symbol, f' {replacement} ')
        return text

    def _normalize_electrical_text(self, text: str) -> str:
        """Normalize electrical terms and measurements - DB'den"""
        # DB'den unit_map al
        unit_map_data = self.pattern_definitions.get('unit_map', {})
        
        # unit_map formatı: {pattern: [replacement]}
        unit_map = {}
        for pattern, replacement_list in unit_map_data.items():
            if replacement_list:
                unit_map[pattern] = replacement_list[0]
        
        for pattern, replacement in unit_map.items():
            text = re.sub(pattern, replacement, text)
        
        text = text.replace('—', '-').replace('"', '"').replace('"', '"')
        text = text.replace('´', "'")
        text = re.sub(r'[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]+', ' ', text)
        return text.strip()

    def extract_images_from_pdf(self, pdf_path: str) -> List[Any]:
        """Extract images from PDF for symbol recognition"""
        try:
            import fitz
            import io
            
            images = []
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    
                    zoom = 2.0
                    mat = fitz.Matrix(zoom, zoom)
                    try:
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                    except:
                        try:
                            pix = page.get_pixmap(zoom=zoom, alpha=False)
                        except:
                            logger.warning(f"Could not get pixmap for page {page_num + 1}")
                            continue
                    
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    binary = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                    denoised = cv2.fastNlMeansDenoising(binary)
                    
                    image_data = {
                        'data': cv2.imencode('.png', denoised)[1].tobytes(),
                        'size': (denoised.shape[1], denoised.shape[0]),
                        'format': 'png',
                        'page': page_num
                    }
                    images.append(image_data)
                    
                    logger.info(f"Successfully extracted image from page {page_num + 1}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process page {page_num + 1}: {e}")
                    continue
            
            pdf_document.close()
            return images
            
        except Exception as e:
            logger.error(f"Image extraction error: {e}")
            return []

    def perform_ocr_on_images(self, images: List[Any]) -> List[str]:
        """Perform OCR on extracted images"""
        try:
            ocr_results = []
            for img_data in images:
                try:
                    nparr = np.frombuffer(img_data['data'], np.uint8)
                    img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    img_cv = clahe.apply(img_cv)
                    img_cv = cv2.fastNlMeansDenoising(img_cv)
                    _, binary = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    img_pil = Image.fromarray(binary)
                    
                    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,-_/\\()[]{}+=<>~!@#$%^&*⏚⊥Ω∆±→←↑↓~≈⎓⌁∿⚡'
                    
                    text = pytesseract.image_to_string(img_pil, config=custom_config)
                    text = self._normalize_electrical_text(text)
                    
                    if 'page' in img_data:
                        text = f"[Page {img_data['page'] + 1}] {text}"
                    
                    ocr_results.append(text)
                    logger.info(f"OCR successful on page {img_data.get('page', 'unknown')}")
                    
                except Exception as e:
                    logger.warning(f"OCR failed: {e}")
                    continue
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            return []

    def detect_components_in_images(self, images: List[Any], circuit_type: str) -> List[ComponentDetection]:
        """Detect electrical components in images - OPTIMIZED VERSION"""
        try:
            detected_components = []
            templates = self.component_templates if hasattr(self, 'component_templates') else {}
            
            logger.info(f"Görsel bileşen analizi başlıyor... Toplam {len(images)} sayfa taranacak.")

            # --- OPTİMİZASYON 1: Şablonları ÖNCEDEN hazırla (Pre-calculate) ---
            cached_templates = {}
            for comp_type, labels in templates.items():
                cached_templates[comp_type] = []
                for label in labels:
                    template = np.zeros((50, 100), dtype=np.uint8)
                    cv2.putText(template, label, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cached_templates[comp_type].append((label, template))

            for i, img_data in enumerate(images):
                if i % 5 == 0 or i == len(images) - 1:
                    logger.info(f"Görsel analiz: Sayfa {i+1}/{len(images)} işleniyor...")

                try:
                    nparr = np.frombuffer(img_data['data'], np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None: continue

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    binary = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                    
                    contours, _ = cv2.findContours(
                        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        if w < 30 or h < 30 or w > 500 or h > 500:
                            continue
                            
                        roi = gray[y:y+h, x:x+w]
                        
                        for comp_type, label_data in cached_templates.items():
                            for label, template in label_data:
                                if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
                                    continue

                                try:
                                    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                                    _, confidence, _, _ = cv2.minMaxLoc(result)
                                    
                                    if confidence > 0.8:
                                        detected_components.append(
                                            ComponentDetection(
                                                component_type=comp_type,
                                                label=label,
                                                position=(x+w//2, y+h//2),
                                                confidence=float(confidence),
                                                bounding_box=(x, y, w, h)
                                            )
                                        )
                                except Exception:
                                    continue

                except Exception as e:
                    logger.warning(f"Sayfa {i+1} görsel analiz hatası: {e}")
                    continue
            
            logger.info(f"Görsel analiz bitti. Toplam tespit: {len(detected_components)}")
            return detected_components
            
        except Exception as e:
            logger.error(f"Component detection error: {e}")
            return []

    def determine_circuit_type(self, text: str, images: List[Any]) -> Tuple[str, float]:
        return "electric", 1.0

    def analyze_criteria(self, text: str, ocr_text: str, detected_components: List[ComponentDetection], 
                        category: str) -> Dict[str, CircuitAnalysisResult]:
        results = {}
        criteria = self.electric_criteria_details.get(category, {})
        
        combined_text = text + " " + ocr_text
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            
            text_matches = re.findall(pattern, combined_text, re.IGNORECASE | re.MULTILINE)
            
            relevant_components = [comp for comp in detected_components 
                                 if self._is_relevant_component(comp, criterion_name)]
            
            if text_matches or relevant_components:
                content_parts = []
                if text_matches:
                    content_parts.append(f"Text: {str(text_matches[:3])}")
                if relevant_components:
                    comp_labels = [comp.label for comp in relevant_components[:5]]
                    content_parts.append(f"Components: {comp_labels}")
                
                content = " | ".join(content_parts)
                found = True
                
                text_score = min(weight * 0.8, len(text_matches) * (weight * 0.2))
                component_score = min(weight * 0.2, len(relevant_components) * (weight * 0.1))
                score = text_score + component_score
                score = min(score, weight)
            else:
                content = "Not found"
                found = False
                score = 0
            
            results[criterion_name] = CircuitAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={
                    "pattern_used": pattern,
                    "text_matches": len(text_matches) if text_matches else 0,
                    "visual_matches": len(relevant_components)
                },
                visual_evidence=relevant_components
            )
        
        return results

    def _is_relevant_component(self, component: ComponentDetection, criterion_name: str) -> bool:
        relevance_map = {
            "direnc_sembol": ["resistor"],
            "kondansator_sembol": ["capacitor"],
            "bobin_sembol": ["inductor"],
            "diyot_sembol": ["diode"],
            "transistor_sembol": ["transistor"],
            "kontaktor_rele": ["relay"],
            "motor_starter": ["motor"],
            "sigorta_sembol": ["fuse"]
        }
        relevant_types = relevance_map.get(criterion_name, [])
        return component.component_type in relevant_types
    
    def calculate_scores(self, analysis_results: Dict[str, Dict[str, CircuitAnalysisResult]], 
                        circuit_type: str) -> Dict[str, Any]:
        category_scores = {}
        total_score = 0

        for category, results in analysis_results.items():
            category_max = self.electric_criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())

            if category_possible > 0:
                raw_percentage = category_earned / category_possible
                adjusted_percentage = math.pow(raw_percentage, 0.7)
                normalized_score = adjusted_percentage * category_max
            else:
                normalized_score = 0

            category_scores[category] = {
                "earned": category_earned,
                "possible": category_possible,
                "normalized": round(normalized_score, 2),
                "max_weight": category_max,
                "percentage": round((category_earned / category_possible * 100), 2) if category_possible > 0 else 0
            }

            total_score += normalized_score

        final_score = min(100, total_score * 1.1)

        return {
            "category_scores": category_scores,
            "total_score": round(final_score, 2),
            "overall_percentage": round(final_score, 2)
        }

    def extract_specific_values(self, text: str, circuit_type: str) -> Dict[str, Any]:
        """Extract specific values - DB'den pattern'ler ile"""
        values = {
            "proje_no": "Not found",
            "sistem_tipi": "Not found",
            "tarih": "Not found",
            "elektrik_paneli": "Not found",
            "voltaj": "Not found",
            "akim": "Not found",
            "guc": "Not found",
            "frekans": "Not found",
            "klemens_blogu": "Not found"
        }
        
        # DB'den pattern'leri al
        extract_values = self.pattern_definitions.get('extract_values', {})
        
        for key in values.keys():
            pattern = extract_values.get(key, '')
            if pattern:
                match = re.search(pattern, text)
                if match:
                    if match.groups():
                        values[key] = next((m for m in match.groups() if m), match.group())
                    else:
                        values[key] = match.group()
        
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict, circuit_type: str) -> List[str]:
        recommendations = []
        
        valid_criteria = sum(1 for category, results in analysis_results.items() 
                           for result in results.values() if result.found)
        total_criteria = sum(len(results) for results in analysis_results.values())
        
        recommendations.append(f"⚡ Elektrik Geçerlilik: %{(valid_criteria/total_criteria)*100:.1f} ({valid_criteria}/{total_criteria} kriter)")

        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 40:
                recommendations.append(f"❌ {category} bölümü yetersiz (%{category_score:.1f})")
                missing_criteria = [name for name, result in results.items() if not result.found]
                if missing_criteria:
                    recommendations.append(f"  Eksik kriterler: {', '.join(missing_criteria)}")
            elif category_score < 70:
                recommendations.append(f"⚠️ {category} bölümü geliştirilmeli (%{category_score:.1f})")
            else:
                recommendations.append(f"✅ {category} bölümü yeterli (%{category_score:.1f})")

        if scores["overall_percentage"] < 70:
            recommendations.append("\n🚨 GENEL ÖNERİLER:")
            recommendations.extend([
                "- Şema IEC veya ANSI standardına uyumlu hale getirilmelidir",
                "- Elektriksel semboller eksiksiz olmalıdır",
                "- Bağlantı hatları net gösterilmelidir",
                "- Bileşenler etiketlenmelidir",
                "- Voltaj, akım ve güç değerleri belirtilmelidir"
            ])

        return recommendations

    def analyze_circuit_diagram(self, pdf_path: str) -> Dict[str, Any]:
        """Ana analiz fonksiyonu"""
        logger.info("Elektrik devre şeması analizi başlatılıyor...")

        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "PDF okunamadı"}

        images = self.extract_images_from_pdf(pdf_path)
        circuit_type, _ = self.determine_circuit_type(text, images)

        logger.info("OCR ve Görsel Analiz tek seferlik çalıştırılıyor...")
        
        ocr_text = ""
        if images:
            ocr_results = self.perform_ocr_on_images(images)
            ocr_text = " ".join(ocr_results)
            
        detected_components = self.detect_components_in_images(images, circuit_type)
        logger.info(f"Görsel analiz tamamlandı. Tespit edilen bileşen sayısı: {len(detected_components)}")

        analysis_results = {}
        for category in self.electric_criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, ocr_text, detected_components, category)

        scores = self.calculate_scores(analysis_results, circuit_type)
        extracted_values = self.extract_specific_values(text, circuit_type)
        recommendations = self.generate_recommendations(analysis_results, scores, circuit_type)

        return {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_details": {"found_criteria": sum(1 for r in analysis_results.values() for res in r.values() if res.found)},
            "category_scores": scores["category_scores"],
            "extracted_values": extracted_values,
            "overall_score": scores["overall_percentage"],
            "recommendations": recommendations,
            "total_score": scores["total_score"],
            "main_issues": []
        }


# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_document_server(text, validation_keywords):
    """Elektrik doküman validasyonu - DB'den"""
    
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
    
    category_found = []
    for i, category in enumerate(critical_terms):
        found = any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category)
        if found:
            logger.info(f"Elektrik Kategori {i+1} bulundu")
        category_found.append(found)
    
    valid = sum(category_found)
    logger.info(f"Doküman validasyonu: {valid}/{len(critical_terms)} kritik kategori")
    return valid >= len(critical_terms) - 1


def check_strong_keywords_first_pages(filepath, validation_keywords):
    """İlk sayfada elektrik özgü kelime kontrolü - OCR - DB'den"""
    
    # DB'den strong keywords al
    strong_keywords = validation_keywords.get('strong_keywords', [])
    
    if not strong_keywords:
        logger.warning("⚠️ Strong keywords bulunamadı, validasyon atlanıyor")
        return True
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=2)
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng')
            all_text += text.lower() + " "
        
        found = [kw for kw in strong_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa: {len(found)} özgü kelime bulundu")
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"OCR hatası: {e}")
        return False


def check_excluded_keywords_first_pages(filepath, validation_keywords):
    """İlk sayfada excluded keyword kontrolü - OCR - DB'den"""
    
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
        
        found = [kw for kw in excluded_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"Excluded: {len(found)} kelime bulundu")
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"Excluded OCR hatası: {e}")
        return False


def get_conclusion_message_elektrik(status, percentage):
    """Sonuç mesajı - Elektrik"""
    if status == "PASS":
        return f"Elektrik devre şeması standartlara uygundur (%{percentage:.0f})"
    return f"Elektrik devre şeması standartlara uygun değil (%{percentage:.0f})"


# ============================================
# FLASK APPLICATION
# ============================================
app = Flask(__name__)

# Database configuration (YENİ)
app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
app.config['SQLALCHEMY_ECHO'] = Config.SQLALCHEMY_ECHO

# Upload configuration
UPLOAD_FOLDER = 'temp_uploads_elektrik'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/elektrik-report', methods=['POST'])
def analyze_elektrik_report():
    """Elektrik Devre Şeması analiz endpoint'i"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided', 'message': 'Lütfen bir elektrik devre şeması sağlayın'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"Elektrik analizi başlatılıyor: {filename}")

            # Create analyzer instance with app context
            analyzer = AdvancedElectricCircuitAnalyzer(app=app)
            
            # 3 AŞAMALI KONTROL
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext == '.pdf':
                logger.info("Aşama 1: Elektrik özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath, analyzer.validation_keywords):
                    logger.info("✅ Aşama 1 geçti")
                else:
                    logger.info("Aşama 2: Excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath, analyzer.validation_keywords):
                        logger.info("❌ Excluded kelimeler bulundu")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya elektrik devre şeması değil',
                            'details': {'filename': filename, 'document_type': 'OTHER_REPORT_TYPE'}
                        }), 400
                    else:
                        # AŞAMA 3: Tam doküman kontrolü
                        logger.info("Aşama 3: Tam doküman kontrolü...")
                        try:
                            with open(filepath, 'rb') as f:
                                pdf_reader = PyPDF2.PdfReader(f)
                                text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
                            
                            if not text or len(text.strip()) < 50 or not validate_document_server(text, analyzer.validation_keywords):
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                                return jsonify({
                                    'error': 'Invalid document type',
                                    'message': 'Yüklediğiniz dosya elektrik devre şeması değil!'
                                }), 400
                        except Exception as e:
                            logger.error(f"Aşama 3 hatası: {e}")
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({'error': 'Analysis failed'}), 500

            # Analizi yap
            logger.info(f"Elektrik analizi yapılıyor: {filename}")
            report = analyzer.analyze_circuit_diagram(filepath)
            
            try:
                os.remove(filepath)
            except:
                pass

            if 'error' in report:
                return jsonify({'error': 'Analysis failed', 'message': report['error']}), 400

            overall_percentage = report.get('overall_score', 0)
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': report.get('analysis_date'),
                'analysis_details': report.get('analysis_details', {}),
                'analysis_id': f"elektrik_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': report.get('category_scores', {}),
                'date_validity': {'is_valid': True, 'message': 'Elektrik için tarih kontrolü uygulanmaz'},
                'extracted_values': report.get('extracted_values', {}),
                'file_type': 'ELEKTRIK_DEVRE_SEMASI',
                'filename': filename,
                'language_info': {'detected_language': 'turkish', 'text_length': 0},
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': report.get('total_score', 0),
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'text_quality': 'good'
                },
                'recommendations': report.get('recommendations', []),
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_elektrik(status, overall_percentage),
                    'main_issues': report.get('main_issues', [])
                }
            }

            return jsonify({
                'success': True,
                'message': 'Elektrik Devre Şeması başarıyla analiz edildi',
                'analysis_service': 'electric_circuit',
                'service_description': 'Elektrik Devre Şeması Analizi',
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
        'service': 'Electric Circuit Analyzer API',
        'version': '1.0.0',
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'report_type': 'ELEKTRIK_DEVRE_SEMASI'
    })


@app.route('/', methods=['GET'])
def index():
    """API bilgileri"""
    return jsonify({
        'service': 'Electric Circuit Analyzer API',
        'version': '1.0.0',
        'description': 'Elektrik Devre Şemalarını analiz eden REST API servisi',
        'endpoints': {
            'POST /api/elektrik-report': 'Elektrik devre şeması analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /': 'Bu bilgi sayfası'
        },
        'usage': {
            'upload_format': 'multipart/form-data',
            'file_field': 'file',
            'supported_types': list(ALLOWED_EXTENSIONS),
            'max_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
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
    logger.info("Elektrik Devre Şeması Analiz Servisi")
    logger.info("=" * 60)
    logger.info(f"📁 Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"📊 Desteklenen formatlar: {', '.join(ALLOWED_EXTENSIONS)}")
    logger.info(f"📏 Maksimum dosya boyutu: {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB")
    logger.info("")
    logger.info("🔗 API Endpoints:")
    logger.info("  POST /api/elektrik-report - Elektrik devre şeması analizi")
    logger.info("  GET /api/health - Servis sağlık kontrolü")
    logger.info("  GET / - API bilgileri")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8001))
    
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)