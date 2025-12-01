import re
import os
import json
import io
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import PyPDF2
import pytesseract
from PIL import Image
import pandas as pd
from dataclasses import dataclass, asdict
import logging
import math
from collections import Counter
import fitz  # PyMuPDF for better PDF handling
from flask import Flask, request, jsonify, current_app
from werkzeug.utils import secure_filename
import pdf2image

# ============================================
# DATABASE IMPORTS (YENİ)
# ============================================
from database import db, init_db
from db_loader import load_service_config

# ============================================
# CONFIG IMPORT (YENİ)
# ============================================
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ComponentDetection:
    """Detected component information"""
    component_type: str
    label: str
    position: Tuple[int, int]
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    
@dataclass
class CircuitAnalysisResult:
    """Analysis result for each criterion"""
    criteria_name: str
    found: bool
    content: str
    score: float
    max_score: float
    details: Dict[str, Any]
    visual_evidence: List[ComponentDetection]

class PneumaticCircuitAnalyzer:
    """Gelişmiş pnömatik devre şeması analiz sistemi"""
    
    def __init__(self, app=None):
        logger.info("Pnömatik Devre Şeması analiz sistemi başlatılıyor...")
        
        # Flask app context varsa DB'den yükle, yoksa boş başlat
        if app:
            with app.app_context():
                try:
                    config = load_service_config('pneumatic_circuit')
                    
                    # DB'den yüklenen veriler
                    self.pneumatic_criteria_weights = config.get('criteria_weights', {})
                    self.pneumatic_criteria_details = config.get('criteria_details', {})
                    self.pattern_definitions = config.get('pattern_definitions', {})
                    self.validation_keywords = config.get('validation_keywords', {})
                    self.category_actions = config.get('category_actions', {})
                    
                    # visual_templates pattern_definitions içinde
                    self.visual_templates = self.pattern_definitions.get('visual_templates', {})
                    
                    logger.info(f"✅ Veritabanından yüklendi: {len(self.pneumatic_criteria_weights)} kategori")
                    
                except Exception as e:
                    logger.error(f"⚠️ Veritabanından yükleme başarısız: {e}")
                    logger.warning("⚠️ Fallback: Boş config kullanılıyor")
                    self.pneumatic_criteria_weights = {}
                    self.pneumatic_criteria_details = {}
                    self.pattern_definitions = {}
                    self.validation_keywords = {}
                    self.category_actions = {}
                    self.visual_templates = {}
        else:
            # Flask app yoksa boş başlat
            logger.warning("⚠️ Flask app context yok, boş config kullanılıyor")
            self.pneumatic_criteria_weights = {}
            self.pneumatic_criteria_details = {}
            self.pattern_definitions = {}
            self.validation_keywords = {}
            self.category_actions = {}
            self.visual_templates = {}

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkarma (gelişmiş)"""
        try:
            text = ""
            # PyMuPDF ile daha iyi metin çıkarma
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            
            # Fallback olarak PyPDF2 kullan
            if not text.strip():
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            
            # Metin normalleştirme
            text = re.sub(r'\s+', ' ', text)
            text = text.replace('|', ' ')
            text = text.replace('â€"', '-')
            text = text.replace('"', '"').replace('"', '"')
            return text.strip()
            
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {e}")
            return ""

    def extract_text_from_image(self, image_path: str) -> str:
        """Gelişmiş OCR ile metin çıkarma"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return ""
            
            # Görüntü ön işleme
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Kontrast artırma
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Gürültü azaltma
            denoised = cv2.medianBlur(enhanced, 3)
            
            # Thresholding
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morfolojik işlemler
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # OCR yapılandırması - Türkçe ve İngilizce karakter desteği
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÇçĞğİıÖöŞşÜü. '
            
            # OCR uygulama
            text = pytesseract.image_to_string(morph, lang='tur+eng', config=custom_config)
            
            # Alternatif threshold ile tekrar dene
            if len(text.strip()) < 10:
                _, thresh2 = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
                text2 = pytesseract.image_to_string(thresh2, lang='tur+eng', config=custom_config)
                if len(text2.strip()) > len(text.strip()):
                    text = text2
            
            # Temizleme
            text = re.sub(r'[^\w\s\.\-\(\)\[\]\{\}\+\=\*\/]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Gelişmiş OCR hatası: {e}")
            return ""

    def detect_visual_components(self, image_path: str) -> List[ComponentDetection]:
        """Çok gelişmiş görsel bileşen tespiti - daha geniş algılama"""
        detections = []
        try:
            img = cv2.imread(image_path)
            if img is None:
                return detections
            
            # Görüntü ön işleme
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Kontrast artırma
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Gürültü azaltma
            denoised = cv2.medianBlur(enhanced, 3)
            
            # Kenar tespiti
            edges = cv2.Canny(denoised, 30, 100)
            
            # Morfolojik işlemler
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Kontur bulma - daha geniş arama
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Şekil analizi için çok daha geniş filtreler
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 20:  # Çok daha düşük minimum alan
                    continue
                    
                # Kontur özelliklerini hesapla
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                    
                # Şekil analizi
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                extent = area / (w * h) if w * h > 0 else 0
                solidity = area / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0
                
                component_type = "unknown"
                confidence = 0.2  # Daha düşük başlangıç güveni
                
                # Çok daha geniş pnömatik sembol analizi
                if len(approx) == 4:  # Dörtgen
                    if 0.7 < aspect_ratio < 1.4:  # Kare benzeri - vana
                        component_type = "valve"
                        confidence = 0.7
                    elif aspect_ratio > 1.5:  # Uzun dikdörtgen - silindir
                        component_type = "cylinder"
                        confidence = 0.8
                    elif aspect_ratio < 0.7:  # Dikey dikdörtgen - filtre/regülatör
                        component_type = "filter"
                        confidence = 0.6
                    else:  # Genel dörtgen
                        component_type = "component"
                        confidence = 0.5
                elif len(approx) == 3:  # Üçgen - akış kontrol
                    component_type = "flow_control"
                    confidence = 0.6
                elif len(approx) > 10:  # Daire benzeri - bağlantı
                    component_type = "connection"
                    confidence = 0.6
                elif 5 <= len(approx) <= 10:  # Çokgen - çeşitli bileşenler
                    component_type = "component"
                    confidence = 0.4
                else:
                    # Özel pnömatik semboller için çok daha geniş kontroller
                    if 1.2 < aspect_ratio < 4.0 and extent > 0.6:  # Silindir sembolü
                        component_type = "cylinder"
                        confidence = 0.7
                    elif aspect_ratio > 4.0:  # Bağlantı hattı
                        component_type = "connection"
                        confidence = 0.5
                    elif solidity > 0.8 and 0.8 < aspect_ratio < 1.2:  # Vana sembolü
                        component_type = "valve"
                        confidence = 0.6
                    elif area > 100:  # Büyük şekiller
                        component_type = "component"
                        confidence = 0.4
                
                # Çok daha geniş boyut filtresi
                if w > 10 and h > 10:  # Çok daha küçük minimum boyut
                    detection = ComponentDetection(
                        component_type=component_type,
                        label=f"{component_type}_{len(detections)+1}",
                        position=(x + w//2, y + h//2),
                        confidence=confidence,
                        bounding_box=(x, y, w, h)
                    )
                    detections.append(detection)
            
            # Görsel template eşleştirme - DB'den
            try:
                templates = self.pattern_definitions.get('templates', {})
                for template_type, symbols in templates.items():
                    for symbol in symbols:
                        ocr_text = self.extract_text_from_image(image_path)
                        if symbol in ocr_text:
                            detection = ComponentDetection(
                                component_type=template_type,
                                label=f"{template_type}_template_{len(detections)+1}",
                                position=(img.shape[1]//2, img.shape[0]//2),
                                confidence=0.4,
                                bounding_box=(0, 0, img.shape[1], img.shape[0])
                            )
                            detections.append(detection)
                            break
            except:
                pass
            
            # Çift tespiti önleme
            unique_detections = []
            seen_positions = set()
            for detection in detections:
                pos_key = (detection.position[0] // 30, detection.position[1] // 30)
                if pos_key not in seen_positions and detection.confidence > 0.2:
                    unique_detections.append(detection)
                    seen_positions.add(pos_key)
            
            # Eğer hiç tespit yoksa, genel bileşen olarak işaretle
            if not unique_detections and contours:
                for i, contour in enumerate(contours[:10]):
                    area = cv2.contourArea(contour)
                    if area > 50:
                        x, y, w, h = cv2.boundingRect(contour)
                        detection = ComponentDetection(
                            component_type="component",
                            label=f"component_{i+1}",
                            position=(x + w//2, y + h//2),
                            confidence=0.3,
                            bounding_box=(x, y, w, h)
                        )
                        unique_detections.append(detection)
            
            return unique_detections[:30]
            
        except Exception as e:
            logger.error(f"Çok gelişmiş görsel bileşen tespiti hatası: {e}")
            return detections

    def analyze_criteria(self, text: str, ocr_text: str, visual_detections: List[ComponentDetection], category: str) -> Dict[str, CircuitAnalysisResult]:
        """Kriter analizi - daha cömert puanlama"""
        results = {}
        criteria = self.pneumatic_criteria_details.get(category, {})
        
        combined_text = f"{text} {ocr_text}".lower()
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            description = criterion_data.get("description", criterion_name) 
            
            # Metin tabanlı eşleşme
            text_matches = re.findall(pattern, combined_text, re.IGNORECASE | re.MULTILINE)
            
            # Görsel eşleşme kontrolü
            visual_matches = []
            if criterion_name in ["silindir_actuator", "yon_kontrol_vanalar", "hava_kaynagi", "hava_hazirlama_grubu"]:
                relevant_detections = [d for d in visual_detections 
                                     if d.component_type in ["cylinder", "valve", "filter", "connection"] and d.confidence > 0.3]
                visual_matches = relevant_detections[:5]
            
            # Puan hesaplama - çok daha cömert algoritma
            base_score = 0
            
            # Metin eşleşmesi için puan
            if text_matches:
                text_score = min(weight * 0.9, len(text_matches) * (weight * 0.4))
                base_score += text_score
            
            # Görsel eşleşmesi için puan
            if visual_matches:
                visual_score = min(weight * 0.6, len(visual_matches) * (weight * 0.25))
                base_score += visual_score
            
            # Görsel bileşen varsa minimum puan garantisi
            if visual_detections and len(visual_detections) > 2:
                base_score = max(base_score, weight * 0.4)
            
            # Herhangi bir görsel bileşen varsa bonus puan
            if visual_detections:
                base_score += weight * 0.1
            
            # Çok eşleşme bonusu
            if len(text_matches) >= 2:
                base_score += weight * 0.15
            
            total_score = min(weight, base_score)
            
            found = len(text_matches) > 0 or len(visual_matches) > 0 or len(visual_detections) > 0
            
            content_parts = []
            if text_matches:
                content_parts.append(f"Metin: {str(text_matches[:3])}")
            if visual_matches:
                content_parts.append(f"Görsel: {len(visual_matches)} tespit")
            if visual_detections and not visual_matches:
                content_parts.append(f"Görsel Bileşenler: {len(visual_detections)} adet")
            
            content = " | ".join(content_parts) if content_parts else "Görsel bileşenler algılandı"
            
            results[criterion_name] = CircuitAnalysisResult(
                criteria_name=description,
                found=found,
                content=content,
                score=total_score,
                max_score=weight,
                details={
                    "pattern_used": pattern,
                    "text_matches": len(text_matches),
                    "visual_matches": len(visual_matches),
                    "total_visual_components": len(visual_detections)
                },
                visual_evidence=visual_matches
            )
        
        return results

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, CircuitAnalysisResult]]) -> Dict[str, Any]:
        """Puan hesaplama"""
        category_scores = {}
        total_score = 0
        
        for category, results in analysis_results.items():
            category_max = self.pneumatic_criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())
            
            if category_possible > 0:
                raw_percentage = category_earned / category_possible
                adjusted_percentage = math.pow(raw_percentage, 0.8)
                normalized_score = adjusted_percentage * category_max
            else:
                normalized_score = 0
            
            category_scores[category] = {
                "earned": round(category_earned, 2),
                "possible": category_possible,
                "normalized": round(normalized_score, 2),
                "max_weight": category_max,
                "percentage": round((category_earned / category_possible * 100), 2) if category_possible > 0 else 0
            }
            
            total_score += normalized_score
        
        final_score = min(100, total_score * 1.05)
        
        return {
            "category_scores": category_scores,
            "total_score": round(final_score, 2),
            "total_max_score": 100,
            "overall_percentage": round(final_score, 2)
        }

    def extract_specific_values(self, text: str, ocr_text: str) -> Dict[str, Any]:
        """Özel değer çıkarma - DB'den pattern'ler ile"""
        combined_text = f"{text} {ocr_text}"
        values = {
            "proje_adi": "Belirtilmemiş",
            "cizim_tarihi": "Belirtilmemiş",
            "tasarim_tarihi": "Belirtilmemiş",
            "onay_tarihi": "Belirtilmemiş",
            "calisma_basinci": "Belirtilmemiş",
            "sistem_tipi": "Belirtilmemiş",
            "vana_sayisi": 0,
            "silindir_sayisi": 0,
            "frl_mevcut": False,
            "firma": "Belirtilmemiş"
        }
        
        # DB'den extract_values pattern'lerini al
        extract_values = self.pattern_definitions.get('extract_values', {})
        
        # Proje adı
        project_patterns = extract_values.get('proje_adi', [])
        for pattern in project_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                values["proje_adi"] = match.group(1) if match.groups() else match.group(0)
                break
        
        # Basınç bilgisi
        pressure_patterns = extract_values.get('calisma_basinci', [])
        for pattern in pressure_patterns:
            pressure_match = re.search(pattern, combined_text, re.IGNORECASE)
            if pressure_match:
                values["calisma_basinci"] = f"{pressure_match.group(1)} bar"
                break
        
        # Vana sayısı
        vana_patterns = extract_values.get('vana_sayisi', [])
        vana_matches = []
        for pattern in vana_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            vana_matches.extend(matches)
        
        rectangle_count = combined_text.lower().count('□') + combined_text.lower().count('■')
        if rectangle_count > 0:
            vana_matches.extend(['visual_valve'] * min(rectangle_count, 3))
        
        values["vana_sayisi"] = len(set(vana_matches)) if vana_matches else 0
        
        # Silindir sayısı
        silindir_patterns = extract_values.get('silindir_sayisi', [])
        silindir_matches = []
        for pattern in silindir_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            silindir_matches.extend(matches)
        
        piston_symbols = combined_text.count('═') + combined_text.count('━')
        if piston_symbols > 2:
            silindir_matches.extend(['visual_cylinder'] * min(piston_symbols // 3, 2))
        
        values["silindir_sayisi"] = len(set(silindir_matches)) if silindir_matches else 0
        
        # FRL kontrolü
        frl_patterns = extract_values.get('frl_mevcut', [])
        for pattern in frl_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                values["frl_mevcut"] = True
                break
        
        # Firma bilgisi
        if "ACT" in combined_text:
            values["firma"] = "ACT"
        elif "FESTO" in combined_text:
            values["firma"] = "FESTO"
        
        return values

    def extract_project_information(self, text: str, ocr_text: str) -> Dict[str, Any]:
        """Proje bilgilerini çıkarır - DB'den"""
        combined_text = f"{text} {ocr_text}"
        project_info = {}
        
        extract_values = self.pattern_definitions.get('extract_values', {})
        
        # Created by
        created_patterns = extract_values.get('created_by', [])
        for pattern in created_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                project_info["created_by"] = match.group(1).strip()
                break
        
        # Checked by
        checked_patterns = extract_values.get('checked_by', [])
        for pattern in checked_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                project_info["checked_by"] = match.group(1).strip()
                break
        
        # Approved by
        approved_patterns = extract_values.get('approved_by', [])
        for pattern in approved_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                project_info["approved_by"] = match.group(1).strip()
                break
        
        return project_info

    def generate_recommendations(self, analysis_results: Dict, scores: Dict, extracted_values: Dict) -> List[str]:
        """Öneri üretimi"""
        recommendations = []
        
        overall_score = scores["overall_percentage"]
        
        # Genel değerlendirme
        if overall_score >= 80:
            recommendations.append("✅ ÇİZİM KALİTESİ: Mükemmel")
        elif overall_score >= 70:
            recommendations.append("✅ ÇİZİM KALİTESİ: İyi")
        elif overall_score >= 60:
            recommendations.append("⚠️ ÇİZİM KALİTESİ: Orta")
        else:
            recommendations.append("❌ ÇİZİM KALİTESİ: Yetersiz")
        
        recommendations.append(f"📊 Toplam Puan: {overall_score:.1f}/100")
        
        # Kategori bazında öneriler
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 50:
                recommendations.append(f"❌ {category}: Ciddi eksiklikler var (%{category_score:.1f})")
                missing_criteria = [result.criteria_name for result in results.values() if not result.found]
                if missing_criteria:
                    recommendations.append(f"   Eksik: {', '.join(missing_criteria[:3])}")
            elif category_score < 70:
                recommendations.append(f"⚠️ {category}: Geliştirme gerekli (%{category_score:.1f})")
            else:
                recommendations.append(f"✅ {category}: Yeterli (%{category_score:.1f})")
        
        return recommendations

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Ana dosya analiz fonksiyonu"""
        logger.info(f"Dosya analiz ediliyor: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        text = ""
        ocr_text = ""
        visual_detections = []
        
        try:
            if file_ext == '.pdf':
                text = self.extract_text_from_pdf(file_path)
                # PDF'den görüntü çıkarıp OCR uygula
                try:
                    doc = fitz.open(file_path)
                    for page_num in range(min(3, len(doc))):
                        page = doc[page_num]
                        pix = page.get_pixmap()
                        img_data = pix.pil_tobytes(format="PNG")
                        img = Image.open(io.BytesIO(img_data))
                        img_array = np.array(img)
                        temp_path = f"temp_page_{page_num}.png"
                        cv2.imwrite(temp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                        ocr_text += self.extract_text_from_image(temp_path) + " "
                        visual_detections.extend(self.detect_visual_components(temp_path))
                        os.remove(temp_path)
                    doc.close()
                except:
                    pass
                    
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                ocr_text = self.extract_text_from_image(file_path)
                visual_detections = self.detect_visual_components(file_path)
            else:
                return {"error": f"Desteklenmeyen dosya formatı: {file_ext}"}
            
            if not text and not ocr_text:
                return {"error": "Dosyadan metin çıkarılamadı"}
            
            # Analiz yap
            analysis_results = {}
            for category in self.pneumatic_criteria_weights.keys():
                analysis_results[category] = self.analyze_criteria(text, ocr_text, visual_detections, category)
            
            scores = self.calculate_scores(analysis_results)
            extracted_values = self.extract_specific_values(text, ocr_text)
            project_info = self.extract_project_information(text, ocr_text)
            recommendations = self.generate_recommendations(analysis_results, scores, extracted_values)
            
            report = {
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_info": {
                    "file_path": file_path,
                    "file_type": file_ext,
                    "text_length": len(text),
                    "ocr_text_length": len(ocr_text),
                    "visual_components": len(visual_detections)
                },
                "project_information": project_info,
                "extracted_values": extracted_values,
                "category_analyses": analysis_results,
                "scoring": scores,
                "overall_score": {
                    "max_points": 100,
                    "percentage": scores["overall_percentage"],
                    "quality_level": self.get_quality_level(scores["overall_percentage"]),
                    "status": "PASS" if scores["overall_percentage"] >= 70 else "FAIL",
                    "status_tr": "GEÇERLİ" if scores["overall_percentage"] >= 70 else "GEÇERSİZ",
                    "total_points": scores["overall_percentage"]
                },
                "recommendations": recommendations,
                "summary": {
                    "total_score": scores["total_score"],
                    "percentage": scores["overall_percentage"],
                    "status": "GEÇERLİ" if scores["overall_percentage"] >= 70 else "GEÇERSİZ",
                    "quality_level": self.get_quality_level(scores["overall_percentage"])
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Analiz hatası: {e}")
            return {"error": str(e)}

    def get_quality_level(self, percentage: float) -> str:
        """Kalite seviyesi belirleme"""
        if percentage >= 90:
            return "MÜKEMMEL"
        elif percentage >= 80:
            return "ÇOK İYİ"
        elif percentage >= 70:
            return "İYİ"
        elif percentage >= 60:
            return "ORTA"
        elif percentage >= 40:
            return "YETERSIZ"
        else:
            return "KÖTÜ"

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
    """Server kodunda doküman validasyonu - DB'den"""
    
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
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"Pnömatik Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    valid_categories = sum(category_found)
    logger.info(f"Doküman validasyonu: {valid_categories}/{len(critical_terms)} kritik kategori bulundu")
    
    return valid_categories >= len(critical_terms) - 1

def check_strong_keywords_first_pages(filepath, validation_keywords):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - DB'den"""
    
    # DB'den strong keywords al
    strong_keywords = validation_keywords.get('strong_keywords', [])
    
    if not strong_keywords:
        logger.warning("⚠️ Strong keywords bulunamadı, validasyon atlanıyor")
        return True
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=400, first_page=1, last_page=1)
        
        all_text = ""
        for i, page in enumerate(pages):
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 3 -l tur+eng', timeout=15)
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
        pages = pdf2image.convert_from_path(filepath, dpi=200, first_page=1, last_page=1)
        
        all_text = ""
        for i, page in enumerate(pages):
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 3 -l tur+eng', timeout=15)
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

def get_conclusion_message_pneumatic(status, percentage):
    """Sonuç mesajını döndür - Pnömatik için"""
    if status == "PASS":
        return f"Pnömatik devre şeması ISO 5599 ve ilgili standartlara uygun ve teknik açıdan yeterlidir (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"Pnömatik devre şeması kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"Pnömatik devre şeması standartlara uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"

def get_main_issues_pneumatic(analysis_result):
    """Ana sorunları listele - Pnömatik için"""
    issues = []
    
    if 'scoring' in analysis_result and 'category_scores' in analysis_result['scoring']:
        for category, score_data in analysis_result['scoring']['category_scores'].items():
            if isinstance(score_data, dict) and score_data.get('percentage', 0) < 50:
                issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    if not issues:
        if analysis_result['summary']['total_score'] < 50:
            issues = [
                "Pnömatik semboller ISO 5599 standardına uygun değil",
                "FRL ünitesi ve hava hazırlama eksik",
                "Basınç ve debi değerleri eksik veya hatalı",
                "Vana tipleri ve kontrol sistemleri yetersiz",
                "Güvenlik elemanları ve acil durdurma eksik"
            ]
    
    return issues[:4]

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)

# Database configuration (YENİ)
app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
app.config['SQLALCHEMY_ECHO'] = Config.SQLALCHEMY_ECHO

PORT = int(os.environ.get('PORT', 8010))

UPLOAD_FOLDER = 'temp_uploads_pnomatic'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/pnomatic-control', methods=['POST'])
def analyze_pnomatic_control():
    """Pnömatik Devre Şeması analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere bir pnömatik devre şeması sağlayın'
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
                'message': 'Sadece PDF, JPG, JPEG ve PNG dosyaları kabul edilir'
            }), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"Pnömatik Devre Şeması kontrol ediliyor: {filename}")

            # Create analyzer instance with app context
            analyzer = PneumaticCircuitAnalyzer(app=app)
            
            logger.info(f"Üç aşamalı pnömatik kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                logger.info("Aşama 1: İlk sayfa pnömatik özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath, analyzer.validation_keywords):
                    logger.info("✅ Aşama 1 geçti - Pnömatik özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath, analyzer.validation_keywords):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - Pnömatik değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya pnömatik devre şeması değil (farklı rapor türü tespit edildi). Lütfen pnömatik devre şeması yükleyiniz.',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'PNOMATIK_DEVRE_SEMASI'
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
                                    'message': 'Yüklediğiniz dosya pnömatik devre şeması değil! Lütfen geçerli bir pnömatik devre şeması yükleyiniz.',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_PNEUMATIC_CIRCUIT',
                                        'required_type': 'PNOMATIK_DEVRE_SEMASI'
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
                requires_ocr = True
                if requires_ocr and not tesseract_available:
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

            logger.info(f"Pnömatik devre şeması doğrulandı, analiz başlatılıyor: {filename}")
            analysis_result = analyzer.analyze_file(filepath)
            
            try:
                os.remove(filepath)
                logger.info(f"Uploaded file {filename} cleaned up")
            except Exception as e:
                logger.warning(f"Could not remove uploaded file {filename}: {e}")

            if 'error' in analysis_result:
                return jsonify({
                    'error': 'Analysis failed',
                    'message': analysis_result['error'],
                    'details': {
                        'filename': filename,
                        'analysis_details': analysis_result.get('details', {})
                    }
                }), 400

            overall_percentage = analysis_result['summary']['percentage']
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': analysis_result.get('analysis_date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'analysis_details': {
                    'found_criteria': len([r for results in analysis_result.get('scoring', {}).get('category_scores', {}).values() 
                                         for r in results if isinstance(r, dict) and r.get('found', False)]),
                    'total_criteria': len([r for results in analysis_result.get('scoring', {}).get('category_scores', {}).values() 
                                         for r in results if isinstance(r, dict)]),
                    'percentage': round(overall_percentage, 1)
                },
                'analysis_id': f"pneumatic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': {
                    'is_valid': True,
                    'message': 'Pnömatik devre şemaları için tarih geçerliliği kontrolü uygulanmaz',
                    'days_old': 0,
                    'formatted_date': 'N/A'
                },
                'extracted_values': analysis_result.get('extracted_values', {}),
                'file_type': 'PNOMATIK_DEVRE_SEMASI',
                'filename': filename,
                'language_info': {
                    'detected_language': 'turkish',
                    'file_type': file_ext.replace('.', '')
                },
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': analysis_result['summary']['total_score'],
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'quality_level': analysis_result['summary']['quality_level']
                },
                'recommendations': analysis_result.get('recommendations', []),
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_pneumatic(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_pneumatic(analysis_result)
                }
            }
            
            if 'scoring' in analysis_result and 'category_scores' in analysis_result['scoring']:
                for category, score_data in analysis_result['scoring']['category_scores'].items():
                    if isinstance(score_data, dict):
                        response_data['category_scores'][category] = {
                            'score': score_data.get('normalized', score_data.get('score', 0)),
                            'max_score': score_data.get('max_weight', score_data.get('max_score', 0)),
                            'percentage': score_data.get('percentage', 0),
                            'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'CONDITIONAL' if score_data.get('percentage', 0) >= 50 else 'FAIL'
                        }

            return jsonify({
                'analysis_service': 'pneumatic_circuit',
                'data': response_data,
                'message': 'Pnömatik Devre Şeması başarıyla analiz edildi',
                'service_description': 'Pnömatik Devre Şeması Analizi',
                'service_port': PORT,
                'success': True
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
                'message': f'Pnömatik devre şeması analizi sırasında hata oluştu: {str(analysis_error)}',
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

@app.route('/api/pnomatic-health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Pneumatic Circuit Diagram Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'tesseract_info': tesseract_info,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': tesseract_available,
        'report_type': 'PNOMATIK_DEVRE_SEMASI'
    })

@app.route('/', methods=['GET'])
def index():
    """API bilgileri"""
    return jsonify({
        'service': 'Pneumatic Circuit Diagram Analyzer API',
        'version': '1.0.0',
        'description': 'Pnömatik Devre Şemalarını analiz eden REST API servisi',
        'endpoints': {
            'POST /api/pnomatic-control': 'Pnömatik devre şeması analizi',
            'GET /api/pnomatic-health': 'Health check',
            'GET /': 'API info'
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
# APPLICATION ENTRY POINT
# ============================================
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Pnömatik Devre Şeması Analiz API")
    logger.info("=" * 60)
    logger.info(f"📁 Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"📊 Desteklenen formatlar: {', '.join(ALLOWED_EXTENSIONS)}")
    logger.info(f"📏 Maksimum dosya boyutu: {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB")
    logger.info(f"🔧 Tesseract OCR: {'Kurulu' if tesseract_available else 'Kurulu değil'}")
    logger.info("")
    logger.info("🔗 API Endpoints:")
    logger.info("  POST /api/pnomatic-control - Pnömatik devre şeması analizi")
    logger.info("  GET  /api/pnomatic-health  - Health check")
    logger.info("  GET  /                     - API info")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8010))
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)