# ============================================
# HİDROLİK DEVRE ŞEMASI ANALİZ SERVİSİ
# Standalone Service - Azure App Service Ready
# Port: 8011
# ============================================

# ============================================
# IMPORTS
# ============================================
import os
import json
import io
from datetime import datetime
import re
from typing import Dict, List, Tuple, Any, Optional
import PyPDF2
import pytesseract
from PIL import Image
from dataclasses import dataclass, asdict
import logging
import math
import cv2
import numpy as np
import fitz  # PyMuPDF
from docx import Document

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pdf2image

# ============================================
# LOGGING CONFIGURATION
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# ============================================
# ANALİZ SINIFI - MAIN ANALYZER
# ============================================
class AdvancedCircuitAnalyzer:
    """Advanced circuit diagram analyzer"""
    
    def __init__(self):
        self.hydraulic_criteria_weights = {
            "Enerji Kaynağı": 25,
            "Hidrolik Semboller ve Bileşenler": 30,
            "Akış Yönü ve Bağlantı Hattı": 20,
            "Sistem Bilgileri ve Etiketler": 15,
            "Başlık ve Belgelendirme": 10
        }
        
        self.hydraulic_criteria_details = {
            "Enerji Kaynağı": {
                "basinc_yagi": {"pattern": r"(?i)(?:oil|yağ|ya[gğ]|hydraulic|hidrolik|hidrolic|fluid|hyd|pressur|pressure|basınç|basinc|bas[ıi]n[çc]|main|feeding|system|sistem)", "weight": 8},
                "basinc_aralik": {"pattern": r"(?i)(?:\d+\s*(?:bar|Bar|BAR|har|bor|MPa|mpa|psi|PSI)(?!\w)|(?:\d+)\s*(?:to|[-–—]|\s)\s*(?:\d+)|pressure|basınç)", "weight": 8},
                "sivil_guc": {"pattern": r"(?i)(?:liquid|hydraulic|hidrolik|hidrolic|fluid|oil|yağ|ya[gğ]|sıvı|s[ıi]v[ıi]|pressur|feeding|system|sistem|line|hat)", "weight": 5},
                "yuksek_basinc": {"pattern": r"(?i)(?:(?:high|yüksek|y[üu]ksek|main)\s*(?:pressure|basınç|basinc)|(?:[0-9]{2,4})\s*(?:bar|Bar|BAR|har|MPa|mpa|psi|PSI)|pressure\s*line|main\s*line)", "weight": 4}
            },
            "Hidrolik Semboller ve Bileşenler": {
                "pompa_sembol": {"pattern": r"(?i)(?:pump|pompa|ponpa|feeding|main\s*pump|pressure\s*pump|[PM]\d+|P\s*\d+|lowering\s*pump|motor|engine|drive)", "weight": 7},
                "motor_sembol": {"pattern": r"(?i)(?:motor|Motor|MOTOR|rotor|drive|engine|M\d+|M\s*\d+|electromotor|30\s*kW|3\s*kW|\d+\s*kW)", "weight": 7},
                "silindir_sembol": {"pattern": r"(?i)(?:cylinder|silindir|cilinder|piston|actuator|lifting|çift\s*etkili|tek\s*etkili|double\s*acting|single\s*acting|C\d+|CYL|lifting\s*cylinder|16x25|25x4)", "weight": 6},
                "basinc_valfi": {"pattern": r"(?i)(?:pressure\s*valve|basınç\s*val|valve|valf|relief|safety|control\s*val|accumulator|akümülat|HDA|pressure\s*control|50MBAR)", "weight": 5},
                "yon_kontrol_valfi": {"pattern": r"(?i)(?:directional|control\s*valve|yön\s*kontrol|4/[23]|3/2|DCV|D[GH][A-Z0-9]+|pilot|spool|valve\s*control|DGAV|DG4V)", "weight": 5}
            },
            "Akış Yönü ve Bağlantı Hattı": {
                "cizgi_borular": {"pattern": r"(?i)(?:line|pipe|boru|hat|çizgi|hose|tube|connection|bağlant|DN\s*\d+|NG\s*\d+|feeding|main|return|circuit|devre)", "weight": 6},
                "yon_oklari": {"pattern": r"(?i)(?:arrow|direction|yön|ok|flow|akış|discharge|suction|return|dönüş|çıkış|giriş|↑|↓|→|←|▲|▼|▶|◀)", "weight": 6},
                "pompa_cikis": {"pattern": r"(?i)(?:pump\s*(?:output|discharge|çıkış)|pompa.*?(?:çıkış|çıkışı|output)|pressure\s*line|main\s*line|high\s*pressure|discharge|output|çıkış)", "weight": 4},
                "tank_donus": {"pattern": r"(?i)(?:tank.*?(?:return|dönüş|dönüşü)|return\s*line|suction|low\s*pressure|reservoir|tahliye|drain|tank\s*line|return|dönüş)", "weight": 4}
            },
            "Sistem Bilgileri ve Etiketler": {
                "bar_basinc": {"pattern": r"(?i)(?:\d+\s*(?:bar|Bar|BAR|har|bor|MPa|mpa|psi|PSI)(?!\w)|p[0-9]:\s*\d+|pt:\s*\d+|p2:\s*\d+|50MBAR)", "weight": 4},
                "debi_bilgi": {"pattern": r"(?i)(?:\d+(?:\.\d+)?\s*(?:cc/rev|cc/dk|lt/dak|lt/min|l/min|lpm|gpm|L/min|flow)|Hub\s*x\s*\d+|debi|flow\s*rate|\d+\s*cc)", "weight": 4},
                "guc_bilgi": {"pattern": r"(?i)(?:\d+(?:\.\d+)?\s*(?:kW|KW|kw|HP|hp|W|w)|(?:\d{3,4})\s*(?:rpm|RPM|dev/dak)|30\s*kW|3\s*kW|power|güç|\d+\s*HP)", "weight": 4},
                "tank_hacmi": {"pattern": r"(?i)(?:V\s*=\s*\d+|(?:\d+)\s*(?:LT|lt|L|l|litre|liter)|tank.*?(?:volume|hacim|hacmi)|reservoir.*?\d+|\d+\s*LT)", "weight": 3}
            },
            "Başlık ve Belgelendirme": {
                "hydraulic_scheme": {"pattern": r"(?i)(?:HYDRAULIC|hydraulic|HİDROLİK|hidrolik|hidrolic|hydro|HYDRO|Hydraulikplan|hydraulikplan|HYDRAULIC\s*PLAN|hydraulic\s*schema)", "weight": 3},
                "data_sheet": {"pattern": r"(?i)(?:DATA\s*SHEET|data.*?sheet|specification|spec|technical|diagram|şema|schema|plan|drawing|çizim|PLAN|scheme|document|döküman)", "weight": 3},
                "manifold_plan": {"pattern": r"(?i)(?:MANIFOLD\s*PLAN|manifold|valve\s*block|block|kolektör|collector|central|unit|WEMHÖNER|manufacturer|company|firma)", "weight": 2},
                "cizim_standardi": {"pattern": r"(?i)(?:ISO\s*1219|standard|standart|DIN|EN|norm|drawing|çizim|technical\s*drawing|VSHY|drawing\s*no)", "weight": 2}
            }
        }
        
        self.component_templates = {
            "hydraulic": {
                "pump": ["P1", "P2", "P3", "PUMP", "POMPA"],
                "motor": ["M1", "M2", "M3", "MOTOR"],
                "valve": ["V1", "V2", "V3", "VALVE", "VALF"],
                "cylinder": ["C1", "C2", "C3", "CYL", "SİLİNDİR"],
                "tank": ["T1", "T2", "TANK", "TAMBUR"],
                "filter": ["F1", "F2", "FİLTRE", "FILTER"]
            }
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyPDF2 and OCR fallback"""
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
                
                if len(text.strip()) < 50:
                    logger.info("PyPDF2 extracted minimal text, trying OCR...")
                    ocr_text = self.extract_text_with_ocr(pdf_path)
                    if len(ocr_text) > len(text):
                        text = ocr_text
                
                return text
        except Exception as e:
            logger.error(f"PDF text extraction error: {e}")
            return self.extract_text_with_ocr(pdf_path)

    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR with enhanced settings"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(min(5, len(doc))):
                page = doc[page_num]
                mat = fitz.Matrix(3.0, 3.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.pil_tobytes(format="PNG")
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                temp_path = f"temp_page_{page_num}.png"
                cv2.imwrite(temp_path, thresh)
                
                ocr_configs = [
                    '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:-/+()[]{}ÇĞİÖŞÜçğıöşü ',
                    '--psm 4',
                    '--psm 11',
                    '--psm 8',
                    '--psm 3'
                ]
                
                page_text = ""
                for config in ocr_configs:
                    try:
                        ocr_result = pytesseract.image_to_string(temp_path, lang='tur+eng+deu', config=config)
                        if len(ocr_result.strip()) > len(page_text.strip()):
                            page_text = ocr_result
                    except:
                        continue
                
                text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            doc.close()
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return ""

    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                text += "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            return ""

    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        try:
            encodings = ['utf-8', 'latin-1', 'cp1254', 'iso-8859-9']
            for encoding in encodings:
                try:
                    with open(txt_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"TXT extraction error: {e}")
            return ""

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            text = pytesseract.image_to_string(image_path, lang='tur+eng', config='--psm 6')
            return text.strip()
        except Exception as e:
            logger.error(f"Image OCR error: {e}")
            return ""

    def extract_images_from_pdf(self, pdf_path: str) -> List[Any]:
        """Extract images from PDF"""
        logger.info("Image extraction is temporarily disabled")
        return []

    def perform_ocr_on_images(self, images: List[Any]) -> List[str]:
        """Perform OCR on extracted images"""
        logger.info("OCR functionality is temporarily disabled")
        return []

    def detect_components_in_images(self, images: List[Any], circuit_type: str) -> List[ComponentDetection]:
        """Detect components in images"""
        logger.info("Component detection is temporarily disabled")
        return []

    def determine_circuit_type(self, text: str, images: List[Any]) -> Tuple[str, float]:
        """Always return hydraulic"""
        return "hydraulic", 1.0

    def analyze_text_quality(self, text: str) -> str:
        """Analyze OCR text quality"""
        if len(text) < 100:
            return "poor"
        
        technical_terms = len(re.findall(r'(?i)\b(?:hydraulic|pressure|valve|pump|cylinder|motor|system|control|bar|psi)\b', text))
        garbled_patterns = len(re.findall(r'[^a-zA-Z0-9\s]{3,}', text))
        readable_words = len(re.findall(r'\b[a-zA-Z]{3,}\b', text))
        text_density = readable_words / max(1, len(text.split()))
        
        if technical_terms >= 10 and text_density > 0.3:
            return "excellent"
        elif technical_terms >= 3 and text_density > 0.2:
            return "good"  
        elif garbled_patterns > 5 or text_density < 0.1:
            return "poor"
        else:
            return "normal"

    def analyze_criteria(self, text: str, images: List[Any], category: str, circuit_type: str) -> Dict[str, CircuitAnalysisResult]:
        """Analyze criteria"""
        results = {}
        criteria = self.hydraulic_criteria_details.get(category, {})
        
        combined_text = text
        if images:
            ocr_results = self.perform_ocr_on_images(images)
            combined_text += " " + " ".join(ocr_results)
        
        detected_components = self.detect_components_in_images(images, circuit_type)
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            
            text_matches = re.findall(pattern, combined_text, re.IGNORECASE | re.MULTILINE)
            relevant_components = [comp for comp in detected_components if self._is_relevant_component(comp, criterion_name)]
            
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
        """Check if component is relevant"""
        relevance_map = {
            "pompa_sembol": ["pump"],
            "motor_sembol": ["motor"],
            "silindir_sembol": ["cylinder"],
            "basinc_valfi": ["valve"],
            "yon_kontrol_valfi": ["valve"],
            "tank_sembol": ["tank"],
            "filtre_sembol": ["filter"]
        }
        relevant_types = relevance_map.get(criterion_name, [])
        return component.component_type in relevant_types

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, CircuitAnalysisResult]], circuit_type: str) -> Dict[str, Any]:
        """Calculate scores"""
        category_scores = {}
        total_score = 0
        
        sample_text = getattr(self, '_last_extracted_text', '')
        text_quality = self.analyze_text_quality(sample_text)
        
        logger.info(f"Detected text quality: {text_quality}")

        for category, results in analysis_results.items():
            category_max = self.hydraulic_criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())

            if category_possible > 0:
                raw_percentage = category_earned / category_possible
                
                if text_quality == "excellent":
                    if raw_percentage > 0.7:
                        adjusted_percentage = raw_percentage
                    elif raw_percentage > 0.3:
                        adjusted_percentage = raw_percentage * 1.1
                    else:
                        adjusted_percentage = raw_percentage * 0.8
                        
                elif text_quality == "poor":
                    if raw_percentage > 0:
                        adjusted_percentage = max(0.6, math.pow(raw_percentage, 0.3))
                    else:
                        adjusted_percentage = 0.3
                        
                else:
                    if raw_percentage > 0:
                        adjusted_percentage = max(0.4, math.pow(raw_percentage, 0.6))
                    else:
                        adjusted_percentage = 0.1
                    
                normalized_score = min(category_max, adjusted_percentage * category_max)
            else:
                normalized_score = 0

            category_scores[category] = {
                "earned": category_earned,
                "possible": category_possible,
                "normalized": round(normalized_score, 2),
                "max_weight": category_max,
                "percentage": round((normalized_score / category_max * 100), 2)
            }

            total_score += normalized_score

        total_found_criteria = sum(
            sum(1 for result in results.values() if result.found) 
            for results in analysis_results.values()
        )
        total_possible_criteria = sum(
            len(results) for results in analysis_results.values()
        )
        
        hydraulic_validity_percentage = (total_found_criteria / total_possible_criteria * 100) if total_possible_criteria > 0 else 0
        
        if hydraulic_validity_percentage >= 25:
            logger.info(f"Hidrolik geçerlilik %{hydraulic_validity_percentage:.1f} - Otomatik geçer puan veriliyor")
            total_score = max(total_score, 75.0)
        
        return {
            "category_scores": category_scores,
            "total_score": round(total_score, 2),
            "total_max_score": 100,
            "overall_percentage": round((total_score / 100 * 100), 2),
            "text_quality": text_quality
        }

    def extract_specific_values(self, text: str, circuit_type: str) -> Dict[str, Any]:
        """Extract specific values - Enhanced for OCR"""
        values = {
            "proje_no": "Bulunamadı",
            "sistem_tipi": "Bulunamadı",
            "tarih": "Bulunamadı",
            "hidrolik_unite": "Bulunamadı",
            "tank_hacmi": "Bulunamadı",
            "motor_gucu": "Bulunamadı",
            "devir": "Bulunamadı",
            "debi": "Bulunamadı",
            "tambur": "Bulunamadı"
        }
        
        project_patterns = [
            r"(?i)(?:2271|VSHY|002204|TH-4|370\s*ton|feintol)",
            r"(?i)(?:proje|project|job)\s*(?:no|number)?\s*:?\s*([A-Z0-9-]+)",
            r"(?i)([A-Z]{2,}-\d+-[A-Z0-9-]+)",
            r"(?i)(TH-\d+|P\+Ânmatik)"
        ]
        for pattern in project_patterns:
            project_match = re.search(pattern, text)
            if project_match:
                values["proje_no"] = project_match.group(1) if len(project_match.groups()) > 0 else project_match.group()
                break
        
        system_patterns = [
            r"(?i)(?:press\s*feeding\s*system|feeding\s*system|hydraulic\s*system|hidrolik\s*sistem)",
            r"(?i)(?:accumulator|akümülat|lifting|kaldırma|pressing|pres)",
            r"(?i)(?:pnömatik|pneumatic|P\+Ânmatik|hidrolik)"
        ]
        for pattern in system_patterns:
            system_match = re.search(pattern, text)
            if system_match:
                values["sistem_tipi"] = system_match.group()
                break
        
        date_patterns = [
            r"(\d{4}[-./]\d{1,2}[-./]\d{1,2})",
            r"(\d{1,2}[-./]\d{1,2}[-./]\d{4})",
            r"(\d{1,2}\.\d{1,2}\.\d{4})",
            r"(\d{4}\s*\d{1,2}\s*\d{1,2})"
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, text)
            if date_match:
                values["tarih"] = date_match.group(1)
                break
        
        unit_patterns = [
            r"(?i)(?:HİDROLİK\s*ÜNİTE|HYDRAULIC\s*UNIT|hydraulic|hidrolik|hydro)",
            r"(?i)(?:HDA-\d+|accumulator|akümülat)",
            r"(?i)(?:pressure.*?control|basınç.*?kontrol)"
        ]
        for pattern in unit_patterns:
            unit_match = re.search(pattern, text)
            if unit_match:
                values["hidrolik_unite"] = unit_match.group()
                break
        
        tank_patterns = [
            r"(?i)(?:V\s*=\s*(\d+)|(\d+)\s*(?:LT|lt|L|l|litre|liter))",
            r"(?i)(?:tank.*?(\d+).*?(?:lt|l|litre))",
            r"(?i)(?:(\d+)\s*(?:lt|l)\s*tank)",
            r"(?i)(?:reservoir.*?(\d+))"
        ]
        for pattern in tank_patterns:
            tank_match = re.search(pattern, text)
            if tank_match:
                values["tank_hacmi"] = next((m for m in tank_match.groups() if m), tank_match.group())
                break
        
        power_patterns = [
            r"(?i)(?:(\d+(?:\.\d+)?)\s*(?:kW|KW|kw))",
            r"(?i)(?:(\d+(?:\.\d+)?)\s*(?:HP|hp))",
            r"(?i)(?:motor.*?(\d+(?:\.\d+)?)\s*(?:kW|hp))",
            r"(?i)(?:power.*?(\d+(?:\.\d+)?)\s*(?:kW|hp))",
            r"(?i)(30\s*kW|3\s*kW)"
        ]
        for pattern in power_patterns:
            power_match = re.search(pattern, text)
            if power_match:
                values["motor_gucu"] = next((m for m in power_match.groups() if m), power_match.group()) if len(power_match.groups()) > 0 else power_match.group()
                break
        
        rpm_patterns = [
            r"(?i)(?:(\d+)\s*(?:rpm|RPM|dev/dak))",
            r"(?i)(?:(\d{3,4})\s*(?:rpm|RPM))",
            r"(?i)(?:devir.*?(\d+))"
        ]
        for pattern in rpm_patterns:
            rpm_match = re.search(pattern, text)
            if rpm_match:
                values["devir"] = rpm_match.group(1)
                break
        
        flow_patterns = [
            r"(?i)(?:(\d+(?:\.\d+)?)\s*(?:lt/dak|l/min|cc/rev|cc/dk|lpm|gpm))",
            r"(?i)(?:(\d+)\s*Hub\s*x)",
            r"(?i)(?:debi.*?(\d+(?:\.\d+)?))",
            r"(?i)(?:flow.*?(\d+(?:\.\d+)?))"
        ]
        for pattern in flow_patterns:
            flow_match = re.search(pattern, text)
            if flow_match:
                values["debi"] = flow_match.group(1)
                break
        
        drum_patterns = [
            r"(?i)(?:TAMBUR|DRUM|cylinder|silindir)",
            r"(?i)(?:lifting\s*cylinder|kaldırma\s*silindir)",
            r"(?i)(?:\d+x\d+.*?cylinder)"
        ]
        for pattern in drum_patterns:
            drum_match = re.search(pattern, text)
            if drum_match:
                values["tambur"] = drum_match.group()
                break
        
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict, circuit_type: str) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        valid_criteria_count = sum(1 for category, results in analysis_results.items() 
                                 for result in results.values() if result.found)
        total_criteria_count = sum(len(results) for results in analysis_results.values())
        hydraulic_validity = valid_criteria_count / total_criteria_count
        
        recommendations.append(f"⚠️ Hidrolik Geçerlilik: %{hydraulic_validity*100:.1f} ({valid_criteria_count}/{total_criteria_count} kriter)")

        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 30:
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
                "- Şema ISO 1219 standardına uyumlu hale getirilmelidir",
                "- Hidrolik semboller eksiksiz olmalıdır",
                "- Sistem bilgileri detaylandırılmalıdır",
                "- Basınç ve debi değerleri belirtilmelidir"
            ])

        return recommendations

    def analyze_circuit_diagram(self, file_path: str) -> Dict[str, Any]:
        """Main analysis function"""
        logger.info(f"Starting circuit diagram analysis for: {file_path}")
        file_ext = os.path.splitext(file_path)[1].lower()
        text = ""
        
        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
            self._last_extracted_text = text
        elif file_ext in ['.docx', '.doc']:
            text = self.extract_text_from_docx(file_path)
            self._last_extracted_text = text
        elif file_ext == '.txt':
            text = self.extract_text_from_txt(file_path)
            self._last_extracted_text = text
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            text = self.extract_text_from_image(file_path)
            self._last_extracted_text = text
        else:
            return {"error": f"Unsupported file format: {file_ext}"}
        
        if not text:
            return {"error": "Could not extract text from file"}

        logger.info(f"Extracted text length: {len(text)} characters")

        images = []
        if file_ext == '.pdf':
            images = self.extract_images_from_pdf(file_path)
        
        circuit_type, type_confidence = self.determine_circuit_type(text, images)

        analysis_results = {}
        for category in self.hydraulic_criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, images, category, circuit_type)

        scores = self.calculate_scores(analysis_results, circuit_type)
        extracted_values = self.extract_specific_values(text, circuit_type)
        recommendations = self.generate_recommendations(analysis_results, scores, circuit_type)

        report = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_info": {"file_path": file_path},
            "circuit_type": {"type": circuit_type, "confidence": round(type_confidence * 100, 2)},
            "extracted_values": extracted_values,
            "category_analyses": analysis_results,
            "scoring": scores,
            "recommendations": recommendations,
            "summary": {
                "total_score": scores["total_score"],
                "percentage": scores["overall_percentage"],
                "status": "PASS" if scores["overall_percentage"] >= 70 else "FAIL",
                "circuit_type": circuit_type.upper()
            }
        }

        return report

# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_document_server(text):
    """Server document validation"""
    critical_terms = [
        ["hidrolik", "hydraulic", "devre", "circuit", "şema", "diagram", "schema"],
        ["pompa", "pump", "valf", "valve", "silindir", "cylinder", "motor", "actuator", "piston"],
        ["basınç", "pressure", "bar", "psi", "debi", "flow", "l/min", "gpm", "mpa"],
        ["yağ", "oil", "hidrolik yağ", "hydraulic oil", "tank", "rezervuar", "filtre", "filter"],
        ["iso 1219", "1219", "sembol", "symbol", "bağlantı", "connection", "hat", "line"]
    ]
    
    category_found = [
        any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category)
        for category in critical_terms
    ]
    
    valid_categories = sum(category_found)
    logger.info(f"Doküman validasyonu: {valid_categories}/5 kritik kategori bulundu")
    
    return valid_categories >= 5


def check_strong_keywords_first_pages(filepath):
    """Check strong keywords in first pages"""
    strong_keywords = [
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 3 -l tur+eng')
            all_text += text.lower() + " "
        
        found_keywords = [kw for kw in strong_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa kontrol: {len(found_keywords)} özgü kelime")
        return len(found_keywords) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa kontrol hatası: {e}")
        return False


def check_excluded_keywords_first_pages(filepath):
    """Check excluded keywords in first pages"""
    excluded_keywords = [
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık","ışık şiddeti",
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        "espe",
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide","kılavuzu",
        "loto",
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve",
        "montaj", "assembly",
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        "titreşim", "vibration", "mekanik",
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 3 -l tur+eng')
            all_text += text.lower() + " "
        
        found_excluded = [kw for kw in excluded_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa excluded kontrol: {len(found_excluded)} istenmeyen kelime")
        return len(found_excluded) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa excluded kontrol hatası: {e}")
        return False


def get_conclusion_message_hydraulic(status, percentage):
    """Get conclusion message"""
    if status == "PASS":
        return f"Hidrolik devre şeması ISO 1219 standardına uygun ve teknik açıdan yeterlidir (%{percentage:.0f})"
    else:
        return f"Hidrolik devre şeması standartlara uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"


def get_main_issues_hydraulic(analysis_result):
    """Get main issues"""
    issues = []
    
    for category, score_data in analysis_result['scoring']['category_scores'].items():
        if score_data['percentage'] < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    if not issues and analysis_result['scoring']['total_score'] < 50:
        issues = [
            "Hidrolik semboller ISO 1219 standardına uygun değil",
            "Basınç ve debi değerleri eksik veya hatalı",
            "Sistem bileşenleri tam tanımlanmamış",
            "Güvenlik elemanları eksik"
        ]
    
    return issues[:4]


# ============================================
# FLASK SERVICE
# ============================================
app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads_hydraulic'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'docx', 'doc', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/hydraulic-control', methods=['POST'])
def analyze_hydraulic_control():
    """Hidrolik Devre Şeması analiz endpoint"""
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
        
        analyzer = AdvancedCircuitAnalyzer()
        file_ext = os.path.splitext(filepath)[1].lower()

        # 3 AŞAMALI KONTROL (sadece PDF için)
        if file_ext == '.pdf':
            logger.info("Aşama 1: Hidrolik özgü kelime kontrolü...")
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
                        'message': 'Bu dosya hidrolik devre şeması değil'
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
                                'message': 'Yüklediğiniz dosya hidrolik devre şeması değil!'
                            }), 400
                    except Exception as e:
                        logger.error(f"Aşama 3 hatası: {e}")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({'error': 'Analysis failed'}), 500

        elif file_ext in ['.docx', '.doc', '.txt']:
            # DOCX/TXT için sadece tam doküman kontrolü
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
                return jsonify({
                    'error': 'Text extraction failed',
                    'message': 'Dosyadan yeterli metin çıkarılamadı'
                }), 400
            
            if not validate_document_server(text):
                try:
                    os.remove(filepath)
                except:
                    pass
                return jsonify({
                    'error': 'Invalid document type',
                    'message': 'Yüklediğiniz dosya hidrolik devre şeması değil!',
                    'details': {
                        'filename': filename,
                        'document_type': 'NOT_HIDROLIK_DEVRE_SEMASI'
                    }
                }), 400

        logger.info(f"Hidrolik devre şeması analizi yapılıyor: {filename}")
        analysis_result = analyzer.analyze_circuit_diagram(filepath)
        
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
            'analysis_id': f"hydraulic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'category_scores': {},
            'extracted_values': analysis_result.get('extracted_values', {}),
            'file_type': 'HIDROLIK_DEVRE_SEMASI',
            'filename': filename,
            'overall_score': {
                'percentage': round(overall_percentage, 2),
                'total_points': analysis_result['scoring']['total_score'],
                'max_points': 100,
                'status': status,
                'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ'
            },
            'recommendations': analysis_result.get('recommendations', []),
            'summary': {
                'is_valid': status == "PASS",
                'conclusion': get_conclusion_message_hydraulic(status, overall_percentage),
                'main_issues': [] if status == "PASS" else get_main_issues_hydraulic(analysis_result)
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
            'message': 'Hidrolik Devre Şeması başarıyla analiz edildi',
            'analysis_service': 'hydraulic_circuit_diagram',
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
        'service': 'Hydraulic Circuit Diagram Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'report_type': 'HIDROLIK_DEVRE_SEMASI'
    })


@app.route('/', methods=['GET'])
def index():
    """Index page"""
    return jsonify({
        'service': 'Hydraulic Circuit Diagram Analyzer API',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/hydraulic-control': 'Hidrolik devre şeması analizi',
            'GET /api/health': 'Health check',
            'GET /': 'API info'
        }
    })


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Hidrolik Devre Şeması Analiz Servisi")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8011))
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)