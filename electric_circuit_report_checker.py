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
    
    def __init__(self):
        self.electric_criteria_weights = {
            "Semboller ve İşaretler": 30,
            "Bağlantı Hatları": 25,
            "Etiketleme ve Numara Sistemleri": 20,
            "Kontrol Panosu / Makine Otomasyon Öğeleri": 15,
            "Şematik Yerleşim": 10
        }
        
        self.electric_criteria_details = {
            "Semboller ve İşaretler": {
                "direnc_sembol": {"pattern": r"(?i)(?:direnç|resistor|ohm|Ω|R\d+|[0-9]+[RKM][0-9]*|zigzag|potansiyometre|pot|trimmer|━+|─+)", "weight": 6},
                "kondansator_sembol": {"pattern": r"(?i)(?:kondansatör|capacitor|C\d+|[0-9]+[µnpF]+|paralel\s*çizgi|elektrolitik|seramik|\|\||═+|◇.*?\|\||◇.*?═+|⬧.*?\|\||⬧.*?═+|⬥.*?\|\||⬥.*?═+|<>.*?\|\||<>.*?═+|[\u25C7\u25C8\u25C6].*?(?:\|\||═+))", "weight": 6},
                "bobin_sembol": {"pattern": r"(?i)(?:bobin|inductor|L\d+|[0-9]+[mH]+|spiral|solenoid|trafo|transformatör|transformer|⤾|⟲|⥀)", "weight": 5},
                "diyot_sembol": {"pattern": r"(?i)(?:diyot|diode|D\d+|LED|zener|köprü|bridge|rectifier|doğrultucu|▶|►|⊳)", "weight": 5},
                "transistor_sembol": {"pattern": r"(?i)(?:transistör|transistor|Q\d+|NPN|PNP|FET|MOSFET|BJT|darlington|⊲|△)", "weight": 4},
                "toprak_sembol": {"pattern": r"(?i)(?:toprak|ground|earth|GND|⏚|⊥|chassis|şasi|PE|↧|⌁)", "weight": 2},
                "sigorta_sembol": {"pattern": r"(?i)(?:sigorta|fuse|F\d+|MCB|RCD|devre\s*kesici|circuit\s*breaker|termik|⚡|═+)", "weight": 2}
            },
            "Bağlantı Hatları": {
                "iletken_baglanti": {"pattern": r"(?i)(?:kablo|wire|cable|hat|line|bağlantı|connection|conductor|iletken|NYA|NYM|H0[57]|━+|─+)", "weight": 8},
                "kesisen_hatlar": {"pattern": r"(?i)(?:kesişen|crossing|köprü|bridge|junction|node|düğüm|bağlantı\s*noktası|●|⊏|⊐)", "weight": 6},
                "baglanti_noktalari": {"pattern": r"(?i)(?:bağlantı\s*noktası|connection\s*point|terminal|node|klemens|terminal\s*block|X\d+|●|○|◯|⊙)", "weight": 6},
                "elektriksel_yon": {"pattern": r"(?i)(?:yön|direction|ok|arrow|akış|flow|akım|current|→|←|↑|↓|⟶|⇾)", "weight": 5}
            },
            "Etiketleme ve Numara Sistemleri": {
                "bilesenlerin_etiketlenmesi": {"pattern": r"(?i)(?:[RCL]\d+|[QDT]\d+|[MKF]\d+|[UIC]\d+|[+-]V(?:cc|dd|ss)|[+-]?\d+V|S[0-9]|K[0-9])", "weight": 6},
                "elektriksel_degerler": {"pattern": r"(?i)(?:\d+(?:\.\d+)?.*?(?:[VvAaMmWwΩ]|volt|amp|watt|ohm|VA|kVA|mA|µA)|[~=]|\~|\∿)", "weight": 5},
                "klemens_numaralari": {"pattern": r"(?i)(?:klemens|terminal|X\d+|TB\d+|[0-9]+\.[0-9]+|L[123N]|PE|[UVWN]\d*)", "weight": 5},
                "kablo_etiketleri": {"pattern": r"(?i)(?:kablo|wire|H\d+|W\d+|[0-9]+[AWG]|NYA|NYM|H0[57]|[0-9xX]+mm²)", "weight": 4}
            },
            "Kontrol Panosu / Makine Otomasyon Öğeleri": {
                "plc_giris_cikis": {"pattern": r"(?i)(?:PLC|I[0-9]+|Q[0-9]+|DI|DO|AI|AO|input|output|giriş|çıkış|[0-9]+[VI][0-9]+)", "weight": 4},
                "kontaktor_rele": {"pattern": r"(?i)(?:kontaktör|contactor|röle|relay|K\d+|KM\d+|NO|NC|coil|bobin|⤾|⟲)", "weight": 4},
                "motor_starter": {"pattern": r"(?i)(?:motor|starter|M\d+|drive|sürücü|inverter|softstarter|DOL|VFD|⊏⊐|▭M)", "weight": 3},
                "buton_sensor": {"pattern": r"(?i)(?:buton|button|sensör|sensor|S\d+|B\d+|switch|anahtar|proximity|PNP|NPN|○|◯|⊙)", "weight": 2},
                "ac_dc_guc": {"pattern": r"(?i)(?:AC|DC|güç|power|[0-9]+[VvAa]|~|⎓|[1-3]~|\+|-|N|PE|L[123]|\∿|=)", "weight": 2}
            },
            "Şematik Yerleşim": {
                "bilgi_akisi": {"pattern": r"(?i)(?:giriş|input|çıkış|output|soldan|sağa|yukarı|aşağı|→|←|↑|↓|⟶|⇾)", "weight": 3},
                "mantikli_dizilim": {"pattern": r"(?i)(?:işleme|process|dönüşüm|transformation|kontrol|control|güç|power|▭|⊏⊐)", "weight": 3},
                "sayfa_basligi": {"pattern": r"(?i)(?:proje|project|tarih|date|çizim|drawing|revizyon|revision|ref|no)", "weight": 2},
                "cerceve_frame": {"pattern": r"(?i)(?:çerçeve|frame|başlık|title|numara|number|sayfa|page|sheet|▭|□)", "weight": 2}
            }
        }
        
        self.component_templates = {
            "electric": {
                "resistor": ["R1", "R2", "R3", "RESISTOR", "DİRENÇ", "POT", "TRIMMER"],
                "capacitor": ["C1", "C2", "C3", "CAPACITOR", "KONDANSATÖR", "ELKO"],
                "inductor": ["L1", "L2", "L3", "INDUCTOR", "BOBİN", "TRAFO"],
                "diode": ["D1", "D2", "D3", "DIODE", "DİYOT", "LED", "ZENER"],
                "transistor": ["Q1", "Q2", "Q3", "TRANSISTOR", "TRANSİSTÖR", "FET", "MOSFET"],
                "relay": ["K1", "K2", "K3", "RELAY", "RÖLE", "KONTAKTÖR"],
                "motor": ["M1", "M2", "M3", "MOTOR", "STARTER", "SÜRÜCÜ"],
                "fuse": ["F1", "F2", "F3", "FUSE", "SİGORTA", "MCB", "RCD"],
                "switch": ["S1", "S2", "S3", "SWITCH", "ANAHTAR", "BUTON"],
                "power": ["V1", "V2", "V3", "POWER", "GÜÇ", "AC", "DC"],
                "ground": ["GND", "GROUND", "TOPRAK", "PE", "EARTH"],
                "terminal": ["X1", "X2", "X3", "TERMINAL", "KLEMENS", "TB"]
            }
        }

    def _preprocess_image_for_ocr(self, img: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            import cv2
            import numpy as np
            
            # Convert PIL Image to OpenCV format
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Noise removal and smoothing
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Increase contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrasted = clahe.apply(denoised)
            
            # Adaptive thresholding with different parameters for symbol detection
            binary = cv2.adaptiveThreshold(
                contrasted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 8
            )
            
            # Morphological operations to enhance symbol shapes
            kernel = np.ones((2,2), np.uint8)
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Edge enhancement
            edges = cv2.Canny(morph, 50, 150)
            enhanced = cv2.addWeighted(morph, 0.7, edges, 0.3, 0)
            
            # Convert back to PIL Image
            return Image.fromarray(enhanced)
        except Exception as e:
            logger.warning(f"Advanced image preprocessing failed: {e}")
            return img

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text and symbols from PDF using PyMuPDF and OCR"""
        try:
            import fitz  # PyMuPDF
            import cv2
            import numpy as np
            import pytesseract
            from PIL import Image
            import io
            
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file does not exist: {pdf_path}")
                return ""
                
            if not os.access(pdf_path, os.R_OK):
                logger.error(f"PDF file is not readable: {pdf_path}")
                return ""
            
            text = ""
            try:
                # Open the PDF file
                pdf_document = fitz.open(pdf_path)
                
                # Configure OCR for better symbol recognition
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,-_/\\()[]{}+=<>~!@#$%^&*⏚⊥Ω∆±→←↑↓~≈⎓⌁∿⚡⤾⟲⥀▶►⊳⊲△↧●○◯⊙⟶⇾▭□⊏⊐|\\'
                
                # Iterate through pages
                for page_num in range(len(pdf_document)):
                    try:
                        # Get the page
                        page = pdf_document[page_num]
                        
                        # First try direct text extraction with higher DPI
                        try:
                            page_text = page.get_text("text", flags=2)  # Using flags for better extraction
                        except:
                            page_text = ""  # If fails, try OCR
                        
                        # If no text found or minimal text, try OCR on the page image
                        if not page_text.strip() or len(page_text.strip()) < 50:
                            # Get page as image with higher resolution
                            zoom = 4.0  # 4x zoom for better OCR
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
                            
                            # Convert to PIL Image
                            img = Image.open(io.BytesIO(img_data))
                            
                            # Apply advanced preprocessing
                            processed_img = self._preprocess_image_for_ocr(img)
                            
                            # Perform OCR with custom configuration
                            page_text = pytesseract.image_to_string(processed_img, config=custom_config)
                        
                        # Clean and normalize text
                        page_text = self._normalize_electrical_text(page_text)
                        text += page_text + "\n"
                        
                        # Log successful page extraction
                        logger.info(f"Successfully extracted text from page {page_num + 1}")
                        
                    except Exception as page_error:
                        logger.warning(f"Failed to process page {page_num + 1}: {str(page_error)}")
                        continue
                
                # Close the PDF
                pdf_document.close()
                
                if not text.strip():
                    logger.warning(f"No text extracted from PDF: {pdf_path}")
                    return ""
                
                # Log successful text extraction
                logger.info(f"Successfully extracted text from PDF: {pdf_path}")
                logger.info(f"Extracted text length: {len(text)} characters")
                    
                return text
                
            except Exception as doc_error:
                logger.error(f"Failed to process PDF document: {str(doc_error)}")
                return ""
                
        except ImportError as imp_error:
            logger.error(f"Required library not found: {str(imp_error)}")
            logger.error("Please install required libraries: pip install PyMuPDF opencv-python Pillow pytesseract")
            return ""
        except Exception as e:
            logger.error(f"PDF text extraction error for {pdf_path}: {str(e)}")
            return ""

    def _process_electrical_symbols(self, text: str) -> str:
        """Process and normalize electrical symbols in text"""
        symbol_map = {
            'Ω': 'ohm',
            '∆': 'delta',
            '±': 'plusminus',
            '→': 'arrow',
            '←': 'arrow',
            '↑': 'arrow',
            '↓': 'arrow',
            '⏚': 'ground',
            '⊥': 'ground',
            '~': 'ac',
            '≈': 'ac',
            '⎓': 'dc',
            '⌁': 'dc',
            '∿': 'sine',
            '⚡': 'power'
        }
        
        for symbol, replacement in symbol_map.items():
            text = text.replace(symbol, f' {replacement} ')
        
        return text

    def _normalize_electrical_text(self, text: str) -> str:
        """Normalize electrical terms and measurements"""
        # Replace common electrical unit variations
        unit_map = {
            r'([0-9]+)\s*[vV]\b': r'\1 volt',
            r'([0-9]+)\s*[aA]\b': r'\1 amp',
            r'([0-9]+)\s*[wW]\b': r'\1 watt',
            r'([0-9]+)\s*[hH][zZ]\b': r'\1 hertz',
            r'([0-9]+)\s*Ω': r'\1 ohm',
            r'([0-9]+)\s*[kK][vV][aA]': r'\1 kva',
            r'([0-9]+)\s*[mM][aA]': r'\1 milliamp',
            r'([0-9]+)\s*[µuU][fF]': r'\1 microfarad',
            r'([0-9]+)\s*[pP][fF]': r'\1 picofarad',
            r'([0-9]+)\s*[mM][hH]': r'\1 millihenry'
        }
        
        for pattern, replacement in unit_map.items():
            text = re.sub(pattern, replacement, text)
        
        # Clean up and normalize text
        text = text.replace('—', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('´', "'")
        text = re.sub(r'[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]+', ' ', text)
        text = text.strip()
        
        return text

    def extract_images_from_pdf(self, pdf_path: str) -> List[Any]:
        """Extract images from PDF for symbol recognition"""
        try:
            import fitz  # PyMuPDF
            import cv2
            import numpy as np
            from PIL import Image
            import io
            
            images = []
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    
                    # Get page as image with higher resolution
                    try:
                        zoom = 2.0  # 2x zoom for better quality
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat)  # Latest method name
                    except:
                        try:
                            pix = page.get_pixmap(zoom=2.0)  # Alternative method
                        except:
                            logger.warning(f"Could not get pixmap for page {page_num + 1}")
                            continue
                    
                    img_data = pix.tobytes("png")
                    
                    # Convert to PIL Image
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Convert to OpenCV format
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    
                    # Apply adaptive thresholding
                    binary = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                    
                    # Noise removal
                    denoised = cv2.fastNlMeansDenoising(binary)
                    
                    # Store processed image data
                    image_data = {
                        'data': cv2.imencode('.png', denoised)[1].tobytes(),
                        'size': (denoised.shape[1], denoised.shape[0]),
                        'format': 'png',
                        'page': page_num
                    }
                    images.append(image_data)
                    
                    logger.info(f"Successfully extracted and processed image from page {page_num + 1}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process page {page_num + 1}: {e}")
                    continue
            
            pdf_document.close()
            return images
            
        except Exception as e:
            logger.error(f"Image extraction error: {e}")
            return []

    def perform_ocr_on_images(self, images: List[Any]) -> List[str]:
        """Perform OCR on extracted images using pytesseract with electrical symbol support"""
        try:
            import pytesseract
            from PIL import Image
            import io
            import numpy as np
            import cv2
            
            ocr_results = []
            for img_data in images:
                try:
                    # Convert image data to numpy array
                    nparr = np.frombuffer(img_data['data'], np.uint8)
                    img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    
                    # Image preprocessing for better OCR
                    # 1. Increase contrast
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    img_cv = clahe.apply(img_cv)
                    
                    # 2. Denoise
                    img_cv = cv2.fastNlMeansDenoising(img_cv)
                    
                    # 3. Thresholding
                    _, binary = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Convert to PIL Image
                    img_pil = Image.fromarray(binary)
                    
                    # Configure OCR for electrical symbols
                    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,-_/\\()[]{}+=<>~!@#$%^&*⏚⊥Ω∆±→←↑↓~≈⎓⌁∿⚡'
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(img_pil, config=custom_config)
                    
                    # Clean and normalize OCR result
                    text = self._normalize_electrical_text(text)
                    
                    # Add page information if available
                    if 'page' in img_data:
                        text = f"[Page {img_data['page'] + 1}] {text}"
                    
                    ocr_results.append(text)
                    logger.info(f"Successfully performed OCR on image from page {img_data.get('page', 'unknown')}")
                    
                except Exception as e:
                    logger.warning(f"OCR failed for image from page {img_data.get('page', 'unknown')}: {e}")
                    continue
            
            return ocr_results
            
        except ImportError as e:
            logger.error(f"OCR dependencies not installed: {e}")
            logger.error("Please install required libraries: pip install pytesseract opencv-python Pillow")
            return []
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            return []

    def detect_components_in_images(self, images: List[Any], circuit_type: str) -> List[ComponentDetection]:
        """Detect electrical components in images"""
        try:
            import cv2
            import numpy as np
            
            detected_components = []
            templates = self.component_templates.get(circuit_type, {})
            
            for img_data in images:
                try:
                    # Convert image data to OpenCV format
                    nparr = np.frombuffer(img_data['data'], np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Apply adaptive thresholding
                    binary = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                    
                    # Find contours
                    contours, _ = cv2.findContours(
                        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # Process each contour
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        roi = gray[y:y+h, x:x+w]
                        
                        # Skip if ROI is too small
                        if w < 20 or h < 20:
                            continue
                        
                        # Perform template matching for each component type
                        for comp_type, labels in templates.items():
                            for label in labels:
                                # Create template text image
                                template = np.zeros((50, 100), dtype=np.uint8)
                                cv2.putText(template, label, (10, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                                
                                # Template matching
                                result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                                _, confidence, _, _ = cv2.minMaxLoc(result)
                                
                                if confidence > 0.8:  # Confidence threshold
                                    detected_components.append(
                                        ComponentDetection(
                                            component_type=comp_type,
                                            label=label,
                                            position=(x+w//2, y+h//2),
                                            confidence=float(confidence),
                                            bounding_box=(x, y, w, h)
                                        )
                                    )
                except Exception as e:
                    logger.warning(f"Component detection failed for an image: {e}")
                    continue
            
            return detected_components
        except Exception as e:
            logger.error(f"Component detection error: {e}")
            return []

    def determine_circuit_type(self, text: str, images: List[Any]) -> Tuple[str, float]:
        return "electric", 1.0

    def analyze_criteria(self, text: str, images: List[Any], category: str, 
                        circuit_type: str) -> Dict[str, CircuitAnalysisResult]:
        results = {}
        criteria = self.electric_criteria_details.get(category, {})
        
        combined_text = text
        if images:
            ocr_results = self.perform_ocr_on_images(images)
            combined_text += " " + " ".join(ocr_results)
        
        detected_components = self.detect_components_in_images(images, circuit_type)
        
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
        total_max_score = 100

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
            "total_max_score": total_max_score,
            "overall_percentage": round((final_score / total_max_score * 100), 2)
        }

    def extract_specific_values(self, text: str, circuit_type: str) -> Dict[str, Any]:
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
        
        project_match = re.search(r"(?:30292390|PROJE\s*NO|PROJECT\s*NO)", text)
        if project_match:
            values["proje_no"] = project_match.group()
        
        system_match = re.search(r"(?i)(?:elektrik\s*şeması|electric\s*circuit|electrical\s*diagram)", text)
        if system_match:
            values["sistem_tipi"] = system_match.group()
        
        date_match = re.search(r"(\d{2}\.\d{2}\.\d{4})", text)
        if date_match:
            values["tarih"] = date_match.group(1)
        
        panel_match = re.search(r"(?i)(?:ELEKTRİK\s*PANELİ|ELECTRICAL\s*PANEL|CONTROL\s*PANEL)", text)
        if panel_match:
            values["elektrik_paneli"] = panel_match.group()
        
        voltage_match = re.search(r"(?i)(?:(\d+)\s*V|(\d+)\s*volt)", text)
        if voltage_match:
            values["voltaj"] = next(m for m in voltage_match.groups() if m)
        
        current_match = re.search(r"(?i)(?:(\d+)\s*A|(\d+)\s*amp)", text)
        if current_match:
            values["akim"] = next(m for m in current_match.groups() if m)
        
        power_match = re.search(r"(?i)(?:(\d+)\s*W|(\d+)\s*watt|(\d+)\s*kW)", text)
        if power_match:
            values["guc"] = next(m for m in power_match.groups() if m)
        
        freq_match = re.search(r"(?i)(?:(\d+)\s*Hz|(\d+)\s*hertz)", text)
        if freq_match:
            values["frekans"] = freq_match.group(1)
        
        terminal_match = re.search(r"(?i)(?:KLEMENS|TERMINAL|TB\d+|X\d+)", text)
        if terminal_match:
            values["klemens_blogu"] = terminal_match.group()
        
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict, circuit_type: str) -> List[str]:
        recommendations = []
        
        valid_criteria_count = sum(1 for category, results in analysis_results.items() 
                                 for result in results.values() if result.found)
        total_criteria_count = sum(len(results) for results in analysis_results.values())
        electric_validity = valid_criteria_count / total_criteria_count
        
        recommendations.append(f"⚡ Elektrik Geçerlilik: Elektrik devre güvenilirlik: %{electric_validity*100:.1f} ({valid_criteria_count}/{total_criteria_count} kriter)")

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
        logger.info("Starting electric circuit diagram analysis...")

        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "Could not read PDF"}

        images = self.extract_images_from_pdf(pdf_path)
        
        circuit_type, type_confidence = self.determine_circuit_type(text, images)
        if circuit_type == "unknown":
            return {"error": "Could not determine circuit type"}

        analysis_results = {}
        criteria_weights = self.electric_criteria_weights

        for category in criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, images, category, circuit_type)

        scores = self.calculate_scores(analysis_results, circuit_type)
        
        extracted_values = self.extract_specific_values(text, circuit_type)
        
        recommendations = self.generate_recommendations(analysis_results, scores, circuit_type)

        report = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_info": {
                "pdf_path": pdf_path
            },
            "circuit_type": {
                "type": circuit_type,
                "confidence": round(type_confidence * 100, 2)
            },
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

    def save_report_to_excel(self, report: Dict, output_path: str):
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_data = {
                'Criterion': ['Total Score', 'Percentage', 'Status', 'Circuit Type'],
                'Value': [
                    report['summary']['total_score'],
                    f"%{report['summary']['percentage']}",
                    report['summary']['status'],
                    report['summary']['circuit_type']
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            values_data = []
            for key, value in report['extracted_values'].items():
                values_data.append({'Criterion': key, 'Value': value})
            pd.DataFrame(values_data).to_excel(writer, sheet_name='Extracted_Values', index=False)

            for category, results in report['category_analyses'].items():
                category_data = []
                for criterion, result in results.items():
                    category_data.append({
                        'Criterion': criterion,
                        'Found': result.found,
                        'Content': result.content,
                        'Score': result.score,
                        'Max Score': result.max_score,
                        'Visual Matches': len(result.visual_evidence)
                    })
                sheet_name = category.replace('/', '_').replace('\\', '_')[:31]
                pd.DataFrame(category_data).to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"Report saved to Excel: {output_path}")

    def save_report_to_json(self, report: Dict, output_path: str):
        json_report = {}
        for key, value in report.items():
            if key == 'category_analyses':
                json_report[key] = {}
                for category, results in value.items():
                    json_report[key][category] = {}
                    for criterion, result in results.items():
                        json_report[key][category][criterion] = asdict(result)
            else:
                json_report[key] = value

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2)

        logger.info(f"Report saved to JSON: {output_path}")

def main():
    """Main function"""
    analyzer = AdvancedElectricCircuitAnalyzer()
    
    pdf_path = "M2407 Gönenli Ø160 Yoğurt Dolum.pdf"
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF dosyası bulunamadı: {pdf_path}")
        print("Lütfen dosya adını ve yolunu kontrol edin.")
        return
    
    # Check if file is readable
    if not os.access(pdf_path, os.R_OK):
        print(f"❌ PDF dosyası okunamıyor: {pdf_path}")
        print("Lütfen dosya izinlerini kontrol edin.")
        return
    
    # Check file size
    try:
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            print(f"❌ PDF dosyası boş: {pdf_path}")
            return
        
        print("🔍 Elektrik Devre Şeması Analizi Başlatılıyor...")
        print("=" * 60)
        print(f"📂 Dosya: {pdf_path}")
        print(f"📊 Boyut: {file_size / 1024:.1f} KB")
        
        # Try to import required libraries
        try:
            import fitz
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError as e:
            print("\n❌ Gerekli kütüphaneler eksik!")
            print("Lütfen aşağıdaki komutları çalıştırın:")
            print("pip install PyMuPDF opencv-python Pillow pytesseract")
            return
        
        report = analyzer.analyze_circuit_diagram(pdf_path)
        
        if "error" in report:
            print(f"\n❌ Hata: {report['error']}")
            print("Lütfen PDF dosyasını kontrol edin ve tekrar deneyin.")
            return
        
        print("\n📊 ANALİZ SONUÇLARI")
        print("=" * 60)
        
        print(f"📅 Analiz Tarihi: {report['analysis_date']}")
        print(f"📋 Toplam Puan: {report['summary']['total_score']}/100")
        print(f"📈 Yüzde: %{report['summary']['percentage']}")
        print(f"🎯 Durum: {report['summary']['status']}")
        print(f"⚡ Elektrik Durumu: {report['summary']['circuit_type']}")
        
        print("\n📋 ÖNEMLİ ÇIKARILAN DEĞERLER")
        print("-" * 40)
        for key, value in report['extracted_values'].items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\n📊 KATEGORİ PUANLARI")
        print("-" * 40)
        for category, score_data in report['scoring']['category_scores'].items():
            print(f"{category}: {score_data['normalized']}/{score_data['max_weight']} (%{score_data['percentage']:.1f})")
        
        print("\n💡 ÖNERİLER")
        print("-" * 40)
        for recommendation in report['recommendations']:
            print(recommendation)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = f"Elektrik_Devre_Analiz_Raporu_{timestamp}.xlsx"
        json_path = f"Elektrik_Devre_Analiz_Raporu_{timestamp}.json"
        
        analyzer.save_report_to_excel(report, excel_path)
        analyzer.save_report_to_json(report, json_path)
        
        print(f"\n💾 Raporlar kaydedildi:")
        print(f"   📊 Excel: {excel_path}")
        print(f"   📄 JSON: {json_path}")
        
        print("\n📋 GENEL DEĞERLENDİRME")
        print("=" * 60)
        percentage = report['summary']['percentage']
        if percentage >= 70:
            print("✅ SONUÇ: GEÇERLİ")
            print(f"🌟 Toplam Başarı: %{percentage:.1f}")
            print("📝 Değerlendirme: Elektrik devre şeması genel olarak yeterli kriterleri sağlamaktadır.")
        else:
            print("❌ SONUÇ: GEÇERSİZ")
            print(f"⚠️ Toplam Başarı: %{percentage:.1f}")
            print("📝 Değerlendirme: Elektrik devre şeması minimum gereklilikleri sağlamamaktadır.")
            print("\n⚠️ EKSİK GEREKLİLİKLER:")
            
            for category, results in report['category_analyses'].items():
                missing_items = []
                for criterion, result in results.items():
                    if not result.found:
                        missing_items.append(criterion)
                
                if missing_items:
                    print(f"\n🔍 {category}:")
                    for item in missing_items:
                        readable_name = {
                            "direnc_sembol": "Direnç Sembolü",
                            "kondansator_sembol": "Kondansatör Sembolü",
                            "bobin_sembol": "Bobin Sembolü",
                            "diyot_sembol": "Diyot Sembolü",
                            "transistor_sembol": "Transistör Sembolü",
                            "toprak_sembol": "Toprak Sembolü",
                            "sigorta_sembol": "Sigorta Sembolü",
                            "iletken_baglanti": "İletken Bağlantı",
                            "kesisen_hatlar": "Kesişen Hatlar",
                            "baglanti_noktalari": "Bağlantı Noktaları",
                            "elektriksel_yon": "Elektriksel Yön",
                            "bilesenlerin_etiketlenmesi": "Bileşenlerin Etiketlenmesi",
                            "elektriksel_degerler": "Elektriksel Değerler",
                            "klemens_numaralari": "Klemens Numaraları",
                            "kablo_etiketleri": "Kablo Etiketleri",
                            "plc_giris_cikis": "PLC Giriş/Çıkış",
                            "kontaktor_rele": "Kontaktör/Röle",
                            "motor_starter": "Motor Starter",
                            "buton_sensor": "Buton/Sensör",
                            "ac_dc_guc": "AC/DC Güç",
                            "bilgi_akisi": "Bilgi Akışı",
                            "mantikli_dizilim": "Mantıklı Dizilim",
                            "sayfa_basligi": "Sayfa Başlığı",
                            "cerceve_frame": "Çerçeve/Frame"
                        }.get(item, item)
                        print(f"   ❌ {readable_name}")
            
            print("\n📌 YAPILMASI GEREKENLER:")
            print("1. Eksik elektrik sembollerini ekleyin")
            print("2. Voltaj, akım ve güç değerlerini belirtin")
            print("3. Bağlantı hatlarını net gösterin")
            print("4. IEC veya ANSI standardına uygun hale getirin")
            print("5. Bileşenleri etiketleyin")
    except Exception as e:
        print(f"\n❌ Beklenmeyen bir hata oluştu: {str(e)}")
        print("Lütfen dosyayı ve sistem yapılandırmanızı kontrol edin.")
        return

if __name__ == "__main__":
    main()