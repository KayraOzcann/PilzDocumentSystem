

# ============================================
# IMPORTS
# ============================================
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


# ============================================
# ANALİZ SINIFI - MAIN ANALYZER
# ============================================
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
        """Extract text from PDF using PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"PDF analizi başlatılıyor - Toplam sayfa: {total_pages}")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    page_text = re.sub(r'\s+', ' ', page_text)
                    page_text = page_text.strip()
                    text += page_text + "\n"
                
                # Metni temizle ve normalize et
                text = self._normalize_electrical_text(text)
                
                logger.info(f"✅ PDF analizi tamamlandı - Metin uzunluğu: {len(text):,} karakter")
                return text
                
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {e}")
            return ""

    def _process_electrical_symbols(self, text: str) -> str:
        """Process and normalize electrical symbols"""
        symbol_map = {
            'Ω': 'ohm', '∆': 'delta', '±': 'plusminus', '→': 'arrow', '←': 'arrow',
            '↑': 'arrow', '↓': 'arrow', '⏚': 'ground', '⊥': 'ground', '~': 'ac',
            '≈': 'ac', '⎓': 'dc', '⌁': 'dc', '∿': 'sine', '⚡': 'power'
        }
        for symbol, replacement in symbol_map.items():
            text = text.replace(symbol, f' {replacement} ')
        return text

    def _normalize_electrical_text(self, text: str) -> str:
        """Normalize electrical terms"""
        unit_map = {
            r'([0-9]+)\s*[vV]\b': r'\1 volt',
            r'([0-9]+)\s*[aA]\b': r'\1 amp',
            r'([0-9]+)\s*[wW]\b': r'\1 watt',
            r'([0-9]+)\s*[hH][zZ]\b': r'\1 hertz',
            r'([0-9]+)\s*Ω': r'\1 ohm'
        }
        for pattern, replacement in unit_map.items():
            text = re.sub(pattern, replacement, text)
        text = text.replace('—', '-').replace('"', '"').replace('"', '"')
        text = re.sub(r'[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]+', ' ', text)
        return text.strip()

    def extract_images_from_pdf(self, pdf_path: str) -> List[Any]:
        """Extract images from PDF - simplified for Azure"""
        return []  # Simplified - OCR yapılmayacak

    def perform_ocr_on_images(self, images: List[Any]) -> List[str]:
        """Perform OCR - simplified"""
        return []

    def detect_components_in_images(self, images: List[Any], circuit_type: str) -> List[ComponentDetection]:
        """Detect components - simplified"""
        return []

    def determine_circuit_type(self, text: str, images: List[Any]) -> Tuple[str, float]:
        return "electric", 1.0

    def analyze_criteria(self, text: str, images: List[Any], category: str, 
                        circuit_type: str) -> Dict[str, CircuitAnalysisResult]:
        results = {}
        criteria = self.electric_criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                content = f"Text: {str(matches[:3])}"
                found = True
                score = min(weight * 0.8, len(matches) * (weight * 0.2))
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
                    "text_matches": len(matches) if matches else 0,
                    "visual_matches": 0
                },
                visual_evidence=[]
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
            "overall_percentage": round((final_score / 100 * 100), 2)
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
        
        patterns = {
            "proje_no": r"(?:30292390|PROJE\s*NO|PROJECT\s*NO)",
            "sistem_tipi": r"(?i)(?:elektrik\s*şeması|electric\s*circuit|electrical\s*diagram)",
            "tarih": r"(\d{2}\.\d{2}\.\d{4})",
            "elektrik_paneli": r"(?i)(?:ELEKTRİK\s*PANELİ|ELECTRICAL\s*PANEL|CONTROL\s*PANEL)",
            "voltaj": r"(?i)(?:(\d+)\s*V|(\d+)\s*volt)",
            "akim": r"(?i)(?:(\d+)\s*A|(\d+)\s*amp)",
            "guc": r"(?i)(?:(\d+)\s*W|(\d+)\s*watt|(\d+)\s*kW)",
            "frekans": r"(?i)(?:(\d+)\s*Hz|(\d+)\s*hertz)",
            "klemens_blogu": r"(?i)(?:KLEMENS|TERMINAL|TB\d+|X\d+)"
        }
        
        for key, pattern in patterns.items():
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
            elif category_score < 70:
                recommendations.append(f"⚠️ {category} bölümü geliştirilmeli (%{category_score:.1f})")
            else:
                recommendations.append(f"✅ {category} bölümü yeterli (%{category_score:.1f})")

        return recommendations

    def analyze_circuit_diagram(self, pdf_path: str) -> Dict[str, Any]:
        """Ana analiz fonksiyonu"""
        logger.info("Elektrik devre şeması analizi başlatılıyor...")

        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "PDF okunamadı"}

        images = self.extract_images_from_pdf(pdf_path)
        circuit_type, _ = self.determine_circuit_type(text, images)

        analysis_results = {}
        for category in self.electric_criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, images, category, circuit_type)

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
# HELPER FUNCTIONS (Server Validasyon)
# ============================================
def map_language_code(lang_code):
    """Dil kodunu tam isme çevir"""
    lang_mapping = {'tr': 'turkish', 'en': 'english', 'de': 'german', 'fr': 'french', 'es': 'spanish', 'it': 'italian'}
    return lang_mapping.get(lang_code, 'turkish')


def validate_document_server(text):
    """Elektrik doküman validasyonu"""
    critical_terms = [
        ["elektrik", "electrical", "circuit", "devre", "şema", "diagram", "voltage", "current"],
        ["kontaktör", "contactor", "röle", "relay", "sigorta", "fuse", "mcb", "rcd", "switch"],
        ["volt", "v", "amper", "a", "watt", "w", "ohm", "ω", "hz", "hertz"],
        ["stop", "start", "emergency", "acil", "güvenlik", "safety", "control", "kontrol"]
    ]
    
    category_found = []
    for i, category in enumerate(critical_terms):
        found = any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category)
        if found:
            logger.info(f"Elektrik Kategori {i+1} bulundu")
        category_found.append(found)
    
    valid = sum(category_found)
    logger.info(f"Doküman validasyonu: {valid}/4 kritik kategori")
    return valid >= 4


def check_strong_keywords_first_pages(filepath):
    """İlk sayfada elektrik özgü kelime kontrolü - OCR"""
    strong_keywords = ["elektrik", "circuit", "electrical", "voltage", "amper", "ohm", "enclosure", "wrp-", "light curtain", "contactors", "controller"]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=2)
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        
        found = [kw for kw in strong_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa: {len(found)} özgü kelime bulundu")
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"OCR hatası: {e}")
        return False


def check_excluded_keywords_first_pages(filepath):
    """İlk sayfada excluded keyword kontrolü - OCR"""
    excluded = ["topraklama direnci", "aydınlatma", "hrc", "espe", "hidrolik", "gürültü", "kullanma", "loto", "lvd", 
                "uygunluk", "isg", "pnömatik", "montaj", "bakım", "titreşim", "at tip", "sertifika"]
    
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
# FLASK SERVİS KATMANI - CONFIGURATION
# ============================================
app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads_elektrik'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================
# FLASK SERVİS KATMANI - API ENDPOINTS
# ============================================
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

            analyzer = AdvancedElectricCircuitAnalyzer()
            
            # 3 AŞAMALI KONTROL
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext == '.pdf':
                logger.info("Aşama 1: Elektrik özgü kelime kontrolü...")
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
                            
                            if not text or len(text.strip()) < 50 or not validate_document_server(text):
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
        }
    })


# ============================================
# APPLICATION ENTRY POINT (Azure-Friendly)
# ============================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Elektrik Devre Şeması Analiz Servisi")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8001))
    
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)