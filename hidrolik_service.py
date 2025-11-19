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
                "basinc_yagi": {"pattern": r"(?i)(?:oil|yağ|hydraulic|hidrolik|fluid|hyd|pressure|basınç|main|feeding|system)", "weight": 8},
                "basinc_aralik": {"pattern": r"(?i)(?:\d+\s*(?:bar|Bar|BAR|MPa|psi|PSI)(?!\w)|pressure|basınç)", "weight": 8},
                "sivil_guc": {"pattern": r"(?i)(?:liquid|hydraulic|hidrolik|fluid|oil|yağ|pressure|feeding|system|line)", "weight": 5},
                "yuksek_basinc": {"pattern": r"(?i)(?:high\s*pressure|yüksek\s*basınç|main\s*line|[0-9]{2,4}\s*(?:bar|Bar|MPa|psi))", "weight": 4}
            },
            "Hidrolik Semboller ve Bileşenler": {
                "pompa_sembol": {"pattern": r"(?i)(?:pump|pompa|feeding|main\s*pump|pressure\s*pump|P\d+|lowering\s*pump|motor)", "weight": 7},
                "motor_sembol": {"pattern": r"(?i)(?:motor|Motor|rotor|drive|engine|M\d+|electromotor|\d+\s*kW)", "weight": 7},
                "silindir_sembol": {"pattern": r"(?i)(?:cylinder|silindir|piston|actuator|lifting|çift\s*etkili|double\s*acting|C\d+|CYL)", "weight": 6},
                "basinc_valfi": {"pattern": r"(?i)(?:pressure\s*valve|basınç\s*val|valve|valf|relief|safety|control|accumulator)", "weight": 5},
                "yon_kontrol_valfi": {"pattern": r"(?i)(?:directional|control\s*valve|yön\s*kontrol|4/[23]|3/2|DCV|pilot|spool)", "weight": 5}
            },
            "Akış Yönü ve Bağlantı Hattı": {
                "cizgi_borular": {"pattern": r"(?i)(?:line|pipe|boru|hat|hose|tube|connection|bağlant|DN\s*\d+|NG\s*\d+|feeding|return)", "weight": 6},
                "yon_oklari": {"pattern": r"(?i)(?:arrow|direction|yön|flow|akış|discharge|suction|return|dönüş|↑|↓|→|←)", "weight": 6},
                "pompa_cikis": {"pattern": r"(?i)(?:pump\s*output|pompa.*?çıkış|pressure\s*line|main\s*line|discharge|output)", "weight": 4},
                "tank_donus": {"pattern": r"(?i)(?:tank.*?return|return\s*line|suction|reservoir|tahliye|drain|tank\s*line)", "weight": 4}
            },
            "Sistem Bilgileri ve Etiketler": {
                "bar_basinc": {"pattern": r"(?i)(?:\d+\s*(?:bar|Bar|BAR|MPa|psi|PSI)(?!\w)|p[0-9]:\s*\d+|pt:\s*\d+)", "weight": 4},
                "debi_bilgi": {"pattern": r"(?i)(?:\d+(?:\.\d+)?\s*(?:cc/rev|lt/dak|lt/min|l/min|lpm|gpm|L/min|flow)|debi)", "weight": 4},
                "guc_bilgi": {"pattern": r"(?i)(?:\d+(?:\.\d+)?\s*(?:kW|KW|HP|W)|(?:\d{3,4})\s*(?:rpm|RPM)|power|güç)", "weight": 4},
                "tank_hacmi": {"pattern": r"(?i)(?:V\s*=\s*\d+|(?:\d+)\s*(?:LT|lt|L|l|litre)|tank.*?volume|reservoir)", "weight": 3}
            },
            "Başlık ve Belgelendirme": {
                "hydraulic_scheme": {"pattern": r"(?i)(?:HYDRAULIC|hydraulic|HİDROLİK|hidrolik|hydro|Hydraulikplan|HYDRAULIC\s*PLAN)", "weight": 3},
                "data_sheet": {"pattern": r"(?i)(?:DATA\s*SHEET|specification|technical|diagram|şema|schema|plan|drawing)", "weight": 3},
                "manifold_plan": {"pattern": r"(?i)(?:MANIFOLD|manifold|valve\s*block|block|kolektör|collector|central)", "weight": 2},
                "cizim_standardi": {"pattern": r"(?i)(?:ISO\s*1219|standard|standart|DIN|EN|norm|drawing|technical)", "weight": 2}
            }
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    page_text = re.sub(r'\s+', ' ', page_text)
                    text += page_text + "\n"
                
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
        """Extract text from PDF using OCR"""
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
                
                ocr_result = pytesseract.image_to_string(temp_path, lang='tur+eng+deu', config='--psm 6')
                text += f"\n--- PAGE {page_num + 1} ---\n{ocr_result}\n"
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            doc.close()
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return ""

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

    def analyze_criteria(self, text: str, category: str) -> Dict[str, CircuitAnalysisResult]:
        """Analyze criteria"""
        results = {}
        criteria = self.hydraulic_criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            
            text_matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if text_matches:
                content = f"Text: {str(text_matches[:3])}"
                found = True
                score = min(weight * 0.8, len(text_matches) * (weight * 0.2))
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
                    "visual_matches": 0
                },
                visual_evidence=[]
            )
        
        return results

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, CircuitAnalysisResult]]) -> Dict[str, Any]:
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

    def extract_specific_values(self, text: str) -> Dict[str, Any]:
        """Extract specific values"""
        values = {
            "proje_no": "Bulunamadı",
            "sistem_tipi": "Bulunamadı",
            "motor_gucu": "Bulunamadı",
            "tank_hacmi": "Bulunamadı"
        }
        
        power_patterns = [
            r"(?i)(?:(\d+(?:\.\d+)?)\s*(?:kW|KW))",
            r"(?i)(30\s*kW|3\s*kW)"
        ]
        for pattern in power_patterns:
            power_match = re.search(pattern, text)
            if power_match:
                values["motor_gucu"] = power_match.group(1) if len(power_match.groups()) > 0 else power_match.group()
                break
        
        tank_patterns = [
            r"(?i)(?:V\s*=\s*(\d+)|(\d+)\s*(?:LT|lt|L))",
        ]
        for pattern in tank_patterns:
            tank_match = re.search(pattern, text)
            if tank_match:
                values["tank_hacmi"] = next((m for m in tank_match.groups() if m), tank_match.group())
                break
        
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 30:
                recommendations.append(f"❌ {category} bölümü yetersiz (%{category_score:.1f})")
            elif category_score < 70:
                recommendations.append(f"⚠️ {category} bölümü geliştirilmeli (%{category_score:.1f})")
            else:
                recommendations.append(f"✅ {category} bölümü yeterli (%{category_score:.1f})")

        return recommendations

    def analyze_circuit_diagram(self, file_path: str) -> Dict[str, Any]:
        """Main analysis function"""
        logger.info(f"Starting circuit diagram analysis for: {file_path}")
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
            self._last_extracted_text = text
        else:
            return {"error": f"Unsupported file format: {file_ext}"}
        
        if not text:
            return {"error": "Could not extract text from file"}

        logger.info(f"Extracted text length: {len(text)} characters")

        analysis_results = {}
        for category in self.hydraulic_criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category)

        scores = self.calculate_scores(analysis_results)
        extracted_values = self.extract_specific_values(text)
        recommendations = self.generate_recommendations(analysis_results, scores)

        report = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_info": {"file_path": file_path},
            "extracted_values": extracted_values,
            "category_analyses": analysis_results,
            "scoring": scores,
            "recommendations": recommendations,
            "summary": {
                "total_score": scores["total_score"],
                "percentage": scores["overall_percentage"],
                "status": "PASS" if scores["overall_percentage"] >= 70 else "FAIL"
            }
        }

        return report

# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_document_server(text):
    """Server document validation"""
    critical_terms = [
        ["hidrolik", "hydraulic", "devre", "circuit", "şema", "diagram"],
        ["pompa", "pump", "valf", "valve", "silindir", "cylinder", "motor"],
        ["basınç", "pressure", "bar", "psi", "debi", "flow"],
        ["yağ", "oil", "hidrolik yağ", "hydraulic oil", "tank"],
        ["iso 1219", "sembol", "symbol", "bağlantı", "connection"]
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
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 3 -l tur+eng', timeout=15)
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
        "aydınlatma", "lighting", "hrc", "cobot", "elektrik", "espe", "gürültü",
        "kullanma", "kılavuz", "loto", "lvd", "isg", "periyodik", "pnömatik",
        "montaj", "bakım", "titreşim", "certificate"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 3 -l tur+eng', timeout=15)
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
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

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