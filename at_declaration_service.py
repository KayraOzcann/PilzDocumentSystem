"""
AT Uygunluk Beyanı Analiz Servisi
==================================
Endpoint: POST /api/at-declaration
Health: GET /api/health
"""

import re, os, json, PyPDF2, logging, cv2, numpy as np, pytesseract
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from PIL import Image
import pdf2image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from langdetect import detect
    LANG_DETECT = True
except:
    LANG_DETECT = False

@dataclass
class ATAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    is_critical: bool
    details: Dict[str, Any]

class ATTypeInspectionAnalyzer:
    def __init__(self):
        self.criteria_weights = {
            "Kritik Bilgiler": 60,
            "Zorunlu Teknik Bilgiler": 25,
            "Standartlar ve Belgeler": 15
        }
        
        self.criteria_details = {
            "Kritik Bilgiler": {
                "uretici_adi": {"pattern": r"(?:üretici|manufacturer|company)[\s:]*([A-Za-zÇŞİĞÜÖıçşığüö\s\.\-&]{8,100})", "weight": 15, "critical": True},
                "uretici_adres": {"pattern": r"(?:adres|address|cd\.\s*no)[\s:]*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{15,200})", "weight": 15, "critical": True},
                "makine_tanimi": {"pattern": r"(?:makine|machine|model|tip)[\s:]*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\-\.]{5,100})", "weight": 15, "critical": True},
                "direktif_atif": {"pattern": r"(?:2006/42|machine directive|direktif)", "weight": 10, "critical": True},
                "yetkili_imza": {"pattern": r"(?:yetkili|authorized|imza|signature)", "weight": 5, "critical": True}
            },
            "Zorunlu Teknik Bilgiler": {
                "uretim_yili": {"pattern": r"([0-9]{4})", "weight": 5, "critical": False},
                "seri_no": {"pattern": r"(?:seri|serial)[\s\w]*(?:no|number)[\s:]*([A-Za-z0-9\-]{2,20})", "weight": 5, "critical": False},
                "beyan_ifadesi": {"pattern": r"(?:beyan|declaration|conform|uygun)", "weight": 5, "critical": False},
                "tarih_yer": {"pattern": r"([0-9]{1,2}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{2,4})", "weight": 5, "critical": False},
                "diger_direktifler": {"pattern": r"(?:2014/30|2014/35|EMC|LVD)", "weight": 5, "critical": False}
            },
            "Standartlar ve Belgeler": {
                "uyumlu_standartlar": {"pattern": r"(?:EN|ISO|IEC)[\s]*[0-9]{3,5}", "weight": 8, "critical": False},
                "teknik_dosya": {"pattern": r"(?:teknik dosya|technical file)", "weight": 4, "critical": False},
                "onaylanmis_kurulus": {"pattern": r"(?:onaylanmış kuruluş|notified body)", "weight": 3, "critical": False}
            }
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() for page in reader.pages)
            return text.strip()
        except:
            return ""
    
    def detect_language(self, text: str) -> str:
        if not LANG_DETECT:
            return 'tr'
        try:
            return detect(text[:500].strip())
        except:
            return 'tr'
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, ATAnalysisResult]:
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            is_critical = criterion_data["critical"]
            
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                content = f"Bulundu: {str(matches[0])[:50]}..."
                found = True
                score = weight
            else:
                content = "Bulunamadı"
                found = False
                score = 0
            
            results[criterion_name] = ATAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                is_critical=is_critical,
                details={"matches_count": len(matches) if matches else 0}
            )
        
        return results
    
    def calculate_scores(self, analysis_results: Dict) -> Dict[str, Any]:
        category_scores = {}
        total_score = 0
        critical_missing = []
        
        for category, results in analysis_results.items():
            category_max = self.criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())
            
            for criterion_name, result in results.items():
                if result.is_critical and not result.found:
                    critical_missing.append(f"{category}: {criterion_name}")
            
            percentage = (category_earned / category_possible * 100) if category_possible > 0 else 0
            normalized_score = (percentage / 100) * category_max
            
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
            "percentage": round(total_score, 2),
            "critical_missing": critical_missing
        }
    
    def extract_specific_values(self, text: str) -> Dict[str, Any]:
        return {
            "manufacturer_name": "Bulunamadı",
            "manufacturer_address": "Bulunamadı",
            "machine_description": "Bulunamadı",
            "machine_model": "Bulunamadı",
            "production_year": "Bulunamadı",
            "serial_number": "Bulunamadı",
            "declaration_date": "Bulunamadı",
            "authorized_person": "Bulunamadı",
            "position": "Bulunamadı",
            "applied_standards": []
        }
    
    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        recommendations = []
        if scores["critical_missing"]:
            recommendations.append("🚨 KRİTİK EKSİKLİKLER:")
            for missing in scores["critical_missing"]:
                recommendations.append(f"  ❌ {missing}")
        else:
            recommendations.append("✅ AT Uygunluk Beyanı yeterli")
        return recommendations
    
    def analyze_at_declaration(self, pdf_path: str) -> Dict[str, Any]:
        text = self.extract_text_from_pdf(pdf_path)
        if not text or len(text.strip()) < 50:
            return {"error": "PDF okunamadı"}
        
        detected_lang = self.detect_language(text)
        extracted_values = self.extract_specific_values(text)
        
        category_analyses = {}
        for category in self.criteria_weights.keys():
            category_analyses[category] = self.analyze_criteria(text, category)
        
        scoring = self.calculate_scores(category_analyses)
        recommendations = self.generate_recommendations(category_analyses, scoring)
        
        percentage = scoring["percentage"]
        has_critical = len(scoring["critical_missing"]) > 0
        
        status = "INVALID" if has_critical else "VALID" if percentage >= 70 else "CONDITIONAL" if percentage >= 50 else "INSUFFICIENT"
        status_tr = {"INVALID": "GEÇERSİZ", "VALID": "GEÇERLİ", "CONDITIONAL": "KOŞULLU", "INSUFFICIENT": "YETERSİZ"}.get(status, "YETERSİZ")
        
        return {
            "analysis_date": datetime.now().isoformat(),
            "file_info": {"detected_language": detected_lang, "text_length": len(text)},
            "extracted_values": extracted_values,
            "category_analyses": category_analyses,
            "scoring": scoring,
            "recommendations": recommendations,
            "summary": {
                "total_score": scoring["total_score"],
                "percentage": percentage,
                "status": status,
                "status_tr": status_tr,
                "critical_missing_count": len(scoring["critical_missing"]),
                "report_type": "AT_UYGUNLUK_BEYANI"
            }
        }

def validate_document_server(text):
    critical_terms = [
        ["AT TİP", "at tip", "ec type", "uygunluk", "beyan"],
        ["SERTİFİKA", "sertifika", "certificate"],
        ["2006/42/EC", "direktif", "directive"],
        ["üretici", "manufacturer"],
        ["muayene", "inspection"]
    ]
    category_found = [any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category) for category in critical_terms]
    return sum(category_found) >= 4

def check_strong_keywords_first_pages(filepath):
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        found = [kw for kw in ["uygunluk", "beyan", "declaration"] if re.search(rf"\b{kw}\b", all_text)]
        return len(found) >= 1
    except:
        return False

def check_excluded_keywords_first_pages(filepath):
    excluded = ["aydınlatma", "hidrolik", "pnömatik", "gürültü", "isg", "hrc", "elektrik", "espe", "kullanma", "loto", "lvd", "montaj", "topraklama", "bakım", "titreşim"]
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=2)
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        found = [kw for kw in excluded if re.search(rf"\b{kw}\b", all_text)]
        return len(found) >= 1
    except:
        return False

def get_conclusion_message_at(status, percentage):
    if status == "PASS":
        return f"AT uygunluk beyanı 2006/42/EC direktifine uygun (%{percentage:.0f})"
    return f"AT uygunluk beyanı direktife uygun değil (%{percentage:.0f})"

def get_main_issues_at(report):
    issues = []
    if report['scoring']['critical_missing']:
        for item in report['scoring']['critical_missing']:
            issues.append(f"Kritik eksik: {item}")
    return issues[:4]

def check_tesseract_installation():
    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract v{version}"
    except:
        return False, "Not installed"

tesseract_available, tesseract_info = check_tesseract_installation()

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads_at'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/at-declaration', methods=['POST'])
def analyze_at_declaration():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        analyzer = ATTypeInspectionAnalyzer()
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.pdf':
            if check_strong_keywords_first_pages(filepath):
                pass
            else:
                if check_excluded_keywords_first_pages(filepath):
                    try: os.remove(filepath)
                    except: pass
                    return jsonify({'error': 'Invalid document type'}), 400
                else:
                    try:
                        with open(filepath, 'rb') as f:
                            text = "".join(page.extract_text() for page in PyPDF2.PdfReader(f).pages)
                        if not text or len(text.strip()) < 50 or not validate_document_server(text):
                            try: os.remove(filepath)
                            except: pass
                            return jsonify({'error': 'Invalid document type'}), 400
                    except:
                        try: os.remove(filepath)
                        except: pass
                        return jsonify({'error': 'Analysis failed'}), 500

        report = analyzer.analyze_at_declaration(filepath)
        try: os.remove(filepath)
        except: pass

        if 'error' in report:
            return jsonify({'error': 'Analysis failed', 'message': report['error']}), 400

        overall_percentage = report['summary']['percentage']
        status = "PASS" if overall_percentage >= 70 else "FAIL"
        
        response_data = {
            'analysis_date': report.get('analysis_date'),
            'analysis_id': f"at_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'category_scores': {},
            'extracted_values': report['extracted_values'],
            'file_type': 'AT_UYGUNLUK_BEYANI',
            'filename': filename,
            'overall_score': {
                'percentage': round(overall_percentage, 2),
                'total_points': report['summary']['total_score'],
                'max_points': 100,
                'status': status,
                'status_tr': report['summary']['status_tr']
            },
            'recommendations': report['recommendations'],
            'summary': {
                'is_valid': status == "PASS",
                'conclusion': get_conclusion_message_at(status, overall_percentage),
                'main_issues': [] if status == "PASS" else get_main_issues_at(report)
            }
        }
        
        for category, score_data in report['scoring']['category_scores'].items():
            response_data['category_scores'][category] = {
                'score': score_data.get('normalized', 0),
                'max_score': score_data.get('max_weight', 0),
                'percentage': score_data.get('percentage', 0),
                'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'FAIL'
            }

        return jsonify({
            'success': True,
            'message': 'AT Uygunluk Beyanı başarıyla analiz edildi',
            'analysis_service': 'at_declaration',
            'data': response_data
        })

    except Exception as e:
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'AT Declaration Analyzer API', 'version': '1.0.0'})

@app.route('/', methods=['GET'])
def index():
    return jsonify({'service': 'AT Declaration Analyzer API', 'version': '1.0.0'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8006))
    logger.info(f"🚀 AT Uygunluk Beyanı Servisi - Port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)