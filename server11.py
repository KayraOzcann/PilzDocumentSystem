import PyPDF2
import cv2
import numpy as np
from PIL import Image
import pdf2image
import pytesseract
from flask import Flask, request, jsonify
import os
import re
from datetime import datetime
from werkzeug.utils import secure_filename
from hydraulic_circuit_diagram_report_checker import AdvancedCircuitAnalyzer
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def validate_document_server(text):
    """Server kodunda doküman validasyonu - Hidrolik Devre Şeması için"""
    
    # Hidrolik devre şemalarında MUTLAKA olması gereken kritik kelimeler
    critical_terms = [
        # Hidrolik temel terimleri
        ["hidrolik", "hydraulic", "devre", "circuit", "şema", "diagram", "schema"],
        
        # Hidrolik bileşenleri ve semboller
        ["pompa", "pump", "valf", "valve", "silindir", "cylinder", "motor", "actuator", "piston"],
        
        # Hidrolik basınç ve akış terimleri
        ["basınç", "pressure", "bar", "psi", "debi", "flow", "l/min", "gpm", "mpa"],
        
        # Hidrolik sıvı ve sistem terimleri
        ["yağ", "oil", "hidrolik yağ", "hydraulic oil", "tank", "rezervuar", "filtre", "filter"],
        
        # ISO standartları ve teknik terimler
        ["iso 1219", "1219", "sembol", "symbol", "bağlantı", "connection", "hat", "line"]
    ]
    
    # Her kategori için kontrol
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"Hidrolik Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    # Tüm kategorilerden en az bir terim bulunmalı
    valid_categories = sum(category_found)
    
    logger.info(f"Doküman validasyonu: {valid_categories}/5 kritik kategori bulundu")
    
    # 5 kategorinin en az 4'ünde terim bulunmalı
    return valid_categories >= 5

def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - Hidrolik Devre Şeması için"""
    strong_keywords = [
        "hidrolik",
        "HİDROLİK",
        "hydraulic",
        "hidrolik yağ",
        "hydraulic oil",
        "iso 1219",
        "1219",
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        
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

def check_excluded_keywords_first_pages(filepath):
    """İlk 1-2 sayfada istenmeyen rapor türlerinin kelimelerini ara"""
    excluded_keywords = [
        # Aydınlatma raporu (eski strong_keywords aydınlatmadan)

        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık","ışık şiddeti"
        
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        
        # Espe raporu  
        "espe",
        
        # Gürültü ölçüm raporu
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # Manuel/kullanma kılavuzu
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide","kılavuzu",
        
        # LOTO raporu
        "loto",
        
        # LVD raporu
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        
        # AT tip muayene
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", 
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE"
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        
        all_text = ""
        for i, page in enumerate(pages):
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 3 -l tur+eng', timeout=15)
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

# Tesseract installation check
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

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads_hydraulic'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/hydraulic-control', methods=['POST'])
def analyze_hydraulic_control():
    """Hidrolik Devre Şeması analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere bir hidrolik devre şeması sağlayın'
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
            logger.info(f"Hidrolik Devre Şeması kontrol ediliyor: {filename}")

            # Create analyzer instance
            analyzer = AdvancedCircuitAnalyzer()
            
            # ÜÇ AŞAMALI HİDROLİK KONTROLÜ
            logger.info(f"Üç aşamalı hidrolik kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                # PDF için üç aşamalı kontrol (OCR dahil)
                logger.info("Aşama 1: İlk sayfa hidrolik özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath):
                    logger.info("✅ Aşama 1 geçti - Hidrolik özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - Hidrolik değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya hidrolik devre şeması değil (farklı rapor türü tespit edildi). Lütfen hidrolik devre şeması yükleyiniz.',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'HIDROLIK_DEVRE_SEMASI'
                            }
                        }), 400
                    else:
                        # AŞAMA 3: PyPDF2 ile tam doküman kontrolü
                        logger.info("Aşama 3: Tam doküman critical terms kontrolü...")
                        try:
                            import PyPDF2
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
                                    'message': 'Yüklediğiniz dosya hidrolik devre şeması değil! Lütfen geçerli bir hidrolik devre şeması yükleyiniz.',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_HYDRAULIC_CIRCUIT',
                                        'required_type': 'HIDROLIK_DEVRE_SEMASI'
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
                # Image files - check if OCR is available
                requires_ocr = True
                if requires_ocr and not tesseract_available:
                    # Clean up file first
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
                            'requires_ocr': True,
                            'installation_help': {
                                'windows': 'https://github.com/UB-Mannheim/tesseract/wiki adresinden Tesseract indirip kurun',
                                'macos': 'brew install tesseract komutunu çalıştırın',
                                'ubuntu': 'sudo apt-get install tesseract-ocr tesseract-ocr-tur komutunu çalıştırın',
                                'centos': 'sudo yum install tesseract tesseract-langpack-tur komutunu çalıştırın'
                            }
                        }
                    }), 500

            # Buraya kadar geldiyse hidrolik devre şeması, şimdi analizi yap
            logger.info(f"Hidrolik devre şeması doğrulandı, analiz başlatılıyor: {filename}")
            analysis_result = analyzer.analyze_circuit_diagram(filepath)
            
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
                        'analysis_details': analysis_result.get('details', {})
                    }
                }), 400

            # Extract key results for API response - HIDROLIK FORMATINDA
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
                'analysis_id': f"hydraulic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': 'N/A',
                'extracted_values': analysis_result.get('extracted_values', {}),
                'file_type': 'HIDROLIK_DEVRE_SEMASI',
                'filename': filename,
                'language_info': {
                    'detected_language': 'turkish',  # Hidrolik şemalar genelde Türkçe/İngilizce karışık
                    'file_type': file_ext.replace('.', '')
                },
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': analysis_result['scoring']['total_score'],
                    'max_points': analysis_result['scoring']['total_max_score'],
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'text_quality': analysis_result['scoring'].get('text_quality', 'good')
                },
                'recommendations': analysis_result.get('recommendations', []),
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_hydraulic(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_hydraulic(analysis_result)
                }
            }
            
            # Add category scores - HIDROLIK FORMATINDA
            for category, score_data in analysis_result['scoring']['category_scores'].items():
                response_data['category_scores'][category] = {
                    'score': score_data['normalized'],
                    'max_score': score_data['max_weight'],
                    'percentage': score_data['percentage'],
                    'status': 'PASS' if score_data['percentage'] >= 70 else 'CONDITIONAL' if score_data['percentage'] >= 50 else 'FAIL'
                }

            return jsonify({
                'success': True,
                'message': 'Hidrolik Devre Şeması başarıyla analiz edildi',
                'analysis_service': 'hydraulic_circuit_diagram',
                'service_description': 'Hidrolik Devre Şeması Analizi',
                'service_port': 5012,
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
                'message': f'Hidrolik devre şeması analizi sırasında hata oluştu: {str(analysis_error)}',
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

def get_conclusion_message_hydraulic(status, percentage):
    """Sonuç mesajını döndür - Hidrolik için"""
    if status == "PASS":
        return f"Hidrolik devre şeması ISO 1219 standardına uygun ve teknik açıdan yeterlidir (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"Hidrolik devre şeması kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"Hidrolik devre şeması standartlara uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"

def get_main_issues_hydraulic(analysis_result):
    """Ana sorunları listele - Hidrolik için"""
    issues = []
    
    for category, score_data in analysis_result['scoring']['category_scores'].items():
        if score_data['percentage'] < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    # Eğer hiç kritik sorun yoksa genel sorunları ekle
    if not issues:
        if analysis_result['scoring']['total_score'] < 50:
            issues = [
                "Hidrolik semboller ISO 1219 standardına uygun değil",
                "Basınç ve debi değerleri eksik veya hatalı",
                "Sistem bileşenleri tam tanımlanmamış",
                "Güvenlik elemanları eksik",
                "Teknik özellikler yetersiz"
            ]
    
    return issues[:4]  # En fazla 4 ana sorun göster

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint - Hidrolik için"""
    return jsonify({
        'status': 'healthy',
        'service': 'Hydraulic Circuit Diagram Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'tesseract_info': tesseract_info,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': tesseract_available,  # Hidrolik'te OCR var (resimler için)
        'report_type': 'HIDROLIK_DEVRE_SEMASI'
    })

@app.route('/api/test-hydraulic', methods=['GET'])
def test_hydraulic_analysis():
    """Test endpoint for debugging - Hidrolik için"""
    try:
        analyzer = AdvancedCircuitAnalyzer()
        test_info = {
            'analyzer_initialized': True,
            'available_methods': [method for method in dir(analyzer) if not method.startswith('_')],
            'criteria_categories': list(analyzer.hydraulic_criteria_weights.keys()) if hasattr(analyzer, 'hydraulic_criteria_weights') else [],
            'criteria_weights': getattr(analyzer, 'hydraulic_criteria_weights', {}),
            'total_possible_score': sum(getattr(analyzer, 'hydraulic_criteria_weights', {}).values()),
            'tesseract_status': tesseract_available,
            'ocr_support': tesseract_available
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

@app.route('/api/hydraulic-categories', methods=['GET'])
def get_hydraulic_categories():
    """Hidrolik analiz kategorilerini döndür"""
    try:
        analyzer = AdvancedCircuitAnalyzer()
        return jsonify({
            'success': True,
            'data': {
                'categories': getattr(analyzer, 'hydraulic_criteria_weights', {}),
                'total_weight': sum(getattr(analyzer, 'hydraulic_criteria_weights', {}).values()),
                'criteria_details': getattr(analyzer, 'criteria_details', {}),
                'standard_reference': 'ISO 1219-1 ve ISO 1219-2'
            }
        })
    except Exception as e:
        return jsonify({
            'error': 'Failed to get categories',
            'message': str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Ana sayfa - API bilgileri - Hidrolik için"""
    return jsonify({
        'service': 'Hydraulic Circuit Diagram Analyzer API',
        'version': '1.0.0',
        'description': 'Hidrolik Devre Şemalarını analiz eden REST API servisi',
        'endpoints': {
            'POST /api/hydraulic-control': 'Hidrolik devre şeması analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /api/test-hydraulic': 'Test analizi',
            'GET /api/hydraulic-categories': 'Analiz kategorileri',
            'GET /': 'Bu bilgi sayfası'
        },
        'usage': {
            'upload_format': 'multipart/form-data',
            'file_field': 'file',
            'supported_types': list(ALLOWED_EXTENSIONS),
            'max_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        },
        'analysis_categories': [
            'Hidrolik Semboller ve Standart Uyumu',
            'Sistem Bileşenleri ve Tanımları',
            'Basınç ve Debi Bilgileri',
            'Güvenlik Elemanları',
            'Teknik Özellikler ve Parametreler',
            'Bağlantı ve Hat Tanımları',
            'Ölçü ve Toleranslar',
            'Dokümantasyon Kalitesi'
        ],
        'scoring': {
            'PASS': '≥70% - ISO 1219 standardına uygun',
            'CONDITIONAL': '50-69% - Kabul edilebilir',
            'FAIL': '<50% - Standarda uygun değil'
        },
        'example_curl': 'curl -X POST -F "file=@hidrolik_schema.pdf" http://localhost:5012/api/hydraulic-control'
    })

if __name__ == '__main__':
    logger.info("Hydraulic Circuit Diagram Analyzer API başlatılıyor...")
    logger.info(f"Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"Tesseract durumu: {tesseract_info}")
    logger.info("API Endpoints:")
    logger.info("  POST /api/hydraulic-control - Hidrolik devre şeması analizi")
    logger.info("  GET /api/health - Servis sağlık kontrolü")
    logger.info("  GET /api/test-hydraulic - Test analizi")
    logger.info("  GET /api/hydraulic-categories - Analiz kategorileri")
    logger.info("  GET / - API bilgileri")

    assigned_port = int(os.environ.get('ASSIGNED_PORT', 5012))
    app.run(host='0.0.0.0', port=assigned_port, debug=False)