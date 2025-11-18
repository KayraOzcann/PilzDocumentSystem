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
from electric_circuit_report_checker import AdvancedElectricCircuitAnalyzer
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
    """Server kodunda doküman validasyonu - Elektrik Devre Şeması için"""
    
    # Elektrik devre şemalarında MUTLAKA olması gereken kritik kelimeler
    critical_terms = [
         # Elektrik temel terimleri
        ["elektrik", "electrical", "circuit", "devre", "şema", "diagram", "voltage", "current"],
        
        # Elektrik bileşenleri
        ["kontaktör", "contactor", "röle", "relay", "sigorta", "fuse", "mcb", "rcd", "switch"],
        
        # Elektrik ölçüm birimleri
        ["volt", "v", "amper", "a", "watt", "w", "ohm", "ω", "hz", "hertz"],
        
        # Elektrik güvenlik ve kontrol
        ["stop", "start", "emergency", "acil", "güvenlik", "safety", "control", "kontrol"]
    ]
    
    # Her kategori için kontrol
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"Elektrik Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    # Tüm kategorilerden en az bir terim bulunmalı
    valid_categories = sum(category_found)
    
    logger.info(f"Doküman validasyonu: {valid_categories}/4 kritik kategori bulundu")
    
    # 4 kategorinin tamamında terim bulunmalı (daha sıkı kontrol)
    return valid_categories >= 4

def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - Elektrik Devre Şeması için"""
    strong_keywords = [
        "elektrik", 
        "circuit", 
        "electrical", 
        "voltage", 
        "amper", 
        "ohm",
        "enclosure",
        "wrp-",
        "light curtain",
        "contactors",
        "controller",
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=2)
        
        all_text = ""
        for i, page in enumerate(pages):
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
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
        # Topraklama raporu (eski strong_keywords)
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        
        # Aydınlatma raporu
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
        
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Espe raporu  
        "espe",
        
        # Hidrolik devre şeması
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219","teknik resim","tasarım",
        
        # Gürültü ölçüm raporu
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # Manuel/kullanma kılavuzu
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide","kılavuzu",
        
        # LOTO raporu
        "loto",
        
        # LVD raporu
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        
        # AT tip muayene (AT uygunluk beyanı)
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=2)
        
        all_text = ""
        for i, page in enumerate(pages):
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
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


app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads_elektrik'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/elektrik-report', methods=['POST'])
def analyze_elektrik_report():
    """Elektrik Devre Şeması analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere bir elektrik devre şeması sağlayın'
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
                'message': 'Sadece PDF, DOCX, DOC ve TXT dosyaları kabul edilir'
            }), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"Elektrik Devre Şeması kontrol ediliyor: {filename}")

            # Create analyzer instance
            analyzer = AdvancedElectricCircuitAnalyzer()
            
            # ÜÇ AŞAMALI ELEKTRİK KONTROLÜ
            logger.info(f"Üç aşamalı elektrik kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                # PDF için üç aşamalı kontrol (OCR dahil)
                logger.info("Aşama 1: İlk sayfa elektrik özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath):
                    logger.info("✅ Aşama 1 geçti - Elektrik özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - Elektrik değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya elektrik devre şeması değil (farklı rapor türü tespit edildi). Lütfen elektrik devre şeması yükleyiniz.',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'ELEKTRIK_DEVRE_SEMASI'
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
                                    'message': 'Yüklediğiniz dosya elektrik devre şeması değil! Lütfen geçerli bir elektrik devre şeması yükleyiniz.',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_ELEKTRIK_CIRCUIT',
                                        'required_type': 'ELEKTRIK_DEVRE_SEMASI'
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

            elif file_ext in ['.docx', '.doc', '.txt']:
                # DOCX/TXT için sadece tam doküman kontrolü
                logger.info(f"DOCX/TXT dosyası için tam doküman kontrolü: {file_ext}")
                text = ""
                if file_ext in ['.docx', '.doc']:
                    # DOCX okuma için basit yaklaşım - analyzer'ın metodunu kullan
                    pass
                elif file_ext == '.txt':
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                
                if text and not validate_document_server(text):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    return jsonify({
                        'error': 'Invalid document type',
                        'message': 'Yüklediğiniz dosya elektrik devre şeması değil! Lütfen geçerli bir elektrik devre şeması yükleyiniz.',
                        'details': {
                            'filename': filename,
                            'document_type': 'NOT_ELEKTRIK_CIRCUIT',
                            'required_type': 'ELEKTRIK_DEVRE_SEMASI'
                        }
                    }), 400

            # Buraya kadar geldiyse elektrik devre şeması, şimdi analizi yap
            logger.info(f"Elektrik devre şeması doğrulandı, analiz başlatılıyor: {filename}")
            report = analyzer.analyze_circuit_diagram(filepath)
            
            # Clean up the uploaded file
            try:
                os.remove(filepath)
                logger.info(f"Uploaded file {filename} cleaned up")
            except Exception as e:
                logger.warning(f"Could not remove uploaded file {filename}: {e}")

            # Check if analysis was successful
            if 'error' in report:
                return jsonify({
                    'error': 'Analysis failed',
                    'message': report['error'],
                    'details': {
                        'filename': filename
                    }
                }), 400

            # Extract key results for API response - ELEKTRİK FORMATINDA
            overall_percentage = report.get('overall_score', 0)
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'analysis_details': report.get('analysis_details', {}),
                'analysis_id': f"elektrik_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': report.get('category_scores', {}),
                'date_validity': {
                    'is_valid': True,  # Elektrik devre şemaları için tarih geçerliliği kontrolü uygulanmaz
                    'message': 'Elektrik devre şemaları için tarih geçerliliği kontrolü uygulanmaz',
                    'days_old': 0,
                    'formatted_date': 'N/A'
                },
                'extracted_values': report.get('extracted_values', {}),
                'file_type': 'ELEKTRIK_DEVRE_SEMASI',
                'filename': filename,
                'language_info': {
                    'detected_language': 'turkish',
                    'text_length': len(str(report))
                },
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
                'analysis_service': 'electric_circuit',
                'data': response_data,
                'message': 'Elektrik Devre Şeması başarıyla analiz edildi',
                'service_description': 'Elektrik Devre Şeması Analizi',
                'service_port': assigned_port,
                'success': True
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
                'message': f'Elektrik devre şeması analizi sırasında hata oluştu: {str(analysis_error)}',
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

def get_conclusion_message_elektrik(status, percentage):
    """Sonuç mesajını döndür - Elektrik için"""
    if status == "PASS":
        return f"Elektrik devre şeması standartlara uygundur (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"Elektrik devre şeması kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"Elektrik devre şeması standartlara uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint - Elektrik için"""
    return jsonify({
        'status': 'healthy',
        'service': 'Electric Circuit Analyzer API',
        'version': '1.0.0',
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': True,  # Elektrik'te OCR var
        'report_type': 'ELEKTRIK_DEVRE_SEMASI'
    })

@app.route('/api/test-elektrik', methods=['GET'])
def test_elektrik_analysis():
    """Test endpoint for debugging - Elektrik için"""
    try:
        analyzer = AdvancedElectricCircuitAnalyzer()
        test_info = {
            'analyzer_initialized': True,
            'available_methods': [method for method in dir(analyzer) if not method.startswith('_')],
            'ocr_support': True  # Elektrik'te OCR var
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

@app.route('/', methods=['GET'])
def index():
    """Ana sayfa - API bilgileri - Elektrik için"""
    return jsonify({
        'service': 'Electric Circuit Analyzer API',
        'version': '1.0.0',
        'description': 'Elektrik Devre Şemalarını analiz eden REST API servisi',
        'endpoints': {
            'POST /api/elektrik-report': 'Elektrik devre şeması analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /api/test-elektrik': 'Test analizi',
            'GET /': 'Bu bilgi sayfası'
        },
        'usage': {
            'upload_format': 'multipart/form-data',
            'file_field': 'file',
            'supported_types': list(ALLOWED_EXTENSIONS),
            'max_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        },
        'analysis_categories': [
            'Elektrik Bileşenleri',
            'Devre Yapısı',
            'Güvenlik Elemanları',
            'Kontrol Sistemi',
            'Dokümantasyon Kalitesi'
        ],
        'scoring': {
            'PASS': '≥70% - Standarda uygun',
            'CONDITIONAL': '50-69% - Kabul edilebilir',
            'FAIL': '<50% - Standarda uygun değil'
        },
        'example_curl': 'curl -X POST -F "file=@elektrik_devre.pdf" http://localhost:5002/api/elektrik-report'
    })

if __name__ == '__main__':
    logger.info("Electric Circuit Analyzer API başlatılıyor...")
    logger.info(f"Upload klasörü: {UPLOAD_FOLDER}")
    logger.info("API Endpoints:")
    logger.info("  POST /api/elektrik-report - Elektrik devre şeması analizi")
    logger.info("  GET /api/health - Servis sağlık kontrolü")
    logger.info("  GET /api/test-elektrik - Test analizi")
    logger.info("  GET / - API bilgileri")

    assigned_port = int(os.environ.get('ASSIGNED_PORT', 5002))
    app.run(host='0.0.0.0', port=assigned_port, debug=False)