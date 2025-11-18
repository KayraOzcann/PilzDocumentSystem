import PyPDF2
import cv2
import numpy as np
from PIL import Image
import pdf2image
import pytesseract
import re
from typing import List
from flask import Flask, request, jsonify
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from at_tip_sertifika2 import ATTipIncelemeAnalyzer
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
    return lang_mapping.get(lang_code, 'english')

def validate_document_server(text):
    """Server kodunda doküman validasyonu - AT Tip İnceleme için"""
    
    # AT Tip İnceleme belgelerinde MUTLAKA olması gereken kritik kelimeler
    critical_terms = [
        # AT Tip temel terimleri (en az 1 tane olmalı)
        ["inceleme", "examination", "sertifika", "certificate", "belge", "document", "at tip", "ec type"],
        
        # Makine direktifi terimleri (en az 1 tane olmalı)  
        ["direktif", "directive", "makine", "machinery", "2006/42/ec", "42/ec", "ek ix", "annex ix"],
        
        # Onaylanmış kuruluş terimleri (mutlaka olmalı)
        ["onaylanmış", "notified", "kuruluş", "body", "notified body", "onaylanmış kuruluş"],
        
        # Belge geçerlilik terimleri (en az 1 tane olmalı)
        ["geçerli", "valid", "yetki", "authority", "onay", "approval", "tarih", "date"]
    ]
    
    # Her kategori için kontrol
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"AT Tip Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    # Tüm kategorilerden en az bir terim bulunmalı
    valid_categories = sum(category_found)
    
    logger.info(f"Doküman validasyonu: {valid_categories}/4 kritik kategori bulundu")
    
    # 4 kategorinin tamamında terim bulunmalı (daha sıkı kontrol)
    return valid_categories >= 4

def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - AT Tip İnceleme için"""
    strong_keywords = [
        "AT TİP",
        "at tip",
        "ec type",
        "SERTİFİKA",
        "sertifika",
        "certificate"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        
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
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        
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
        "pnömatik", "pneumatic", "lubricator","inflate","psi","bar","oil","regis","r102","regulator","dump valve","oil",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # Aydınlatma
        "aydınlatma", "lighting",  "illumination",  "lux",  "lümen",  "lumen",  "ts en 12464",  "en 12464", "ışık",  "ışık şiddeti"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=200, first_page=1, last_page=2)
        
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
        return len(found_excluded) >= 2
        
    except Exception as e:
        logger.warning(f"İlk sayfa excluded kontrol hatası: {e}")
        return False

app = Flask(__name__)

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

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads_at_tip'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'docx', 'doc', 'txt'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/ati-inceleme-report', methods=['POST'])
def analyze_at_tip_certificate():
    """AT Tip İncelemesi Belgesi analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere bir AT Tip İncelemesi Belgesi sağlayın'
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
            logger.info(f"AT Tip İncelemesi Belgesi kontrol ediliyor: {filename}")

            # Create analyzer instance
            analyzer = ATTipIncelemeAnalyzer()
            
            # ÜÇ AŞAMALI AT TİP KONTROLÜ
            logger.info(f"Üç aşamalı AT Tip kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()
            requires_ocr = file_ext in ['.jpg', '.jpeg', '.png']

            if file_ext == '.pdf':
                # PDF için üç aşamalı kontrol (OCR dahil)
                logger.info("Aşama 1: İlk sayfa AT Tip özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath):
                    logger.info("✅ Aşama 1 geçti - AT Tip özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - AT Tip değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya AT Tip İnceleme belgesi değil (farklı rapor türü tespit edildi). Lütfen AT Tip İnceleme belgesi yükleyiniz.',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'AT_TIP_INCELEME_BELGESI'
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
                                    'message': 'Yüklediğiniz dosya AT Tip İnceleme belgesi değil! Lütfen geçerli bir AT Tip İnceleme belgesi yükleyiniz.',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_AT_TIP_DOCUMENT',
                                        'required_type': 'AT_TIP_INCELEME_BELGESI'
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
                        'message': 'Yüklediğiniz dosya AT Tip İnceleme belgesi değil! Lütfen geçerli bir AT Tip İnceleme belgesi yükleyiniz.',
                        'details': {
                            'filename': filename,
                            'document_type': 'NOT_AT_TIP_DOCUMENT',
                            'required_type': 'AT_TIP_INCELEME_BELGESI'
                        }
                    }), 400

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

            # Buraya kadar geldiyse AT Tip belgesi, şimdi analizi yap
            logger.info(f"AT Tip belgesi doğrulandı, analiz başlatılıyor: {filename}")
            analysis_result = analyzer.analyze_type_examination_certificate(filepath)
            
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

            # Extract key results for API response - TOPRAKLAMA FORMATINDA
            overall_percentage = analysis_result['scoring']['percentage']
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
                'analysis_id': f"at_tip_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': 'N/A',
                'extracted_values': analysis_result.get('extracted_values', {}),
                'file_type': 'AT_TIP_INCELEME_BELGESI',
                'filename': filename,
                'language_info': {
                    'detected_language': map_language_code(analysis_result['file_info']['detected_language']),
                    'text_length': analysis_result['file_info']['text_length']
                },
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': analysis_result['scoring']['total_score'],
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'text_quality': 'good' if len(analysis_result.get('extracted_values', {})) > 3 else 'fair'
                },
                'recommendations': analysis_result.get('recommendations', []),
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_at_tip(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_at_tip(analysis_result)
                }
            }
            
            # Add category scores - TOPRAKLAMA FORMATINDA
            for category, score_data in analysis_result['scoring']['category_scores'].items():
                response_data['category_scores'][category] = {
                    'score': score_data['normalized'],
                    'max_score': score_data['max_weight'],
                    'percentage': score_data['percentage'],
                    'status': 'PASS' if score_data['percentage'] >= 70 else 'CONDITIONAL' if score_data['percentage'] >= 50 else 'FAIL'
                }

            return jsonify({
                'success': True,
                'message': 'AT Tip İncelemesi Belgesi başarıyla analiz edildi',
                'analysis_service': 'at_tip',
                'service_description': 'AT Tip İncelemesi Belgesi Analizi',
                'service_port': 5019,
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
                'message': f'AT Tip İncelemesi analizi sırasında hata oluştu: {str(analysis_error)}',
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

def get_conclusion_message_at_tip(status, percentage):
    """Sonuç mesajını döndür - AT Tip için"""
    if status == "PASS":
        return f"AT Tip İncelemesi Belgesi 2006/42/EC Makine Direktifi'ne uygundur (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"AT Tip İncelemesi Belgesi kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"AT Tip İncelemesi Belgesi direktife uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"

def get_main_issues_at_tip(analysis_result):
    """Ana sorunları listele - AT Tip için"""
    issues = []
    
    for category, score_data in analysis_result['scoring']['category_scores'].items():
        if score_data['percentage'] < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    # Kritik eksiklikleri ekle
    if 'critical_missing' in analysis_result['scoring']:
        for critical in analysis_result['scoring']['critical_missing']:
            issues.append(f"Kritik alan eksik: {critical}")
    
    # Eğer hiç kritik sorun yoksa genel sorunları ekle
    if not issues:
        if analysis_result['scoring']['total_score'] < 50:
            issues = [
                "Onaylanmış kuruluş bilgileri eksik",
                "İmalatçı bilgileri eksik",
                "Makine tipi ve model bilgileri eksik",
                "Direktif referansı eksik",
                "Standart referansları eksik"
            ]
    
    return issues[:4]  # En fazla 4 ana sorun göster

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint - AT Tip için"""
    return jsonify({
        'status': 'healthy',
        'service': 'AT Tip İncelemesi Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'tesseract_info': tesseract_info,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': True,
        'report_type': 'AT_TIP_INCELEME_BELGESI'
    })

@app.route('/api/test-at-tip', methods=['GET'])
def test_at_tip_analysis():
    """Test endpoint for debugging - AT Tip için"""
    try:
        analyzer = ATTipIncelemeAnalyzer()
        test_info = {
            'analyzer_initialized': True,
            'available_methods': [method for method in dir(analyzer) if not method.startswith('_')],
            'criteria_categories': list(analyzer.criteria_weights.keys()),
            'criteria_weights': analyzer.criteria_weights,
            'total_possible_score': sum(analyzer.criteria_weights.values()),
            'ocr_support': True,
            'tesseract_available': tesseract_available
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

@app.route('/api/at-tip-categories', methods=['GET'])
def get_at_tip_categories():
    """AT Tip analiz kategorilerini döndür - Topraklama formatında"""
    try:
        analyzer = ATTipIncelemeAnalyzer()
        return jsonify({
            'success': True,
            'data': {
                'categories': analyzer.criteria_weights,
                'total_weight': sum(analyzer.criteria_weights.values()),
                'criteria_details': {
                    category: list(criteria.keys()) 
                    for category, criteria in analyzer.criteria_details.items()
                },
                'standard_reference': '2006/42/EC Makine Direktifi Ek IX'
            }
        })
    except Exception as e:
        return jsonify({
            'error': 'Failed to get categories',
            'message': str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Ana sayfa - API bilgileri - AT Tip için"""
    return jsonify({
        'service': 'AT Tip İncelemesi Analyzer API',
        'version': '1.0.0',
        'description': '2006/42/EC Makine Direktifi Ek IX AT Tip İncelemesi Belgelerini analiz eden REST API servisi',
        'endpoints': {
            'POST /api/ati-inceleme-report': 'AT Tip İncelemesi Belgesi analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /api/test-at-tip': 'Test analizi',
            'GET /api/at-tip-categories': 'Analiz kategorileri',
            'GET /': 'Bu bilgi sayfası'
        },
        'usage': {
            'upload_format': 'multipart/form-data',
            'file_field': 'file',
            'supported_types': list(ALLOWED_EXTENSIONS),
            'max_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        },
        'analysis_categories': [
            'Genel Bilgiler',
            'Onaylanmış Kuruluş Bilgileri',
            'İmalatçı Bilgileri',
            'Makine Tanımı',
            'Belge Bilgileri',
            'Direktif ve Standartlar',
            'Geçerlilik ve Yetki'
        ],
        'scoring': {
            'PASS': '≥70% - Direktife uygun',
            'CONDITIONAL': '50-69% - Kabul edilebilir',
            'FAIL': '<50% - Direktife uygun değil'
        },
        'example_curl': 'curl -X POST -F "file=@at_tip_belgesi.pdf" http://localhost:5019/api/ati-inceleme-report'
    })

if __name__ == '__main__':
    logger.info("AT Tip İncelemesi Analyzer API başlatılıyor...")
    logger.info(f"Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"Tesseract durumu: {tesseract_info}")
    logger.info("API Endpoints:")
    logger.info("  POST /api/ati-inceleme-report - AT Tip İncelemesi analizi")
    logger.info("  GET /api/health - Servis sağlık kontrolü")
    logger.info("  GET /api/test-at-tip - Test analizi")
    logger.info("  GET /api/at-tip-categories - Analiz kategorileri")
    logger.info("  GET / - API bilgileri")
    
    # Get port from environment variable (set by main_api_gateway.py)
    assigned_port = int(os.environ.get('ASSIGNED_PORT', 5019))
    app.run(host='0.0.0.0', port=assigned_port, debug=False)