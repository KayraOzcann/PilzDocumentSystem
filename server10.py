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
from pnomatic_report_checker import PneumaticCircuitAnalyzer
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
    """Server kodunda doküman validasyonu - Pnömatik Devre Şeması için"""
    
    # Pnömatik devre şemalarında MUTLAKA olması gereken kritik kelimeler
    critical_terms = [
        # Pnömatik temel terimleri
        ["pnömatik", "pnomatik", "pneumatic", "hava", "air", "basınçlı hava", "compressed air"],
        
        # Pnömatik bileşenleri ve semboller
        ["silindir", "cylinder", "valf", "valve", "vana", "frl", "lubricator", "regulator", "filter"],
        
        # Pnömatik basınç ve akış terimleri
        ["basınç", "pressure", "psi", "bar", "debi", "flow", "cfm", "l/min"],
        
        # Pnömatik kontrol elemanları
        ["kontrol", "control", "yön kontrol", "directional control", "hız kontrol", "speed control"],
        
        # ISO standartları ve teknik terimler
        ["iso 5599", "5599", "iso 1219", "sembol", "symbol", "bağlantı", "connection", "port"]
    ]
    
    # Her kategori için kontrol
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"Pnömatik Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    # Tüm kategorilerden en az bir terim bulunmalı
    valid_categories = sum(category_found)
    
    logger.info(f"Doküman validasyonu: {valid_categories}/5 kritik kategori bulundu")
    
    # 5 kategorinin en az 4'ünde terim bulunmalı
    return valid_categories >= 4

def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - Pnömatik Devre Şeması için"""
    strong_keywords = [
        "pnömatik",
        "pnomatik", 
        "pneumatic",
        "lubricator",
        "inflate",
        "psi",
        "bar",
        "regis",
        "r102",
        "regulator",
        "dump valve"
    ]
    
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

def check_excluded_keywords_first_pages(filepath):
    """İlk 1-2 sayfada istenmeyen rapor türlerinin kelimelerini ara"""
    excluded_keywords = [
        # Aydınlatma raporu
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
        
        # Hidrolik devre şeması (eski strong_keywords hidrolikten)
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219", "teknik resim", "tasarım",
        
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        
        # Espe raporu  
        "espe",
        
        # Gürültü ölçüm raporu
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # Manuel/kullanma kılavuzu
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide", "kılavuzu",
        
        # LOTO raporu
        "loto",
        
        # LVD raporu
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        
        # AT tip muayene
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration","TİTREŞİM",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=200, first_page=1, last_page=1)
        
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
UPLOAD_FOLDER = 'temp_uploads_pnomatic'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

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

            # Create analyzer instance
            analyzer = PneumaticCircuitAnalyzer()
            
            # ÜÇ AŞAMALI PNÖMATİK KONTROLÜ
            logger.info(f"Üç aşamalı pnömatik kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                # PDF için üç aşamalı kontrol (OCR dahil)
                logger.info("Aşama 1: İlk sayfa pnömatik özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath):
                    logger.info("✅ Aşama 1 geçti - Pnömatik özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath):
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

            # Buraya kadar geldiyse pnömatik devre şeması, şimdi analizi yap
            logger.info(f"Pnömatik devre şeması doğrulandı, analiz başlatılıyor: {filename}")
            analysis_result = analyzer.analyze_file(filepath)
            
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

            # Extract key results for API response - PNÖMATİK FORMATINDA
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
                    'is_valid': True,  # Pnömatik devre şemaları için tarih geçerliliği kontrolü uygulanmaz
                    'message': 'Pnömatik devre şemaları için tarih geçerliliği kontrolü uygulanmaz',
                    'days_old': 0,
                    'formatted_date': 'N/A'
                },
                'extracted_values': analysis_result.get('extracted_values', {}),
                'file_type': 'PNOMATIK_DEVRE_SEMASI',
                'filename': filename,
                'language_info': {
                    'detected_language': 'turkish',  # Pnömatik şemalar genelde Türkçe/İngilizce karışık
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
            
            # Add category scores - PNÖMATİK FORMATINDA
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
    
    # Eğer hiç kritik sorun yoksa genel sorunları ekle
    if not issues:
        if analysis_result['summary']['total_score'] < 50:
            issues = [
                "Pnömatik semboller ISO 5599 standardına uygun değil",
                "FRL ünitesi ve hava hazırlama eksik",
                "Basınç ve debi değerleri eksik veya hatalı",
                "Vana tipleri ve kontrol sistemleri yetersiz",
                "Güvenlik elemanları ve acil durdurma eksik"
            ]
    
    return issues[:4]  # En fazla 4 ana sorun göster

@app.route('/api/pnomatic-health', methods=['GET'])
def health_check():
    """Health check endpoint - Pnömatik için"""
    return jsonify({
        'status': 'healthy',
        'service': 'Pneumatic Circuit Diagram Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'tesseract_info': tesseract_info,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': tesseract_available,  # Pnömatik'te OCR var (resimler için)
        'report_type': 'PNOMATIK_DEVRE_SEMASI'
    })

@app.route('/api/pnomatic-validate', methods=['GET'])
def validate_file():
    """Dosya doğrulama endpoint'i"""
    filename = request.args.get('filename', '')
    
    if not filename:
        return jsonify({
            'valid': False,
            'message': 'Dosya adı belirtilmedi'
        }), 400
    
    # Check file extension
    is_valid = allowed_file(filename)
    file_extension = os.path.splitext(filename)[1].lower()
    requires_ocr = file_extension in ['.jpg', '.jpeg', '.png']
    
    response = {
        'valid': is_valid,
        'filename': filename,
        'file_extension': file_extension,
        'requires_ocr': requires_ocr,
        'tesseract_available': tesseract_available
    }
    
    if not is_valid:
        response['message'] = f'Desteklenmeyen dosya türü: {file_extension}'
        response['supported_formats'] = list(ALLOWED_EXTENSIONS)
    elif requires_ocr and not tesseract_available:
        response['valid'] = False
        response['message'] = 'OCR gerekli ama Tesseract kurulu değil'
        response['tesseract_error'] = tesseract_info
    else:
        response['message'] = 'Dosya geçerli ve analiz edilebilir'
    
    return jsonify(response)

@app.route('/api/pnomatic-info', methods=['GET'])
def get_analysis_info():
    """Analiz bilgileri endpoint'i"""
    return jsonify({
        'service_name': 'Pnömatik Devre Şeması Analiz Servisi',
        'description': 'Pnömatik devre şemalarını analiz eder ve puanlar',
        'version': '1.0.0',
        'analysis_categories': {
            'Temel Sistem Bileşenleri': {
                'weight': 25,
                'description': 'Hava kaynağı, FRL, basınç göstergeleri, susturucu'
            },
            'Pnömatik Semboller ve Vana Sistemleri': {
                'weight': 30,
                'description': 'Silindirler, yön kontrol vanaları, hız/basınç kontrol vanaları'
            },
            'Akış Yönü ve Bağlantı Hatları': {
                'weight': 20,
                'description': 'Besleme hatları, çalışma hatları, egzoz hatları, yön okları'
            },
            'Sistem Bilgileri ve Teknik Parametreler': {
                'weight': 15,
                'description': 'Çalışma basıncı, hava tüketimi, strok boyutları, vana tipleri'
            },
            'Dokümantasyon ve Standart Uygunluk': {
                'weight': 10,
                'description': 'ISO standartları, çizim bilgileri, proje bilgileri, firma bilgileri'
            }
        },
        'scoring_system': {
            'total_points': 100,
            'pass_threshold': 70,
            'quality_levels': {
                '90-100': 'MÜKEMMEL',
                '80-89': 'ÇOK İYİ',
                '70-79': 'İYİ',
                '60-69': 'ORTA',
                '40-59': 'YETERSIZ',
                '0-39': 'KÖTÜ'
            }
        },
        'supported_features': [
            'PDF text extraction',
            'OCR for images',
            'Visual component detection',
            'Project information extraction',
            'Valve and cylinder counting',
            'Multi-language support (Turkish/English)',
            'Detailed scoring and recommendations'
        ],
        'api_endpoints': {
            'POST /api/pnomatic-control': 'Pnömatik devre şeması analizi',
            'GET /api/pnomatic-health': 'Sistem sağlık kontrolü',
            'GET /api/pnomatic-validate': 'Dosya geçerlilik kontrolü',
            'GET /api/pnomatic-info': 'Analiz bilgileri'
        }
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large',
        'message': 'Dosya boyutu çok büyük (maksimum 50MB)',
        'max_size': '50MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'API endpoint bulunamadı',
        'available_endpoints': [
            'POST /api/pnomatic-control',
            'GET /api/pnomatic-health',
            'GET /api/pnomatic-validate',
            'GET /api/pnomatic-info'
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        'error': 'Method not allowed',
        'message': 'Bu endpoint için izin verilmeyen HTTP metodu'
    }), 405

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': 'Sunucu iç hatası'
    }), 500

if __name__ == '__main__':
    logger.info("🔧 Pnömatik Devre Şeması Analiz API başlatılıyor...")
    logger.info(f"📁 Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"📊 Tesseract OCR durumu: {'Mevcut' if tesseract_available else 'Mevcut değil'}")
    logger.info(f"📋 Desteklenen formatlar: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Get port from environment variable (set by main_api_gateway.py)
    assigned_port = int(os.environ.get('ASSIGNED_PORT', 5011))
    logger.info(f"🚀 Server {assigned_port} portunda başlatılıyor...")
    
    app.run(
        host='0.0.0.0',
        port=assigned_port,
        debug=False,
        threaded=True
    )