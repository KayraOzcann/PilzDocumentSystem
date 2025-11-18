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
from isg_periyodik_kontrol_checker import ISGPeriyodikKontrolAnalyzer
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
    """Server kodunda doküman validasyonu - İSG Periyodik Kontrol için"""
    
    # İSG Periyodik Kontrol raporlarında MUTLAKA olması gereken kritik kelimeler
    critical_terms = [
        # İSG temel terimleri
        ["isg", "iş sağlığı", "güvenlik", "periyodik", "kontrol", "periodic", "inspection", "denetim"],
        
        # Ölçüm türleri ve parametreler
        ["gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic", "aydınlatma", "lux"],
        
        # Laboratuvar ve rapor bilgileri
        ["laboratuvar", "laboratory", "ölçüm", "measurement", "rapor", "report", "analiz", "analysis"],
        
        # Yasal ve standart referanslar
        ["yönetmelik", "regulation", "standart", "standard", "limit", "sınır", "değer", "value"],
        
        # Çevre ve iş hijyeni terimleri
        ["çevre", "environment", "iş hijyeni", "occupational hygiene", "sağlık", "health", "risk", "assessment"]
    ]
    
    # Her kategori için kontrol
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"İSG Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    # Tüm kategorilerden en az bir terim bulunmalı
    valid_categories = sum(category_found)
    
    logger.info(f"Doküman validasyonu: {valid_categories}/5 kritik kategori bulundu")
    
    # 5 kategorinin en az 4'ünde terim bulunmalı
    return valid_categories >= 4

def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - İSG Periyodik Kontrol için"""
    strong_keywords = [
        "isg",
        "periyodik",
        "periodic",
        "inspection",
        "denetim",
        "çevre laboratuvarı",
        "iş hi̇jyeni̇ olçum, test ve analiz", "iş hi̇jyeni̇ ölçüm, test ve analiz",
        "is güvenliği", "iş güvenligi",
        "tetratest",
        "turkak", "türkak", 
        "akredite", "akredıte",
        "accreditation agency",
        "yeterlilik bölge",
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
        # Aydınlatma raporu
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
        
        # Hidrolik devre şeması
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219","teknik resim","tasarım",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve",
        
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
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type",
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
UPLOAD_FOLDER = 'temp_uploads_isg'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/isg-control', methods=['POST'])
def analyze_isg_control():
    """İSG Periyodik Kontrol analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere bir İSG periyodik kontrol raporu sağlayın'
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
            logger.info(f"İSG Periyodik Kontrol raporu kontrol ediliyor: {filename}")

            # Create analyzer instance
            analyzer = ISGPeriyodikKontrolAnalyzer()
            
            # ÜÇ AŞAMALI İSG KONTROLÜ
            logger.info(f"Üç aşamalı İSG kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                # PDF için üç aşamalı kontrol (OCR dahil)
                logger.info("Aşama 1: İlk sayfa İSG özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath):
                    logger.info("✅ Aşama 1 geçti - İSG özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - İSG değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya İSG periyodik kontrol raporu değil (farklı rapor türü tespit edildi). Lütfen İSG periyodik kontrol raporu yükleyiniz.',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'ISG_PERIYODIK_KONTROL_RAPORU'
                            }
                        }), 400
                    else:
                        # AŞAMA 3: PyPDF2 ile tam doküman kontrolü
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
                            
                            if not validate_document_server(text):
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                                return jsonify({
                                    'error': 'Invalid document type',
                                    'message': 'Yüklediğiniz dosya İSG periyodik kontrol raporu değil! Lütfen geçerli bir İSG periyodik kontrol raporu yükleyiniz.',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_ISG_REPORT',
                                        'required_type': 'ISG_PERIYODIK_KONTROL_RAPORU'
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

            # Buraya kadar geldiyse İSG raporu, şimdi analizi yap
            logger.info(f"İSG periyodik kontrol raporu doğrulandı, analiz başlatılıyor: {filename}")
            report = analyzer.analyze_isg_kontrol(filepath)
            
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
                        'filename': filename,
                        'analysis_details': report.get('details', {})
                    }
                }), 400

            # Extract key results for API response - İSG FORMATINDA
            overall_percentage = report['summary']['percentage']
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': report.get('analysis_date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'analysis_details': {
                    'found_criteria': len([c for c in report['scoring']['category_scores'].values() if isinstance(c, dict) and c.get('score', 0) > 0]),
                    'total_criteria': len(report['scoring']['category_scores']),
                    'percentage': round(overall_percentage, 1)
                },
                'analysis_id': f"isg_kontrol_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': {
                    'is_valid': report['date_validity']['is_valid'],
                    'message': report['date_validity']['message'],
                    'days_old': report['date_validity']['days_old'],
                    'formatted_date': report['date_validity']['formatted_date']
                },
                'extracted_values': {
                    'firma_adi': report['extracted_values']['firma_adi'],
                    'olcum_tarihi': report['extracted_values']['olcum_tarihi'],
                    'rapor_numarasi': report['extracted_values']['rapor_numarasi'],
                    'laboratuvar': report['extracted_values']['laboratuvar'],
                    'adres': report['extracted_values']['adres'],
                    'gurultu_seviye': report['extracted_values']['gurultu_seviye'],
                    'aydinlatma_seviye': report['extracted_values']['aydinlatma_seviye'],
                    'genel_degerlendirme': report['extracted_values']['genel_degerlendirme']
                },
                'file_type': 'ISG_PERIYODIK_KONTROL_RAPORU',
                'filename': filename,
                'language_info': {
                    'detected_language': report['file_info']['detected_language'].upper(),
                    'file_type': file_ext.replace('.', '')
                },
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': report['summary']['total_score'],
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'report_type': report['summary']['report_type']
                },
                'recommendations': report['recommendations'],
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_isg(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_isg(report)
                }
            }
            
            # Add category scores - İSG FORMATINDA
            for category, score_data in report['scoring']['category_scores'].items():
                if isinstance(score_data, dict):
                    response_data['category_scores'][category] = {
                        'score': score_data.get('score', 0),
                        'max_score': score_data.get('max_score', score_data.get('weight', 0)),
                        'percentage': score_data.get('percentage', 0),
                        'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'CONDITIONAL' if score_data.get('percentage', 0) >= 50 else 'FAIL'
                    }

            return jsonify({
                'analysis_service': 'isg_periyodik_kontrol',
                'data': response_data,
                'message': 'İSG Periyodik Kontrol Raporu başarıyla analiz edildi',
                'service_description': 'İSG Periyodik Kontrol Analizi',
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
                'message': f'İSG periyodik kontrol raporu analizi sırasında hata oluştu: {str(analysis_error)}',
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

def get_conclusion_message_isg(status, percentage):
    """Sonuç mesajını döndür - İSG için"""
    if status == "PASS":
        return f"İSG periyodik kontrol raporu İş Sağlığı ve Güvenliği Yönetmeliği gereksinimlerine uygundur (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"İSG raporu kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"İSG raporu yönetmelik gereksinimlerine uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"

def get_main_issues_isg(report):
    """Ana sorunları listele - İSG için"""
    issues = []
    
    for category, score_data in report['scoring']['category_scores'].items():
        if isinstance(score_data, dict) and score_data.get('percentage', 0) < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    # Eğer hiç kritik sorun yoksa genel sorunları ekle
    if not issues:
        if report['summary']['total_score'] < 50:
            issues = [
                "Gürültü ölçüm sonuçları eksik veya yetersiz",
                "Aydınlatma seviyesi kontrolü yapılmamış",
                "Laboratuvar akreditasyon bilgileri eksik",
                "Ölçüm tarihi geçerlilik süresi aşmış",
                "Yasal limit değerlerle karşılaştırma eksik"
            ]
    
    return issues[:4]  # En fazla 4 ana sorun göster

@app.route('/api/isg-health', methods=['GET'])
def health_check():
    """Health check endpoint - İSG için"""
    return jsonify({
        'status': 'healthy',
        'service': 'ISG Periodic Control Report Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'tesseract_info': tesseract_info,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': tesseract_available,
        'report_type': 'ISG_PERIYODIK_KONTROL_RAPORU',
        'regulations': 'İş Sağlığı ve Güvenliği Yönetmeliği'
    })

@app.route('/api/isg-validate', methods=['POST'])
def validate_isg_report():
    """İSG Periyodik Kontrol raporu hızlı geçerlilik kontrolü"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Dosya sağlanmadı'
            }), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Dosya seçilmedi'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Sadece PDF, JPG, JPEG ve PNG dosyaları kabul edilir'
            }), 400

        # Check OCR requirement for image files
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension in ['.jpg', '.jpeg', '.png'] and not tesseract_available:
            return jsonify({
                'valid': False,
                'error': 'OCR not available for image files',
                'message': 'Resim dosyalarını analiz edebilmek için Tesseract OCR kurulumu gereklidir'
            }), 400

        # Temporary file processing
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            analyzer = ISGPeriyodikKontrolAnalyzer()
            report = analyzer.analyze_isg_kontrol(filepath)
            
            # Clean up
            os.remove(filepath)
            
            if "error" in report:
                return jsonify({
                    'valid': False,
                    'error': report['error']
                }), 500

            # Quick validation response
            date_valid = report['date_validity']['is_valid']
            score_valid = report['summary']['percentage'] >= 50
            is_valid = date_valid and score_valid
            
            return jsonify({
                'valid': is_valid,
                'status': report['summary']['status'],
                'score': round(report['summary']['percentage'], 1),
                'date_valid': date_valid,
                'date_message': report['date_validity']['message'],
                'quick_assessment': {
                    'firma_found': report['extracted_values']['firma_adi'] != 'Bulunamadı',
                    'measurement_date_found': report['extracted_values']['olcum_tarihi'] != 'Bulunamadı',
                    'report_number_found': report['extracted_values']['rapor_numarasi'] != 'Bulunamadı',
                    'laboratory_found': report['extracted_values']['laboratuvar'] != 'Bulunamadı',
                    'date_within_validity': date_valid
                }
            }), 200

        except Exception as e:
            # Clean up in case of error
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
            raise e

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'valid': False,
            'error': f'Doğrulama hatası: {str(e)}'
        }), 500

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

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    logger.info("İSG Periyodik Kontrol Analiz API başlatılıyor...")
    logger.info(f"Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"Tesseract OCR durumu: {'Kurulu' if tesseract_available else 'Kurulu değil'}")
    logger.info("API endpoint'leri:")
    logger.info("  POST /api/isg-control  - İSG Periyodik Kontrol analizi")
    logger.info("  POST /api/isg-validate - Hızlı geçerlilik kontrolü") 
    logger.info("  GET  /api/isg-health   - Sağlık kontrolü")
    
    assigned_port = int(os.environ.get('ASSIGNED_PORT', 5010))
    app.run(host='0.0.0.0', port=assigned_port, debug=False)