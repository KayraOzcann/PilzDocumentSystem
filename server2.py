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
from espe_report_checker import ESPEReportAnalyzer
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
    """Server kodunda doküman validasyonu - ESPE için"""
    
    # ESPE raporlarında MUTLAKA olması gereken kritik kelimeler
    critical_terms = [
        # ESPE temel terimleri
        ["espe", "electro-sensitive", "koruyucu", "protective", "equipment", "ekipman"],
        
        # Güvenlik sistemi terimleri
        ["güvenlik", "safety", "sistem", "system", "emniyet", "security", "korunum", "protection"],
        
        # Elektrik ve sensör terimleri
        ["elektrik", "electrical", "sensör", "sensor", "dedektör", "detector", "monitoring", "izleme"],
        
        # Risk ve analiz terimleri
        ["risk", "analiz", "analysis", "değerlendirme", "assessment", "kontrol", "control"],
        
        # Standart ve test terimleri
        ["standart", "standard", "test", "muayene", "inspection", "performans", "performance"]
    ]
    
    # Her kategori için kontrol
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"ESPE Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    # Tüm kategorilerden en az bir terim bulunmalı
    valid_categories = sum(category_found)
    
    logger.info(f"Doküman validasyonu: {valid_categories}/5 kritik kategori bulundu")
    
    # 5 kategorinin en az 4'ünde terim bulunmalı
    return valid_categories >= 4

def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - ESPE için"""
    strong_keywords = [
        "espe"
    ]
    
    try:
        # PIL güvenlik sınırını artır
        Image.MAX_IMAGE_PIXELS = None
        
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        
        all_text = ""
        for i, page in enumerate(pages):
            # Büyük görüntüleri küçült
            if page.size[0] > 2000 or page.size[1] > 2000:
                page = page.resize((1500, int(1500 * page.size[1] / page.size[0])), Image.Resampling.LANCZOS)
                
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
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # AT Uygunluk Beyanı
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        
        # LVD raporu
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        
        # LOTO raporu
        "loto",
        
        # Manuel/kullanım kılavuzu
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide", "kılavuzu",
        
        # Gürültü ölçüm raporu (eski strong_keywords gürültüden)
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama"
    ]
    
    try:
        # PIL güvenlik sınırını artır
        Image.MAX_IMAGE_PIXELS = None
        
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=2)
        
        all_text = ""
        for i, page in enumerate(pages):
            # Büyük görüntüleri küçült
            if page.size[0] > 2000 or page.size[1] > 2000:
                page = page.resize((1500, int(1500 * page.size[1] / page.size[0])), Image.Resampling.LANCZOS)
                
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
UPLOAD_FOLDER = 'temp_uploads_espe'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/espe-report', methods=['POST'])
def analyze_espe_report():
    """ESPE Raporu analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere bir ESPE raporu sağlayın'
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
            logger.info(f"ESPE Raporu kontrol ediliyor: {filename}")

            # Create analyzer instance
            analyzer = ESPEReportAnalyzer()
            
            # ÜÇ AŞAMALI ESPE KONTROLÜ
            logger.info(f"Üç aşamalı ESPE kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                # PDF için üç aşamalı kontrol (OCR dahil)
                logger.info("Aşama 1: İlk sayfa ESPE özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath):
                    logger.info("✅ Aşama 1 geçti - ESPE özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - ESPE değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya ESPE raporu değil (farklı rapor türü tespit edildi). Lütfen ESPE raporu yükleyiniz.',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'ESPE_RAPORU'
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
                                    'message': 'Yüklediğiniz dosya ESPE raporu değil! Lütfen geçerli bir ESPE raporu yükleyiniz.'
                                }), 400
                            
                            if not validate_document_server(text):
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                                return jsonify({
                                    'error': 'Invalid document type',
                                    'message': 'Yüklediğiniz dosya ESPE raporu değil! Lütfen geçerli bir ESPE raporu yükleyiniz.',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_ESPE_REPORT',
                                        'required_type': 'ESPE_RAPORU'
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

            # Buraya kadar geldiyse ESPE raporu, şimdi analizi yap
            logger.info(f"ESPE raporu doğrulandı, analiz başlatılıyor: {filename}")
            docx_path = None  # ESPE için DOCX gerekmeyebilir
            report = analyzer.generate_detailed_report(filepath, docx_path)
            
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

            # Extract key results for API response - ESPE FORMATINDA
            # Get overall score correctly from nested structure
            overall_score = 0
            if report.get('puanlama'):
                overall_score = report['puanlama'].get('total_score', 0)
            elif report.get('ozet'):
                overall_score = report['ozet'].get('toplam_puan', 0)
            
            status = "PASS" if overall_score >= 70 else "FAIL"
            
            # Determine failure reason
            failure_reason = None
            if status != "PASS":
                tarih_gecerliligi = report.get('tarih_gecerliligi', {})
                if tarih_gecerliligi.get('durum') == 'GEÇERSİZ':
                    failure_reason = tarih_gecerliligi.get('mesaj', 'Tarih geçerliliği sorunu')
                else:
                    failure_reason = "Yetersiz puan"
            
            response_data = {
                'analysis_date': report.get('analiz_tarihi', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'analysis_details': {
                    'found_criteria': len([c for c in report.get('puanlama', {}).get('category_scores', {}).values() if isinstance(c, dict) and c.get('earned', 0) > 0]),
                    'total_criteria': len(report.get('puanlama', {}).get('category_scores', {})),
                    'percentage': round(overall_score, 1)
                },
                'analysis_id': f"espe_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': report.get('tarih_gecerliligi', {}),
                'extracted_values': report.get('cikarilan_degerler', {}),
                'file_type': 'ESPE_RAPORU',
                'filename': filename,
                'language_info': {
                    'detected_language': 'TURKISH',
                    'file_type': file_ext.replace('.', '')
                },
                'overall_score': {
                    'percentage': round(overall_score, 2),
                    'total_points': overall_score,
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'text_quality': 'good' if len(report.get('cikarilan_degerler', {})) > 3 else 'fair'
                },
                'recommendations': report.get('oneriler', []),
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_espe(status, overall_score),
                    'main_issues': [] if status == "PASS" else get_main_issues_espe(report)
                }
            }
            
            # Add category scores - ESPE FORMATINDA
            if report.get('puanlama') and report['puanlama'].get('category_scores'):
                for category, score_data in report['puanlama']['category_scores'].items():
                    if isinstance(score_data, dict):
                        response_data['category_scores'][category] = {
                            'percentage': score_data.get('percentage', 0),
                            'points_earned': score_data.get('earned', 0),
                            'max_points': score_data.get('max_weight', 0),
                            'normalized_score': score_data.get('normalized', 0),
                            'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'CONDITIONAL' if score_data.get('percentage', 0) >= 50 else 'FAIL'
                        }
            
            return jsonify({
                'analysis_service': 'espe',
                'data': response_data,
                'message': 'ESPE Raporu başarıyla analiz edildi',
                'service_description': 'ESPE Rapor Analizi',
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
                'message': f'ESPE raporu analizi sırasında hata oluştu: {str(analysis_error)}',
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

def get_conclusion_message_espe(status, percentage):
    """Sonuç mesajını döndür - ESPE için"""
    if status == "PASS":
        return f"ESPE raporu elektro-hassas koruyucu ekipman standartlarına uygun ve yeterli kriterleri sağlamaktadır (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"ESPE raporu kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"ESPE raporu standartlara uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"

def get_main_issues_espe(report):
    """Ana sorunları listele - ESPE için"""
    issues = []
    
    if report.get('puanlama') and report['puanlama'].get('category_scores'):
        for category, score_data in report['puanlama']['category_scores'].items():
            if isinstance(score_data, dict) and score_data.get('percentage', 0) < 50:
                issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    # Eğer hiç kritik sorun yoksa genel sorunları ekle
    if not issues:
        overall_score = 0
        if report.get('puanlama'):
            overall_score = report['puanlama'].get('total_score', 0)
        elif report.get('ozet'):
            overall_score = report['ozet'].get('toplam_puan', 0)
            
        if overall_score < 50:
            issues = [
                "ESPE cihazının teknik spesifikasyonları eksik",
                "Güvenlik sistemi performans testleri yetersiz",
                "Risk analizi ve değerlendirmesi eksik",
                "Standart uygunluk belgelendirmesi yetersiz",
                "Bakım ve kalibrasyon prosedürleri eksik"
            ]
    
    return issues[:4]  # En fazla 4 ana sorun göster

@app.route('/api/espe-health', methods=['GET'])
def health_check():
    """Health check endpoint - ESPE için"""
    return jsonify({
        'status': 'healthy',
        'service': 'ESPE Report Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'tesseract_info': tesseract_info,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': tesseract_available,
        'report_type': 'ESPE_RAPORU'
    })

@app.route('/api/espe-info', methods=['GET'])
def service_info():
    return jsonify({
        'service': 'ESPE Report Analysis API',
        'version': '1.0.0',
        'description': 'API for analyzing ESPE (Electro-Sensitive Protective Equipment) reports',
        'endpoints': {
            '/api/espe-report': 'POST - Upload and analyze ESPE report',
            '/api/espe-health': 'GET - Health check',
            '/api/espe-info': 'GET - Service information'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '32MB',
        'scoring': {
            'PASS': '≥70% - ESPE standartlarına uygun',
            'CONDITIONAL': '50-69% - Kabul edilebilir',
            'FAIL': '<50% - Standartlara uygun değil'
        }
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large',
        'message': 'Dosya boyutu 32MB limitini aşıyor'
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
    
    logger.info("ESPE Report Analysis Server başlatılıyor...")
    logger.info(f"Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"Tesseract OCR durumu: {'Kurulu' if tesseract_available else 'Kurulu değil'}")
    logger.info("API endpoint'leri:")
    logger.info("  POST /api/espe-report - ESPE raporu analizi")
    logger.info("  GET  /api/espe-health - Sağlık kontrolü")
    logger.info("  GET  /api/espe-info   - Servis bilgileri")
    
    assigned_port = int(os.environ.get('ASSIGNED_PORT', 5003))
    app.run(host='0.0.0.0', port=assigned_port, debug=False)