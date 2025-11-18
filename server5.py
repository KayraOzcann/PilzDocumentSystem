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
from manuel_report_checker import ManualReportAnalyzer
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
    """Server kodunda doküman validasyonu - Manuel/Kullanım Kılavuzu için"""
    
    # Manuel/Kullanım Kılavuzu raporlarında MUTLAKA olması gereken kritik kelimeler
    critical_terms = [
        # Manuel/kılavuz temel terimleri
        ["kullanma", "kılavuz", "manual", "instruction", "talimat", "guide", "kılavuzu", "user manual"],
        
        # Güvenlik ve uyarı terimleri
        ["güvenlik", "safety", "uyarı", "warning", "dikkat", "attention", "tehlike", "danger"],
        
        # Kurulum ve montaj terimleri
        ["kurulum", "installation", "montaj", "assembly", "setup", "kurma", "takma", "yerleştirme"],
        
        # Kullanım ve işletim terimleri
        ["kullanım", "operation", "işletim", "çalıştırma", "kullanma", "nasıl kullanılır", "how to use"],
        
        # Bakım ve arıza terimleri
        ["bakım", "maintenance", "temizlik", "cleaning", "arıza", "troubleshooting", "onarım", "repair"]
    ]
    
    # Her kategori için kontrol
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"Manuel Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    # Tüm kategorilerden en az bir terim bulunmalı
    valid_categories = sum(category_found)
    
    logger.info(f"Doküman validasyonu: {valid_categories}/5 kritik kategori bulundu")
    
    # 5 kategorinin en az 4'ünde terim bulunmalı
    return valid_categories >= 4

def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - Manuel için"""
    strong_keywords = [
        "kullanma",
        "kılavuz",
        "manual",
        "instruction",
        "talimat",
        "guide",
        "kılavuzu"
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
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # AT Uygunluk Beyanı
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        
        # LVD raporu
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        
        # LOTO raporu (eski strong_keywords LOTO'dan)
        "loto",
        
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        
        # Espe raporu  
        "espe",
        
        # Gürültü ölçüm raporu
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204","topraklama", "TOPRAKLAMA DİRENCİ",
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300 , first_page=1, last_page=1)
        
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

def format_analysis_response(report):
    """Format the analysis response like manuel_report_checker.py output"""
    lines = []
    
    # Header
    lines.append("📊 ANALİZ SONUÇLARI")
    lines.append("=" * 60)
    
    # Basic info
    lines.append(f"📅 Analiz Tarihi: {report['analiz_tarihi']}")
    lang = report['dosya_bilgisi']['detected_language'].upper()
    lines.append(f"🔍 Tespit Edilen Dil: {lang}")
    lines.append(f"📋 Toplam Puan: {report['ozet']['toplam_puan']}/100")
    lines.append(f"📈 Yüzde: %{report['ozet']['yuzde']}")
    lines.append(f"🎯 Durum: {report['ozet']['durum']}")
    lines.append(f"📄 Rapor Tipi: {report['ozet']['rapor_tipi']}")
    lines.append("")
    
    return "\n".join(lines)

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads_manuel'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/manuel-report', methods=['POST'])
def analyze_manuel_report():
    """Manuel/Kullanım Kılavuzu analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere bir kullanım kılavuzu sağlayın'
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
            logger.info(f"Manuel/Kullanım Kılavuzu kontrol ediliyor: {filename}")

            # Create analyzer instance
            analyzer = ManualReportAnalyzer()
            
            # ÜÇ AŞAMALI MANUEL KONTROLÜ
            logger.info(f"Üç aşamalı manuel kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                # PDF için üç aşamalı kontrol (OCR dahil)
                logger.info("Aşama 1: İlk sayfa manuel özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath):
                    logger.info("✅ Aşama 1 geçti - Manuel özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - Manuel değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya kullanım kılavuzu değil (farklı rapor türü tespit edildi). Lütfen kullanım kılavuzu yükleyiniz.',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'KULLANIM_KILAVUZU'
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
                                    'message': 'Yüklediğiniz dosya kullanım kılavuzu değil! Lütfen geçerli bir kullanım kılavuzu yükleyiniz.',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_MANUAL',
                                        'required_type': 'KULLANIM_KILAVUZU'
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

            # Buraya kadar geldiyse manuel/kılavuz, şimdi analizi yap
            logger.info(f"Manuel/kullanım kılavuzu doğrulandı, analiz başlatılıyor: {filename}")
            report = analyzer.analyze_manual_report(filepath)
            
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

            # Extract key results for API response - MANUEL FORMATINDA
            overall_percentage = report.get('ozet', {}).get('yuzde', 0)
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': report.get('analiz_tarihi', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'analysis_details': {
                    'found_criteria': len([c for c in report.get('puanlama', {}).get('category_scores', {}).values() if isinstance(c, dict) and c.get('score', 0) > 0]),
                    'total_criteria': len(report.get('puanlama', {}).get('category_scores', {})),
                    'percentage': round(overall_percentage, 1)
                },
                'analysis_id': f"manuel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': {
                    'is_valid': True,  # Manuel raporlar için tarih geçerliliği yok
                    'message': 'Manuel raporlar için tarih geçerliliği kontrolü uygulanmaz',
                    'days_old': 0,
                    'formatted_date': 'N/A'
                },
                'extracted_values': report.get('cikarilan_degerler', {}),
                'file_type': 'KULLANIM_KILAVUZU',
                'filename': filename,
                'language_info': {
                    'detected_language': report.get('dosya_bilgisi', {}).get('detected_language', 'turkish').upper(),
                    'file_type': file_ext.replace('.', '')
                },
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': report.get('ozet', {}).get('toplam_puan', 0),
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'text_quality': 'good' if len(report.get('cikarilan_degerler', {})) > 3 else 'fair'
                },
                'recommendations': report.get('oneriler', []),
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_manuel(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_manuel(report)
                }
            }
            
            # Add category scores - MANUEL FORMATINDA
            if 'puanlama' in report and 'category_scores' in report['puanlama']:
                for category, score_data in report['puanlama']['category_scores'].items():
                    if isinstance(score_data, dict):
                        response_data['category_scores'][category] = {
                            'score': score_data.get('normalized', score_data.get('score', 0)),
                            'max_score': score_data.get('max_weight', score_data.get('max_score', 0)),
                            'percentage': score_data.get('percentage', 0),
                            'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'CONDITIONAL' if score_data.get('percentage', 0) >= 50 else 'FAIL'
                        }

            return jsonify({
                'analysis_service': 'manuel_report',
                'data': response_data,
                'message': 'Manuel/Kullanım Kılavuzu başarıyla analiz edildi',
                'service_description': 'Manuel/Kullanım Kılavuzu Analizi',
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
                'message': f'Manuel/kullanım kılavuzu analizi sırasında hata oluştu: {str(analysis_error)}',
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

def get_conclusion_message_manuel(status, percentage):
    """Sonuç mesajını döndür - Manuel için"""
    if status == "PASS":
        return f"Kullanım kılavuzu yeterli kriterleri sağlamakta ve kullanıcı güvenliği açısından uygundur (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"Kullanım kılavuzu kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"Kullanım kılavuzu yetersiz kriterlere sahip, kapsamlı iyileştirme gerekli (%{percentage:.0f})"

def get_main_issues_manuel(report):
    """Ana sorunları listele - Manuel için"""
    issues = []
    
    if 'puanlama' in report and 'category_scores' in report['puanlama']:
        for category, score_data in report['puanlama']['category_scores'].items():
            if isinstance(score_data, dict) and score_data.get('percentage', 0) < 50:
                issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    # Eğer hiç kritik sorun yoksa genel sorunları ekle
    if not issues:
        if report.get('ozet', {}).get('toplam_puan', 0) < 50:
            issues = [
                "Güvenlik bilgileri ve uyarıları eksik",
                "Kurulum ve montaj talimatları yetersiz",
                "Kullanım adımları net şekilde açıklanmamış",
                "Bakım ve temizlik bilgileri eksik",
                "Arıza giderme bölümü yetersiz"
            ]
    
    return issues[:4]  # En fazla 4 ana sorun göster

@app.route('/api/manuel-health', methods=['GET'])
def health_check():
    """Health check endpoint - Manuel için"""
    return jsonify({
        'status': 'healthy',
        'service': 'Manual Report Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'tesseract_info': tesseract_info,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': tesseract_available,
        'report_type': 'KULLANIM_KILAVUZU'
    })

@app.route('/api/manuel-info', methods=['GET'])
def service_info():
    return jsonify({
        'service': 'Manual Report Analysis API',
        'version': '1.0.0',
        'description': 'API for analyzing User Manual (Kullanma Kılavuzu) reports',
        'endpoints': {
            '/api/manuel-report': 'POST - Upload and analyze Manual report PDF',
            '/api/manuel-health': 'GET - Health check',
            '/api/manuel-info': 'GET - Service information'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '32MB',
        'scoring_categories': [
            'Genel Bilgiler (10 points)',
            'Giriş ve Amaç (5 points)',
            'Güvenlik Bilgileri (15 points)',
            'Ürün Tanıtımı (10 points)',
            'Kurulum ve Montaj Bilgileri (15 points)',
            'Kullanım Talimatları (20 points)',
            'Bakım ve Temizlik (10 points)',
            'Arıza Giderme (15 points)'
        ],
        'total_points': 100,
        'scoring': {
            'PASS': '≥70% - Yeterli kullanım kılavuzu',
            'CONDITIONAL': '50-69% - Kabul edilebilir',
            'FAIL': '<50% - Yetersiz kılavuz'
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
    
    logger.info("Manual Report Analysis Server başlatılıyor...")
    logger.info(f"Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"Tesseract OCR durumu: {'Kurulu' if tesseract_available else 'Kurulu değil'}")
    logger.info("API endpoint'leri:")
    logger.info("  POST /api/manuel-report - Manuel/kullanım kılavuzu analizi")
    logger.info("  GET  /api/manuel-health - Sağlık kontrolü")
    logger.info("  GET  /api/manuel-info  - Servis bilgileri")
    
    assigned_port = int(os.environ.get('ASSIGNED_PORT', 5006))
    app.run(host='0.0.0.0', port=assigned_port, debug=False)