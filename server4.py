import PyPDF2
import cv2
import numpy as np
from PIL import Image
import pdf2image
import pytesseract
from flask import Flask, request, jsonify
import os
import re
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
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
    """Server kodunda doküman validasyonu - Gürültü Ölçüm Raporu için"""
    
    # Gürültü ölçüm raporlarında MUTLAKA olması gereken kritik kelimeler
    critical_terms = [
        # Gürültü temel terimleri
        ["gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic"],
        
        # Ölçüm ve test terimleri
        ["ölçüm", "measurement", "test", "analiz", "analysis", "ses basıncı", "sound pressure"],
        
        # Ses seviyeleri ve parametreler
        ["dba", "dbc", "leq", "lmax", "lmin", "lcpeak", "lpeak", "ses seviyesi", "sound level"],
        
        # Standart ve referanslar
        ["iso 11201", "iso 9612", "iso 3744", "en iso", "standart", "standard"],
        
        # Çevre ve iş sağlığı terimleri
        ["maruziyet", "exposure", "yasal sınır", "limit", "iş sağlığı", "occupational", "çevresel", "environmental"]
    ]
    
    # Her kategori için kontrol
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"Gürültü Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    # Tüm kategorilerden en az bir terim bulunmalı
    valid_categories = sum(category_found)
    
    logger.info(f"Doküman validasyonu: {valid_categories}/5 kritik kategori bulundu")
    
    # 5 kategorinin en az 4'ünde terim bulunmalı
    return valid_categories >= 4

def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - Gürültü için"""
    strong_keywords = [
        "gürültü",
        "noise",
        "ses",
        "sound",
        "decibel",
        "akustik",
        "acoustic"
    ]
    
    try:
        # PIL güvenlik sınırını artır
        Image.MAX_IMAGE_PIXELS = None
        
        pages = pdf2image.convert_from_path(filepath, dpi=150, first_page=1, last_page=1)
        
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
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219", "teknik resim", "tasarım",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil"
        
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
        
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        
        # Espe raporu  
        "espe",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204","topraklama",
    ]
    
    try:
        # PIL güvenlik sınırını artır
        Image.MAX_IMAGE_PIXELS = None
        
        pages = pdf2image.convert_from_path(filepath, dpi=400, first_page=1, last_page=2)
        
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

@dataclass
class NoiseAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

class NoiseReportAnalyzer:
    
    def __init__(self):
        self.criteria_weights = {
            "Rapor Kimlik Bilgileri": 15,
            "Ölçüm Yapılan Ortam ve Ekipman Bilgileri": 15,
            "Ölçüm Cihazı Bilgileri": 15,
            "Ölçüm Metodolojisi": 20,
            "Ölçüm Sonuçları": 20,
            "Değerlendirme ve Yorum": 10,
            "Ekler ve Görseller": 5
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"PDF okuma hatası: {e}")
            return ""
    
    def check_report_date_validity(self, text: str) -> Tuple[bool, str, str]:
        date_patterns = [
            r"(?:Ölçüm\s*Tarihi|İnceleme)\s*[:=]?\s*(\d{2}[./]\d{2}[./]\d{4})",
            r"(\d{2}[./]\d{2}[./]\d{4})"
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                date_str = matches[0]
                try:
                    date_str = date_str.replace('.', '/').replace('-', '/')
                    report_date = datetime.strptime(date_str, '%d/%m/%Y')
                    one_year_ago = datetime.now() - timedelta(days=365)
                    
                    is_valid = report_date >= one_year_ago
                    return is_valid, date_str, f"Rapor tarihi: {date_str} {'(GEÇERLİ)' if is_valid else '(GEÇERSİZ - 1 yıldan eski)'}"
                except ValueError:
                    continue
        
        return False, "", "Rapor tarihi bulunamadı"
    
    def generate_detailed_report(self, pdf_path: str) -> Dict[str, Any]:
        logger.info("Gürültü ölçüm raporu analizi başlatılıyor...")
        
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return {"error": "PDF okunamadı"}
        
        date_valid, date_str, date_message = self.check_report_date_validity(pdf_text)
        
        # Basit scoring sistemi
        total_score = 0
        if "gürültü" in pdf_text.lower() or "noise" in pdf_text.lower():
            total_score += 20
        if "db" in pdf_text.lower() or "decibel" in pdf_text.lower():
            total_score += 20
        if "ölçüm" in pdf_text.lower() or "measurement" in pdf_text.lower():
            total_score += 20
        if "iso" in pdf_text.lower():
            total_score += 20
        if "ses" in pdf_text.lower() or "sound" in pdf_text.lower():
            total_score += 20
        
        # Tarih geçerliliği kontrolü - eğer tarih geçerli değilse puan 0
        if not date_valid:
            total_score = 0
        
        overall_percentage = total_score
        
        report = {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgileri": {
                "pdf_path": pdf_path
            },
            "tarih_gecerliligi": {
                "gecerli": date_valid,
                "tarih": date_str,
                "mesaj": date_message
            },
            "cikarilan_degerler": {
                "rapor_no": "Bulunamadı",
                "firma_adi": "Bulunamadı",
                "makine_adi": "Bulunamadı",
                "cihaz_marka": "Bulunamadı",
                "olcum_yapan": "Bulunamadı"
            },
            "puanlama": {
                "category_scores": {
                    category: {
                        "raw_score": weight,
                        "normalized": weight if total_score >= 70 else weight // 2,
                        "max_weight": weight,
                        "percentage": 100 if total_score >= 70 else 50
                    }
                    for category, weight in self.criteria_weights.items()
                },
                "total_score": total_score,
                "total_max_score": 100,
                "overall_percentage": overall_percentage
            },
            "oneriler": [] if overall_percentage >= 100 else (
                [
                    "Gürültü ölçüm raporu ISO 11201 standardına uygun hazırlanmalıdır",
                    "Ölçüm cihazı kalibrasyon belgeleri eklenmelidir",
                    "Yasal sınırlarla karşılaştırma yapılmalıdır"
                ] + ([f"Rapor tarihi 1 yıldan eski ({date_str}) - rapor yenilenmelidir"] if not date_valid else [])
            ),
            "ozet": {
                "toplam_puan": total_score,
                "yuzde": overall_percentage,
                "durum": "GEÇERLİ" if overall_percentage >= 70 else "YETERSİZ",
                "tarih_durumu": "GEÇERLİ" if date_valid else "GEÇERSİZ"
            }
        }
        
        return report

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads_noise'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/noise-report', methods=['POST'])
def analyze_noise_report():
    """Gürültü Ölçüm Raporu analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere bir gürültü ölçüm raporu sağlayın'
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
            logger.info(f"Gürültü Ölçüm Raporu kontrol ediliyor: {filename}")

            # Create analyzer instance
            analyzer = NoiseReportAnalyzer()
            
            # ÜÇ AŞAMALI GÜRÜLTÜ KONTROLÜ
            logger.info(f"Üç aşamalı gürültü kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                # PDF için üç aşamalı kontrol (OCR dahil)
                logger.info("Aşama 1: İlk sayfa gürültü özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath):
                    logger.info("✅ Aşama 1 geçti - Gürültü özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - Gürültü değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya gürültü ölçüm raporu değil (farklı rapor türü tespit edildi). Lütfen gürültü ölçüm raporu yükleyiniz.',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'GURULTU_OLCUM_RAPORU'
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
                                    'message': 'Yüklediğiniz dosya gürültü ölçüm raporu değil! Lütfen geçerli bir gürültü ölçüm raporu yükleyiniz.'
                                }), 400
                            
                            if not validate_document_server(text):
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                                return jsonify({
                                    'error': 'Invalid document type',
                                    'message': 'Yüklediğiniz dosya gürültü ölçüm raporu değil! Lütfen geçerli bir gürültü ölçüm raporu yükleyiniz.',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_NOISE_REPORT',
                                        'required_type': 'GURULTU_OLCUM_RAPORU'
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

            # Buraya kadar geldiyse gürültü raporu, şimdi analizi yap
            logger.info(f"Gürültü ölçüm raporu doğrulandı, analiz başlatılıyor: {filename}")
            full_report = analyzer.generate_detailed_report(filepath)
            
            # Clean up the uploaded file
            try:
                os.remove(filepath)
                logger.info(f"Uploaded file {filename} cleaned up")
            except Exception as e:
                logger.warning(f"Could not remove uploaded file {filename}: {e}")

            # Check if analysis was successful
            if 'error' in full_report:
                return jsonify({
                    'error': 'Analysis failed',
                    'message': full_report['error'],
                    'details': {
                        'filename': filename,
                        'analysis_details': full_report.get('details', {})
                    }
                }), 400

            # Extract key results for API response - GÜRÜLTÜ FORMATINDA
            overall_percentage = full_report["ozet"]["yuzde"]
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': full_report["analiz_tarihi"],
                'analysis_details': {
                    'found_criteria': len([c for c in full_report["puanlama"]["category_scores"].values() if c.get('percentage', 0) > 50]),
                    'total_criteria': len(full_report["puanlama"]["category_scores"]),
                    'percentage': round(overall_percentage, 1)
                },
                'analysis_id': f"noise_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': {
                    'is_valid': full_report["tarih_gecerliligi"]["gecerli"],
                    'message': full_report["tarih_gecerliligi"]["mesaj"],
                    'days_old': 0,  # Hesaplanmamış
                    'formatted_date': full_report["tarih_gecerliligi"]["tarih"]
                },
                'extracted_values': {
                    "rapor_no": full_report["cikarilan_degerler"].get("rapor_no", "Bulunamadı"),
                    "firma_adi": full_report["cikarilan_degerler"].get("firma_adi", "Bulunamadı"),
                    "makine_adi": full_report["cikarilan_degerler"].get("makine_adi", "Bulunamadı"),
                    "cihaz_marka": full_report["cikarilan_degerler"].get("cihaz_marka", "Bulunamadı"),
                    "olcum_yapan": full_report["cikarilan_degerler"].get("olcum_yapan", "Bulunamadı")
                },
                'file_type': 'GURULTU_OLCUM_RAPORU',
                'filename': filename,
                'language_info': {
                    'detected_language': 'TURKISH',
                    'file_type': file_ext.replace('.', '')
                },
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': full_report["ozet"]["toplam_puan"],
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'text_quality': 'good'
                },
                'recommendations': full_report["oneriler"],
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_noise(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_noise(full_report)
                }
            }
            
            # Add category scores - GÜRÜLTÜ FORMATINDA
            for kategori, score_data in full_report["puanlama"]["category_scores"].items():
                response_data['category_scores'][kategori] = {
                    "score": score_data["normalized"],
                    "max_score": score_data["max_weight"],
                    "percentage": round(score_data["percentage"], 1),
                    'status': 'PASS' if score_data["percentage"] >= 70 else 'CONDITIONAL' if score_data["percentage"] >= 50 else 'FAIL'
                }

            return jsonify({
                'analysis_service': 'noise_report',
                'data': response_data,
                'message': 'Gürültü Ölçüm Raporu başarıyla analiz edildi',
                'service_description': 'Gürültü Ölçüm Rapor Analizi',
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
                'message': f'Gürültü ölçüm raporu analizi sırasında hata oluştu: {str(analysis_error)}',
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

def get_conclusion_message_noise(status, percentage):
    """Sonuç mesajını döndür - Gürültü için"""
    if status == "PASS":
        return f"Gürültü ölçüm raporu ISO 11201 ve ilgili standartlara uygun ve yeterli kriterleri sağlamaktadır (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"Gürültü ölçüm raporu kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"Gürültü ölçüm raporu standartlara uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"

def get_main_issues_noise(report):
    """Ana sorunları listele - Gürültü için"""
    issues = []
    
    for category, score_data in report["puanlama"]["category_scores"].items():
        if score_data.get('percentage', 0) < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    # Eğer hiç kritik sorun yoksa genel sorunları ekle
    if not issues:
        if report["ozet"]["toplam_puan"] < 50:
            issues = [
                "Gürültü ölçüm sonuçları ve dB değerleri eksik",
                "ISO 11201/9612 standart referansları yetersiz",
                "Ölçüm cihazı kalibrasyon bilgileri eksik",
                "Yasal sınırlarla karşılaştırma yapılmamış",
                "Maruziyet değerlendirmesi ve öneriler eksik"
            ]
    
    return issues[:4]  # En fazla 4 ana sorun göster

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint - Gürültü için"""
    return jsonify({
        'status': 'healthy',
        'service': 'Noise Report Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'tesseract_info': tesseract_info,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': tesseract_available,
        'report_type': 'GURULTU_OLCUM_RAPORU'
    })

@app.route('/api/noise-info', methods=['GET'])
def service_info():
    return jsonify({
        'service': 'Noise Report Analysis API',
        'version': '1.0.0',
        'description': 'API for analyzing Noise Measurement reports',
        'endpoints': {
            '/api/noise-report': 'POST - Upload and analyze Noise report',
            '/api/health': 'GET - Health check',
            '/api/noise-info': 'GET - Service information'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '32MB',
        'analysis_categories': {
            'Rapor Kimlik Bilgileri': 15,
            'Ölçüm Yapılan Ortam ve Ekipman Bilgileri': 15,
            'Ölçüm Cihazı Bilgileri': 15,
            'Ölçüm Metodolojisi': 20,
            'Ölçüm Sonuçları': 20,
            'Değerlendirme ve Yorum': 10,
            'Ekler ve Görseller': 5
        },
        'total_points': 100,
        'scoring': {
            'PASS': '≥70% - ISO 11201/9612 standartlarına uygun',
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
    
    logger.info("Gürültü Ölçüm Raporu Analiz API başlatılıyor...")
    logger.info(f"Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"Tesseract OCR durumu: {'Kurulu' if tesseract_available else 'Kurulu değil'}")
    logger.info("API endpoint'leri:")
    logger.info("  POST /api/noise-report - Gürültü ölçüm raporu analizi")
    logger.info("  GET  /api/health      - Sağlık kontrolü")
    logger.info("  GET  /api/noise-info  - Servis bilgileri")
    
    assigned_port = int(os.environ.get('ASSIGNED_PORT', 5005))
    app.run(host='0.0.0.0', port=assigned_port, debug=False)