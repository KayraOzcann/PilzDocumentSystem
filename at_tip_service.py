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
import logging
from dataclasses import dataclass
from typing import Dict, Any, List
from docx import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ATTypeCertAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

class ATTypeCertificateAnalyzer:
    def __init__(self):
        logger.info("AT Type Certificate Analyzer başlatılıyor...")
        
        self.criteria_weights = {
            "Sertifika Başlığı ve Tip": 15,
            "Notified Body Bilgileri": 15,
            "Makina/Ürün Tanımı": 15,
            "Üretici Bilgileri": 10,
            "Uygulanabilir Standartlar": 15,
            "Sertifika Numarası ve Tarihi": 10,
            "İmza ve Onay": 10,
            "Teknik Dosya Referansı": 10
        }
        
        self.criteria_details = {
            "Sertifika Başlığı ve Tip": {
                "certificate_title": {"pattern": r"(EC\s*TYPE[\s-]*EXAMINATION\s*CERTIFICATE|AT\s*TİP\s*İNCELEME\s*SERTİFİKASI|TYPE[\s-]*EXAMINATION|TIP\s*MUAYENE)", "weight": 8},
                "directive_ref": {"pattern": r"(Directive\s*2006/42/EC|Makina\s*Direktifi|Machinery\s*Directive|2006/42/EC)", "weight": 7}
            },
            "Notified Body Bilgileri": {
                "notified_body": {"pattern": r"(Notified\s*Body|Onaylanmış\s*Kuruluş|Akredite\s*Kurum)", "weight": 8},
                "body_number": {"pattern": r"(NB\s*\d{4}|Notified\s*Body\s*No\.?\s*\d{4}|No\.?\s*\d{4})", "weight": 7}
            },
            "Makina/Ürün Tanımı": {
                "machine_description": {"pattern": r"(machine\s*description|makina\s*tanımı|product\s*description|ürün\s*açıklaması)", "weight": 8},
                "machine_type": {"pattern": r"(type|model|tip|seri)", "weight": 7}
            },
            "Üretici Bilgileri": {
                "manufacturer": {"pattern": r"(manufacturer|üretici|yapımcı)", "weight": 5},
                "address": {"pattern": r"(address|adres|street|sokak)", "weight": 5}
            },
            "Uygulanabilir Standartlar": {
                "standards": {"pattern": r"(EN\s*ISO\s*\d+|ISO\s*\d+|EN\s*\d+|IEC\s*\d+)", "weight": 10},
                "harmonized": {"pattern": r"(harmonized\s*standard|uyumlaştırılmış\s*standart)", "weight": 5}
            },
            "Sertifika Numarası ve Tarihi": {
                "cert_number": {"pattern": r"(certificate\s*no\.?|sertifika\s*no\.?|cert\.?\s*no\.?)[\s:]*([A-Z0-9\-/]+)", "weight": 5},
                "issue_date": {"pattern": r"(issue\s*date|date\s*of\s*issue|düzenleme\s*tarihi|tarih)[\s:]*(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})", "weight": 5}
            },
            "İmza ve Onay": {
                "signature": {"pattern": r"(signature|imza|signed\s*by|onaylayan)", "weight": 5},
                "approval": {"pattern": r"(approved|onaylanmıştır|certified|sertifikalandırılmıştır)", "weight": 5}
            },
            "Teknik Dosya Referansı": {
                "tech_file": {"pattern": r"(technical\s*file|teknik\s*dosya|documentation|dokümantasyon)", "weight": 5},
                "assessment": {"pattern": r"(assessment|değerlendirme|examination|inceleme|muayene)", "weight": 5}
            }
        }

    def extract_text_with_ocr(self, file_path: str) -> str:
        """OCR ile metin çıkarımı"""
        try:
            images = pdf2image.convert_from_path(file_path, dpi=300)
            text = ""
            for i, image in enumerate(images):
                logger.info(f"OCR işleniyor: Sayfa {i+1}")
                text += pytesseract.image_to_string(image, lang='eng+tur')
            return text
        except Exception as e:
            logger.error(f"OCR hatası: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkarımı"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {e}")
            return ""

    def extract_text_from_docx(self, docx_path: str) -> str:
        """DOCX'den metin çıkarımı"""
        try:
            doc = Document(docx_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            logger.error(f"DOCX hatası: {e}")
            return ""

    def extract_text_from_txt(self, txt_path: str) -> str:
        """TXT'den metin çıkarımı"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""

    def extract_text_from_file(self, file_path: str) -> str:
        """Dosya türüne göre metin çıkarımı"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
            if not text.strip():
                logger.info("PDF'de metin bulunamadı, OCR ile devam ediliyor...")
                text = self.extract_text_with_ocr(file_path)
            return text
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            return pytesseract.image_to_string(Image.open(file_path), lang='eng+tur')
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            logger.error(f"Desteklenmeyen dosya formatı: {file_ext}")
            return ""

    def analyze_criteria(self, text: str, category: str) -> Dict[str, ATTypeCertAnalysisResult]:
        """Kriterleri analiz et"""
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                content = f"Bulunan: {matches[:3]}"
                found = True
                score = min(weight, len(matches) * (weight // 2)) if weight > 2 else min(weight, len(matches))
            else:
                content = "Bulunamadı"
                found = False
                score = 0
            
            results[criterion_name] = ATTypeCertAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_count": len(matches) if matches else 0}
            )
        return results

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, ATTypeCertAnalysisResult]]) -> Dict[str, Any]:
        """Puanları hesapla"""
        category_scores = {}
        total_score = 0
        
        for category, results in analysis_results.items():
            category_max = self.criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())
            
            if category_possible > 0:
                percentage = (category_earned / category_possible) * 100
                normalized_score = (percentage / 100) * category_max
            else:
                percentage = 0
                normalized_score = 0
            
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
            "percentage": round(total_score, 2)
        }

    def extract_specific_values(self, text: str) -> Dict[str, Any]:
        """Spesifik değerleri çıkar"""
        values = {
            "certificate_number": "Bulunamadı",
            "issue_date": "Bulunamadı",
            "notified_body": "Bulunamadı",
            "manufacturer": "Bulunamadı",
            "machine_type": "Bulunamadı",
            "directive": "Bulunamadı"
        }
        
        # Sertifika numarası
        match = re.search(r"certificate\s*no\.?[\s:]*([A-Z0-9\-/]+)", text, re.IGNORECASE)
        if match:
            values["certificate_number"] = match.group(1).strip()
        
        # Tarih
        match = re.search(r"(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})", text)
        if match:
            values["issue_date"] = match.group(1)
        
        # Notified Body
        match = re.search(r"(Notified\s*Body|NB)[\s:]*([\w\s]+?)(?=\n|$)", text, re.IGNORECASE)
        if match:
            values["notified_body"] = match.group(2).strip()[:50]
        
        # Üretici
        match = re.search(r"manufacturer[\s:]*([^\n]+)", text, re.IGNORECASE)
        if match:
            values["manufacturer"] = match.group(1).strip()[:50]
        
        # Makina tipi
        match = re.search(r"(type|model)[\s:]*([A-Z0-9\-]+)", text, re.IGNORECASE)
        if match:
            values["machine_type"] = match.group(2).strip()
        
        # Direktif
        if re.search(r"2006/42/EC", text):
            values["directive"] = "2006/42/EC (Machinery Directive)"
        
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        total_percentage = scores["percentage"]
        
        if total_percentage >= 70:
            recommendations.append(f"✅ AT Type Certificate GEÇERLİ (Toplam: %{total_percentage:.1f})")
        else:
            recommendations.append(f"❌ AT Type Certificate GEÇERSİZ (Toplam: %{total_percentage:.1f})")
        
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 40:
                recommendations.append(f"🔴 {category} bölümü yetersiz (%{category_score:.1f})")
            elif category_score < 70:
                recommendations.append(f"🟡 {category} bölümü geliştirilmeli (%{category_score:.1f})")
            else:
                recommendations.append(f"🟢 {category} bölümü yeterli (%{category_score:.1f})")
        
        return recommendations

    def validate_document(self, text: str) -> bool:
        """Doküman validasyonu"""
        keywords = ["type", "certificate", "examination", "EC", "notified", "body", "machinery", "directive"]
        found = sum(1 for kw in keywords if re.search(kw, text, re.IGNORECASE))
        return found >= 4

    def analyze_report(self, file_path: str) -> Dict[str, Any]:
        """Ana analiz fonksiyonu"""
        logger.info("AT Type Certificate analizi başlatılıyor...")
        
        if not os.path.exists(file_path):
            return {"error": f"Dosya bulunamadı: {file_path}"}
        
        text = self.extract_text_from_file(file_path)
        if not text.strip():
            return {"error": "Dosyadan metin çıkarılamadı"}
        
        if not self.validate_document(text):
            return {"error": "YANLIŞ DOKÜMAN: Bu dosya AT Type Certificate değil!"}
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category)
        
        scores = self.calculate_scores(analysis_results)
        extracted_values = self.extract_specific_values(text)
        recommendations = self.generate_recommendations(analysis_results, scores)
        
        final_status = "PASS" if scores["percentage"] >= 70 else "FAIL"
        
        return {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgisi": {"file_path": file_path, "file_type": os.path.splitext(file_path)[1].lower()},
            "cikarilan_degerler": extracted_values,
            "kategori_analizleri": analysis_results,
            "puanlama": scores,
            "oneriler": recommendations,
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["percentage"],
                "durum": final_status,
                "rapor_tipi": "AT_TYPE_CERTIFICATE"
            }
        }

def validate_document_server(text):
    """Server kodunda doküman validasyonu - AT Type Certificate için"""
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
    
    category_found = []
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"AT Type Cert Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    valid_categories = sum(category_found)
    logger.info(f"Doküman validasyonu: {valid_categories}/4 kritik kategori bulundu")
    return valid_categories >= 3

def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - AT Type Certificate için"""
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
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l eng+tur', timeout=15)
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
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l eng+tur', timeout=15)
            all_text += text.lower() + " "
        
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

UPLOAD_FOLDER = 'temp_uploads_at_type_cert'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/at-type-cert-report', methods=['POST'])
def analyze_at_type_cert_report():
    """AT Type Certificate analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided', 'message': 'Lütfen analiz edilmek üzere bir AT Type Certificate sağlayın'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'message': 'Lütfen bir dosya seçin'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type', 'message': 'Sadece PDF, DOCX, DOC ve TXT dosyaları kabul edilir'}), 400
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"AT Type Certificate kontrol ediliyor: {filename}")
            
            analyzer = ATTypeCertificateAnalyzer()
            
            # ÜÇ AŞAMALI KONTROL
            logger.info(f"Üç aşamalı AT Type Certificate kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext == '.pdf':
                logger.info("Aşama 1: İlk sayfa AT Type Certificate özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath):
                    logger.info("✅ Aşama 1 geçti - AT Type Certificate özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - AT Type Certificate değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya AT Type Certificate değil (farklı rapor türü tespit edildi). Lütfen AT Type Certificate yükleyiniz.',
                            'details': {'filename': filename, 'document_type': 'OTHER_REPORT_TYPE', 'required_type': 'AT_TYPE_CERTIFICATE'}
                        }), 400
                    else:
                        logger.info("Aşama 3: Tam doküman critical terms kontrolü...")
                        try:
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
                                return jsonify({'error': 'Text extraction failed', 'message': 'Dosyadan yeterli metin çıkarılamadı'}), 400
                            
                            if not validate_document_server(text):
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                                return jsonify({
                                    'error': 'Invalid document type',
                                    'message': 'Yüklediğiniz dosya AT Type Certificate değil! Lütfen geçerli bir AT Type Certificate yükleyiniz.',
                                    'details': {'filename': filename, 'document_type': 'NOT_AT_TYPE_CERT', 'required_type': 'AT_TYPE_CERTIFICATE'}
                                }), 400
                        except Exception as e:
                            logger.error(f"Aşama 3 hatası: {e}")
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({'error': 'Analysis failed', 'message': 'Dosya analizi sırasında hata oluştu'}), 500
            
            elif file_ext in ['.docx', '.doc', '.txt']:
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
                    return jsonify({'error': 'Text extraction failed', 'message': 'Dosyadan yeterli metin çıkarılamadı'}), 400
                
                if not validate_document_server(text):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    return jsonify({
                        'error': 'Invalid document type',
                        'message': 'Yüklediğiniz dosya AT Type Certificate değil! Lütfen geçerli bir AT Type Certificate yükleyiniz.',
                        'details': {'filename': filename, 'document_type': 'NOT_AT_TYPE_CERT', 'required_type': 'AT_TYPE_CERTIFICATE'}
                    }), 400
            
            logger.info(f"AT Type Certificate doğrulandı, analiz başlatılıyor: {filename}")
            analysis_result = analyzer.analyze_report(filepath)
            
            try:
                os.remove(filepath)
                logger.info(f"Uploaded file {filename} cleaned up")
            except Exception as e:
                logger.warning(f"Could not remove uploaded file {filename}: {e}")
            
            if 'error' in analysis_result:
                return jsonify({'error': 'Analysis failed', 'message': analysis_result['error'], 'details': {'filename': filename}}), 400
            
            overall_percentage = analysis_result['puanlama']['percentage']
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': analysis_result.get('analiz_tarihi', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'analysis_id': f"at_type_cert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'filename': filename,
                'file_type': 'AT_TYPE_CERTIFICATE',
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': analysis_result['puanlama']['total_score'],
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ'
                },
                'category_scores': {},
                'extracted_values': analysis_result.get('cikarilan_degerler', {}),
                'recommendations': analysis_result.get('oneriler', []),
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': f"AT Type Certificate {'geçerli' if status == 'PASS' else 'geçersiz'} (%{overall_percentage:.0f})"
                }
            }
            
            for category, score_data in analysis_result['puanlama']['category_scores'].items():
                response_data['category_scores'][category] = {
                    'score': score_data['normalized'],
                    'max_score': score_data['max_weight'],
                    'percentage': score_data['percentage'],
                    'status': 'PASS' if score_data['percentage'] >= 70 else 'CONDITIONAL' if score_data['percentage'] >= 50 else 'FAIL'
                }
            
            return jsonify({
                'success': True,
                'message': 'AT Type Certificate başarıyla analiz edildi',
                'analysis_service': 'at_type_cert',
                'service_description': 'AT Type Certificate Analizi',
                'service_port': 8015,
                'data': response_data
            })
        
        except Exception as analysis_error:
            try:
                if 'filepath' in locals():
                    os.remove(filepath)
            except:
                pass
            
            logger.error(f"Analiz hatası: {str(analysis_error)}")
            return jsonify({
                'error': 'Analysis failed',
                'message': f'AT Type Certificate analizi sırasında hata oluştu: {str(analysis_error)}',
                'details': {'error_type': type(analysis_error).__name__, 'file_processed': filename if 'filename' in locals() else 'unknown'}
            }), 500
    
    except Exception as e:
        logger.error(f"API endpoint hatası: {str(e)}")
        return jsonify({'error': 'Server error', 'message': f'Sunucu hatası: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AT Type Certificate Analyzer API',
        'version': '1.0.0',
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': True,
        'report_type': 'AT_TYPE_CERTIFICATE'
    })

@app.route('/', methods=['GET'])
def index():
    """Ana sayfa - API bilgileri"""
    return jsonify({
        'service': 'AT Type Certificate Analyzer API',
        'version': '1.0.0',
        'description': 'AT Type Certificate belgelerini analiz eden REST API servisi',
        'endpoints': {
            'POST /api/at-type-cert-report': 'AT Type Certificate analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /': 'Bu bilgi sayfası'
        }
    })

if __name__ == '__main__':
    logger.info("AT Type Certificate Analyzer API başlatılıyor...")
    assigned_port = int(os.environ.get('PORT', 8015))
    app.run(host='0.0.0.0', port=assigned_port, debug=False)