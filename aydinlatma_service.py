# ============================================
# AYDINLATMA ÖLÇÜM RAPORU ANALİZ SERVİSİ
# Standalone Service - Azure App Service Ready
# Port: 8008
# ============================================

# ============================================
# IMPORTS
# ============================================
import os
import json
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Any
import PyPDF2
from docx import Document
from dataclasses import dataclass
import logging
import cv2
import numpy as np
from PIL import Image
import pdf2image
import pytesseract

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# ============================================
# LOGGING CONFIGURATION
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# LANGUAGE DETECTION (Optional)
# ============================================
try:
    from langdetect import detect
    LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False
    logger.warning("langdetect modülü bulunamadı - dil tespiti devre dışı")

# ============================================
# TESSERACT CHECK
# ============================================
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

# ============================================
# ANALİZ SINIFI - DATA CLASSES
# ============================================
@dataclass
class LightingAnalysisResult:
    """Aydınlatma Ölçüm Raporu analiz sonucu veri sınıfı"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

# ============================================
# ANALİZ SINIFI - MAIN ANALYZER
# ============================================
class LightingReportAnalyzer:
    """Aydınlatma Ölçüm Raporu analiz sınıfı"""
    
    def __init__(self):
        logger.info("Aydınlatma Ölçüm Raporu analiz sistemi başlatılıyor...")
        
        self.criteria_weights = {
            "Genel Rapor Bilgileri": 10,
            "Ölçüm Metodu ve Standart Referansları": 15,
            "Ölçüm Sonuç Tablosu": 25,
            "Uygunluk Değerlendirmesi": 20,
            "Görsel ve Teknik Dokümantasyon": 5,
            "Ölçüm Cihazı Bilgileri": 10,
            "Sonuç ve Öneriler": 15
        }
        
        self.criteria_details = {
            "Genel Rapor Bilgileri": {
                "proje_adi_numarasi": {"pattern": r"(?:Proje\s*Ad[ıi]|Project\s*Name|Proje\s*No|Project\s*Number)", "weight": 2},
                "olcum_rapor_tarihleri": {"pattern": r"(?:Ölçüm\s*Tarih|Measurement\s*Date|Rapor\s*Tarih|Report\s*Date|\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})", "weight": 2},
                "tesis_bolge_alan": {"pattern": r"(?:Tesis|Facility|Fabrika|Factory|Ofis|Office|Bina|Building)", "weight": 1},
                "rapor_no_revizyon": {"pattern": r"(?:Rapor\s*No|Report\s*No|Rev|Revizyon|Version)", "weight": 2},
                "olcumu_yapan_firma": {"pattern": r"(?:Ölçümü\s*Yapan|Measured\s*By|Firma|Company)", "weight": 1},
                "onay_imza": {"pattern": r"(?:Onay|Approval|İmza|Signature|Sorumlu|Yetkili)", "weight": 2}
            },
            "Ölçüm Metodu ve Standart Referansları": {
                "olcum_cihazi": {"pattern": r"(?:Lüksmetre|Luxmeter|Lux\s*Meter|Işık\s*Ölçer|Light\s*Meter|Model|Seri\s*No)", "weight": 4},
                "kalibrasyon_bilgi": {"pattern": r"(?:Kalibrasyon|Calibration|Sertifika|Certificate|Geçerlilik)", "weight": 3},
                "olcum_yontemi": {"pattern": r"(?:Ölçüm\s*Yöntem|Measurement\s*Method|Prosedür|Methodology|Ortlama)", "weight": 4},
                "standartlar": {"pattern": r"(?:TS\s*EN\s*12464|ISO\s*8995|İş\s*Sağlığ[ıi]|Standart|Norm)", "weight": 4}
            },
            "Ölçüm Sonuç Tablosu": {
                "tablo_yapisi": {"pattern": r"(?:Tablo|Table|Liste|List|Sıra\s*No|Row\s*No)", "weight": 5},
                "calisma_alani": {"pattern": r"(?:Çalışma\s*Alan[ıi]|Work\s*Area|Bölge\s*Ad[ıi]|Konum|Nokta)", "weight": 4},
                "olculen_degerler": {"pattern": r"(?:Lüks|Lux|lx|Aydınlatma\s*Şiddet|Illumination|Light\s*Level|Ölçülen)", "weight": 8},
                "hedeflenen_degerler": {"pattern": r"(?:Hedeflenen|Target|İstenen|Gerekli|Minimum|Standart)", "weight": 4},
                "uygunluk_durumu": {"pattern": r"(?:Uygun|Suitable|Uygunsuz|Not\s*Suitable|PASS|FAIL|OK|NOK)", "weight": 4}
            },
            "Uygunluk Değerlendirmesi": {
                "toplu_degerlendirme": {"pattern": r"(?:Genel\s*Değerlendirme|Overall\s*Evaluation|Özet|Summary|Sonuç)", "weight": 5},
                "limit_disi_degerler": {"pattern": r"(?:Limit\s*Dış[ıi]|Out\s*of\s*Limit|Uygunsuz|Eksik|Fazla)", "weight": 5},
                "risk_belirtme": {"pattern": r"(?:Risk|Tehlike|Hazard|Göz\s*Yorgunluk|Verimlilik|Güvenlik)", "weight": 5},
                "duzeltici_faaliyet": {"pattern": r"(?:Düzeltici\s*Faaliyet|Corrective\s*Action|İyileştirme|Öneri)", "weight": 5}
            },
            "Görsel ve Teknik Dokümantasyon": {
                "alan_fotograflari": {"pattern": r"(?:Fotoğraf|Photo|Görsel|Visual|Resim|Image)", "weight": 1},
                "cihaz_fotograflari": {"pattern": r"(?:Cihaz\s*Fotoğraf|Device\s*Photo|Ölçüm\s*Cihaz)", "weight": 1},
                "kroki_sema": {"pattern": r"(?:Kroki|Sketch|Şema|Schema|Plan|Layout|Çizim)", "weight": 1},
                "armatur_teknik": {"pattern": r"(?:Armatür|Fixture|Lamba|Lamp|LED|Fotometrik|Lümen)", "weight": 2}
            },
            "Ölçüm Cihazı Bilgileri": {
                "cihaz_detay": {"pattern": r"(?:Marka|Brand|Model|Tip|Type|Seri|Serial|Kalibrasyon)", "weight": 5},
                "cihaz_ozellikleri": {"pattern": r"(?:Hassasiyet|Accuracy|Precision|Range|Aralık|Çözünürlük)", "weight": 3},
                "cihaz_durumu": {"pattern": r"(?:Durum|Status|Çalışır|Working|Aktif|Geçerli|Uygun)", "weight": 2}
            },
            "Sonuç ve Öneriler": {
                "genel_uygunluk": {"pattern": r"(?:Genel\s*Sonuç|Overall\s*Result|Uygun|Suitable|PASS|FAIL)", "weight": 4},
                "standart_atif": {"pattern": r"(?:Standart|Standard|TS\s*EN|ISO|Referans|Uygunluk)", "weight": 3},
                "iyilestirme_onerileri": {"pattern": r"(?:İyileştirme|Improvement|Öneri|Recommendation|Aksiyon)", "weight": 4},
                "tekrar_olcum": {"pattern": r"(?:Tekrar\s*Ölçüm|Re[\\-\\s]*Measurement|Periyot|Period|Kontrol)", "weight": 4}
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Metin dilini tespit et"""
        if not LANGUAGE_DETECTION_AVAILABLE:
            return 'tr'
        
        try:
            sample_text = text[:500].strip()
            if not sample_text:
                return 'tr'
                
            detected_lang = detect(sample_text)
            logger.info(f"Tespit edilen dil: {detected_lang}")
            return detected_lang
            
        except Exception as e:
            logger.warning(f"Dil tespiti başarısız: {e}")
            return 'tr'
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkarma"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                all_text = ""
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"PDF analizi başlatılıyor - Toplam sayfa: {total_pages}")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    page_text = re.sub(r'\s+', ' ', page_text)
                    page_text = page_text.replace('|', ' ')
                    page_text = page_text.strip()
                    all_text += page_text + "\n"
                
                all_text = all_text.replace('—', '-')
                all_text = all_text.replace('"', '"').replace('"', '"')
                all_text = re.sub(r'[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]+', ' ', all_text)
                all_text = all_text.strip()
                
                logger.info(f"✅ PDF analizi tamamlandı: {len(all_text):,} karakter")
                return all_text
                
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {e}")
            return ""

    def extract_text_from_docx(self, docx_path: str) -> str:
        """DOCX'den metin çıkarma"""
        try:
            doc = Document(docx_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                text += "\n"
            
            text = re.sub(r'\s+', ' ', text).strip()
            logger.info(f"DOCX'den {len(text)} karakter metin çıkarıldı")
            return text
            
        except Exception as e:
            logger.error(f"DOCX metin çıkarma hatası: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """TXT dosyasından metin çıkarma"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            if not text:
                encodings = ['cp1254', 'iso-8859-9', 'latin1']
                for encoding in encodings:
                    try:
                        with open(txt_path, 'r', encoding=encoding) as file:
                            text = file.read()
                        if text:
                            break
                    except:
                        continue
            
            logger.info(f"TXT'den {len(text)} karakter metin çıkarıldı")
            return text.strip()
            
        except Exception as e:
            logger.error(f"TXT metin çıkarma hatası: {e}")
            return ""
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, LightingAnalysisResult]:
        """Kriterleri analiz et"""
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                content = f"Bulunan: {str(matches[:3])}"
                found = True
                
                if weight <= 2:
                    score = min(weight, len(matches))
                else:
                    score = min(weight, len(matches) * (weight // 2))
                    score = max(score, weight // 2)
                    
            else:
                content = "Bulunamadı"
                found = False
                score = 0
            
            results[criterion_name] = LightingAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_count": len(matches) if matches else 0}
            )
        
        return results

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, LightingAnalysisResult]]) -> Dict[str, Any]:
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
        """Aydınlatma raporuna özgü değerleri çıkar"""
        values = {
            "rapor_numarasi": "Bulunamadı",
            "proje_adi": "Bulunamadı",
            "olcum_tarihi": "Bulunamadı", 
            "rapor_tarihi": "Bulunamadı",
            "olcum_cihazi": "Bulunamadı",
            "tesis_adi": "Bulunamadı",
            "genel_uygunluk": "Bulunamadı"
        }
        
        # RAPOR NUMARASI
        report_no_patterns = [
            r"(?i)(?:RAPOR\s*NO|REPORT\s*NO)\s*[:=]?\s*([A-Z0-9\-\/]{3,20})",
        ]
        
        for pattern in report_no_patterns:
            match = re.search(pattern, text)
            if match:
                result = match.group(1).strip()
                if 3 <= len(result) <= 20:
                    values["rapor_numarasi"] = result
                    break

        # PROJE ADI
        project_patterns = [
            r"(?i)(?:PROJE\s*ADI|PROJECT\s*NAME)\s*[:=]?\s*(.*?(?:ÖLÇÜM|TEST|RAPOR|REPORT))",
        ]
        
        for pattern in project_patterns:
            match = re.search(pattern, text)
            if match:
                result = match.group(1).strip()
                result = re.sub(r'\s+', ' ', result)
                if 3 <= len(result) <= 100:
                    values["proje_adi"] = result
                    break
        
        # ÖLÇÜM TARİHİ
        measurement_date_patterns = [
            r"(?i)(?:ÖLÇÜM\s*TARİH[İI]?|MEASUREMENT\s*DATE)\s*[:=]\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})",
            r"\b(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{4})\b"
        ]
        
        for pattern in measurement_date_patterns:
            match = re.search(pattern, text)
            if match:
                values["olcum_tarihi"] = match.group(1)
                break
        
        # RAPOR TARİHİ
        report_date_patterns = [
            r"(?i)(?:RAPOR\s*TARİH[İI]?|REPORT\s*DATE)\s*[:=]\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})",
        ]
        
        for pattern in report_date_patterns:
            match = re.search(pattern, text)
            if match:
                values["rapor_tarihi"] = match.group(1)
                break
    
        if values["rapor_tarihi"] == "Bulunamadı" and values["olcum_tarihi"] != "Bulunamadı":
            values["rapor_tarihi"] = "Rapor tarihi ayrı belirtilmemiş"

        # ÖLÇÜM CİHAZI
        device_patterns = [
            r"(?i)(?:LÜKSMETRE|LUXMETER|LUX\s*METER)\s*[:=]?\s*([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9\s\.\-]{2,50})",
        ]
        
        for pattern in device_patterns:
            match = re.search(pattern, text)
            if match:
                result = match.group(1).strip()
                if 2 <= len(result) <= 35:
                    values["olcum_cihazi"] = result
                    break
        
        # TESİS ADI
        facility_patterns = [
            r"(?i)(?:TESİS\s*ADI|FACILITY\s*NAME|FABRİKA|FACTORY)\s*[:=]?\s*([A-ZÇĞİÖŞÜ0-9][A-ZÇĞİÖŞÜ0-9\.\&\s\-]{3,60})",
        ]

        for pattern in facility_patterns:
            matches = re.findall(pattern, text)
            for result in matches:
                result = result.strip()
                if 3 <= len(result) <= 60: 
                    values["tesis_adi"] = result
                    break
            if values["tesis_adi"] != "Bulunamadı":
                break
        
        # GENEL UYGUNLUK
        compliance_patterns = [
            r"(?i)\b(UYGUN|SUITABLE|PASS)\b",
            r"(?i)\b(UYGUNSUZ|NOT\s*SUITABLE|FAIL)\b",
        ]
        
        for pattern in compliance_patterns:
            match = re.search(pattern, text)
            if match:
                result = match.group(1).strip().upper()
                if result in ["UYGUN", "SUITABLE", "PASS"]:
                    values["genel_uygunluk"] = "UYGUN"
                elif result in ["UYGUNSUZ", "NOT SUITABLE", "FAIL"]:
                    values["genel_uygunluk"] = "UYGUNSUZ"
                break
        
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        total_percentage = scores["percentage"]
        
        if total_percentage >= 70:
            recommendations.append(f"✅ Aydınlatma Ölçüm Raporu GEÇERLİ (Toplam: %{total_percentage:.1f})")
        else:
            recommendations.append(f"❌ Aydınlatma Ölçüm Raporu GEÇERSİZ (Toplam: %{total_percentage:.1f})")
        
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 40:
                recommendations.append(f"🔴 {category} bölümü yetersiz (%{category_score:.1f})")
            elif category_score < 70:
                recommendations.append(f"🟡 {category} bölümü geliştirilmeli (%{category_score:.1f})")
            else:
                recommendations.append(f"🟢 {category} bölümü yeterli (%{category_score:.1f})")
        
        return recommendations

    def analyze_lighting_report(self, file_path: str) -> Dict[str, Any]:
        """Ana Aydınlatma Ölçüm Raporu analiz fonksiyonu"""
        logger.info("Aydınlatma Ölçüm Raporu analizi başlatılıyor...")
        
        if not os.path.exists(file_path):
            return {"error": f"Dosya bulunamadı: {file_path}"}
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            text = self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            text = self.extract_text_from_txt(file_path)
        else:
            return {"error": f"Desteklenmeyen dosya formatı: {file_ext}"}
        
        if not text:
            return {"error": "Dosyadan metin çıkarılamadı"}
        
        detected_lang = self.detect_language(text)
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category)
        
        scores = self.calculate_scores(analysis_results)
        extracted_values = self.extract_specific_values(text)
        recommendations = self.generate_recommendations(analysis_results, scores)
        
        final_status = "PASS" if scores["percentage"] >= 70 else "FAIL"
        
        report = {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgisi": {
                "file_path": file_path,
                "file_type": file_ext,
                "detected_language": detected_lang
            },
            "cikarilan_degerler": extracted_values,
            "kategori_analizleri": analysis_results,
            "puanlama": scores,
            "oneriler": recommendations,
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["percentage"],
                "durum": final_status,
                "rapor_tipi": "AYDINLATMA_OLCUM_RAPORU"
            }
        }
        
        return report
    
# ============================================
# HELPER FUNCTIONS (Server Validasyon)
# ============================================
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
    """Server kodunda doküman validasyonu - Aydınlatma için"""
    
    critical_terms = [
         # Aydınlatma temel terimleri
        ["aydınlatma", "lighting", "illumination", "ışık", "lumen", "ışık şiddeti"],
        
        # Aydınlatma ölçüm birimleri
        ["lux", "cd/m2", "candela", "luminance", "illuminance"],
        
        # Aydınlatma standartları
        ["ts en 12464", "en 12464", "12464", "iso 8995", "cibse"],
        
        # Aydınlatma ekipmanları
        ["led", "fluorescent", "floresan", "armatur", "luminaire", "ballast"],

        # Aydınlatma türleri/çeşitleri (en az 1 tane olmalı)
        ["genel aydınlatma", "general lighting", "task lighting", "görev aydınlatması", "accent lighting", "emergency lighting", "acil aydınlatma"]
    ]
    
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"Aydınlatma Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    valid_categories = sum(category_found)
    logger.info(f"Doküman validasyonu: {valid_categories}/5 kritik kategori bulundu")
    
    return valid_categories >= 4


def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - Aydınlatma için"""
    strong_keywords = [
        "aydınlatma",
        "lighting",
        "illumination", 
        "lux",
        "lümen",
        "lumen",
        "ts en 12464",
        "en 12464",
        "ışık",
        "ışık şiddeti"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=200, first_page=1, last_page=2)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        
        found_keywords = [kw for kw in strong_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa kontrol: {len(found_keywords)} özgü kelime: {found_keywords}")
        return len(found_keywords) >= 2
        
    except Exception as e:
        logger.warning(f"İlk sayfa kontrol hatası: {e}")
        return False


def check_excluded_keywords_first_pages(filepath):
    """İlk 1-2 sayfada istenmeyen rapor türlerinin kelimelerini ara"""
    excluded_keywords = [
        # HRC raporu (eski strong_keywords)
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
        "uygunluk", "beyan", "muayene", "conformity", "declaration",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=200, first_page=1, last_page=2)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        
        found_excluded = [kw for kw in excluded_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa excluded kontrol: {len(found_excluded)} istenmeyen kelime: {found_excluded}")
        return len(found_excluded) >= 2
        
    except Exception as e:
        logger.warning(f"İlk sayfa excluded kontrol hatası: {e}")
        return False


def get_conclusion_message_aydinlatma(status, percentage):
    """Sonuç mesajını döndür - Aydınlatma için"""
    if status == "PASS":
        return f"Aydınlatma ölçüm raporu TS EN 12464-1 ve İSG mevzuatına uygundur (%{percentage:.0f})"
    else:
        return f"Aydınlatma raporu standartlara uygun değil (%{percentage:.0f})"


def get_main_issues_aydinlatma(report):
    """Ana sorunları listele - Aydınlatma için"""
    issues = []
    
    for category, score_data in report['puanlama']['category_scores'].items():
        if score_data['percentage'] < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    if not issues and report['puanlama']['total_score'] < 50:
        issues = [
            "Aydınlatma seviyesi ölçüm sonuçları eksik",
            "Ölçüm cihazı kalibrasyon bilgileri eksik",
            "Çalışma alanı tanımı ve sınıflandırması eksik",
            "Yasal uygunluk değerlendirmesi yapılmamış"
        ]
    
    return issues[:4]


# ============================================
# FLASK SERVİS KATMANI - CONFIGURATION
# ============================================
app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads_aydinlatma'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================
# FLASK SERVİS KATMANI - API ENDPOINTS
# ============================================
@app.route('/api/aydinlatma-report', methods=['POST'])
def analyze_aydinlatma_report():
    """Aydınlatma Ölçüm Raporu analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere bir aydınlatma raporu sağlayın'
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected', 'message': 'Lütfen bir dosya seçin'}), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Sadece PDF, DOCX, DOC ve TXT dosyaları kabul edilir'
            }), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"Aydınlatma Ölçüm Raporu kontrol ediliyor: {filename}")

            analyzer = LightingReportAnalyzer()
            
            logger.info(f"Üç aşamalı aydınlatma kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                logger.info("Aşama 1: İlk sayfa aydınlatma özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath):
                    logger.info("✅ Aşama 1 geçti - Aydınlatma özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - Aydınlatma değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya aydınlatma raporu değil (farklı rapor türü tespit edildi).',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'AYDINLATMA_OLCUM_RAPORU'
                            }
                        }), 400
                    else:
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
                                    'message': 'Yüklediğiniz dosya aydınlatma ölçüm raporu değil!',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_AYDINLATMA_REPORT',
                                        'required_type': 'AYDINLATMA_OLCUM_RAPORU'
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
                        'message': 'Yüklediğiniz dosya aydınlatma ölçüm raporu değil!',
                        'details': {
                            'filename': filename,
                            'document_type': 'NOT_AYDINLATMA_REPORT',
                            'required_type': 'AYDINLATMA_OLCUM_RAPORU'
                        }
                    }), 400

            logger.info(f"Aydınlatma raporu doğrulandı, analiz başlatılıyor: {filename}")
            report = analyzer.analyze_lighting_report(filepath)
            
            try:
                os.remove(filepath)
                logger.info(f"Uploaded file {filename} cleaned up")
            except Exception as e:
                logger.warning(f"Could not remove uploaded file {filename}: {e}")

            if 'error' in report:
                return jsonify({
                    'error': 'Analysis failed',
                    'message': report['error'],
                    'details': {'filename': filename}
                }), 400

            overall_percentage = report['ozet']['yuzde']
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': report.get('analiz_tarihi', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'analysis_details': {
                    'found_criteria': len([c for results in report['kategori_analizleri'].values() for c in results.values() if c.found]),
                    'total_criteria': len([c for results in report['kategori_analizleri'].values() for c in results.values()]),
                    'percentage': round(overall_percentage, 1)
                },
                'analysis_id': f"aydinlatma_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'extracted_values': report['cikarilan_degerler'],
                'file_type': 'AYDINLATMA_OLCUM_RAPORU',
                'filename': filename,
                'language_info': {
                    'detected_language': report['dosya_bilgisi'].get('detected_language', 'turkish').upper(),
                    'file_type': file_ext.replace('.', '')
                },
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': report['ozet']['toplam_puan'],
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'text_quality': 'good'
                },
                'recommendations': report['oneriler'],
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_aydinlatma(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_aydinlatma(report)
                }
            }
            
            for category, score_data in report['puanlama']['category_scores'].items():
                response_data['category_scores'][category] = {
                    'score': score_data.get('normalized', 0),
                    'max_score': score_data.get('max_weight', 0),
                    'percentage': score_data.get('percentage', 0),
                    'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'CONDITIONAL' if score_data.get('percentage', 0) >= 50 else 'FAIL'
                }

            return jsonify({
                'success': True,
                'message': 'Aydınlatma Ölçüm Raporu başarıyla analiz edildi',
                'analysis_service': 'aydinlatma',
                'service_description': 'Aydınlatma Ölçüm Raporu Analizi',
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
                'message': f'Aydınlatma raporu analizi sırasında hata oluştu: {str(analysis_error)}',
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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint - Aydınlatma için"""
    return jsonify({
        'status': 'healthy',
        'service': 'Lighting Report Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'report_type': 'AYDINLATMA_OLCUM_RAPORU',
        'standard': 'TS EN 12464-1'
    })


@app.route('/', methods=['GET'])
def index():
    """Ana sayfa - API bilgileri - Aydınlatma için"""
    return jsonify({
        'service': 'Lighting Report Analyzer API',
        'version': '1.0.0',
        'description': 'Aydınlatma Ölçüm Raporlarını analiz eden REST API servisi',
        'endpoints': {
            'POST /api/aydinlatma-report': 'Aydınlatma raporu analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /': 'Bu bilgi sayfası'
        },
        'usage': {
            'upload_format': 'multipart/form-data',
            'file_field': 'file',
            'supported_types': list(ALLOWED_EXTENSIONS),
            'max_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        },
        'scoring': {
            'PASS': '≥70% - TS EN 12464-1 standardına uygun',
            'CONDITIONAL': '50-69% - Kabul edilebilir',
            'FAIL': '<50% - Standarda uygun değil'
        }
    })


# ============================================
# APPLICATION ENTRY POINT (Azure-Friendly)
# ============================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Aydınlatma Ölçüm Raporu Analiz Servisi")
    logger.info("=" * 60)
    logger.info(f"📁 Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"📊 Desteklenen formatlar: {', '.join(ALLOWED_EXTENSIONS)}")
    logger.info(f"📏 Maksimum dosya boyutu: {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB")
    logger.info(f"🔍 Tesseract OCR: {'Kurulu' if tesseract_available else 'Kurulu değil'}")
    logger.info("")
    logger.info("🔗 API Endpoints:")
    logger.info("  POST /api/aydinlatma-report - Aydınlatma raporu analizi")
    logger.info("  GET /api/health - Servis sağlık kontrolü")
    logger.info("  GET / - API bilgileri")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8008))
    
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )