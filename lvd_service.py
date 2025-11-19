# ============================================
# LVD TOPRAKLAMA SÜREKLİLİK RAPORU ANALİZ SERVİSİ
# Standalone Service - Azure App Service Ready
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
import pandas as pd
from dataclasses import dataclass, asdict
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
class GroundingContinuityCriteria:
    """Topraklama Süreklilik rapor kriterleri veri sınıfı"""
    genel_rapor_bilgileri: Dict[str, Any]
    olcum_metodu_standart_referanslari: Dict[str, Any]
    olcum_sonuc_tablosu: Dict[str, Any]
    uygunluk_degerlendirmesi: Dict[str, Any]
    gorsel_teknik_dokumantasyon: Dict[str, Any]
    sonuc_oneriler: Dict[str, Any]

@dataclass
class GroundingAnalysisResult:
    """Topraklama Süreklilik analiz sonucu veri sınıfı"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

# ============================================
# ANALİZ SINIFI - MAIN ANALYZER
# ============================================
class GroundingContinuityReportAnalyzer:
    """Topraklama Süreklilik rapor analiz sınıfı"""
    
    def __init__(self):
        logger.info("LVD Topraklama Süreklilik analiz sistemi başlatılıyor...")
        
        self.criteria_weights = {
            "Genel Rapor Bilgileri": 15,
            "Ölçüm Metodu ve Standart Referansları": 15,
            "Ölçüm Sonuç Tablosu": 25,
            "Uygunluk Değerlendirmesi": 20,
            "Görsel ve Teknik Dökümantasyon": 10,
            "Sonuç ve Öneriler": 15
        }
        
        self.criteria_details = {
            "Genel Rapor Bilgileri": {
                "proje_adi_numarasi": {"pattern": r"(?:Project\s*(?:Name|No)|Proje\s*(?:Ad[ıi]|No)|Report\s*Title|Document\s*Title|E\d{2}\.\d{3}|C\d{2}\.\d{3}|T\d{2,3}[-.]?\d{3,4})", "weight": 3},
                "olcum_tarihi": {"pattern": r"(?:Measurement\s*Date|Ölçüm\s*Tarihi|Test\s*Date|Date\s*of\s*(?:Test|Measurement)|Measured\s*on|Tested\s*on|\d{1,2}[./\-]\d{1,2}[./\-]\d{4})", "weight": 3},
                "rapor_tarihi": {"pattern": r"(?:Report\s*Date|Rapor\s*Tarihi|Issue\s*Date|Document\s*Date|Prepared\s*on|Created\s*on|Date|Tarih|\d{1,2}[./\-]\d{1,2}[./\-]\d{4})", "weight": 3},
                "tesis_bolge_hat": {"pattern": r"(?:Customer|Müşteri|Client|Facility|Tesis|Plant|Factory|Company|Firma|Toyota|DANONE|Ford|BOSCH)", "weight": 2},
                "rapor_numarasi": {"pattern": r"(?:Report\s*No|Rapor\s*No|Document\s*No|Belge\s*No|E\d{2}\.\d{3}|C\d{2}\.\d{3}|SM\s*\d+|MCC\d+)", "weight": 2},
                "revizyon": {"pattern": r"(?:Version|Revizyon|Rev\.?|v)\s*[:=]?\s*(\d+|[A-Z])", "weight": 1},
                "firma_personel": {"pattern": r"(?:Prepared\s*by|Hazırlayan|Performed\s*by|Ölçümü\s*Yapan|Consultant|Engineer|PILZ)", "weight": 1}
            },
            "Ölçüm Metodu ve Standart Referansları": {
                "olcum_cihazi": {"pattern": r"(?:Measuring\s*Instrument|Ölçüm\s*Cihaz[ıi]|Test\s*Equipment|Multimeter|Multimetre|Ohmmeter|Instrument|Equipment|Device|Tester|Fluke|Metrix|Chauvin|Megger|Hioki)", "weight": 6},
                "kalibrasyon": {"pattern": r"(?:Calibration|Kalibrasyon|Kalibre|Certificate|Sertifika|Cal\s*Date)", "weight": 4},
                "standartlar": {"pattern": r"(?:EN\s*60204[-\s]*1?|IEC\s*60364|Standard|Standart)", "weight": 5}
            },
            "Ölçüm Sonuç Tablosu": {
                "sira_numarasi": {"pattern": r"(?:S[ıi]ra\s*(?:No|Numaras[ıi])|^\s*\d+\s)", "weight": 3},
                "makine_hat_bolge": {"pattern": r"(?:8X45|8X50|8X9J|9J73|8X52|8X60|8X62|8X70)\s*(?:R[1-9])?\s*(?:Hatt[ıi]|Line|Zone|Bölge)", "weight": 3},
                "olcum_noktasi": {"pattern": r"(?:Robot\s*\d+\.\s*Eksen\s*Motoru|Kalemtraş|Lift\s*and\s*Shift|Motor|Ekipman|Equipment|Device)", "weight": 3},
                "rlo_degeri": {"pattern": r"(\d+[.,]?\d*)\s*(?:mΩ|mohm|ohm|Ω)", "weight": 5},
                "yuk_iletken_kesiti": {"pattern": r"(?:4x4|4x2[.,]5|4x6|4x10|Yük\s*İletken|Load\s*Conductor|PE\s*İletken|PE\s*Conductor)", "weight": 4},
                "referans_degeri": {"pattern": r"(?:500\s*mΩ|500\s*ohm|500\s*Ω|EN\s*60204|IEC\s*60364)", "weight": 3},
                "uygunluk_durumu": {"pattern": r"(?:UYGUN|OK|PASS|Compliant|Uygun)", "weight": 4},
                "kesit_uygunlugu": {"pattern": r"(?:UYGUN|OK|PASS|Compliant|Uygun)", "weight": 3}
            },
            "Uygunluk Değerlendirmesi": {
                "toplam_olcum_nokta": {"pattern": r"(?:222|220|200|Toplam.*\d+)", "weight": 5},
                "uygun_nokta_sayisi": {"pattern": r"(?:211|210|UYGUN)", "weight": 5},
                "uygunsuz_isaretleme": {"pattern": r"\*D\.Y", "weight": 5, "reverse_logic": True},
                "standart_referans_uygunluk": {"pattern": r"(?:500\s*mΩ|EN\s*60204)", "weight": 5}
            },
            "Görsel ve Teknik Dökümantasyon": {
                "cihaz_baglanti_fotografi": {"pattern": r"(?:Cihaz.*Fotoğraf|Bağlant[ıi].*Fotoğraf|Ölçüm.*Cihaz|Photo|Image|Figure|Resim|Görsel)", "weight": 10}
            },
            "Sonuç ve Öneriler": {
                "genel_uygunluk": {"pattern": r"(?:Genel\s*Uygunluk|Sonuç|UYGUN|UYGUNSUZ|Result|Conclusion|Compliant|Non-compliant)", "weight": 8},
                "standart_atif": {"pattern": r"(?:EN\s*60204|IEC\s*60364|Standart.*Atıf|Standart.*Referans|Standard.*Reference)", "weight": 7}
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
    
    def get_language_name(self, lang_code: str) -> str:
        """Dil kodunu dil adına çevir"""
        lang_names = {
            'tr': 'Türkçe',
            'en': 'İngilizce', 
            'de': 'Almanca',
            'fr': 'Fransızca',
            'es': 'İspanyolca',
            'it': 'İtalyanca',
            'pt': 'Portekizce',
            'ru': 'Rusça'
        }
        return lang_names.get(lang_code, f'Bilinmeyen ({lang_code})')
    
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
                all_text = all_text.replace('´', "'")
                all_text = re.sub(r'[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]+', ' ', all_text)
                all_text = all_text.strip()
                
                logger.info(f"✅ PDF analizi tamamlandı: {len(all_text):,} karakter")
                return all_text
                
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {e}")
            return ""
    
    def normalize_date_string(self, date_str: str) -> str:
        """Tarih string'ini DD/MM/YYYY formatına çevir"""
        if not date_str or date_str == "Bulunamadı":
            return date_str
            
        month_names = {
            'jan': '01', 'january': '01', 'feb': '02', 'february': '02', 
            'mar': '03', 'march': '03', 'apr': '04', 'april': '04',
            'may': '05', 'jun': '06', 'june': '06', 'jul': '07', 'july': '07',
            'aug': '08', 'august': '08', 'sep': '09', 'september': '09',
            'oct': '10', 'october': '10', 'nov': '11', 'november': '11',
            'dec': '12', 'december': '12',
            'ocak': '01', 'şubat': '02', 'subat': '02', 'mart': '03',
            'nisan': '04', 'mayıs': '05', 'mayis': '05', 'haziran': '06',
            'temmuz': '07', 'ağustos': '08', 'agustos': '08',
            'eylül': '09', 'eylul': '09', 'ekim': '10',
            'kasım': '11', 'kasim': '11', 'aralık': '12', 'aralik': '12'
        }
        
        date_str = date_str.strip()
        
        if re.match(r'\d{1,2}[./\-]\d{1,2}[./\-]\d{4}', date_str):
            return date_str.replace('.', '/').replace('-', '/')
        
        if re.match(r'\d{4}[./\-]\d{1,2}[./\-]\d{1,2}', date_str):
            parts = re.split(r'[./\-]', date_str)
            return f"{parts[2].zfill(2)}/{parts[1].zfill(2)}/{parts[0]}"
        
        month_pattern = r'(\d{1,2})\s+([a-zA-ZğıüşçöĞIÜŞÇÖ]+)\s+(\d{4})'
        match = re.match(month_pattern, date_str, re.IGNORECASE)
        if match:
            day, month_name, year = match.groups()
            month_name_lower = month_name.lower()
            if month_name_lower in month_names:
                month_num = month_names[month_name_lower]
                return f"{day.zfill(2)}/{month_num}/{year}"
        
        return date_str.replace('.', '/').replace('-', '/')
    
    def check_date_validity(self, text: str, file_path: str = None) -> Tuple[bool, str, str, str]:
        """Tarih bilgilerini çıkar - sadece tarih tespiti için (1 yıl kısıtlaması kaldırıldı)"""
        
        olcum_patterns = [
            r"(?:Ölçüm\s*Tarihi|Test\s*Tarihi|Ölçüm\s*Yapıldığı\s*Tarih)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Ölçüm|Test).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4}).*?(?:ölçüm|test)",
            r"(?:Measurement\s*Date|Test\s*Date|Date\s*of\s*(?:Test|Measurement))\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Measured\s*on|Tested\s*on)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4}).*?(?:measurement|test)",
            r"(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})",
            r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
            r"(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})"
        ]
        
        rapor_patterns = [
            r"(?:Rapor\s*Tarihi|Belge\s*Tarihi|Hazırlanma\s*Tarihi)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Rapor|Belge).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Hazırlayan|Hazırlandı)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Report\s*Date|Document\s*Date|Issue\s*Date|Prepared\s*Date)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Prepared\s*on|Issued\s*on|Created\s*on)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Date|Tarih)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})",
            r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
            r"(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})"
        ]
        
        olcum_tarihi = None
        rapor_tarihi = None
        
        for pattern in olcum_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                olcum_tarihi = matches[0]
                break
        
        for pattern in rapor_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                rapor_tarihi = matches[0]
                break
        
        if not rapor_tarihi and file_path and os.path.exists(file_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            rapor_tarihi = file_mod_time.strftime("%d/%m/%Y")
        elif not rapor_tarihi:
            rapor_tarihi = datetime.now().strftime("%d/%m/%Y")
        
        try:
            if olcum_tarihi:
                olcum_tarihi_clean = self.normalize_date_string(olcum_tarihi)
                rapor_tarihi_clean = self.normalize_date_string(rapor_tarihi)
                
                olcum_date = datetime.strptime(olcum_tarihi_clean, '%d/%m/%Y')
                rapor_date = datetime.strptime(rapor_tarihi_clean, '%d/%m/%Y')
                
                tarih_farki = (rapor_date - olcum_date).days
                
                is_valid = True
                status_message = f"Ölçüm: {olcum_tarihi_clean}, Rapor: {rapor_tarihi_clean}, Fark: {tarih_farki} gün (GEÇERLİ)"
                
                return is_valid, olcum_tarihi_clean, rapor_tarihi_clean, status_message
            else:
                return True, "Bulunamadı", rapor_tarihi, "Ölçüm tarihi bulunamadı ama tarih kısıtlaması yok"
                
        except ValueError as e:
            logger.error(f"Tarih parse hatası: {e}")
            return True, olcum_tarihi or "Bulunamadı", rapor_tarihi, f"Tarih formatı hatası ama tarih kısıtlaması yok: {e}"
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, GroundingAnalysisResult]:
        """Belirli kategori kriterlerini analiz etme"""
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            reverse_logic = criterion_data.get("reverse_logic", False)
            
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                if reverse_logic:
                    content = f"Uygunsuzluk tespit edildi: {str(matches[:3])}"
                    found = True
                    score = weight // 3
                else:
                    content = str(matches[0]) if len(matches) == 1 else str(matches)
                    found = True
                    score = weight
            else:
                if reverse_logic:
                    content = "Uygunsuzluk bulunamadı - Tüm ölçümler uygun"
                    found = True
                    score = weight
                else:
                    general_patterns = {
                        "proje_adi_numarasi": r"(C\d+\.\d+|Proje|Project|SM\s*\d+)",
                        "tesis_bolge_hat": r"(Tesis|Makine|Hat|Bölge|Line)",
                        "olcum_cihazi": r"(Multimetre|Ohmmetre|Ölçüm|Cihaz)",
                        "kalibrasyon": r"(Kalibrasyon|Kalibre|Cert|Sertifika)",
                        "standartlar": r"(EN\s*60204|IEC\s*60364|Standard|Standart)",
                        "rlo_degeri": r"(\d+[.,]?\d*\s*(?:mΩ|mohm|ohm))",
                        "uygunluk_durumu": r"(UYGUN|OK|NOK|Uygun|Değil)",
                        "risk_belirtme": r"(Risk|Tehlike|Uygunsuz|Problem)",
                        "genel_uygunluk": r"(Sonuç|Result|Uygun|Geçer|Pass|Fail)"
                    }
                    
                    general_pattern = general_patterns.get(criterion_name)
                    if general_pattern:
                        general_matches = re.findall(general_pattern, text, re.IGNORECASE)
                        if general_matches:
                            content = f"Genel eşleşme bulundu: {general_matches[0]}"
                            found = True
                            score = weight // 2
                        else:
                            content = "Bulunamadı"
                            found = False
                            score = 0
                    else:
                        content = "Bulunamadı"
                        found = False
                        score = 0
            
            results[criterion_name] = GroundingAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_found": len(matches) if matches else 0}
            )
        
        return results
    
    def extract_specific_values(self, text: str, file_path: str = None) -> Dict[str, Any]:
        """Spesifik değerleri çıkarma"""
        values = {}
        
        if file_path:
            filename = os.path.basename(file_path)
            
            proje_patterns = [
                r'(C\d{2}\.\d{3})', r'(E\d{2}\.\d{3})', r'(T\d{2,3}[-\.]?\d{3,4})',
                r'(\d{4,6})', r'([A-Z]\d{2,3}[.-]\d{3,4})'
            ]
            
            proje_no = "Bulunamadı"
            for pattern in proje_patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    proje_no = match.group(1)
                    break
            values["proje_no"] = proje_no
        
        value_patterns = {
            "rapor_numarasi": [
                r"(?:Report\s*No|Rapor\s*No|Report\s*Number)\s*[:=]\s*([A-Z0-9.-]+)",
                r"(E\d{2}\.\d{3})", r"(C\d{2}\.\d{3})", r"SM\s*(\d+)"
            ],
            "olcum_cihazi": [
                r"(?:Measuring\s*Instrument|Ölçüm\s*Cihaz[ıi])\s*[:=]\s*([^\n\r]+)",
                r"(Fluke\s*\d+[A-Z]*)", r"(Metrix\s*[A-Z0-9]+)"
            ]
        }
        
        for key, pattern_list in value_patterns.items():
            if key not in values:
                found_value = "Bulunamadı"
                for pattern in pattern_list:
                    matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if matches:
                        if isinstance(matches[0], tuple):
                            value = [m for m in matches[0] if m.strip()]
                            if value:
                                found_value = value[0].strip()
                                break
                        else:
                            found_value = matches[0].strip()
                            break
                values[key] = found_value
        
        self.analyze_measurement_data(text, values)
        return values
    
    def analyze_measurement_data(self, text: str, values: Dict[str, Any]):
        """Ölçüm verilerini analiz et"""
        rlo_patterns = [
            r"(\d+[.,]?\d*)\s*(?:mΩ|mohm|ohm|Ω)",
            r"(\d+)\s*(?:4x[2-9](?:[.,]\d+)?|4x4)\s*(?:[2-9](?:[.,]\d+)?|4)\s*500"
        ]
        
        rlo_values = []
        for pattern in rlo_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value_str = str(match).replace(',', '.')
                    rlo_values.append(float(value_str))
                except:
                    continue
        
        if rlo_values:
            values["rlo_min"] = f"{min(rlo_values):.1f} mΩ"
            values["rlo_max"] = f"{max(rlo_values):.1f} mΩ"
            values["rlo_ortalama"] = f"{sum(rlo_values)/len(rlo_values):.1f} mΩ"
        else:
            values["rlo_min"] = "Bulunamadı"
            values["rlo_max"] = "Bulunamadı"
            values["rlo_ortalama"] = "Bulunamadı"
        
        kesit_patterns = [r"4x4", r"4x2[.,]5", r"4x6", r"4x10"]
        total_kesit_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in kesit_patterns)
        values["toplam_olcum_nokta"] = total_kesit_count
        
        uygun_matches = re.findall(r"UYGUNUYGUN", text)
        values["uygun_nokta_sayisi"] = len(uygun_matches)
        
        if len(uygun_matches) == values["toplam_olcum_nokta"] and values["toplam_olcum_nokta"] > 0:
            values["genel_sonuc"] = "TÜM NOKTALAR UYGUN"
        else:
            values["genel_sonuc"] = f"{values['toplam_olcum_nokta'] - len(uygun_matches)} NOKTA UYGUNSUZ"
    
    def calculate_scores(self, analysis_results: Dict[str, Dict[str, GroundingAnalysisResult]]) -> Dict[str, Any]:
        """Puanları hesaplama"""
        category_scores = {}
        total_score = 0
        total_max_score = 100
        
        for category, results in analysis_results.items():
            category_max = self.criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())
            
            normalized_score = (category_earned / category_possible * category_max) if category_possible > 0 else 0
            
            category_scores[category] = {
                "earned": category_earned,
                "possible": category_possible,
                "normalized": round(normalized_score, 2),
                "max_weight": category_max,
                "percentage": round((category_earned / category_possible * 100), 2) if category_possible > 0 else 0
            }
            
            total_score += normalized_score
        
        return {
            "category_scores": category_scores,
            "total_score": round(total_score, 2),
            "total_max_score": total_max_score,
            "overall_percentage": round((total_score / total_max_score * 100), 2)
        }
    
    def generate_recommendations(self, analysis_results: Dict, scores: Dict, date_valid: bool) -> List[str]:
        """Öneriler oluşturma"""
        recommendations = []
        
        if not date_valid:
            recommendations.append("ℹ️ BİLGİ: Tarih bilgilerinde eksiklik veya format hatası var")
            recommendations.append("- Bu durum artık rapor geçerliliğini etkilemez")
        
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 50:
                recommendations.append(f"❌ {category} bölümü yetersiz (%{category_score:.1f})")
                missing_criteria = [name for name, result in results.items() if not result.found]
                if missing_criteria:
                    recommendations.append(f"  Eksik kriterler: {', '.join(missing_criteria)}")
            elif category_score < 80:
                recommendations.append(f"⚠️ {category} bölümü geliştirilmeli (%{category_score:.1f})")
            else:
                recommendations.append(f"✅ {category} bölümü yeterli (%{category_score:.1f})")
        
        if scores["overall_percentage"] < 70:
            recommendations.append("\n🚨 GENEL ÖNERİLER:")
            recommendations.append("- Rapor EN 60204-1 standardına tam uyumlu hale getirilmelidir")
            recommendations.append("- Eksik bilgiler tamamlanmalıdır")
        
        if scores["overall_percentage"] >= 70 and date_valid:
            recommendations.append("\n✅ RAPOR BAŞARILI")
            recommendations.append("- Tüm gerekli kriterler sağlanmıştır")
        
        return recommendations
    
    def generate_detailed_report(self, file_path: str) -> Dict[str, Any]:
        """Detaylı rapor oluşturma"""
        logger.info("Topraklama Süreklilik rapor analizi başlatılıyor...")
        
        text = self.extract_text_from_pdf(file_path)
        if not text:
            return {"error": "Dosya okunamadı"}
        
        detected_language = self.detect_language(text)
        language_name = self.get_language_name(detected_language)
        logger.info(f"📖 Belge dili: {language_name}")
        
        date_valid, olcum_tarihi, rapor_tarihi, date_message = self.check_date_validity(text, file_path)
        
        extracted_values = self.extract_specific_values(text, file_path)
        extracted_values['detected_language'] = detected_language
        extracted_values['language_name'] = language_name
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category)
        
        scores = self.calculate_scores(analysis_results)
        
        final_status = "PASSED" if scores["overall_percentage"] >= 70 else "FAILED"
        fail_reason = f"Toplam puan yetersiz (%{scores['overall_percentage']:.1f} < 70)" if final_status == "FAILED" else None
        
        recommendations = self.generate_recommendations(analysis_results, scores, date_valid)
        
        report = {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgileri": {
                "file_path": file_path,
                "file_type": os.path.splitext(file_path)[1],
                "detected_language": detected_language
            },
            "tarih_gecerliligi": {
                "gecerli": date_valid,
                "olcum_tarihi": olcum_tarihi,
                "rapor_tarihi": rapor_tarihi,
                "mesaj": date_message
            },
            "cikarilan_degerler": extracted_values,
            "kategori_analizleri": analysis_results,
            "puanlama": scores,
            "oneriler": recommendations,
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["overall_percentage"],
                "final_durum": final_status,
                "tarih_durumu": "BİLGİ AMAÇLI" if not date_valid else "GEÇERLİ",
                "gecme_durumu": "PASSED" if final_status == "PASSED" else "FAILED",
                "fail_nedeni": fail_reason
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
    """Server kodunda doküman validasyonu - LVD için"""
    
    critical_terms = [
        ["süreklilik", "continuity", "iletkenler", "conductors", "bağlantı", "connection", "devamlılık"],
        ["kesit", "cross section", "iletken kesiti", "conductor cross section", "mm2", "mm²", "kesit uygunluğu"],
        ["lvd", "en 60204-1", "60204-1", "60204", "makine güvenliği", "machine safety"],
        ["ölçüm", "measurement", "test", "ohm", "direnç", "resistance", "milliohm", "mΩ"],
        ["elektrik güvenliği", "electrical safety", "koruyucu iletken", "protective conductor", "pe", "protective earth"]
    ]
    
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"LVD Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    valid_categories = sum(category_found)
    logger.info(f"Doküman validasyonu: {valid_categories}/5 kritik kategori bulundu")
    
    return valid_categories >= 4


def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - LVD için"""
    strong_keywords = [
        "lvd", "TOPRAKLAMA SÜREKLİLİK", "topraklama süreklilik",
        "TOPRAKLAMA İLETKENLERİ", "TOPRAKLAMA ILETKENLERI",
        "topraklama iletkenleri", "topraklama sureklilik",
        "KESİT UYGUNLUĞU", "kesit uygunlugu", "kesit uygunluğu"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        
        found_keywords = [kw for kw in strong_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa kontrol: {len(found_keywords)} özgü kelime: {found_keywords}")
        return len(found_keywords) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa kontrol hatası: {e}")
        return False


def check_excluded_keywords_first_pages(filepath):
    """İlk 1-2 sayfada istenmeyen rapor türlerinin kelimelerini ara"""
    excluded_keywords = [
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen",
        "hidrolik", "HİDROLİK", "hydraulic",
        "pnömatik", "pnomatik", "pneumatic",
        "isg", "periyodik", "kontrol",
        "uygunluk", "beyan", "conformity", "declaration",
        "hrc", "cobot", "robot",
        "elektrik", "devre", "şema", "circuit",
        "espe", "gürültü", "noise", "ses",
        "kullanma", "kılavuz", "manual",
        "loto", "montaj", "assembly",
        "topraklama direnci", "grounding", "earthing",
        "bakım", "maintenance", "bakim",
        "titreşim", "vibration",
        "AT TİP", "at tip", "certificate"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=2)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        
        logger.info(f"OCR okunan text: {all_text[:500]}...")
        
        found_excluded = [kw for kw in excluded_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa excluded kontrol: {len(found_excluded)} istenmeyen kelime: {found_excluded}")
        return len(found_excluded) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa excluded kontrol hatası: {e}")
        return False


def get_conclusion_message_lvd(status, percentage):
    """Sonuç mesajını döndür - LVD için"""
    if status == "PASS":
        return f"LVD topraklama raporu EN 60204-1 standardına uygun (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"LVD topraklama raporu kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"LVD topraklama raporu standartlara uygun değil (%{percentage:.0f})"


def get_main_issues_lvd(report):
    """Ana sorunları listele - LVD için"""
    issues = []
    
    for category, score_data in report['puanlama']['category_scores'].items():
        if isinstance(score_data, dict) and score_data.get('percentage', 0) < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    if not issues and report['ozet']['toplam_puan'] < 50:
        issues = [
            "Topraklama süreklilik ölçüm sonuçları eksik",
            "EN 60204-1 standart referansları yetersiz",
            "Ölçüm cihazı kalibrasyon bilgileri eksik",
            "Uygunluk değerlendirmesi yapılmamış"
        ]
    
    return issues[:4]


# ============================================
# FLASK SERVİS KATMANI - CONFIGURATION
# ============================================
app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads_lvd'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================
# FLASK SERVİS KATMANI - API ENDPOINTS
# ============================================
@app.route('/api/lvd-report', methods=['POST'])
def analyze_lvd_report():
    """LVD Topraklama Süreklilik raporu analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere bir LVD topraklama raporu sağlayın'
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected', 'message': 'Lütfen bir dosya seçin'}), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Sadece PDF, JPG, JPEG ve PNG dosyaları kabul edilir'
            }), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"LVD raporu kontrol ediliyor: {filename}")

            analyzer = GroundingContinuityReportAnalyzer()
            
            logger.info(f"Üç aşamalı LVD kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                logger.info("Aşama 1: İlk sayfa LVD özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath):
                    logger.info("✅ Aşama 1 geçti - LVD özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - LVD değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya LVD topraklama raporu değil (farklı rapor türü tespit edildi).',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'LVD_TOPRAKLAMA_RAPORU'
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
                                    'message': 'Yüklediğiniz dosya LVD topraklama raporu değil!',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_LVD_REPORT',
                                        'required_type': 'LVD_TOPRAKLAMA_RAPORU'
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
                if not tesseract_available:
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    
                    return jsonify({
                        'error': 'OCR not available',
                        'message': 'Tesseract OCR kurulu değil. Resim dosyalarını analiz edebilmek için Tesseract kurulumu gereklidir.',
                        'details': {'tesseract_error': tesseract_info, 'file_type': file_ext}
                    }), 500

            logger.info(f"LVD topraklama raporu doğrulandı, analiz başlatılıyor: {filename}")
            report = analyzer.generate_detailed_report(filepath)
            
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
                    'found_criteria': len([c for c in report['puanlama']['category_scores'].values() if isinstance(c, dict) and c.get('score', 0) > 0]),
                    'total_criteria': len(report['puanlama']['category_scores']),
                    'percentage': round(overall_percentage, 1)
                },
                'analysis_id': f"lvd_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'date_validity': {
                    'is_valid': report['tarih_gecerliligi']['gecerli'],
                    'measurement_date': report['tarih_gecerliligi']['olcum_tarihi'],
                    'report_date': report['tarih_gecerliligi']['rapor_tarihi'],
                    'message': report['tarih_gecerliligi']['mesaj']
                },
                'extracted_values': report['cikarilan_degerler'],
                'file_type': 'LVD_TOPRAKLAMA_RAPORU',
                'filename': filename,
                'language_info': {
                    'detected_language': report['dosya_bilgileri'].get('detected_language', 'turkish').upper(),
                    'file_type': file_ext.replace('.', '')
                },
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': report['ozet']['toplam_puan'],
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ',
                    'text_quality': 'good' if len(report.get('cikarilan_degerler', {})) > 3 else 'fair'
                },
                'recommendations': report['oneriler'],
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_lvd(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_lvd(report)
                }
            }
            
            for category, score_data in report['puanlama']['category_scores'].items():
                if isinstance(score_data, dict):
                    response_data['category_scores'][category] = {
                        'score': score_data.get('normalized', score_data.get('score', 0)),
                        'max_score': score_data.get('max_weight', score_data.get('max_score', 0)),
                        'percentage': score_data.get('percentage', 0),
                        'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'CONDITIONAL' if score_data.get('percentage', 0) >= 50 else 'FAIL'
                    }

            return jsonify({
                'success': True,
                'message': 'LVD Topraklama Raporu başarıyla analiz edildi',
                'analysis_service': 'lvd',
                'service_description': 'LVD Topraklama Rapor Analizi',
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
                'message': f'LVD topraklama raporu analizi sırasında hata oluştu: {str(analysis_error)}',
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
    """Health check endpoint - LVD için"""
    return jsonify({
        'status': 'healthy',
        'service': 'LVD Report Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'report_type': 'LVD_TOPRAKLAMA_RAPORU',
        'standard': 'EN 60204-1'
    })


@app.route('/', methods=['GET'])
def index():
    """Ana sayfa - API bilgileri - LVD için"""
    return jsonify({
        'service': 'LVD Topraklama Süreklilik Rapor Analyzer API',
        'version': '1.0.0',
        'description': 'LVD Topraklama Süreklilik Raporlarını analiz eden REST API servisi',
        'endpoints': {
            'POST /api/lvd-report': 'LVD raporu analizi',
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
            'PASS': '≥70% - EN 60204-1 standardına uygun',
            'CONDITIONAL': '50-69% - Kabul edilebilir',
            'FAIL': '<50% - Standarda uygun değil'
        }
    })


# ============================================
# APPLICATION ENTRY POINT (Azure-Friendly)
# ============================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("LVD Topraklama Süreklilik Rapor Analiz Servisi")
    logger.info("=" * 60)
    # ... logging ...
    
    port = int(os.environ.get('PORT', 8007))
    
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )