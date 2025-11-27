import re
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import PyPDF2
from dataclasses import dataclass, asdict
import logging
import cv2
import numpy as np
from PIL import Image
import pdf2image
import pytesseract
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Azure için Tesseract ve Poppler path'leri (Azure'da otomatik bulunur)
# Lokal test için path'ler gerekirse environment variable'dan alınabilir
"""try:
    pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD', 'tesseract')
except:
    pass"""

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
    """Topraklama Süreklilik analiz sonucu"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

class GroundingContinuityReportAnalyzer:
    """Topraklama Süreklilik rapor analiz sınıfı"""
    
    def __init__(self):
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

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkarma"""
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
    
    def normalize_date_string(self, date_str: str) -> str:
        """Tarih string'ini DD/MM/YYYY formatına çevir"""
        if not date_str or date_str == "Bulunamadı":
            return date_str
            
        # Ay isimleri çeviri tablosu
        month_names = {
            # İngilizce ay isimleri
            'jan': '01', 'january': '01',
            'feb': '02', 'february': '02', 
            'mar': '03', 'march': '03',
            'apr': '04', 'april': '04',
            'may': '05',
            'jun': '06', 'june': '06',
            'jul': '07', 'july': '07',
            'aug': '08', 'august': '08',
            'sep': '09', 'september': '09',
            'oct': '10', 'october': '10',
            'nov': '11', 'november': '11',
            'dec': '12', 'december': '12',
            
            # Türkçe ay isimleri
            'ocak': '01',
            'şubat': '02', 'subat': '02',
            'mart': '03',
            'nisan': '04',
            'mayıs': '05', 'mayis': '05',
            'haziran': '06',
            'temmuz': '07',
            'ağustos': '08', 'agustos': '08',
            'eylül': '09', 'eylul': '09',
            'ekim': '10',
            'kasım': '11', 'kasim': '11',
            'aralık': '12', 'aralik': '12'
        }
        
        # Çeşitli tarih formatlarını normalize et
        date_str = date_str.strip()
        
        # DD/MM/YYYY veya DD.MM.YYYY veya DD-MM-YYYY formatları
        if re.match(r'\d{1,2}[./\-]\d{1,2}[./\-]\d{4}', date_str):
            return date_str.replace('.', '/').replace('-', '/')
        
        # YYYY/MM/DD formatı
        if re.match(r'\d{4}[./\-]\d{1,2}[./\-]\d{1,2}', date_str):
            parts = re.split(r'[./\-]', date_str)
            return f"{parts[2].zfill(2)}/{parts[1].zfill(2)}/{parts[0]}"
        
        # DD Month YYYY formatı (örn: "18 Apr 2023" veya "18 Nisan 2023")
        month_pattern = r'(\d{1,2})\s+([a-zA-ZğıüşçöĞIÜŞÇÖ]+)\s+(\d{4})'
        match = re.match(month_pattern, date_str, re.IGNORECASE)
        if match:
            day, month_name, year = match.groups()
            month_name_lower = month_name.lower()
            if month_name_lower in month_names:
                month_num = month_names[month_name_lower]
                return f"{day.zfill(2)}/{month_num}/{year}"
        
        # Eğer hiçbir format eşleşmezse orijinal string'i döndür
        return date_str.replace('.', '/').replace('-', '/')
    
    def check_date_validity(self, text: str, file_path: str = None) -> Tuple[bool, str, str, str]:
        """Tarih bilgilerini çıkar - sadece tarih tespiti için (1 yıl kısıtlaması kaldırıldı)"""
        
        # Ölçüm tarihi arama - çok kapsamlı pattern'lar
        olcum_patterns = [
            # Türkçe formatlar
            r"(?:Ölçüm\s*Tarihi|Test\s*Tarihi|Ölçüm\s*Yapıldığı\s*Tarih)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Ölçüm|Test).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4}).*?(?:ölçüm|test)",
            
            # İngilizce formatlar
            r"(?:Measurement\s*Date|Test\s*Date|Date\s*of\s*(?:Test|Measurement))\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Measured\s*on|Tested\s*on)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4}).*?(?:measurement|test)",
            
            # Genel formatlar
            r"(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})",
            r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
            r"(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})"
        ]
        
        # Rapor tarihi arama - çok kapsamlı pattern'lar
        rapor_patterns = [
            # Türkçe formatlar
            r"(?:Rapor\s*Tarihi|Belge\s*Tarihi|Hazırlanma\s*Tarihi)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Rapor|Belge).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Hazırlayan|Hazırlandı)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            
            # İngilizce formatlar
            r"(?:Report\s*Date|Document\s*Date|Issue\s*Date|Prepared\s*Date)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Prepared\s*on|Issued\s*on|Created\s*on)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            
            # Genel formatlar
            r"(?:Date|Tarih)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})",
            r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
            r"(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})"
        ]
        
        olcum_tarihi = None
        rapor_tarihi = None
        
        # Ölçüm tarihini bul
        for pattern in olcum_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                olcum_tarihi = matches[0]
                break
        
        # Rapor tarihini bul
        for pattern in rapor_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                rapor_tarihi = matches[0]
                break
        
        # Eğer tarihler bulunamazsa dosya modifikasyon tarihini kullan
        if not rapor_tarihi and file_path and os.path.exists(file_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            rapor_tarihi = file_mod_time.strftime("%d/%m/%Y")
        elif not rapor_tarihi:
            rapor_tarihi = datetime.now().strftime("%d/%m/%Y")
        
        try:
            if olcum_tarihi:
                # Tarih formatlarını normalize et ve ay isimlerini çevir
                olcum_tarihi_clean = self.normalize_date_string(olcum_tarihi)
                rapor_tarihi_clean = self.normalize_date_string(rapor_tarihi)
                
                olcum_date = datetime.strptime(olcum_tarihi_clean, '%d/%m/%Y')
                rapor_date = datetime.strptime(rapor_tarihi_clean, '%d/%m/%Y')
                
                # Tarih farkını hesapla (bilgi amaçlı)
                tarih_farki = (rapor_date - olcum_date).days
                
                # 1 yıl koşulu kaldırıldı - her zaman geçerli
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
                    # Uygunsuzluk bulundu - düşük puan
                    content = f"Uygunsuzluk tespit edildi: {str(matches[:3])}"
                    found = True
                    score = weight // 3
                else:
                    content = str(matches[0]) if len(matches) == 1 else str(matches)
                    found = True
                    score = weight
            else:
                if reverse_logic:
                    # Uygunsuzluk bulunamadı - tam puan (iyi bir şey)
                    content = "Uygunsuzluk bulunamadı - Tüm ölçümler uygun"
                    found = True
                    score = weight
                else:
                    # İkincil arama - daha genel pattern
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
        """Spesifik değerleri çıkarma - Dosya adından da bilgi çıkar"""
        values = {}
        
        # Önce dosya adından bilgileri çıkar
        if file_path:
            filename = os.path.basename(file_path)
            
            # Proje numarası pattern'leri - farklı formatlar için
            proje_patterns = [
                r'(C\d{2}\.\d{3})',
                r'(E\d{2}\.\d{3})',
                r'(T\d{2,3}[-\.]?\d{3,4})',
                r'(\d{4,6})',
                r'([A-Z]\d{2,3}[.-]\d{3,4})'
            ]
            
            # Rapor numarası pattern'leri
            rapor_patterns = [
                r'SM\s*(\d+)',
                r'MCC(\d+)',
                r'Report\s*No[\s:]*([A-Z0-9.-]+)',
                r'Rapor[\s:]*([A-Z0-9.-]+)'
            ]
            
            # Müşteri/firma bilgisi
            musteri_patterns = [
                r'Toyota',
                r'DANONE',
                r'Ford',
                r'BOSCH',
                r'P&G'
            ]
            
            # Dosya adından proje no çıkar
            proje_no = "Bulunamadı"
            for pattern in proje_patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    proje_no = match.group(1)
                    break
            values["proje_no"] = proje_no
            
            # Dosya adından rapor numarası çıkar
            rapor_no = "Bulunamadı"
            for pattern in rapor_patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    rapor_no = match.group(1)
                    break
            values["rapor_numarasi"] = rapor_no
            
            # Müşteri bilgisi
            musteri = "Bulunamadı"
            for pattern in musteri_patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    musteri = pattern
                    break
            values["musteri"] = musteri
            
            # Revizyon bilgisi
            revizyon_match = re.search(r'[vV](\d+)', filename)
            values["revizyon"] = f"v{revizyon_match.group(1)}" if revizyon_match else "v0"
        
        # Önemli değerler için pattern'ler - çok daha kapsamlı
        value_patterns = {
            # Proje adı/numarası için kapsamlı pattern'ler
            "proje_adi": [
                r"(?:Project\s*Name|Proje\s*Ad[ıi])\s*[:=]\s*([^\n\r]+)",
                r"(?:Project\s*No|Proje\s*No|Project\s*Number)\s*[:=]\s*([A-Z0-9.-]+)",
                r"(?:Report\s*Title|Rapor\s*Başl[ıi]ğ[ıi])\s*[:=]\s*([^\n\r]+)",
                r"(?:Document\s*Title|Belge\s*Başl[ıi]ğ[ıi])\s*[:=]\s*([^\n\r]+)",
                r"(LVD\s+[Öö]lç[üu]m[^,\n]*)",
                r"(Topraklama\s+S[üu]reklilik[^,\n]*)",
                r"(Grounding\s+Continuity[^,\n]*)",
                r"([A-Z][a-z]+\s*-\s*[A-Z][a-z]+.*?[Öö]lç[üu]m)",
                r"(E\d{2}\.\d{3}\s*-[^,\n]+)"
            ],
            
            # Rapor numarası için kapsamlı pattern'ler
            "rapor_numarasi": [
                r"(?:Report\s*No|Rapor\s*No|Report\s*Number)\s*[:=]\s*([A-Z0-9.-]+)",
                r"(?:Document\s*No|Belge\s*No)\s*[:=]\s*([A-Z0-9.-]+)",
                r"(E\d{2}\.\d{3})",
                r"(C\d{2}\.\d{3})",
                r"(T\d{2,3}[-.]?\d{3,4})",
                r"SM\s*(\d+)",
                r"MCC(\d+)",
                r"^\s*([A-Z]\d{2,3}[.-]\d{3,4})"
            ],
            
            # Ölçüm cihazı için çok kapsamlı pattern'ler
            "olcum_cihazi": [
                r"(?:Measuring\s*Instrument|Ölçüm\s*Cihaz[ıi]|Test\s*Equipment)\s*[:=]\s*([^\n\r]+)",
                r"(?:Multimeter|Multimetre|Ohmmeter|Ohmmetre)\s*[:=]?\s*([A-Z0-9\s.-]+)",
                r"(?:Instrument|Cihaz)\s*[:=]\s*([^\n\r]+)",
                r"(?:Equipment|Ekipman)\s*[:=]\s*([^\n\r]+)",
                r"(?:Device|Alet)\s*[:=]\s*([^\n\r]+)",
                r"(?:Tester|Test\s*Cihaz[ıi])\s*[:=]?\s*([A-Z0-9\s.-]+)",
                r"(Fluke\s*\d+[A-Z]*)",
                r"(Metrix\s*[A-Z0-9]+)",
                r"(Chauvin\s*Arnoux\s*[A-Z0-9]+)",
                r"(Megger\s*[A-Z0-9]+)",
                r"(Hioki\s*[A-Z0-9]+)",
                r"([A-Z][a-z]+\s*\d+[A-Z]*)",
                r"(MΩ\s*metre|mΩ\s*metre|Loop\s*Tester|Continuity\s*Tester)"
            ],
            
            # Tesis/müşteri bilgisi
            "tesis_adi": [
                r"(?:Customer|Müşteri|Client)\s*[:=]\s*([^\n\r]+)",
                r"(?:Facility|Tesis|Plant|Factory)\s*[:=]\s*([^\n\r]+)",
                r"(?:Company|Firma|Corporation)\s*[:=]\s*([^\n\r]+)",
                r"(Toyota[^\n\r]*)",
                r"(DANONE[^\n\r]*)",
                r"(Ford[^\n\r]*)",
                r"(BOSCH[^\n\r]*)",
                r"(?:8X45|8X50|8X9J|9J73)\s*(?:R1|R2|R3)?\s*Hatt[ıi]",
                r"([A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Factory|Plant|Facility))"
            ],
            
            # Firma/personel bilgisi
            "firma_personel": [
                r"(?:Prepared\s*by|Hazırlayan|Consultant)\s*[:=]\s*([^\n\r]+)",
                r"(?:Performed\s*by|Ölçümü\s*Yapan)\s*[:=]\s*([^\n\r]+)",
                r"(?:Company|Firma)\s*[:=]\s*([^\n\r]+)",
                r"(?:Engineer|Mühendis)\s*[:=]\s*([^\n\r]+)",
                r"(PILZ[^\n\r]*)",
                r"([A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Engineering|Mühendislik))"
            ],
            
            # Tarih pattern'leri - çok kapsamlı
            "olcum_tarihi": [
                # Türkçe formatlar
                r"(?:Ölçüm\s*Tarihi|Test\s*Tarihi|Ölçüm\s*Yapıldığı\s*Tarih)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Ölçüm|Test).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"Tarih\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                
                # İngilizce formatlar
                r"(?:Measurement\s*Date|Test\s*Date|Date\s*of\s*(?:Test|Measurement))\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Measured\s*on|Tested\s*on)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Date|When)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                
                # Çeşitli formatlar
                r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})\s*(?:tarihinde|on|at|de)",
                r"(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})",
                r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
                r"(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})",
                
                # Tablo içindeki tarihler
                r"(?:Measurement|Ölçüm).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            ],
            
            "rapor_tarihi": [
                # Türkçe formatlar
                r"(?:Rapor\s*Tarihi|Belge\s*Tarihi|Hazırlanma\s*Tarihi)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Rapor|Belge).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Hazırlayan|Hazırlandı)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                
                # İngilizce formatlar  
                r"(?:Report\s*Date|Document\s*Date|Issue\s*Date|Prepared\s*Date)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Prepared\s*on|Issued\s*on|Created\s*on)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Report|Document).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                
                # Çeşitli formatlar
                r"(?:Date|Tarih)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})",
                r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
                r"(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})",
                
                # Tablo başlığı veya footer'daki tarihler
                r"(?:Created|Issued|Published)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            ]
        }
        
        # Metinden değerleri çıkar - her pattern listesi için
        for key, pattern_list in value_patterns.items():
            if key not in values:
                found_value = "Bulunamadı"
                
                # Pattern listesinde her pattern'i dene
                for pattern in pattern_list:
                    matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if matches:
                        if isinstance(matches[0], tuple):
                            # Tuple içindeki boş olmayan ilk değeri al
                            value = [m for m in matches[0] if m.strip()]
                            if value:
                                found_value = value[0].strip()
                                break
                        else:
                            found_value = matches[0].strip()
                            break
                
                values[key] = found_value
        
        # Ölçüm verilerini analiz et
        self.analyze_measurement_data(text, values)
        
        return values
    
    def analyze_measurement_data(self, text: str, values: Dict[str, Any]):
        """Ölçüm verilerini analiz et"""
        # RLO değerlerini topla - daha geniş pattern
        rlo_patterns = [
            r"(\d+[.,]?\d*)\s*(?:mΩ|mohm|ohm|Ω)",
            r"(\d+)\s*(?:4x[2-9](?:[.,]\d+)?|4x4)\s*(?:[2-9](?:[.,]\d+)?|4)\s*500",
            r"(\d+)\s*(?:mΩ|mohm|ohm|Ω)"
        ]
        
        rlo_values = []
        for pattern in rlo_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Virgülü noktaya çevir ve sayıya çevir
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
        
        # Kesit bilgilerini analiz et - daha geniş pattern
        kesit_patterns = [
            r"4x4",
            r"4x2[.,]5", 
            r"4x6",
            r"4x10",
            r"Yük\s*İletken",
            r"Load\s*Conductor",
            r"PE\s*İletken",
            r"PE\s*Conductor"
        ]
        
        total_kesit_count = 0
        for pattern in kesit_patterns:
            count = len(re.findall(pattern, text, re.IGNORECASE))
            total_kesit_count += count
        
        values["toplam_olcum_nokta"] = total_kesit_count
        
        # Uygunluk durumlarını say
        uygun_pattern = r"UYGUNUYGUN"
        uygun_matches = re.findall(uygun_pattern, text)
        values["uygun_nokta_sayisi"] = len(uygun_matches)
        
        # Uygunsuz ölçümleri tespit et
        self.find_non_compliant_measurements(text, values)
        
        # Genel sonuç
        if len(uygun_matches) == values["toplam_olcum_nokta"] and values["toplam_olcum_nokta"] > 0:
            values["genel_sonuc"] = "TÜM NOKTALAR UYGUN"
        else:
            values["genel_sonuc"] = f"{values['toplam_olcum_nokta'] - len(uygun_matches)} NOKTA UYGUNSUZ"
        
        # Hat/bölge bilgileri
        hat_pattern = r"(8X45|8X50|8X9J|9J73|8X52|8X60|8X62|8X70)\s*(?:R[1-9])?\s*Hatt[ıi]"
        hat_matches = re.findall(hat_pattern, text, re.IGNORECASE)
        if hat_matches:
            unique_hats = list(set(hat_matches))
            values["makine_hatlari"] = ", ".join(unique_hats)
        else:
            values["makine_hatlari"] = "Bulunamadı"
    
    def find_non_compliant_measurements(self, text: str, values: Dict[str, Any]):
        """Uygunsuz ölçümleri tespit et"""
        # 500 mΩ'dan büyük değerleri ve D.Y. değerlerini bul
        lines = text.split('\n')
        non_compliant = []
        
        for i, line in enumerate(lines):
            # Sıra numarası kontrolü
            sira_match = re.search(r'(\d+)\s', line)
            if sira_match:
                sira = sira_match.group(1)
                
                # Yüksek RLO değeri kontrolü (>500 mΩ) - daha geniş pattern
                high_rlo_patterns = [
                    r'(\d{3,4})\s*(?:4x[2-9](?:[.,]\d+)?|4x4)\s*(?:[2-9](?:[.,]\d+)?|4)\s*500(\d+)\s*mΩ\s*<\s*500\s*mΩ',
                    r'(\d{3,4})\s*(?:mΩ|mohm|ohm|Ω)',
                    r'(\d{3,4})[.,]?\d*\s*(?:mΩ|mohm|ohm|Ω)'
                ]
                
                for pattern in high_rlo_patterns:
                    high_rlo_match = re.search(pattern, line, re.IGNORECASE)
                    if high_rlo_match:
                        try:
                            rlo_value = float(str(high_rlo_match.group(1)).replace(',', '.'))
                            if rlo_value > 500:
                                # Hat ve ekipman bilgisi - daha geniş pattern
                                hat_patterns = [
                                    r'(8X\d+R?\d*)\s*(?:Hatt[ıi]|Line|Zone)?\s*(.*?)(?:\s+\d+)',
                                    r'(8X\d+R?\d*)\s*(.*?)(?:\s+\d+)',
                                    r'(Line\s*\d+|Zone\s*\d+)\s*(.*?)(?:\s+\d+)'
                                ]
                                
                                for hat_pattern in hat_patterns:
                                    hat_match = re.search(hat_pattern, line, re.IGNORECASE)
                                    if hat_match:
                                        hat = hat_match.group(1)
                                        ekipman = hat_match.group(2).strip()
                                        non_compliant.append({
                                            'sira': sira,
                                            'rlo': f"{rlo_value:.1f} mΩ",
                                            'hat': hat,
                                            'ekipman': ekipman,
                                            'durum': 'Yüksek Direnç'
                                        })
                                        break
                                break
                        except:
                            continue
                
                # D.Y. (Değer Yok) kontrolü - daha geniş pattern
                if '*D.Y' in line or 'D.Y' in line or 'N/A' in line or 'N/A' in line:
                    hat_patterns = [
                        r'(8X\d+R?\d*)\s*(?:Hatt[ıi]|Line|Zone)?\s*(.*?)(?:\s+|$)',
                        r'(8X\d+R?\d*)\s*(.*?)(?:\s+|$)',
                        r'(Line\s*\d+|Zone\s*\d+)\s*(.*?)(?:\s+|$)'
                    ]
                    
                    for hat_pattern in hat_patterns:
                        hat_match = re.search(hat_pattern, line, re.IGNORECASE)
                        if hat_match:
                            hat = hat_match.group(1)
                            ekipman = hat_match.group(2).strip()
                            non_compliant.append({
                                'sira': sira,
                                'rlo': 'D.Y.',
                                'hat': hat,
                                'ekipman': ekipman,
                                'durum': 'Ölçüm Yapılamadı'
                            })
                            break
        
        values["uygunsuz_olcumler"] = non_compliant
    
    def calculate_scores(self, analysis_results: Dict[str, Dict[str, GroundingAnalysisResult]]) -> Dict[str, Any]:
        """Puanları hesaplama"""
        category_scores = {}
        total_score = 0
        total_max_score = 100
        
        for category, results in analysis_results.items():
            category_max = self.criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())
            
            # Kategori puanını ağırlığa göre normalize et
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
    
    def generate_detailed_report(self, file_path: str) -> Dict[str, Any]:
        """Detaylı rapor oluşturma"""
        logger.info("Topraklama Süreklilik rapor analizi başlatılıyor...")
        
        # Dosyadan metin çıkar
        text = self.extract_text_from_pdf(file_path)
        if not text:
            return {"error": "Dosya okunamadı"}
        
        # Tarih geçerliliği kontrolü (1 yıl kuralı kaldırıldı)
        date_valid, olcum_tarihi, rapor_tarihi, date_message = self.check_date_validity(text, file_path)
        
        # Spesifik değerleri çıkar
        extracted_values = self.extract_specific_values(text, file_path)
        
        # Her kategori için analiz yap
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category)
        
        # Puanları hesapla
        scores = self.calculate_scores(analysis_results)
        
        # Final karar: Sadece puanla karar ver (tarih kısıtlaması kaldırıldı)
        final_status = "PASSED"
        if scores["overall_percentage"] < 70:
            final_status = "FAILED"
            fail_reason = f"Toplam puan yetersiz (%{scores['overall_percentage']:.1f} < 70)"
        else:
            fail_reason = None
        
        # Öneriler oluştur
        recommendations = self.generate_recommendations(analysis_results, scores, date_valid)
        
        report = {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgileri": {
                "file_path": file_path,
                "file_type": os.path.splitext(file_path)[1]
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
    
    def generate_recommendations(self, analysis_results: Dict, scores: Dict, date_valid: bool) -> List[str]:
        """Öneriler oluşturma"""
        recommendations = []
        
        # Tarih kontrolü bilgi amaçlı (artık kritik değil)
        if not date_valid:
            recommendations.append("ℹ️ BİLGİ: Tarih bilgilerinde eksiklik veya format hatası var")
            recommendations.append("- Bu durum artık rapor geçerliliğini etkilemez")
        
        # Kategori bazlı öneriler
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 50:
                recommendations.append(f"❌ {category} bölümü yetersiz (%{category_score:.1f})")
                
                # Eksik kriterler
                missing_criteria = [name for name, result in results.items() if not result.found]
                if missing_criteria:
                    recommendations.append(f"  Eksik kriterler: {', '.join(missing_criteria)}")
                
                # Kategori özel öneriler
                if category == "Genel Rapor Bilgileri":
                    recommendations.append("  - Proje adı ve numarası eksiksiz belirtilmelidir")
                    recommendations.append("  - Ölçüm ve rapor tarihleri açıkça belirtilmelidir")
                    recommendations.append("  - Rapor numarası ve revizyon bilgisi eklenmeli")
                
                elif category == "Ölçüm Metodu ve Standart Referansları":
                    recommendations.append("  - Ölçüm cihazı marka/model bilgileri eklenmeli")
                    recommendations.append("  - Kalibrasyon sertifikası bilgileri verilmeli")
                    recommendations.append("  - EN 60204-1 Tablo 10 referansı yapılmalı")
                
                elif category == "Ölçüm Sonuç Tablosu":
                    recommendations.append("  - Tüm ölçüm noktaları için RLO değerleri belirtilmeli")
                    recommendations.append("  - Yük ve PE iletken kesitleri girilmeli")
                    recommendations.append("  - EN 60204 Tablo 10 referans değerleri eklenmeli")
                    recommendations.append("  - Uygunluk durumu her nokta için belirtilmeli")
                
                elif category == "Uygunluk Değerlendirmesi":
                    recommendations.append("⚠️ Uygunsuz noktalar için teknik açıklama ekleyin")
                    recommendations.append("📊 Toplam ölçüm sayısı ve uygunluk oranını belirtin")
                    recommendations.append("🔍 500 mΩ limit değeri aşımlarını işaretleyin")
                
                elif category == "Görsel ve Teknik Dökümantasyon":
                    recommendations.append("  - Ölçüm yapılan alan fotoğrafları eklenmeli")
                    recommendations.append("  - Ölçüm cihazı ve bağlantı fotoğrafları çekilmeli")
                    recommendations.append("  - Ölçüm noktalarının kroki/şeması hazırlanmalı")
                
                elif category == "Sonuç ve Öneriler":
                    recommendations.append("  - Genel uygunluk sonucu açıkça belirtilmeli")
                    recommendations.append("  - Standartlara atıf yapılmalı")
                    recommendations.append("  - İyileştirme önerileri detaylandırılmalı")
                    recommendations.append("  - Tekrar ölçüm periyodu önerilmeli")
            
            elif category_score < 80:
                recommendations.append(f"⚠️ {category} bölümü geliştirilmeli (%{category_score:.1f})")
            
            else:
                recommendations.append(f"✅ {category} bölümü yeterli (%{category_score:.1f})")
        
        # Genel öneriler
        if scores["overall_percentage"] < 70:
            recommendations.append("\n🚨 GENEL ÖNERİLER:")
            recommendations.append("- Rapor EN 60204-1 standardına tam uyumlu hale getirilmelidir")
            recommendations.append("- IEC 60364 standart referansları eklenmeli")
            recommendations.append("- Eksik bilgiler tamamlanmalıdır")
            recommendations.append("- Ölçüm sonuçları tablo formatında düzenlenmeli")
        
        # Başarılı durumda
        if scores["overall_percentage"] >= 70 and date_valid:
            recommendations.append("\n✅ RAPOR BAŞARILI")
            recommendations.append("- Tüm gerekli kriterler sağlanmıştır")
            recommendations.append("- Rapor standarltara uygun olarak hazırlanmıştır")
        
        return recommendations

# ============================================================================
# FLASK SERVER FUNCTIONS - Server.py'den alınan kodlar
# ============================================================================

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

def validate_document_server(text):
    """Server kodunda doküman validasyonu - LVD için"""
    
    # LVD raporlarında MUTLAKA olması gereken kritik kelimeler
    critical_terms = [
        # Topraklama süreklilik temel terimleri
        ["süreklilik", "continuity", "iletkenler", "conductors", "bağlantı", "connection", "devamlılık"],
        
        # Kesit uygunluk terimleri
        ["kesit", "cross section", "iletken kesiti", "conductor cross section", "mm2", "mm²", "kesit uygunluğu"],
        
        # LVD ve standart terimleri
        ["lvd", "en 60204-1", "60204-1", "60204", "makine güvenliği", "machine safety"],
        
        # Ölçüm ve test terimleri
        ["ölçüm", "measurement", "test", "ohm", "direnç", "resistance", "milliohm", "mΩ"],
        
        # Elektrik güvenlik terimleri
        ["elektrik güvenliği", "electrical safety", "koruyucu iletken", "protective conductor", "pe", "protective earth"]
    ]
    
    # Her kategori için kontrol
    category_found = []
    
    for i, category in enumerate(critical_terms):
        found_in_category = False
        for term in category:
            if re.search(rf"\b{term}\b", text, re.IGNORECASE):
                found_in_category = True
                logger.info(f"LVD Kategori {i+1} bulundu: '{term}'")
                break
        category_found.append(found_in_category)
    
    # Tüm kategorilerden en az bir terim bulunmalı
    valid_categories = sum(category_found)
    
    logger.info(f"Doküman validasyonu: {valid_categories}/5 kritik kategori bulundu")
    
    # 5 kategorinin en az 4'ünde terim bulunmalı
    return valid_categories >= 4

def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara - LVD için"""
    strong_keywords = [
        "lvd",
        "TOPRAKLAMA SÜREKLİLİK", 
        "topraklama süreklilik",
        "TOPRAKLAMA İLETKENLERİ", "TOPRAKLAMA ILETKENLERI",
        "topraklama iletkenleri",
        "topraklama sureklilik",
        "KESİT UYGUNLUĞU", "kesit uygunlugu", "kesit uygunluğu", "kesıt uygunluğu",
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
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # AT Uygunluk Beyanı (eski strong_keywords AT'den)
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        
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
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",

        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "TOPRAKLAMA DİRENCİ",
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

def get_conclusion_message_lvd(status, percentage):
    """Sonuç mesajını döndür - LVD için"""
    if status == "PASS":
        return f"LVD topraklama raporu EN 60204-1 standardına uygun ve elektrik güvenlik gereksinimlerini karşılamaktadır (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"LVD topraklama raporu kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"LVD topraklama raporu standartlara uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"

def get_main_issues_lvd(report):
    """Ana sorunları listele - LVD için"""
    issues = []
    
    for category, score_data in report['puanlama']['category_scores'].items():
        if isinstance(score_data, dict) and score_data.get('percentage', 0) < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    # Eğer hiç kritik sorun yoksa genel sorunları ekle
    if not issues:
        if report['ozet']['toplam_puan'] < 50:
            issues = [
                "Topraklama süreklilik ölçüm sonuçları eksik",
                "EN 60204-1 standart referansları yetersiz",
                "Ölçüm cihazı kalibrasyon bilgileri eksik",
                "Uygunluk değerlendirmesi yapılmamış",
                "Teknik dokümantasyon ve görsel belgeler eksik"
            ]
    
    return issues[:4]

# ============================================================================
# FLASK APP - Azure App Service için
# ============================================================================

app = Flask(__name__)

# Azure için port yapılandırması
PORT = int(os.environ.get('PORT', 8007))

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads_lvd'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/lvd-report', methods=['POST'])
def analyze_lvd_report():
    """LVD (Topraklama Süreklilik) raporu analiz API endpoint'i - 3 Aşamalı Validasyon"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Lütfen analiz edilmek üzere bir LVD topraklama raporu sağlayın'
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
            logger.info(f"LVD (Topraklama Süreklilik) raporu kontrol ediliyor: {filename}")

            # Create analyzer instance
            analyzer = GroundingContinuityReportAnalyzer()
            
            # ÜÇ AŞAMALI LVD KONTROLÜ
            logger.info(f"Üç aşamalı LVD kontrolü başlatılıyor: {filename}")
            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.pdf':
                # PDF için üç aşamalı kontrol (OCR dahil)
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
                            'message': 'Bu dosya LVD topraklama raporu değil (farklı rapor türü tespit edildi). Lütfen LVD topraklama raporu yükleyiniz.',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'LVD_TOPRAKLAMA_RAPORU'
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
                                    'message': 'Yüklediğiniz dosya LVD topraklama raporu değil! Lütfen geçerli bir LVD topraklama raporu yükleyiniz.',
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

            # Buraya kadar geldiyse LVD raporu, şimdi analizi yap
            logger.info(f"LVD topraklama raporu doğrulandı, analiz başlatılıyor: {filename}")
            report = analyzer.generate_detailed_report(filepath)
            
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

            # Extract key results for API response - LVD FORMATINDA
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
                    'detected_language': 'TURKISH',
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
            
            # Add category scores - LVD FORMATINDA
            for category, score_data in report['puanlama']['category_scores'].items():
                if isinstance(score_data, dict):
                    response_data['category_scores'][category] = {
                        'score': score_data.get('normalized', score_data.get('score', 0)),
                        'max_score': score_data.get('max_weight', score_data.get('max_score', 0)),
                        'percentage': score_data.get('percentage', 0),
                        'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'CONDITIONAL' if score_data.get('percentage', 0) >= 50 else 'FAIL'
                    }

            return jsonify({
                'analysis_service': 'lvd_report',
                'data': response_data,
                'message': 'LVD Topraklama Raporu başarıyla analiz edildi',
                'service_description': 'LVD Topraklama Rapor Analizi',
                'service_port': PORT,
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

@app.route('/api/lvd-health', methods=['GET'])
def health_check():
    """Health check endpoint - LVD için"""
    return jsonify({
        'status': 'healthy',
        'service': 'LVD Report Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'tesseract_info': tesseract_info,
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': tesseract_available,
        'report_type': 'LVD_TOPRAKLAMA_RAPORU',
        'standard': 'EN 60204-1'
    })

@app.route('/api/lvd-info', methods=['GET'])
def api_info():
    """LVD API bilgileri"""
    return jsonify({
        'api_name': 'LVD (Topraklama Süreklilik) Rapor Analiz API',
        'description': 'Topraklama süreklilik ölçüm raporlarını analiz eder ve uygunluk değerlendirmesi yapar',
        'standard': 'EN 60204-1',
        'endpoints': {
            'POST /api/lvd-report': {
                'description': 'LVD raporunu analiz eder',
                'parameters': {
                    'file': 'Analiz edilecek rapor dosyası (PDF, JPG, JPEG, PNG)'
                },
                'response': 'Detaylı analiz raporu ve puanlama'
            },
            'GET /api/lvd-health': {
                'description': 'API sağlık durumu kontrolü',
                'response': 'Servis durumu bilgisi'
            },
            'GET /api/lvd-info': {
                'description': 'API bilgileri ve kullanım kılavuzu',
                'response': 'API dokümantasyonu'
            }
        },
        'analysis_criteria': {
            'Genel Rapor Bilgileri': {
                'weight': 15,
                'includes': ['Proje adı/numarası', 'Ölçüm tarihi', 'Rapor tarihi', 'Tesis/bölge/hat', 'Rapor numarası', 'Revizyon', 'Firma/personel']
            },
            'Ölçüm Metodu ve Standart Referansları': {
                'weight': 15,
                'includes': ['Ölçüm cihazı', 'Kalibrasyon', 'Standartlar (EN 60204-1)']
            },
            'Ölçüm Sonuç Tablosu': {
                'weight': 25,
                'includes': ['Sıra numarası', 'Makine/hat/bölge', 'Ölçüm noktası', 'RLO değeri', 'Yük iletken kesiti', 'Referans değeri', 'Uygunluk durumu']
            },
            'Uygunluk Değerlendirmesi': {
                'weight': 20,
                'includes': ['Toplam ölçüm nokta', 'Uygun nokta sayısı', 'Uygunsuz işaretleme', 'Standart referans uygunluk']
            },
            'Görsel ve Teknik Dökümantasyon': {
                'weight': 10,
                'includes': ['Cihaz bağlantı fotoğrafı', 'Görsel dokümantasyon']
            },
            'Sonuç ve Öneriler': {
                'weight': 15,
                'includes': ['Genel uygunluk', 'Standart atıf']
            }
        },
        'scoring': {
            'PASS': '≥70% - EN 60204-1 standardına uygun',
            'CONDITIONAL': '50-69% - Kabul edilebilir',
            'FAIL': '<50% - Standarda uygun değil'
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
    
    logger.info("LVD Rapor Analiz API başlatılıyor...")
    logger.info(f"Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"Tesseract OCR durumu: {'Kurulu' if tesseract_available else 'Kurulu değil'}")
    logger.info(f"Port: {PORT}")
    logger.info("API endpoint'leri:")
    logger.info("  POST /api/lvd-report - LVD raporu analizi")
    logger.info("  GET  /api/lvd-health  - Sağlık kontrolü")
    logger.info("  GET  /api/lvd-info    - API bilgileri")
    
    app.run(host='0.0.0.0', port=PORT, debug=False)