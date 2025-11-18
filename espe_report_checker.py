import re
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import PyPDF2
from docx import Document
from dataclasses import dataclass, asdict
import logging

try:
    from langdetect import detect
    LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False
    print("⚠️ Dil tespiti için: pip install langdetect")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ESPECriteria:
    """ESPE rapor kriterleri veri sınıfı"""
    genel_rapor_bilgileri: Dict[str, Any]
    koruma_cihazi_bilgileri: Dict[str, Any]
    makine_durus_performansi: Dict[str, Any]
    guvenlik_mesafesi_hesabi: Dict[str, Any]
    gorsel_teknik_dokumantasyon: Dict[str, Any]
    sonuc_oneriler: Dict[str, Any]

@dataclass
class ESPEAnalysisResult:
    """ESPE analiz sonucu veri sınıfı"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

class ESPEReportAnalyzer:
    """ESPE rapor analiz sınıfı"""
    
    def __init__(self):
        self.criteria_weights = {
            "Genel Rapor Bilgileri": 10,
            "Koruma Cihazı (ESPE) Bilgileri": 10,
            "Makine Duruş Performansı Ölçümü": 25,
            "Güvenlik Mesafesi Hesabı": 25,
            "Görsel ve Teknik Dökümantasyon": 5,
            "Sonuç ve Öneriler": 10
        }
        
        # Çoklu dil pattern'leri - EN ISO 13855 standardına göre
        self.criteria_details = {
            "Genel Rapor Bilgileri": {
                "proje_adi_numarasi": {"pattern": r"(?:Proje\s*(?:Ad[ıi]|No|Numaras[ıi])[:]*\s*([A-Z]?\d+(?:\.\d+)?)|Project\s*(?:Name|No|Number)[:]*\s*([A-Z]?\d+)|C\d{2}\.\d{3})", "weight": 2},
                "olcum_tarihi": {"pattern": r"(?:Ölçüm\s*Tarihi|Measurement\s*Date|Messdatum|\d{1,2}[./]\d{1,2}[./]\d{4})", "weight": 2},
                "rapor_tarihi": {"pattern": r"(?:Rapor\s*Tarihi|Report\s*Date|Berichtsdatum|\d{1,2}[./]\d{1,2}[./]\d{4})", "weight": 1},
                "makine_adi": {"pattern": r"(?:Makine\s*Ad[ıi][:]*\s*(T\d+\s*-\s*MCC\d+|T\d+|MCC\d+)|Machine\s*Name[:]*\s*(T\d+\s*-\s*MCC\d+))", "weight": 2},
                "hat_bolge": {"pattern": r"(?:Hat|Line|Linie|Bölge|Area|Bereich|Zone|Jaws|\d+\.?\s*Hat)", "weight": 1},
                "olcum_yapan": {"pattern": r"(?:Hazırlayan|Ölçümü\s*Yapan|Prepared\s*by|Measured\s*by|Erstellt\s*von|Gemessen\s*von|Pilz|Firma|Company)", "weight": 1},
                "imza_onay": {"pattern": r"(?:İmza|Signature|Onay|Approval|İnceleyen|Reviewed)", "weight": 1}
            },
            "Koruma Cihazı (ESPE) Bilgileri": {
                "cihaz_tipi": {"pattern": r"(?:Işık\s*Perdesi|Light\s*Curtain|Lichtvorhang|ESPE|Safety\s*Device|Alan\s*Tarayıcı)", "weight": 3},
                "kategori": {"pattern": r"(?:Kategori|Category|Kategorie|Cat\s*[234])", "weight": 2},
                "koruma_yuksekligi": {"pattern": r"(?:Koruma\s*Yüksekliği|Protection\s*Height|Schutzhöhe|\d{3,4}\s*mm)", "weight": 3},
                "cozunurluk": {"pattern": r"(?:Çözünürlük|Resolution|Auflösung|d\s*değeri|\d{1,2}\s*mm)", "weight": 2}
            },
            "Makine Duruş Performansı Ölçümü": {
                "olcum_metodu": {"pattern": r"(?:Ölçüm\s*Metodu|Test\s*Prosedürü|Measurement\s*Method|Test\s*Procedure|ESPE\s*Ölçüm|Yapılan.*?Ölçüm)", "weight": 4},
                "test_sayisi": {"pattern": r"(?:Test\s*Sayısı|Tekrarlanabilirlik|Repeatability|Test\s*Count|\d+\s*test|\d+\s*ölçüm)", "weight": 4},
                "durus_suresi_min": {"pattern": r"Min\s*(\d{2,3})|Minimum\s*(\d{2,3})|En\s*Az\s*(\d{2,3})", "weight": 6},
                "durus_suresi_max": {"pattern": r"Maks?\.?\s*(\d{2,3})|Max\.?\s*(\d{2,3})|Maximum\s*(\d{2,3})|En\s*Fazla\s*(\d{2,3})", "weight": 6},
                "durus_mesafesi": {"pattern": r"(?:Duruş\s*Mesafesi|Durma\s*Mesafesi|Stopping\s*Distance|Anhalteweg|STD|\d{2,4}\s*mm)", "weight": 5}
            },
            "Güvenlik Mesafesi Hesabı": {
                "formula_s": {"pattern": r"S\s*=\s*\([^)]*[KT][^)]*\)", "weight": 8},
                "k_sabiti": {"pattern": r"(?:K\s*=\s*(\d{4})|2000\s*mm/s|1600\s*mm/s)", "weight": 5},
                "c_sabiti": {"pattern": r"C\s*=\s*8\s*[×x*]\s*\(\s*d\s*[-–]\s*14\s*\)", "weight": 4},
                "t_durus_suresi": {"pattern": r"(?:T\s*[:=]|Duruş\s*Süresi|Stopping\s*Time)", "weight": 4},
                "uygunluk_kontrolu": {"pattern": r"(?:Mevcut\s*mesafe|≥|>=|UYGUN|SUITABLE|UYGUNSUZ|UNSUITABLE)", "weight": 2},
                "alternatif_hesap": {"pattern": r"(?:500\s*mm|K\s*=\s*1600)", "weight": 2}
            },
            "Görsel ve Teknik Dökümantasyon": {
                "makine_espe_fotograf": {"pattern": r"(?:Görsel|Fotoğraf|Resim|Photo|Image|Picture|Bild|Foto)", "weight": 3},
                "mesafe_olcumu_gorseli": {"pattern": r"(?:Mesafe|Distance|Ölçüm|Measurement).*?(?:Görsel|işaretli|Marked)", "weight": 2}
            },
            "Sonuç ve Öneriler": {
                "tehlike_tanimi": {"pattern": r"(?:Tehlikeli?\s*Hareket|Dangerous\s*Movement|Gefährliche\s*Bewegung|Tehlike|fikstür|pres|kapı|hareket)", "weight": 3},
                "uygunluk_degerlendirme": {"pattern": r"(?:Uygun|Suitable|Geeignet|Uygunsuz|Unsuitable|Ungeeignet)", "weight": 2},
                "iyilestirme_onerileri": {"pattern": r"(?:Öneri|Recommendation|Empfehlung|İyileştir|Improve|Verbessern|mesafe\s*arttır)", "weight": 3},
                "en_iso_baglanti": {"pattern": r"EN\s*ISO\s*13855", "weight": 2}
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Metnin dilini tespit et"""
        if not LANGUAGE_DETECTION_AVAILABLE:
            return "unknown"
        
        try:
            # Kısa metin örnekleri al
            sample_text = " ".join(text.split()[:100])
            detected_lang = detect(sample_text)
            
            # Ana diller
            if detected_lang in ['tr', 'turkish']:
                return 'turkish'
            elif detected_lang in ['en', 'english']:
                return 'english'
            elif detected_lang in ['de', 'german']:
                return 'german'
            else:
                return detected_lang
                
        except Exception as e:
            logger.warning(f"Dil tespiti hatası: {e}")
            return "unknown"

    def get_multilingual_patterns(self, criterion: str, detected_lang: str) -> List[str]:
        """Tespit edilen dile göre ek pattern'ler döndür"""
        additional_patterns = {
            'turkish': {
                'proje_adi_numarasi': [r"Proje\s*No", r"Belge\s*No", r"Döküman"],
                'makine_adi': [r"Makine\s*Adı", r"Ekipman", r"Cihaz"],
                'olcum_tarihi': [r"Ölçüm\s*Tarihi", r"Test\s*Tarihi"],
                'tehlike_tanimi': [r"Tehlikeli", r"Risk", r"Tehlike"]
            },
            'english': {
                'proje_adi_numarasi': [r"Project\s*No", r"Document\s*No", r"Report\s*No"],
                'makine_adi': [r"Machine\s*Name", r"Equipment", r"Device"],
                'olcum_tarihi': [r"Measurement\s*Date", r"Test\s*Date"],
                'tehlike_tanimi': [r"Dangerous", r"Hazardous", r"Risk"]
            },
            'german': {
                'proje_adi_numarasi': [r"Projekt\s*Nr", r"Dokument\s*Nr", r"Bericht\s*Nr"],
                'makine_adi': [r"Maschinen\s*Name", r"Ausrüstung", r"Gerät"],
                'olcum_tarihi': [r"Messdatum", r"Testdatum"],
                'tehlike_tanimi': [r"Gefährlich", r"Risiko", r"Gefahr"]
            }
        }
        
        return additional_patterns.get(detected_lang, {}).get(criterion, [])

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
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """DOCX'den metin çıkarma"""
        try:
            doc = Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"DOCX okuma hatası: {e}")
            return ""
    
    def check_report_date_validity(self, text: str) -> Tuple[bool, str, str]:
        """Rapor tarihinin geçerliliğini kontrol etme"""
        date_patterns = [
            r"Ölçüm\s*Tarihi\s*[:=]\s*(\d{2}[./]\d{2}[./]\d{4})",
            r"(\d{2}[./]\d{2}[./]\d{4})"
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                date_str = matches[0]
                try:
                    # Tarih formatını normalize et
                    date_str = date_str.replace('.', '/').replace('-', '/')
                    report_date = datetime.strptime(date_str, '%d/%m/%Y')
                    one_year_ago = datetime.now() - timedelta(days=365)
                    
                    is_valid = report_date >= one_year_ago
                    return is_valid, date_str, f"Rapor tarihi: {date_str} {'(GEÇERLİ)' if is_valid else '(GEÇERSİZ - 1 yıldan eski)'}"
                except ValueError:
                    continue
        
        return False, "", "Rapor tarihi bulunamadı"
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, ESPEAnalysisResult]:
        """Belirli kategori kriterlerini analiz etme"""
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        # Dil tespiti yap
        detected_lang = self.detect_language(text)
        logger.info(f"Tespit edilen dil: {detected_lang}")
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            
            # Ana pattern ile ara
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            # Eğer bulunamadıysa, dile özel ek pattern'ler dene
            if not matches:
                additional_patterns = self.get_multilingual_patterns(criterion_name, detected_lang)
                for add_pattern in additional_patterns:
                    matches = re.findall(add_pattern, text, re.IGNORECASE | re.MULTILINE)
                    if matches:
                        break
            
            # Özel durum: Durma zamanları için tablo formatı arama
            if not matches and criterion_name in ['durus_suresi_min', 'durus_suresi_max']:
                # Tablo formatında sayı arama - Min/Max ile birlikte
                table_patterns = [
                    r"Min\s*(\d{2,3})",
                    r"Maks?\s*(\d{2,3})", 
                    r"(\d{2,3})\s*(?=.*(?:Min|Maks?))",
                    r"(\d{2,3})\s*(?=.*\d{2,3}.*(?:Min|Maks?))",
                    r"(?:Durma\s*Zamanı|STT).*?(\d{2,3})"
                ]
                
                for table_pattern in table_patterns:
                    table_matches = re.findall(table_pattern, text, re.IGNORECASE | re.MULTILINE)
                    if table_matches:
                        matches = table_matches
                        break
            
            # Özel durum: Ölçüm metodu için geniş arama
            if not matches and criterion_name == 'olcum_metodu':
                method_patterns = [
                    r"(?:Yapılan.*?Ölçüm|ESPE.*?Ölçüm|Test.*?Prosedür)",
                    r"(?:Durma\s*Zamanı|STT|STD)",
                    r"(?:Tehlikeli\s*Hareket|Mevcut\s*Emniyet)"
                ]
                
                for method_pattern in method_patterns:
                    method_matches = re.findall(method_pattern, text, re.IGNORECASE | re.MULTILINE)
                    if method_matches:
                        matches = method_matches
                        break
            
            if matches:
                content = str(matches[0]) if len(matches) == 1 else str(matches)
                found = True
                score = weight
            else:
                # Genel fallback pattern'ler
                general_patterns = {
                    "proje_adi_numarasi": r"[A-Z]\d{2}\.\d{3}",
                    "makine_adi": r"(?:T\d+|MCC\d+)",
                    "cihaz_tipi": r"(?:Light|Işık|Licht|ESPE)",
                    "marka_model": r"(?:DataLogic|SAFEasy|Pilz|Sick|Banner|Omron|Keyence)",
                    "formula_s": r"S\s*=",
                    "tehlike_tanimi": r"(?:Dangerous|Tehlike|Gefahr|fikstür|kapı|hareket)",
                    "k_sabiti": r"(?:2000|1600)",
                    "uygunluk_kontrolu": r"(?:UYGUN|SUITABLE|UYGUNSUZ|UNSUITABLE)",
                    "olcum_metodu": r"(?:ESPE|Ölçüm|Test|Measurement)",
                    "durus_suresi_min": r"(?:Min|\d{2,3}(?=.*\d{2,3}))",
                    "durus_suresi_max": r"(?:Maks?|\d{2,3}(?=.*Min))",
                    "test_sayisi": r"(?:Test|Tekrar|\d+)"
                }
                
                general_pattern = general_patterns.get(criterion_name)
                if general_pattern:
                    general_matches = re.findall(general_pattern, text, re.IGNORECASE)
                    if general_matches:
                        content = f"Genel eşleşme bulundu: {general_matches[0]}"
                        found = True
                        score = weight // 2  # Kısmi puan
                    else:
                        content = "Bulunamadı"
                        found = False
                        score = 0
                else:
                    content = "Bulunamadı"
                    found = False
                    score = 0
            
            results[criterion_name] = ESPEAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_found": len(matches) if matches else 0}
            )
        
        return results
    
    def extract_specific_values(self, text: str) -> Dict[str, Any]:
        """Spesifik değerleri çıkarma"""
        values = {}
        
        # Çoklu dil değer pattern'leri - basit ve genel
        value_patterns = {
            "proje_no": r"(C\d{2}\.\d{3})",
            "olcum_tarihi": r"(\d{1,2}[./]\d{1,2}[./]\d{4})",
            "makine_adi": r"(?:Makine\s*Ad[ıi][:]*\s*(T\d+\s*-\s*MCC\d+))",
            "hat_bolge": r"(?:Jaws\s*\d+|Hat.*?[:=]\s*([^\n\r]+))",
            "koruma_yuksekligi": r"(\d{3,4})\s*mm",
            "cozunurluk": r"(\d{1,2})\s*mm",
            "durus_suresi_min": r"Min\.?\s*(\d{2,3})|Minimum\s*(\d{2,3})",
            "durus_suresi_max": r"Maks?\.?\s*(\d{2,3})|Max\.?\s*(\d{2,3})|Maximum\s*(\d{2,3})|(\d{2,3})\s*(?=.*Maks?)|(\d{2,3})\s*(?=.*Max)",
            "mevcut_mesafe": r"(\d{2,4})\s*mm",
            "hesaplanan_mesafe": r"(?:S\s*=\s*(\d{2,4})|(\d{2,4})\s*mm)",
            "durum": r"(UYGUNSUZ|UYGUN)",
            "tehlikeli_hareket": r"(?:Takım\s*tezgahı|kapı\s*kapanma|fikstür|tehlikeli\s*hareket)",
            "k_sabiti": r"(?:K\s*=\s*(\d{4})|2000|1600)",
            "formula_s": r"(S\s*=\s*\([^)]+\))",
            "formula_c": r"(C\s*=\s*8\s*[×x*]\s*\([^)]+\))",
            "en_iso_13855": r"(EN\s*ISO\s*13855)"
        }
        
        # Dil tespiti
        detected_lang = self.detect_language(text)
        
        for key, pattern in value_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Özel durum: durum için UYGUNSUZ varsa onu tercih et
                if key == "durum" and len(matches) > 1:
                    uygunsuz_found = any("UYGUNSUZ" in str(m).upper() for m in matches)
                    if uygunsuz_found:
                        values[key] = "UYGUNSUZ"
                    else:
                        # İlk grubu al (eğer gruplar varsa)
                        if isinstance(matches[0], tuple):
                            values[key] = next((m for m in matches[0] if m), matches[0][0]).strip()
                        else:
                            values[key] = matches[0].strip()
                else:
                    # İlk grubu al (eğer gruplar varsa)
                    if isinstance(matches[0], tuple):
                        values[key] = next((m for m in matches[0] if m), matches[0][0]).strip()
                    else:
                        values[key] = matches[0].strip()
            else:
                # Fallback: Basit pattern'ler
                fallback_patterns = {
                    "proje_no": r"C\d{2}\.\d{3}",
                    "olcum_tarihi": r"\d{1,2}[./]\d{1,2}[./]\d{4}",
                    "cihaz_tipi": r"(?:Işık\s*Perdesi|Light\s*Curtain|Lichtvorhang)",
                    "durum": r"(?:durumu[:]*\s*(UYGUNSUZ|UYGUN)|(?:UYGUNSUZ|UYGUN)(?=\s|$))",
                    "makine_adi": r"(?:T\d+\s*-\s*MCC\d+)",
                    "hat_bolge": r"(?:Jaws|\d+\.?\s*Hat)",
                    "koruma_yuksekligi": r"\d{3,4}\s*mm",
                    "cozunurluk": r"\d{1,2}\s*mm",
                    "durus_suresi_min": r"(?:Min|(\d{2,3})(?=.*\d{2,3}))",
                    "durus_suresi_max": r"(?:Maks?|(\d{2,3})(?=.*Min))",
                    "olcum_metodu": r"(?:ESPE|Ölçüm|Test|Measurement)"
                }
                
                fallback_pattern = fallback_patterns.get(key)
                if fallback_pattern:
                    fallback_matches = re.findall(fallback_pattern, text, re.IGNORECASE)
                    if fallback_matches:
                        values[key] = fallback_matches[0].strip()
                    else:
                        values[key] = "Bulunamadı"
                else:
                    values[key] = "Bulunamadı"
        
        return values
    
    def validate_extracted_values(self, extracted_values: Dict[str, Any]) -> Dict[str, float]:
        """Çıkarılan değerlerin geçerliliğini kontrol ederek puan azaltma faktörü hesapla"""
        validation_scores = {}
        
        # Kritik değerlerin kontrolleri
        validations = {
            # Boş veya "Bulunamadı" değerler
            "durus_suresi_min": 0.0 if not extracted_values.get("durus_suresi_min") or extracted_values.get("durus_suresi_min") == "Bulunamadı" else 1.0,
            "durus_suresi_max": 0.0 if not extracted_values.get("durus_suresi_max") or extracted_values.get("durus_suresi_max") == "Bulunamadı" else 1.0,
            
            # Makine adı kontrol (T ile başlamalı)
            "makine_adi": 1.0 if extracted_values.get("makine_adi", "").startswith("T") else 0.5,
            
            # UYGUNSUZ durumu tespit edilmeli
            "durum": 0.5 if extracted_values.get("durum", "").upper() in ["UYGUN", "SUITABLE"] else 1.0,
            
            # Sayısal değerlerin mantıklı olması
            "koruma_yuksekligi": 1.0 if extracted_values.get("koruma_yuksekligi", "0").isdigit() and int(extracted_values.get("koruma_yuksekligi", "0")) > 100 else 0.5,
            "cozunurluk": 1.0 if extracted_values.get("cozunurluk", "0").isdigit() and int(extracted_values.get("cozunurluk", "0")) > 5 else 0.5,
        }
        
        return validations

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, ESPEAnalysisResult]], extracted_values: Dict[str, Any]) -> Dict[str, Any]:
        """Puanları hesaplama - çıkarılan değerlerin geçerliliğini de kontrol ederek"""
        category_scores = {}
        total_score = 0
        total_max_score = 100
        
        # Değer geçerlilik kontrolü
        validation_scores = self.validate_extracted_values(extracted_values)
        
        for category, results in analysis_results.items():
            category_max = self.criteria_weights[category]
            category_earned = 0
            category_possible = sum(result.max_score for result in results.values())
            
            # Her kriter için puanı hesapla
            for criterion_name, result in results.items():
                base_score = result.score
                
                # Eğer bu kriter için geçerlilik kontrolü varsa uygula
                if criterion_name in validation_scores:
                    validation_factor = validation_scores[criterion_name]
                    adjusted_score = base_score * validation_factor
                    category_earned += adjusted_score
                else:
                    category_earned += base_score
            
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
    
    def generate_detailed_report(self, pdf_path: str, docx_path: str = None) -> Dict[str, Any]:
        """Detaylı rapor oluşturma"""
        logger.info("ESPE rapor analizi başlatılıyor...")
        
        # PDF'den metin çıkar
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return {"error": "PDF okunamadı"}
        
        # Dil tespiti
        detected_language = self.detect_language(pdf_text)
        logger.info(f"Tespit edilen belge dili: {detected_language}")
        
        # Tarih geçerliliği kontrolü
        date_valid, date_str, date_message = self.check_report_date_validity(pdf_text)
        
        # Spesifik değerleri çıkar
        extracted_values = self.extract_specific_values(pdf_text)
        
        # Her kategori için analiz yap
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(pdf_text, category)
        
        # Puanları hesapla
        scores = self.calculate_scores(analysis_results, extracted_values)
        
        # Öneriler oluştur
        recommendations = self.generate_recommendations(analysis_results, scores)
        
        report = {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgileri": {
                "pdf_path": pdf_path,
                "docx_path": docx_path,
                "tespit_edilen_dil": detected_language
            },
            "tarih_gecerliligi": {
                "durum": "GEÇERLİ" if date_valid else "GEÇERSİZ",
                "tarih": date_str,
                "mesaj": date_message
            },
            "cikarilan_degerler": extracted_values,
            "kategori_analizleri": analysis_results,
            "puanlama": scores,
            "overall_score": {
                "date_validity": "GEÇERLİ" if date_valid else "GEÇERSİZ",
                "failure_reason": None if scores["overall_percentage"] >= 70 else ("Tarih geçerliliği sorunu" if not date_valid else "Yetersiz puan"),
                "final_status": "PASSED" if scores["overall_percentage"] >= 70 else "FAILED",
                "max_points": 100,
                "pass_status": "PASSED" if scores["overall_percentage"] >= 70 else "FAILED",
                "percentage": scores["overall_percentage"],
                "status": "PASS" if scores["overall_percentage"] >= 70 else "FAIL",
                "status_tr": "GEÇERLİ" if scores["overall_percentage"] >= 70 else "GEÇERSİZ",
                "total_points": scores["overall_percentage"]
            },
            "oneriler": recommendations,
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["overall_percentage"],
                "durum": "GEÇERLİ" if scores["overall_percentage"] >= 70 else "YETERSİZ",
                "tarih_durumu": "GEÇERLİ" if date_valid else "GEÇERSİZ",
                "dil": detected_language
            }
        }
        
        return report
    
    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """Öneriler oluşturma"""
        recommendations = []
        
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 50:
                recommendations.append(f"❌ {category} bölümü yetersiz (%{category_score:.1f})")
                
                # Eksik kriterler
                missing_criteria = [name for name, result in results.items() if not result.found]
                if missing_criteria:
                    recommendations.append(f"  Eksik kriterler: {', '.join(missing_criteria)}")
            
            elif category_score < 80:
                recommendations.append(f"⚠️ {category} bölümü geliştirilmeli (%{category_score:.1f})")
            
            else:
                recommendations.append(f"✅ {category} bölümü yeterli (%{category_score:.1f})")
        
        # Genel öneriler
        if scores["overall_percentage"] < 70:
            recommendations.append("\n🚨 GENEL ÖNERİLER:")
            recommendations.append("- Rapor EN ISO 13855 standardına tam uyumlu hale getirilmelidir")
            recommendations.append("- Eksik bilgiler tamamlanmalıdır")
            recommendations.append("- Formül hesaplamaları detaylandırılmalıdır")
        
        return recommendations
    
def main():
    """Ana fonksiyon"""
    analyzer = ESPEReportAnalyzer()
    
    # Dosya yolları
    pdf_path = "T4-MCC1201 - ESPE Kontrol Raporu.pdf"
    docx_path = "ESPE_Rapor_Kriterleri_Puanlama.docx"
    
    # Dosyaların varlığını kontrol et
    if not os.path.exists(pdf_path):
        print(f"❌ PDF dosyası bulunamadı: {pdf_path}")
        return
    
    print("🔍 ESPE Rapor Analizi Başlatılıyor...")
    print("=" * 60)
    
    # Analizi çalıştır
    report = analyzer.generate_detailed_report(pdf_path, docx_path)
    
    if "error" in report:
        print(f"❌ Hata: {report['error']}")
        return
    
    # Sonuçları göster
    print("\n📊 ANALİZ SONUÇLARI")
    print("=" * 60)
    
    print(f"📅 Analiz Tarihi: {report['analiz_tarihi']}")
    print(f"🌐 Tespit Edilen Dil: {report['ozet']['dil'].title()}")
    print(f"📋 Toplam Puan: {report['ozet']['toplam_puan']}/100")
    print(f"📈 Yüzde: %{report['ozet']['yuzde']}")
    print(f"🎯 Durum: {report['ozet']['durum']}")
    print(f"📆 Tarih Durumu: {report['ozet']['tarih_durumu']}")
    
    print(f"\n⚠️ Tarih Kontrolü: {report['tarih_gecerliligi']['mesaj']}")
    
    print("\n📋 ÖNEMLİ ÇIKARILAN DEĞERLER")
    print("-" * 40)
    important_values = ['proje_no', 'olcum_tarihi', 'makine_adi', 'marka', 'model', 
                       'koruma_yuksekligi', 'cozunurluk', 'durus_suresi_min', 'durus_suresi_max',
                       'mevcut_mesafe', 'hesaplanan_mesafe', 'durum', 'tehlikeli_hareket', 
                       'k_sabiti', 'formula_s', 'en_iso_13855']
    
    for key in important_values:
        if key in report['cikarilan_degerler']:
            print(f"{key.replace('_', ' ').title()}: {report['cikarilan_degerler'][key]}")
    
    print("\n📊 KATEGORİ PUANLARI")
    print("-" * 40)
    for category, score_data in report['puanlama']['category_scores'].items():
        print(f"{category}: {score_data['normalized']}/{score_data['max_weight']} (%{score_data['percentage']:.1f})")
    
    print("\n💡 ÖNERİLER")
    print("-" * 40)
    for recommendation in report['oneriler']:
        print(recommendation)

if __name__ == "__main__":
    main()