import re
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import PyPDF2
from docx import Document
import pandas as pd
from dataclasses import dataclass, asdict
import logging
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    OFFLINE_TRANSLATION_AVAILABLE = True
except ImportError:
    OFFLINE_TRANSLATION_AVAILABLE = False
    print("⚠️ Offline çeviri desteği için: pip install transformers torch sentencepiece")

try:
    from langdetect import detect
    LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False
    print("⚠️ Dil tespiti için: pip install langdetect")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LOTOCriteria:
    """LOTO rapor kriterleri veri sınıfı"""
    genel_rapor_bilgileri: Dict[str, Any]
    tesis_makine_tanimi: Dict[str, Any]
    loto_politikasi_degerlendirmesi: Dict[str, Any]
    enerji_kaynaklari_analizi: Dict[str, Any]
    izolasyon_noktalari_prosedurler: Dict[str, Any]
    teknik_degerlendirme_sonuclar: Dict[str, Any]
    dokumantasyon_referanslar: Dict[str, Any]

@dataclass
class LOTOAnalysisResult:
    """LOTO analiz sonucu veri sınıfı"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

class LOTOReportAnalyzer:
    """LOTO rapor analiz sınıfı"""
    
    def __init__(self):
        # Çeviri özelliğini devre dışı bırak (çoğu LOTO raporu Türkçe)
        self.translation_models = {}
        self.language_detector = None
        
        # Sadece dil tespiti kullan, çeviri yapma
        logger.info("LOTO analiz sistemi başlatılıyor (Türkçe optimized)...")
        
        self.criteria_weights = {
            "Genel Rapor Bilgileri": 10,
            "Tesis ve Makine Tanımı": 10,
            "LOTO Politikası Değerlendirmesi": 10,
            "Enerji Kaynakları Analizi": 25,
            "İzolasyon Noktaları ve Prosedürler": 25,
            "Teknik Değerlendirme ve Sonuçlar": 15,
            "Dokümantasyon ve Referanslar": 5
        }
        
        self.criteria_details = {
            "Genel Rapor Bilgileri": {
                "proje_adi_belge_no": {"pattern": r"(?:Proje\s*Ad[ıi]|Project\s*Name|Belge\s*(?:No|Numaras[ıi])|Document\s*(?:No|Number)|LOTO|Lockout|Tagout|Lock\s*out|Tag\s*out)", "weight": 2},
                "rapor_tarihi_versiyon": {"pattern": r"(?:Rapor\s*Tarihi|Report\s*Date|Date|Tarih|Versiyon|Version|Rev\.?|v)\s*[:=]?\s*(\d{1,2}[./]\d{1,2}[./]\d{4}|\d+|[A-Z])", "weight": 2},
                "hazirlayan_firma": {"pattern": r"(?:Hazırlayan|Prepared\s*by|Company|Firma|Consultant|Contractor)\s*[:=]?\s*([^\n\r]+)", "weight": 2},
                "musteri_bilgileri": {"pattern": r"(?:Müşteri|Customer|Client|Tesis\s*Ad[ıi]|Facility\s*Name|Plant\s*Name|Adres|Address|Location)", "weight": 2},
                "imza_onay": {"pattern": r"(?:İmza|Signature|Onay|Approval|İnceleyen|Reviewed|Authorized|Yetkili|Checked\s*by|Approved\s*by)", "weight": 2}
            },
            "Tesis ve Makine Tanımı": {
                "tesis_bilgileri": {"pattern": r"(?:Tesis|Facility|Plant|Factory|Site)\s*(?:Ad[ıi]|Name|Lokasyon|Location|Information)", "weight": 2},
                "makine_tanimi": {"pattern": r"(?:Makine|Machine|Equipment)\s*(?:Tan[ıi]m[ıi]|Description|Details|Information|ne\s*işe\s*yarad[ıi]ğ[ıi]|what\s*it\s*does)", "weight": 2},
                "makine_teknik_bilgi": {"pattern": r"(?:Üretici|Manufacturer|Seri\s*No|Serial\s*(?:No|Number)|Model|Üretim\s*Tarihi|Production\s*Date|Ekipman\s*Tipi|Equipment\s*Type)", "weight": 2},
                "makine_fotograflari": {"pattern": r"(?:Fotoğraf|Photo|Image|Görsel|Picture|Genel\s*Görünüm|General\s*View|Visual|Figure)", "weight": 2},
                "lokasyon_konumu": {"pattern": r"(?:Lokasyon|Location|Konum|Position|Site|Tesisteki\s*konum|Plant\s*location)", "weight": 2}
            },
            "LOTO Politikası Değerlendirmesi": {
                "mevcut_politika": {"pattern": r"(?:Politika|Policy|LOTO\s*Policy|Prosedür|Procedure|Mevcut.*?politika|Current.*?policy|Existing.*?policy)", "weight": 2},
                "politika_uygunluk": {"pattern": r"(?:Kontrol\s*Listesi|Checklist|Check\s*list|16\s*madde|16\s*items|Evet|Hayır|Yes|No|M\.D|Pass|Fail)", "weight": 3},
                "prosedur_degerlendirme": {"pattern": r"(?:Prosedür|Procedure|5\s*madde|5\s*items|Değerlendirme|Assessment|İnceleme|Review|Evaluation)", "weight": 2},
                "personel_gorusme": {"pattern": r"(?:Personel|Personnel|Staff|Görüşme|Interview|Çalışan|Employee|Worker|7\s*madde|7\s*items)", "weight": 2},
                "egitim_durumu": {"pattern": r"(?:Eğitim|Training|Education|Kurs|Course|LOTO.*?eğitim|LOTO.*?training)", "weight": 1}
            },
            "Enerji Kaynakları Analizi": {
                "enerji_kaynagi_tanimlama": {"pattern": r"(?:Enerji\s*Kaynağ[ıi]|Energy\s*Source|Power\s*Source|Elektrik|Electric|Electrical|Pn[öo]matik|Pneumatic|Hidrolik|Hydraulic|Su|Water|Steam|Thermal|Mechanical)", "weight": 6},
                "izolasyon_cihazi_bilgi": {"pattern": r"(?:İzolasyon\s*Cihaz[ıi]|Isolation.*?Device|Isolating.*?Device|Switch|Valve|Vana|Şalter|Breaker|Disconnect)", "weight": 6},
                "cihaz_durumu_kontrol": {"pattern": r"(?:Çalış[ıt][ıa]rılabilirlik|Operability|Kilitlenebilirlik|Lockability|Lockable|Tahliye\s*edilebilirlik|Drainable|Working|Lock|Drain|Test)", "weight": 6},
                "kilitleme_ekipman": {"pattern": r"(?:Kilit|Lock|Padlock|Etiket|Tag|Label|Valf\s*Kit|Valve\s*Kit|Ölçüm\s*Cihaz[ıi]|Measuring\s*Device|Tester)", "weight": 4},
                "uygunsuz_enerji_tablosu": {"pattern": r"(?:Uygunsuz\s*Enerji|Unsuitable.*?Energy|Hazardous.*?Energy|Enerji.*?Özet|Energy.*?Summary|Energy.*?Table)", "weight": 3}
            },
            "İzolasyon Noktaları ve Prosedürler": {
                "izolasyon_noktalari_tablo": {"pattern": r"(?:İzolasyon\s*Nokta|Isolation.*?Point|Isolation.*?Location|Layout|Şema|Diagram|Scheme|Drawing)", "weight": 6},
                "prosedur_detaylari": {"pattern": r"(?:Prosedür\s*Detay|Procedure.*?Detail|Step.*?by.*?step|Enerji\s*Kesme|Energy.*?Cut|Energy.*?Shut.*?off|Ad[ıi]m|Step)", "weight": 6},
                "mevcut_prosedur_analiz": {"pattern": r"(?:Mevcut\s*Prosedür|Current.*?Procedure|Existing.*?Procedure|Var\s*olan|As.*?is)", "weight": 4},
                "tavsiyeler": {"pattern": r"(?:Tavsiye|Recommendation|Suggest|İyileştirme|Improvement|Enhance|Yeni\s*Ekipman|New.*?Equipment)", "weight": 5},
                "izolasyon_fotograflari": {"pattern": r"(?:İzolasyon.*?Fotoğraf|Isolation.*?Photo|Kilit.*?Etiket|Lock.*?Tag|Valf.*?Kit|Valve.*?Kit|Visual.*?Evidence)", "weight": 4}
            },
            "Teknik Değerlendirme ve Sonuçlar": {
                "kabul_edilebilirlik": {"pattern": r"(?:Kabul\s*Edilebilir|Acceptable|Accept|LOTO\s*Uygun|LOTO.*?Suitable|Suitable|Evet|Hayır|Yes|No|Pass|Fail)", "weight": 4},
                "bulgular_yorumlar": {"pattern": r"(?:BULGULAR|FINDINGS|YORUMLAR|COMMENTS|Bulgu|Finding|Yorum|Comment|Observation|Eksiklik|Deficiency|Tehlike|Hazard|Risk|gözlemlenmiştir|öngörülmektedir|sebebiyet|değiştirilmesi\s*gerekmektedir|observed|noted|identified)", "weight": 3},
                "sonuc_tablolari": {"pattern": r"(?:Sonuç\s*Tablo|Result.*?Table|Summary.*?Table|Makine\s*Özet|Machine.*?Summary|Conclusion)", "weight": 3},
                "oneriler": {"pattern": r"(?:Öneri|Recommendation|Recommend|İyileştirme|Improvement|Improve|Genel\s*Değerlendirme|General.*?Assessment|gerekmektedir|konmalıdır|yapılmalı|sağlanmalı|gerçekleşmeli|LOTO\s*uygunluğunun\s*sağlanması|tahliye\s*yapabilen|kilitlenebilen|should\s*be|must\s*be|need\s*to)", "weight": 3},
                "mevzuat_uygunlugu": {"pattern": r"(?:2006/42/EC|2009/104/EC|98/37/EC|2014/35/EU|Direktif|Directive|Mevzuat|Regulation|Compliance|Standard|EN\s*ISO)", "weight": 2}
            },
            "Dokümantasyon ve Referanslar": {
                "mevzuat_referanslari": {"pattern": r"(?:2006/42/EC|2009/104/EC|98/37/EC|2014/35/EU|AB\s*Direktif|EU.*?Directive|European.*?Directive|Makine\s*Emniyeti|Machinery\s*Safety|İş\s*Ekipmanları|Work\s*Equipment|Direktifi?|Mevzuat\s*[Rr]eferans|Legal.*?Requirement|Yasal.*?Mevzuat|Legal.*?Reference|Tablo.*?AB.*?Mevzuat|Regulation)", "weight": 3},
                "normatif_referanslar": {"pattern": r"(?:EN\s*ISO|ISO|12100|60204|4414|14118|13849|13855|Standard|Norm|Technical.*?Standard|Safety.*?Standard)", "weight": 2}
            }
        }
    
    def init_translation_models(self):
        """Offline çeviri modellerini başlat"""
        try:
            logger.info("Offline çeviri modelleri yükleniyor...")
            
            # Facebook NLLB modeli - daha küçük ve hızlı
            model_name = "facebook/nllb-200-distilled-600M"
            
            try:
                logger.info("NLLB çeviri modeli kontrol ediliyor...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="./models")
                
                # NLLB için pipeline oluştur
                translator = pipeline('translation', 
                                    model=model, 
                                    tokenizer=tokenizer,
                                    device=-1)
                
                self.translation_models['nllb'] = {
                    'tokenizer': tokenizer,
                    'model': model,
                    'pipeline': translator
                }
                logger.info("✅ NLLB çeviri modeli hazır (200+ dil destekli)")
                
            except Exception as e:
                logger.warning(f"⚠️ NLLB modeli yüklenemedi: {str(e)[:100]}...")
                logger.info("Alternatif olarak Google Translate API'si kullanılabilir")
                
            if len(self.translation_models) > 0:
                logger.info(f"Çeviri sistemi aktif")
            else:
                logger.info("Çeviri modelleri yüklenemedi, sadece Türkçe desteklenecek")
                
        except Exception as e:
            logger.error(f"Çeviri modelleri başlatılamadı: {e}")
            logger.info("Çeviri özelliği devre dışı, sadece Türkçe desteklenecek")
    
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
    
    def translate_to_turkish(self, text: str, source_lang: str) -> str:
        """Metni Türkçe'ye çevir - Temel İngilizce desteği"""
        if source_lang != 'tr' and source_lang == 'en':
            logger.info(f"İngilizce belgede temel terim çevirisi uygulanıyor...")
            
            # Temel LOTO terimlerini çevir
            translation_map = {
                r'\bLockout\s+Tagout\b': 'LOTO',
                r'\bLock\s+out\b': 'LOTO',
                r'\bTag\s+out\b': 'LOTO', 
                r'\bEnergy\s+Source\b': 'Enerji Kaynağı',
                r'\bEnergy\s+Sources\b': 'Enerji Kaynakları',
                r'\bIsolation\s+Device\b': 'İzolasyon Cihazı',
                r'\bIsolation\s+Point\b': 'İzolasyon Noktası',
                r'\bIsolation\s+Points\b': 'İzolasyon Noktaları',
                r'\bProcedure\b': 'Prosedür',
                r'\bPolicy\b': 'Politika',
                r'\bTraining\b': 'Eğitim',
                r'\bPersonnel\b': 'Personel',
                r'\bEmployee\b': 'Çalışan',
                r'\bEquipment\b': 'Ekipman',
                r'\bMachine\b': 'Makine',
                r'\bFacility\b': 'Tesis',
                r'\bPlant\b': 'Tesis',
                r'\bManufacturer\b': 'Üretici',
                r'\bSerial\s+Number\b': 'Seri Numarası',
                r'\bModel\b': 'Model',
                r'\bElectrical\b': 'Elektrik',
                r'\bElectric\b': 'Elektrik', 
                r'\bPneumatic\b': 'Pnömatik',
                r'\bHydraulic\b': 'Hidrolik',
                r'\bMechanical\b': 'Mekanik',
                r'\bValve\b': 'Vana',
                r'\bSwitch\b': 'Şalter',
                r'\bBreaker\b': 'Kesici',
                r'\bLock\b': 'Kilit',
                r'\bTag\b': 'Etiket',
                r'\bAcceptable\b': 'Kabul Edilebilir',
                r'\bSuitable\b': 'Uygun',
                r'\bRecommendation\b': 'Tavsiye',
                r'\bRecommendations\b': 'Tavsiyeler',
                r'\bImprovement\b': 'İyileştirme',
                r'\bFinding\b': 'Bulgu',
                r'\bFindings\b': 'Bulgular',
                r'\bComment\b': 'Yorum',
                r'\bComments\b': 'Yorumlar',
                r'\bObservation\b': 'Gözlem',
                r'\bAssessment\b': 'Değerlendirme',
                r'\bEvaluation\b': 'Değerlendirme',
                r'\bAnalysis\b': 'Analiz',
                r'\bSummary\b': 'Özet',
                r'\bConclusion\b': 'Sonuç',
                r'\bResult\b': 'Sonuç',
                r'\bResults\b': 'Sonuçlar',
                r'\bCompliance\b': 'Uygunluk',
                r'\bStandard\b': 'Standart',
                r'\bRegulation\b': 'Mevzuat',
                r'\bDirective\b': 'Direktif',
                r'\bSafety\b': 'Güvenlik',
                r'\bHazard\b': 'Tehlike',
                r'\bRisk\b': 'Risk',
                r'\bProject\s+Name\b': 'Proje Adı',
                r'\bReport\s+Date\b': 'Rapor Tarihi',
                r'\bPrepared\s+by\b': 'Hazırlayan',
                r'\bCustomer\b': 'Müşteri',
                r'\bClient\b': 'Müşteri',
                r'\bAddress\b': 'Adres',
                r'\bLocation\b': 'Lokasyon',
                r'\bDocument\s+Number\b': 'Belge Numarası',
                r'\bVersion\b': 'Versiyon',
                r'\bRevision\b': 'Revizyon',
                r'\bApproved\s+by\b': 'Onaylayan',
                r'\bChecked\s+by\b': 'Kontrol Eden',
                r'\bReviewed\s+by\b': 'İnceleyen',
                r'\bSignature\b': 'İmza',
                r'\bDate\b': 'Tarih'
            }
            
            # Terim çevirilerini uygula
            for english_term, turkish_term in translation_map.items():
                text = re.sub(english_term, turkish_term, text, flags=re.IGNORECASE)
            
            logger.info("Temel terim çevirisi tamamlandı")
            return text
        elif source_lang != 'tr':
            logger.info(f"Tespit edilen dil: {source_lang.upper()} - Temel çeviri desteği yok, orijinal metin kullanılıyor")
        
        return text
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkarma - PyPDF2 ve OCR ile"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    page_text = re.sub(r'\s+', ' ', page_text)
                    page_text = page_text.replace('|', ' ')
                    text += page_text + "\n"
                
                text = text.replace('—', '-')
                text = text.replace('"', '"').replace('"', '"')
                text = text.replace('´', "'")
                text = re.sub(r'[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]+', ' ', text)
                text = text.strip()
                
                if len(text) > 50:
                    logger.info("Metin PyPDF2 ile çıkarıldı")
                    return text
                
                logger.info("PyPDF2 ile yeterli metin bulunamadı, OCR deneniyor...")
                return self.extract_text_with_ocr(pdf_path)
                
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {e}")
            logger.info("OCR'a geçiliyor...")
            return self.extract_text_with_ocr(pdf_path)

    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """OCR ile metin çıkarma"""
        try:
            images = convert_from_path(pdf_path, dpi=300)
            
            all_text = ""
            for i, image in enumerate(images):
                try:
                    text = pytesseract.image_to_string(image, lang='tur+eng')
                    text = re.sub(r'\s+', ' ', text)
                    text = text.replace('|', ' ')
                    all_text += text + "\n"
                    
                    logger.info(f"OCR ile sayfa {i+1}'den {len(text)} karakter çıkarıldı")
                    
                except Exception as page_error:
                    logger.error(f"Sayfa {i+1} OCR hatası: {page_error}")
                    continue
            
            all_text = all_text.replace('—', '-')
            all_text = all_text.replace('"', '"').replace('"', '"')
            all_text = all_text.replace('´', "'")
            all_text = re.sub(r'[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]+', ' ', all_text)
            all_text = all_text.strip()
            
            logger.info(f"OCR toplam metin uzunluğu: {len(all_text)}")
            return all_text
            
        except Exception as e:
            logger.error(f"OCR metin çıkarma hatası: {e}")
            return ""
    
    def detect_document_type(self, text: str) -> str:
        """Belge türünü tespit et: 'analysis_report' veya 'procedure_document'"""
        
        # Analiz raporu belirtileri
        analysis_indicators = [
            r"(?:analiz|analysis)\s+(?:rapor|report)",
            r"(?:bulgular|findings)",
            r"(?:sonuç|result|conclusion)",
            r"(?:değerlendirme|assessment|evaluation)",
            r"(?:kabul\s*edilebilir|acceptable)",
            r"(?:uygun|suitable|compliant)",
            r"(?:mevzuat|regulation|directive)",
            r"(?:teknik\s*değerlendirme|technical\s*assessment)"
        ]
        
        # Prosedür dökümanı belirtileri  
        procedure_indicators = [
            r"(?:prosedür|procedure)",
            r"(?:talimat|instruction)",
            r"(?:adım|step)",
            r"(?:zone|alan)\s*\d+",
            r"(?:bakım|maintenance)\s+(?:operasyon|operation)",
            r"turn\s+off",
            r"cut\s+off",
            r"attach\s+(?:a\s+)?(?:lock|kilit)",
            r"obtaining\s+(?:the\s+)?necessary\s+permissions"
        ]
        
        analysis_count = sum(1 for pattern in analysis_indicators 
                           if re.search(pattern, text, re.IGNORECASE))
        
        procedure_count = sum(1 for pattern in procedure_indicators 
                            if re.search(pattern, text, re.IGNORECASE))
        
        logger.info(f"Analiz göstergeleri: {analysis_count}, Prosedür göstergeleri: {procedure_count}")
        
        if procedure_count > analysis_count:
            return "procedure_document"
        else:
            return "analysis_report"

    def analyze_criteria(self, text: str, category: str, document_type: str = "analysis_report") -> Dict[str, LOTOAnalysisResult]:
        """Kriterleri analiz et - belge türüne göre uyarlanmış"""
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                content = f"Bulunan: {str(matches[:3])}"
                found = True
                
                # İzolasyon noktaları tablosu ve cihaz durumu kontrol varsa tam puan ver
                if criterion_name in ["izolasyon_noktalari_tablo", "cihaz_durumu_kontrol"]:
                    score = weight  # Tam puan
                else:
                    score = min(weight, len(matches) * (weight // 2))
                    score = max(score, weight // 2)
            else:
                content = "Bulunamadı"
                found = False
                score = 0
                
                # Prosedür dökümanı için özel durumlar
                if document_type == "procedure_document":
                    score = self.handle_procedure_document_scoring(criterion_name, text, weight)
                    if score > 0:
                        found = True
                        content = "Prosedür dökümanından çıkarıldı"
            
            results[criterion_name] = LOTOAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={
                    "pattern_used": pattern,
                    "matches_count": len(matches) if matches else 0,
                    "document_type": document_type
                }
            )
        
        return results
    
    def handle_procedure_document_scoring(self, criterion_name: str, text: str, weight: int) -> int:
        """Prosedür dökümanı için özel puanlama mantığı"""
        
        # Prosedür dökümanlarında bu kriterler farklı şekilde değerlendirilir
        procedure_adaptations = {
            # Teknik değerlendirme kriterleri - prosedürde bunlar olmasalar da puan ver
            "kabul_edilebilirlik": weight,  # Prosedür varsa zaten "kabul edilmiş" demektir
            "bulgular_yorumlar": weight // 2,  # Kısmi puan
            "sonuc_tablolari": weight // 2,  # Kısmi puan  
            "oneriler": weight,  # Prosedür kendisi bir öneri
            
            # İzolasyon kriterleri - prosedürde adımlar var
            "izolasyon_noktalari_tablo": weight if re.search(r"fig|figure|diagram|şema", text, re.IGNORECASE) else 0,
            "prosedur_detaylari": weight,  # Prosedür dökümanının ana içeriği
            "tavsiyeler": weight,  # Prosedür kendisi tavsiye niteliğinde
            
            # Makine tanımı - prosedürde genelde yoktur ama kısmi puan
            "makine_tanimi": weight // 2 if re.search(r"line|hat|ekipman|equipment", text, re.IGNORECASE) else 0,
            "tesis_bilgileri": weight // 2 if re.search(r"zone|alan|facility", text, re.IGNORECASE) else 0,
            
            # Enerji analizi - prosedürde energy cutoff adımları var
            "uygunsuz_enerji_tablosu": weight if re.search(r"energy|enerji", text, re.IGNORECASE) else 0,
            
            # Mevzuat - prosedür dökümanı genelde mevzuata uygun olarak hazırlanır
            "mevzuat_uygunlugu": weight // 2,
            "mevzuat_referanslari": weight // 2,
        }
        
        return procedure_adaptations.get(criterion_name, 0)

    def check_date_validity(self, text: str) -> Dict[str, Any]:
        """Rapor tarihini bul (1 yıl kuralı artık yok)"""
        date_patterns = [
            r"(?:Rapor\s*Tarihi|Report\s*Date|Date\s*of\s*Report)\s*[:=]?\s*(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})",
            r"(?:Tarih|Date|Issue\s*Date|Prepared\s*on)\s*[:=]?\s*(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})",
            r"(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})",
            r"(\d{4})[./\-](\d{1,2})[./\-](\d{1,2})",
            # İngilizce formatlar için ek pattern'lar
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
            r"(\d{1,2})\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})"
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(str(match[2])) == 4:  # DD/MM/YYYY format
                        day, month, year = int(match[0]), int(match[1]), int(match[2])
                    else:  # YYYY/MM/DD format
                        year, month, day = int(match[0]), int(match[1]), int(match[2])
                    
                    if 1 <= day <= 31 and 1 <= month <= 12 and 2020 <= year <= 2030:
                        report_date = datetime(year, month, day)
                        current_date = datetime.now()
                        date_diff = current_date - report_date
                        
                        return {
                            "found": True,
                            "report_date": report_date.strftime("%d.%m.%Y"),
                            "days_old": date_diff.days,
                            "is_valid": True,  # Artık hep geçerli
                            "validity_reason": "Tarih bulundu"
                        }
                except:
                    continue
        
        return {
            "found": False,
            "report_date": "Bulunamadı",
            "days_old": 0,
            "is_valid": True,  # Tarih bulunamasa da artık geçerli sayalım
            "validity_reason": "Rapor tarihi bulunamadı ama kabul edilebilir"
        }

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, LOTOAnalysisResult]]) -> Dict[str, Any]:
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
            "proje_adi": "Bulunamadı",
            "rapor_tarihi": "Bulunamadı",
            "hazirlayan_firma": "Bulunamadı",
            "kabul_durumu": "Bulunamadı"
        }
        
        # Proje adı için daha geniş pattern'lar
        project_patterns = [
            r"(?:Proje\s*Ad[ıi]|Project\s*Name)\s*[:=]\s*([^\n\r]+)",
            r"(?:Belge\s*Ad[ıi]|Document\s*Title|Report\s*Title)\s*[:=]\s*([^\n\r]+)",
            r"LOTO.*?(?:Report|Rapor).*?([A-Z][A-Za-z\s0-9]+)",
            r"Lockout.*?Tagout.*?([A-Z][A-Za-z\s0-9]+)",
            r"(?:Title|Başlık)\s*[:=]\s*([^\n\r]+)"
        ]
        
        for pattern in project_patterns:
            project_match = re.search(pattern, text, re.IGNORECASE)
            if project_match:
                values["proje_adi"] = project_match.group(1).strip()[:50]
                break
        
        # Rapor tarihi için daha geniş pattern'lar
        date_patterns = [
            r"(?:Rapor\s*Tarihi|Report\s*Date|Date\s*of\s*Report)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Tarih|Date)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Issue\s*Date|Prepared\s*on)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})"
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                values["rapor_tarihi"] = date_match.group(1)
                break
        
        # Hazırlayan firma için daha geniş pattern'lar
        company_patterns = [
            r"(?:Raporu\s*Hazırlayan|Hazırlayan|Prepared\s*by|Consultant|Company|Contractor|Firma)\s*[:=]?\s*([^\n\r]+)",
            r"(?:Prepared\s*for|Client|Customer|Müşteri)\s*[:=]?\s*([^\n\r]+)",
            r"PILZ\s+MAKİNE\s+EMNİYET\s+OTOMASYON",
            r"PILZ.*?OTOMASYON",
            r"(?:Prepared|Hazırlayan).*?(PILZ[^\n\r]*)",
            r"(PILZ\s+[A-Z\s]+OTOMASYON)",
            r"(?:Engineering|Consultant|Mühendislik)\s*[:=]?\s*([^\n\r]+)"
        ]
        
        for pattern in company_patterns:
            company_match = re.search(pattern, text, re.IGNORECASE)
            if company_match:
                if len(company_match.groups()) > 0:
                    values["hazirlayan_firma"] = company_match.group(1).strip()[:50]
                else:
                    values["hazirlayan_firma"] = company_match.group().strip()[:50]
                break
        
        # Kabul durumu için pattern'lar
        acceptance_patterns = [
            r"(?:Kabul\s*Edilebilir|Acceptable|Accept)\s*[:=]?\s*(EVET|YES|HAYIR|NO|True|False)",
            r"(?:Compliance|Uygunluk)\s*[:=]?\s*(UYGUN|UYGUNSUZ|SUITABLE|UNSUITABLE|COMPLIANT|NON.*?COMPLIANT)",
            r"(?:Status|Durum|Result|Sonuç)\s*[:=]?\s*(PASS|FAIL|GEÇERLİ|GEÇERSİZ|APPROVED|REJECTED)",
            r"(UYGUN|UYGUNSUZ|SUITABLE|UNSUITABLE|PASS|FAIL|GEÇERLİ|GEÇERSİZ)"
        ]
        
        for pattern in acceptance_patterns:
            acceptance_match = re.search(pattern, text, re.IGNORECASE)
            if acceptance_match:
                values["kabul_durumu"] = acceptance_match.group(1).upper()
                break
        
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict, date_validity: Dict, document_type: str = "analysis_report") -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        
        # Tarih kontrolü artık yok, sadece bilgi amaçlı
        if date_validity["found"]:
            recommendations.append(f"� Rapor tarihi: {date_validity['report_date']}")
        else:
            recommendations.append("📅 Rapor tarihi: Tespit edilemedi")
        
        total_percentage = scores["percentage"]
        
        # Belge türüne göre eşik değerleri
        pass_threshold = 50 if document_type == "procedure_document" else 70
        
        if total_percentage >= pass_threshold:
            if document_type == "procedure_document":
                recommendations.append(f"✅ LOTO Prosedürü GEÇERLİ (Toplam: %{total_percentage:.1f})")
                recommendations.append("📝 Bu bir prosedür dökümanıdır, analiz raporu değil")
            else:
                recommendations.append(f"✅ LOTO Raporu GEÇERLİ (Toplam: %{total_percentage:.1f})")
        else:
            if document_type == "procedure_document":
                recommendations.append(f"❌ LOTO Prosedürü EKSİK (Toplam: %{total_percentage:.1f})")
                recommendations.append("📝 Bu bir prosedür dökümanıdır, analiz raporu değil")
            else:
                recommendations.append(f"❌ LOTO Raporu GEÇERSİZ (Toplam: %{total_percentage:.1f})")
        
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            # Prosedür dökümanı için daha esnek değerlendirme
            min_threshold = 30 if document_type == "procedure_document" else 40
            good_threshold = 50 if document_type == "procedure_document" else 70
            
            if category_score < min_threshold:
                recommendations.append(f"🔴 {category} bölümü yetersiz (%{category_score:.1f})")
                missing_items = [name for name, result in results.items() if not result.found]
                if missing_items:
                    recommendations.append(f"   Eksik: {', '.join(missing_items[:3])}")
            elif category_score < good_threshold:
                recommendations.append(f"🟡 {category} bölümü geliştirilmeli (%{category_score:.1f})")
            else:
                recommendations.append(f"🟢 {category} bölümü yeterli (%{category_score:.1f})")
        
        if total_percentage < pass_threshold:
            if document_type == "procedure_document":
                recommendations.extend([
                    "",
                    "💡 PROSEDÜR İYİLEŞTİRME ÖNERİLERİ:",
                    "- Daha detaylı adımlar eklenebilir",
                    "- Görsel şemalar artırılabilir",
                    "- Güvenlik uyarıları güçlendirilebilir",
                    "- Kontrol listesi eklenebilir"
                ])
            else:
                recommendations.extend([
                    "",
                    "💡 İYİLEŞTİRME ÖNERİLERİ:",
                    "- Enerji kaynakları detaylı tanımlanmalı",
                    "- İzolasyon noktaları eksiksiz belirtilmeli",
                    "- LOTO prosedürü adımları detaylandırılmalı",
                    "- Teknik değerlendirme ve sonuçlar güçlendirilmeli",
                    "- Görsel dokümantasyon artırılmalı"
                ])
        
        return recommendations

    def analyze_loto_report(self, pdf_path: str) -> Dict[str, Any]:
        """Ana LOTO rapor analiz fonksiyonu"""
        logger.info("LOTO rapor analizi başlatılıyor...")
        
        if not os.path.exists(pdf_path):
            return {"error": f"PDF dosyası bulunamadı: {pdf_path}"}
        
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "PDF'den metin çıkarılamadı"}
        
        detected_lang = self.detect_language(text)
        
        if detected_lang != 'tr' and detected_lang == 'en':
            logger.info(f"{detected_lang.upper()} dilinden Türkçe'ye çeviriliyor...")
            text = self.translate_to_turkish(text, detected_lang)
        
        # Belge türünü tespit et
        document_type = self.detect_document_type(text)
        logger.info(f"Tespit edilen belge türü: {document_type}")
        
        date_validity = self.check_date_validity(text)
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category, document_type)
        
        # Mevzuat uygunluğu bulunursa dokümantasyon bölümündeki mevzuat referanslarına da puan ver
        if ("Teknik Değerlendirme ve Sonuçlar" in analysis_results and 
            "mevzuat_uygunlugu" in analysis_results["Teknik Değerlendirme ve Sonuçlar"] and
            analysis_results["Teknik Değerlendirme ve Sonuçlar"]["mevzuat_uygunlugu"].found and
            "Dokümantasyon ve Referanslar" in analysis_results and
            "mevzuat_referanslari" in analysis_results["Dokümantasyon ve Referanslar"] and
            not analysis_results["Dokümantasyon ve Referanslar"]["mevzuat_referanslari"].found):
            
            # Mevzuat referanslarına otomatik tam puan ver
            mevzuat_ref = analysis_results["Dokümantasyon ve Referanslar"]["mevzuat_referanslari"]
            mevzuat_ref.found = True
            mevzuat_ref.content = "Teknik değerlendirmede mevzuat uygunluğu bulundu"
            mevzuat_ref.score = mevzuat_ref.max_score
        
        scores = self.calculate_scores(analysis_results)
        extracted_values = self.extract_specific_values(text)
        recommendations = self.generate_recommendations(analysis_results, scores, date_validity, document_type)
        
        # Prosedür dökümanı için daha düşük eşik değeri
        pass_threshold = 50 if document_type == "procedure_document" else 70
        final_status = "PASS" if scores["percentage"] >= pass_threshold else "FAIL"
        
        report = {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgisi": {
                "pdf_path": pdf_path,
                "detected_language": detected_lang,
                "document_type": document_type,
                "pass_threshold": pass_threshold
            },
            "tarih_gecerliligi": date_validity,
            "cikarilan_degerler": extracted_values,
            "kategori_analizleri": analysis_results,
            "puanlama": scores,
            "oneriler": recommendations,
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["percentage"],
                "durum": final_status,
                "rapor_tipi": "LOTO",
                "belge_turu": document_type
            }
        }
        
        return report

    def save_report_to_excel(self, report: Dict, output_path: str):
        """Raporu Excel'e kaydet"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_data = {
                'Kriter': ['Toplam Puan', 'Yüzde', 'Durum', 'Rapor Tipi', 'Tarih Geçerliliği'],
                'Değer': [
                    report['ozet']['toplam_puan'],
                    f"%{report['ozet']['yuzde']}",
                    report['ozet']['durum'],
                    report['ozet']['rapor_tipi'],
                    "Geçerli" if report['tarih_gecerliligi']['is_valid'] else "Geçersiz"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Özet', index=False)
            
            values_data = []
            for key, value in report['cikarilan_degerler'].items():
                values_data.append({'Kriter': key, 'Değer': str(value)})
            pd.DataFrame(values_data).to_excel(writer, sheet_name='Çıkarılan_Değerler', index=False)
            
            for category, results in report['kategori_analizleri'].items():
                category_data = []
                for criterion, result in results.items():
                    category_data.append({
                        'Kriter': criterion,
                        'Bulundu': result.found,
                        'İçerik': result.content,
                        'Puan': result.score,
                        'Maksimum Puan': result.max_score
                    })
                sheet_name = category.replace('/', '_').replace('\\', '_')[:31]
                pd.DataFrame(category_data).to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"Rapor Excel'e kaydedildi: {output_path}")

    def save_report_to_json(self, report: Dict, output_path: str):
        """Raporu JSON'a kaydet"""
        json_report = {}
        for key, value in report.items():
            if key == 'kategori_analizleri':
                json_report[key] = {}
                for category, results in value.items():
                    json_report[key][category] = {}
                    for criterion, result in results.items():
                        json_report[key][category][criterion] = asdict(result)
            else:
                json_report[key] = value

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2)

        logger.info(f"Rapor JSON'a kaydedildi: {output_path}")

def main():
    """Ana fonksiyon"""
    analyzer = LOTOReportAnalyzer()

    pdf_path = "Loto talimatı.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF dosyası bulunamadı: {pdf_path}")
        return
    
    print("🔒 LOTO Rapor Analizi Başlatılıyor...")
    print("=" * 60)
    
    report = analyzer.analyze_loto_report(pdf_path)
    
    if "error" in report:
        print(f"❌ Hata: {report['error']}")
        return
    
    print("\n📊 ANALİZ SONUÇLARI")
    print("=" * 60)
    
    print(f"📅 Analiz Tarihi: {report['analiz_tarihi']}")
    print(f"🔍 Tespit Edilen Dil: {report['dosya_bilgisi']['detected_language'].upper()}")
    print(f"📋 Toplam Puan: {report['ozet']['toplam_puan']}/100")
    print(f"📈 Yüzde: %{report['ozet']['yuzde']}")
    print(f"🎯 Durum: {report['ozet']['durum']}")
    print(f"📄 Rapor Tipi: {report['ozet']['rapor_tipi']}")
    
    print(f"\n📅 TARİH GEÇERLİLİĞİ")
    print("-" * 40)
    date_info = report['tarih_gecerliligi']
    print(f"Rapor Tarihi: {date_info['report_date']}")
    print(f"Yaş: {date_info['days_old']} gün")
    print(f"Geçerlilik: {date_info['validity_reason']}")
    
    print("\n📋 ÖNEMLİ ÇIKARILAN DEĞERLER")
    print("-" * 40)
    for key, value in report['cikarilan_degerler'].items():
        display_name = {
            "proje_adi": "Proje Adı",
            "rapor_tarihi": "Rapor Tarihi", 
            "hazirlayan_firma": "Hazırlayan Firma",
            "kabul_durumu": "Kabul Durumu"
        }.get(key, key.replace('_', ' ').title())
        print(f"{display_name}: {value}")
    
    print("\n📊 KATEGORİ PUANLARI VE DETAYLAR")
    print("=" * 60)
    for category, score_data in report['puanlama']['category_scores'].items():
        percentage = score_data['percentage']
        print(f"\n🔍 {category}: {score_data['normalized']}/{score_data['max_weight']} (%{percentage:.1f})")
        print("-" * 50)
        
        # Bu kategorinin analiz sonuçlarını göster
        if category in report['kategori_analizleri']:
            category_analysis = report['kategori_analizleri'][category]
            for criterion_name, criterion_result in category_analysis.items():
                criterion_display = criterion_name.replace('_', ' ').title()
                if hasattr(criterion_result, 'found') and criterion_result.found:
                    print(f"  ✅ {criterion_display}: {criterion_result.score}/{criterion_result.max_score} puan")
                else:
                    print(f"  ❌ {criterion_display}: 0/{criterion_result.max_score} puan - BULUNAMADI")
    
    print("\n" + "=" * 60)
    
    print("\n💡 ÖNERİLER VE DEĞERLENDİRME")
    print("-" * 40)
    for recommendation in report['oneriler']:
        print(recommendation)
    
    print("\n📋 GENEL DEĞERLENDİRME")
    print("=" * 60)
    
    if report['ozet']['yuzde'] >= 70:
        print("✅ SONUÇ: GEÇERLİ")
        print(f"🌟 Toplam Başarı: %{report['ozet']['yuzde']:.1f}")
        print("📝 Değerlendirme: LOTO raporu genel olarak yeterli kriterleri sağlamaktadır.")
    else:
        print("❌ SONUÇ: GEÇERSİZ")
        print(f"⚠️ Toplam Başarı: %{report['ozet']['yuzde']:.1f}")
        print("📝 Değerlendirme: LOTO raporu minimum gereklilikleri sağlamamaktadır.")
        
        print("\n⚠️ EKSİK GEREKLİLİKLER:")
        for category, results in report['kategori_analizleri'].items():
            missing_items = []
            for criterion, result in results.items():
                if not result.found:
                    missing_items.append(criterion)
            
            if missing_items:
                print(f"\n🔍 {category}:")
                for item in missing_items:
                    readable_name = item.replace('_', ' ').title()
                    print(f"   ❌ {readable_name}")
        
        print("\n📌 YAPILMASI GEREKENLER:")
        print("1. Eksik belgelendirmeleri tamamlayın")
        print("2. Enerji kaynakları ve izolasyon noktalarını detaylandırın")
        print("3. LOTO prosedürlerini eksiksiz tanımlayın")
        print("4. Teknik değerlendirme ve sonuçları güçlendirin")
        print("5. Görsel dokümantasyonu artırın")
        print("6. Mevzuat referanslarını ekleyin")

if __name__ == "__main__":
    main()
