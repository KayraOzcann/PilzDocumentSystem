"""
Bakım Talimatları Analiz Servisi
=================================
Azure App Service için optimize edilmiş standalone servis

Endpoint: POST /api/bakimtalimatlari-report
Health Check: GET /api/health
Port: 8014
"""

# ============================================
# IMPORTS
# ============================================
import os
import re
from datetime import datetime
from typing import Dict, List, Any
import PyPDF2
from docx import Document
from dataclasses import dataclass
import logging
import pytesseract
import cv2
import numpy as np
from PIL import Image
import pdf2image

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# ============================================
# LOGGING CONFIGURATION
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# LANGUAGE DETECTION
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
# DATA CLASSES
# ============================================
@dataclass
class MaintenanceAnalysisResult:
    """Bakım Talimatları analiz sonucu veri sınıfı"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

# ============================================
# ANALYZER CLASS
# ============================================
class MaintenanceReportAnalyzer:
    """Bakım Talimatları rapor analiz sınıfı"""
    
    def __init__(self):
        logger.info("Bakım Talimatları analiz sistemi başlatılıyor...")
        
        self.criteria_weights = {
            "Genel Bilgiler": 10,
            "Güvenlik Önlemleri": 15,
            "Bakım Türleri ve Operasyon Çeşitleri": 20,
            "Adım Adım Bakım İşlemleri": 10,
            "Teknik Veriler": 10,
            "Yedek Parça ve Sarf Malzemeleri": 15,
            "Kayıt ve İzlenebilirlik": 10,
            "Çevre ve Atık Yönetimi": 5,
            "Ekler": 5
        }
        
        self.criteria_details = {
            "Genel Bilgiler": {
                "makine_tanimi": {"pattern": r"(?:Makine|Machine|Model|Seri\s*No|Serial\s*Number|Üretici|Manufacturer|Marka|Brand|Ekipman|Equipment|Cihaz|Device|Sistem|System|Ürün|Product|Tip|Type)", "weight": 4},
                "belge_kapsami": {"pattern": r"(?:Kapsam|Scope|Geçerli|Valid|Bu\s*talimat|This\s*instruction|Coverage|Amaç|Purpose|Hedef|Target|Kullan|Use|Uygula|Apply|Alan|Field|Bölüm|Section)", "weight": 3},
                "yetkili_personel": {"pattern": r"(?:Yetkili|Authorized|Teknisyen|Technician|Eğitim|Training|Yetkin|Qualified|Personel|Personnel|Sorumlu|Responsible|Operatör|Operator|Uzman|Expert|Mühendis|Engineer)", "weight": 3}
            },
            "Güvenlik Önlemleri": {
                "genel_guvenlik_uyari": {"pattern": r"(?:Güvenlik\s*Uyar[ıi]|Safety\s*Warning|Elektrik\s*Çarp|Electric\s*Shock|Hareketli\s*Parça|Moving\s*Parts|S[ıi]cak\s*Yüzey|Hot\s*Surface|Tehlike|Danger|Hazard|Risk|Dikkat|Caution|Warning|Alert)", "weight": 4},
                "kkd_zorunluluk": {"pattern": r"(?:KKD|PPE|Gözlük|Goggle|Eldiven|Glove|Kulaklık|Earplug|Baret|Helmet|Koruyucu\s*Donan[ıi]m|Protective\s*Equipment|Maske|Mask|İş\s*Güvenlik|Work\s*Safety|Koruyucu|Protective)", "weight": 4},
                "loto_kilitleme": {"pattern": r"(?:LOTO|Lock\s*Out|Tag\s*Out|Kilitle|Lock|Kapama|Shutdown|Enerji\s*Kes|Energy\s*Cut|İzolasyon|Isolation|Güvenli\s*Durdur|Safe\s*Stop|Devreden\s*Çıkar|Disconnect|Anahtar|Switch)", "weight": 4},
                "artik_riskler": {"pattern": r"(?:Art[ıi]k\s*Risk|Residual\s*Risk|Kalan\s*Tehlike|Remaining\s*Hazard|Tehlike|Hazard|Risk\s*Değerlendirme|Risk\s*Assessment|Güvenlik\s*Riski|Safety\s*Risk)", "weight": 3}
            },
            "Bakım Türleri ve Operasyon Çeşitleri": {
                "periyodik_bakim": {"pattern": r"(?:Periyodik|Periodic|Günlük|Daily|Haftal[ıi]k|Weekly|Ayl[ıi]k|Monthly|Y[ıi]ll[ıi]k|Yearly|Yağlama|Lubrication|Filtre|Filter|Rutin|Routine|Düzenli|Regular|Planlı|Planned|Programlı|Scheduled)", "weight": 6},
                "duzeltici_bakim": {"pattern": r"(?:Düzeltici|Corrective|Ar[ıi]za|Failure|Müdahale|Intervention|Onar[ıi]m|Repair|Tamir|Fix|Çözüm|Solution|Giderme|Troubleshoot|Acil|Emergency|Reaktif|Reactive)", "weight": 5},
                "ongorul_bakim": {"pattern": r"(?:Öngörülü|Predictive|Sensör|Sensor|Titreşim|Vibration|S[ıi]cakl[ıi]k|Temperature|Analiz|Analysis|İzleme|Monitoring|Tahmin|Prediction|Kondisyon|Condition|Durum\s*İzleme|Condition\s*Monitoring)", "weight": 4},
                "operasyon_cesitleri": {"pattern": r"(?:Ayarlama|Adjustment|Temizlik|Cleaning|Dezenfeksiyon|Disinfection|Sökme|Disassembly|İzolasyon|Isolation|Reset|Kalibrasyon|Calibration|Test|Kontrol|Check|Muayene|Inspection|Değişim|Replacement)", "weight": 5}
            },
            "Adım Adım Bakım İşlemleri": {
                "islem_sirasi": {"pattern": r"(?:Ad[ıi]m|Step|Sıra|Order|İşlem|Process|Prosedür|Procedure|Resim|Picture|Görsel|Image|Sıral[ıi]|Sequential|Aşama|Phase|Basamak|Stage|Numara|Number)", "weight": 3},
                "gerekli_aletler": {"pattern": r"(?:Alet|Tool|Cihaz|Device|Yedek\s*Parça|Spare\s*Part|Malzeme|Material|Equipment|Teçhizat|Ekipman|Gereç|Apparatus|Takım|Kit|Set|Donan[ıi]m|Hardware)", "weight": 3},
                "kontrol_listesi": {"pattern": r"(?:Kontrol\s*Liste|Check\s*List|Liste|List|Form|Checklist|Doğrulama|Verification|Onay|Approval|İmza|Signature|Teyit|Confirm|Sınar|Validate)", "weight": 2},
                "fonksiyon_testleri": {"pattern": r"(?:Test|Fonksiyon|Function|Doğrulama|Verification|Kontrol|Check|Muayene|Inspection|Deneme|Trial|Çal[ıi]şma|Operation|Performans|Performance|Kalite|Quality)", "weight": 2}
            },
            "Teknik Veriler": {
                "yaglama_bilgileri": {"pattern": r"(?:Yağlama|Lubrication|Yağ\s*Tür|Oil\s*Type|Yağ\s*Miktar|Oil\s*Amount|Gres|Grease|Lubricant|Yağl[ıi]|Oiled|Viscosity|Viskozite|Motor\s*Yağ|Engine\s*Oil)", "weight": 3},
                "tork_ayar_degerleri": {"pattern": r"(?:Tork|Torque|Ayar|Setting|Ölçü|Measure|Tolerans|Tolerance|Boşluk|Clearance|Değer|Value|Parametre|Parameter|Spec|Specification|Limit|S[ıi]n[ıi]r)", "weight": 3},
                "basinc_degerleri": {"pattern": r"(?:Bas[ıi]nç|Pressure|Elektrik|Electric|Pnömatik|Pneumatic|Hidrolik|Hydraulic|Bar|PSI|Volt|Ampere|Watt|kPa|MPa|V|A|W|Power|Güç)", "weight": 2},
                "sarf_malzeme_omru": {"pattern": r"(?:Ömür|Life|Sarf|Consumable|Filtre|Filter|Kay[ıi]ş|Belt|Conta|Gasket|Değişim|Change|Replace|Renewal|Yenileme|Service\s*Life|Working\s*Life|Kullan[ıi]m\s*Ömrü)", "weight": 2}
            },
            "Yedek Parça ve Sarf Malzemeleri": {
                "orijinal_parca_listesi": {"pattern": r"(?:Orijinal|Original|Yedek\s*Parça|Spare\s*Part|Parça\s*No|Part\s*Number|Liste|List|Katalog|Catalog|Code|Kod|OEM|Genuine|Gerçek|Authentic)", "weight": 6},
                "kritik_stok": {"pattern": r"(?:Kritik|Critical|Stok|Stock|Bulundur|Keep|Tavsiye|Recommend|Rezerv|Reserve|Minimum|Emergency|Acil|Zorunlu|Must|Required|Gerekli)", "weight": 5},
                "yanlis_parca_riski": {"pattern": r"(?:Yanl[ıi]ş|Wrong|Risk|Tehlike|Hazard|Uyar[ıi]|Warning|Dikkat|Attention|Caution|Uyumlu|Compatible|Uyumsuz|Incompatible|Doğru|Correct)", "weight": 4}
            },
            "Kayıt ve İzlenebilirlik": {
                "bakim_formu": {"pattern": r"(?:Bak[ıi]m\s*Form|Maintenance\s*Form|Kay[ıi]t|Record|Tarih|Date|Sorumlu|Responsible|Log|Rapor|Report|Dosya|File|Belge|Document|Archive|Arşiv)", "weight": 4},
                "ariza_kayitlari": {"pattern": r"(?:Ar[ıi]za\s*Kay[ıi]t|Failure\s*Record|Takip|Track|Değiştir|Replace|Geçmiş|History|Trend|İstatistik|Statistics|Analiz|Analysis|Log)", "weight": 3},
                "yasal_izlenebilirlik": {"pattern": r"(?:Yasal|Legal|İzlenebilir|Traceable|Gereklilik|Requirement|Uygunluk|Compliance|Standart|Standard|Sertifika|Certificate|Audit|Denetim|Regulation|Regulasyon)", "weight": 3}
            },
            "Çevre ve Atık Yönetimi": {
                "atik_bertaraf": {"pattern": r"(?:At[ıi]k|Waste|Bertaraf|Disposal|Yağ|Oil|Filtre|Filter|Akü|Battery|İmha|Destruction|Geri\s*Dönüşüm|Recycling|Çevre|Environment)", "weight": 2},
                "cevre_koruma": {"pattern": r"(?:Çevre|Environment|Zarar|Damage|İmha|Destruction|Geri\s*Dönüşüm|Recycling|Kirletici|Pollutant|Temiz|Clean|Yeşil|Green|Sürdürülebilir|Sustainable)", "weight": 2},
                "talimat_yontem": {"pattern": r"(?:Talimat|Instruction|Yöntem|Method|Prosedür|Procedure|Guide|Kılavuz|Guideline|Protocol|Protokol|Manual|El\s*Kitab[ıi])", "weight": 1}
            },
            "Ekler": {
                "resimli_sema": {"pattern": r"(?:Resim|Picture|Şema|Scheme|Diyagram|Diagram|Çizim|Drawing|Plan|Grafik|Graphic|Chart|Tablo|Table|Figure|Şekil|Image|Görsel)", "weight": 1},
                "ariza_teshis": {"pattern": r"(?:Ar[ıi]za\s*Teşhis|Fault\s*Diagnosis|Sorun|Problem|Neden|Cause|Çözüm|Solution|Troubleshoot|Debug|Hata|Error|Issue|Mesele)", "weight": 1},
                "iletisim_bilgi": {"pattern": r"(?:İletişim|Contact|Üretici|Manufacturer|Servis|Service|Telefon|Phone|E-mail|Mail|Address|Adres|Support|Destek|Help|Yard[ıi]m|Hotline)", "weight": 3}
            }
        }
    
    def detect_language(self, text: str) -> str:
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
        if source_lang != 'tr':
            logger.info(f"Tespit edilen dil: {source_lang.upper()} - Çeviri yapılmıyor, orijinal metin kullanılıyor")
        return text
    
    def extract_text_from_file(self, file_path: str) -> str:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            logger.error(f"Desteklenmeyen dosya formatı: {file_ext}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den sadece PyPDF2 ile metin çıkarma"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                all_text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    page_text = re.sub(r'\s+', ' ', page_text)
                    all_text += page_text + "\n"
                return all_text.strip()
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {e}")
            return ""

    def extract_text_from_docx(self, docx_path: str) -> str:
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
            return text.strip()
        except Exception as e:
            logger.error(f"DOCX metin çıkarma hatası: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except:
            try:
                with open(txt_path, 'r', encoding='cp1254') as file:
                    return file.read().strip()
            except Exception as e:
                logger.error(f"TXT metin çıkarma hatası: {e}")
                return ""
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, MaintenanceAnalysisResult]:
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
                    score = max(score, weight // 2)  # ✅ Minimum yarı puan
            else:
                content = "Bulunamadı"
                found = False
                score = 0
            
            results[criterion_name] = MaintenanceAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_count": len(matches) if matches else 0}
            )
        
        return results

    def calculate_scores(self, analysis_results: Dict) -> Dict[str, Any]:
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
        """✅ TAM DETAYLI PATTERN'LAR - 200+ SATIR"""
        values = {
            "makine_adi": "Bulunamadı",
            "makine_modeli": "Bulunamadı",
            "seri_numarasi": "Bulunamadı",
            "bakim_turu": "Bulunamadı",
            "yetkili_personel": "Bulunamadı"
        }
        
        # ========================= MAKİNE ADI =========================
        machine_title_patterns = [
            r"([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,40})\s+(?:MAKİNESİ|MACHINE)\s+(?:BAKIM|KULLANIM|SERVİS|MAINTENANCE|SERVICE|OPERATION)",
            r"([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,40})\s+(?:CİHAZI|DEVICE)\s+(?:BAKIM|KULLANIM|SERVİS|MAINTENANCE|SERVICE)",
            r"([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,40})\s+(?:SİSTEMİ|SYSTEM)\s+(?:BAKIM|KULLANIM|SERVİS|MAINTENANCE|SERVICE)",
            r"([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,40})\s+(?:EKİPMANI|EQUIPMENT)\s+(?:BAKIM|KULLANIM|SERVİS|MAINTENANCE|SERVICE)",
            r"([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,40})\s+(?:BAKIM|MAINTENANCE)\s+(?:TALİMAT|KILAVUZ|PROSEDÜR|INSTRUCTION|MANUAL|PROCEDURE)",
            r"([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,40})\s+(?:SERVİS|SERVICE)\s+(?:TALİMAT|KILAVUZ|MANUAL|GUIDE)",
            r"([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,40})\s+(?:KULLANIM|OPERATION|USER)\s+(?:KILAVUZ|MANUAL|GUIDE)"
        ]
        
        machine_field_patterns = [
            r"(?i)(?:ÜRÜN\s*ADI|ÜRÜN\s*TANIM|PRODUCT\s*NAME|PRODUCT\s*DESCRIPTION)\s*[:=]\s*([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,50})(?=\s*$|\s+(?:PROJE|MODEL|TİP|PROJECT|TYPE|\n))",
            r"(?i)(?:MAKİNE\s*ADI|MACHINE\s*NAME|MACHINE\s*TITLE)\s*[:=]\s*([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,50})(?=\s*$|\s+(?:PROJE|MODEL|PROJECT|\n))",
            r"(?i)(?:EKİPMAN\s*ADI|EQUIPMENT\s*NAME|EQUIPMENT\s*TITLE)\s*[:=]\s*([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,50})(?=\s*$|\s+(?:PROJE|MODEL|PROJECT|\n))",
            r"(?i)(?:CİHAZ\s*ADI|DEVICE\s*NAME|DEVICE\s*TITLE)\s*[:=]\s*([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,50})(?=\s*$|\s+(?:PROJE|MODEL|PROJECT|\n))",
            r"(?i)(?:SİSTEM\s*ADI|SYSTEM\s*NAME|SYSTEM\s*TITLE)\s*[:=]\s*([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,50})(?=\s*$|\s+(?:PROJE|MODEL|PROJECT|\n))"
        ]
        
        for pattern in machine_title_patterns:
            match = re.search(pattern, text)
            if match:
                result = match.group(1).strip()
                if 3 <= len(result) <= 50 and not result.lower().startswith(("bu", "şu", "o ")):
                    values["makine_adi"] = result
                    break
        
        if values["makine_adi"] == "Bulunamadı":
            for pattern in machine_field_patterns:
                match = re.search(pattern, text)
                if match:
                    result = match.group(1).strip()
                    cleanup_words = ["PROJESİ", "PROJECT", "SİSTEMİ", "SYSTEM", "EKİPMANI", "EQUIPMENT", "TALİMATI", "INSTRUCTION"]
                    for cleanup_word in cleanup_words:
                        result = re.sub(rf"\b{cleanup_word}\b", "", result, flags=re.IGNORECASE).strip()
                    result = re.sub(r'\s+', ' ', result).strip()
                    if 3 <= len(result) <= 50 and not result.lower().startswith(("bu", "şu", "o ")):
                        values["makine_adi"] = result
                        break
        
        # ========================= MODEL =========================
        model_title_patterns = [
            r"(?:MODEL|TİP|TYPE)\s+([A-Z0-9-]{2,20})\s+(?:MAKİNESİ|CİHAZI|SİSTEMİ|BAKIM|SERVİS)",
            r"([A-Z0-9-]{2,20})\s+(?:MODEL|TİP|TYPE)\s+(?:BAKIM|SERVİS|KULLANIM|MAINTENANCE)",
            r"([A-Z0-9-]{2,20})\s+(?:SERİSİ|SERIES)\s+(?:BAKIM|SERVİS|KULLANIM)"
        ]
        
        model_field_patterns = [
            r"(?i)(?:MODEL\s*NO|MODEL\s*NUMARASI|MODEL\s*NUMBER)\s*[:=]\s*([A-Z0-9-]{2,20})(?=\s|$|\n)",
            r"(?i)(?:MODEL\s*KODU|MODEL\s*CODE)\s*[:=]\s*([A-Z0-9-]{2,20})(?=\s|$|\n)",
            r"(?i)(?:MODEL)\s*[:=]\s*([A-Z0-9-]{2,20})(?=\s|$|\n)",
            r"(?i)(?:TİP|TYPE)\s*[:=]\s*([A-Z0-9-]{2,20})(?=\s|$|\n)",
            r"(?i)(?:TİP\s*NO|TYPE\s*NO|TİP\s*NUMARASI|TYPE\s*NUMBER)\s*[:=]\s*([A-Z0-9-]{2,20})(?=\s|$|\n)"
        ]
        
        for pattern in model_title_patterns:
            match = re.search(pattern, text)
            if match:
                result = match.group(1).strip()
                if 2 <= len(result) <= 25:
                    values["makine_modeli"] = result
                    break
        
        if values["makine_modeli"] == "Bulunamadı":
            for pattern in model_field_patterns:
                match = re.search(pattern, text)
                if match:
                    result = match.group(1).strip()
                    filter_words = ["konveyör", "tipi", "çeşit", "türü", "xxx", "tbd", "n/a"]
                    if not any(word in result.lower() for word in filter_words):
                        if 2 <= len(result) <= 25:
                            values["makine_modeli"] = result
                            break
        
        # ========================= SERİ NO =========================
        serial_title_patterns = [
            r"(?:SERİ|SERIAL)\s+(?:NO|NUMBER|NUMARASI)[\s:=]*([A-Z0-9-]{3,25})",
            r"(?:S/N|SN)[\s:=]*([A-Z0-9-]{3,25})"
        ]
        
        serial_field_patterns = [
            r"(?i)(?:SERİ\s*NO|SERİ\s*NUMARASI|SERIAL\s*NUMBER|S/N)\s*[:=]\s*([A-Z0-9-]{3,25})(?=\s+(?:Üretim|Tarih|Rev|Revizyon|\n)|$)",
            r"(?i)(?:SERİ\s*KODU|SERIAL\s*CODE)\s*[:=]\s*([A-Z0-9-]{3,25})(?=\s+(?:Üretim|Tarih|Rev|\n)|$)",
            r"(?i)(?:SERIAL)\s*[:=]\s*([A-Z0-9-]{3,25})(?=\s|$)"
        ]
        
        placeholder_patterns = [
            r"^[X]{2,}$", r"^[X-]{2,}$", r".*[X]{3,}.*", r"^[-]{2,}$",
            r"(?i)^(tbd|n/a|na|null|none|boş|yok)$", r"(?i).*rev[x0-9].*",
            r"^[0]{3,}$", r"(?i)^(sample|örnek|example).*"
        ]
        
        for pattern in serial_title_patterns:
            match = re.search(pattern, text)
            if match:
                result = match.group(1).strip()
                is_placeholder = any(re.match(p, result) for p in placeholder_patterns)
                if not is_placeholder and 3 <= len(result) <= 30:
                    values["seri_numarasi"] = result
                    break
        
        if values["seri_numarasi"] == "Bulunamadı":
            for pattern in serial_field_patterns:
                match = re.search(pattern, text)
                if match:
                    result = match.group(1).strip()
                    is_placeholder = any(re.match(p, result) for p in placeholder_patterns)
                    if not is_placeholder and 3 <= len(result) <= 30:
                        values["seri_numarasi"] = result
                        break
        
        # ========================= BAKIM TÜRÜ =========================
        maintenance_title_patterns = [
            r"(PERİYODİK\s*BAKIM|PERIODIC\s*MAINTENANCE)\s+(?:TALİMAT|PROSEDÜR|MANUAL)",
            r"(DÜZELTİCİ\s*BAKIM|CORRECTIVE\s*MAINTENANCE)\s+(?:TALİMAT|PROSEDÜR|MANUAL)",
            r"(ÖNGÖRÜLÜ\s*BAKIM|PREDICTIVE\s*MAINTENANCE)\s+(?:TALİMAT|PROSEDÜR|MANUAL)",
            r"(GÜNLÜK\s*BAKIM|DAILY\s*MAINTENANCE)\s+(?:TALİMAT|PROSEDÜR|MANUAL)",
            r"(HAFTALIK\s*BAKIM|WEEKLY\s*MAINTENANCE)\s+(?:TALİMAT|PROSEDÜR|MANUAL)",
            r"(AYLIK\s*BAKIM|MONTHLY\s*MAINTENANCE)\s+(?:TALİMAT|PROSEDÜR|MANUAL)",
            r"(YILLIK\s*BAKIM|YEARLY\s*MAINTENANCE|ANNUAL\s*MAINTENANCE)\s+(?:TALİMAT|PROSEDÜR|MANUAL)"
        ]
        
        maintenance_word_patterns = [
            r"\b(Periyodik\s*Bakım|Periodic\s*Maintenance)\b",
            r"\b(Düzeltici\s*Bakım|Corrective\s*Maintenance)\b",
            r"\b(Öngörülü\s*Bakım|Predictive\s*Maintenance)\b",
            r"\b(Önleyici\s*Bakım|Preventive\s*Maintenance)\b",
            r"\b(Günlük\s*Bakım|Daily\s*Maintenance)\b",
            r"\b(Haftalık\s*Bakım|Weekly\s*Maintenance)\b",
            r"\b(Aylık\s*Bakım|Monthly\s*Maintenance)\b",
            r"\b(Yıllık\s*Bakım|Yearly\s*Maintenance|Annual\s*Maintenance)\b",
            r"\b(Acil\s*Bakım|Emergency\s*Maintenance)\b",
            r"\b(Planlı\s*Bakım|Planned\s*Maintenance)\b"
        ]
        
        for pattern in maintenance_title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                if len(result) <= 35:
                    values["bakim_turu"] = result
                    break
        
        if values["bakim_turu"] == "Bulunamadı":
            for pattern in maintenance_word_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result = match.group().strip()
                    if len(result) <= 35:
                        values["bakim_turu"] = result
                        break
        
        # ========================= YETKİLİ PERSONEL =========================
        personnel_title_patterns = [
            r"(?:SORUMLU|RESPONSIBLE)[\s:=]*([A-ZÇĞİÖŞÜ][a-züçğıöş\s]{2,30})(?=\s|$|\n)",
            r"(?:TEKNİSYEN|TECHNICIAN)[\s:=]*([A-ZÇĞİÖŞÜ][a-züçğıöş\s]{2,30})(?=\s|$|\n)",
            r"(?:OPERATÖR|OPERATOR)[\s:=]*([A-ZÇĞİÖŞÜ][a-züçğıöş\s]{2,30})(?=\s|$|\n)"
        ]
        
        personnel_field_patterns = [
            r"(?i)(?:YETKİLİ\s*PERSONEL|AUTHORIZED\s*PERSONNEL|AUTHORIZED\s*STAFF)\s*[:=]\s*([A-ZÇĞİÖŞÜ][a-züçğıöş\s]{2,30})(?=\s|$)",
            r"(?i)(?:TEKNİSYEN|TECHNICIAN|TECHNICAL\s*STAFF)\s*[:=]\s*([A-ZÇĞİÖŞÜ][a-züçğıöş\s]{2,30})(?=\s|$)",
            r"(?i)(?:SORUMLU\s*PERSONEL|RESPONSIBLE\s*PERSONNEL|RESPONSIBLE\s*STAFF|RESPONSIBLE\s*PERSON)\s*[:=]\s*([A-ZÇĞİÖŞÜ][a-züçğıöş\s]{2,30})(?=\s|$)",
            r"(?i)(?:BAKIM\s*SORUMLUSU|MAINTENANCE\s*RESPONSIBLE|MAINTENANCE\s*SUPERVISOR|MAINTENANCE\s*MANAGER)\s*[:=]\s*([A-ZÇĞİÖŞÜ][a-züçğıöş\s]{2,30})(?=\s|$)",
            r"(?i)(?:OPERATÖR|OPERATOR|MACHINE\s*OPERATOR)\s*[:=]\s*([A-ZÇĞİÖŞÜ][a-züçğıöş\s]{2,30})(?=\s|$)",
            r"(?i)(?:UZMAN|EXPERT|SPESİYALİST|SPECIALIST|TECHNICAL\s*EXPERT)\s*[:=]\s*([A-ZÇĞİÖŞÜ][a-züçğıöş\s]{2,30})(?=\s|$)"
        ]
        
        for pattern in personnel_title_patterns:
            match = re.search(pattern, text)
            if match:
                result = match.group(1).strip()
                filter_words = ["saklanması", "gereken", "yapılması", "bulunması", "olması", "edilmesi", "sağlanması"]
                if not any(word in result.lower() for word in filter_words):
                    if len(result.split()) >= 2 or (len(result.split()) == 1 and len(result) >= 4):
                        if 3 <= len(result) <= 35:
                            values["yetkili_personel"] = result
                            break
        
        if values["yetkili_personel"] == "Bulunamadı":
            for pattern in personnel_field_patterns:
                match = re.search(pattern, text)
                if match:
                    result = match.group(1).strip()
                    filter_words = ["saklanması", "gereken", "yapılması", "bulunması", "olması", "edilmesi", "sağlanması"]
                    if not any(word in result.lower() for word in filter_words):
                        if len(result.split()) >= 2 or (len(result.split()) == 1 and len(result) >= 4):
                            if 3 <= len(result) <= 35:
                                values["yetkili_personel"] = result
                                break
        
        return values

    def validate_maintenance_document(self, text: str) -> bool:
        """✅ GENİŞLETİLMİŞ VALIDASYON - 50+ KEYWORD"""
        maintenance_keywords = [
            "bakım", "maintenance", "talimat", "instruction", "kılavuz", "manual", "guide",
            "makine", "machine", "ekipman", "equipment", "cihaz", "device", "sistem", "system",
            "güvenlik", "safety", "tehlike", "hazard", "risk", "dikkat", "caution", "warning",
            "prosedür", "procedure", "işlem", "process", "operasyon", "operation", "çalışma", "work",
            "onarım", "repair", "servis", "service", "tamir", "fix", "düzeltici", "corrective",
            "kontrol", "check", "muayene", "inspection", "test", "doğrulama", "verification",
            "periyodik", "periodic", "günlük", "daily", "haftalık", "weekly", "aylık", "monthly",
            "parça", "part", "yedek", "spare", "malzeme", "material", "sarf", "consumable",
            "yağlama", "lubrication", "filtre", "filter", "ayarlama", "adjustment", "kalibrasyon", "calibration",
            "alet", "tool", "teçhizat", "apparatus", "donanım", "hardware"
        ]
        
        found_keywords = 0
        found_words = []
        for keyword in maintenance_keywords:
            if re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
                found_keywords += 1
                found_words.append(keyword)
        
        logger.info(f"Doküman validasyonu: {found_keywords} anahtar kelime bulundu: {found_words[:10]}")
        return found_keywords >= 2

    def generate_improvement_actions(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """✅ DİNAMİK İYİLEŞTİRME ÖNERİLERİ"""
        actions = []
        sorted_categories = sorted(scores["category_scores"].items(), key=lambda x: x[1]["percentage"])
        
        category_actions = {
            "Genel Bilgiler": [
                "Makine tanımını (ad, model, seri numarası, üretici) netleştiriniz",
                "Belge kapsamını ve hangi makine(ler) için geçerli olduğunu belirtiniz",
                "Yetkili personel niteliklerini (eğitilmiş teknisyen vb.) tanımlayınız"
            ],
            "Güvenlik Önlemleri": [
                "Genel güvenlik uyarılarını (elektrik çarpması, hareketli parçalar) ekleyiniz",
                "KKD gerekliliklerini (gözlük, eldiven, kulaklık) detaylandırınız",
                "LOTO prosedürlerini ve makineyi güvenli duruma getirme adımlarını belirtiniz",
                "Artık riskleri ve bakım sırasındaki tehlikeleri açıklayınız"
            ],
            "Bakım Türleri ve Operasyon Çeşitleri": [
                "Periyodik bakım türlerini (günlük, haftalık, aylık, yıllık) tanımlayınız",
                "Düzeltici bakım prosedürlerini ve arıza sonrası müdahaleleri açıklayınız",
                "Öngörülü bakım yöntemlerini (sensör verileri, titreşim analizi) ekleyiniz",
                "Operasyon çeşitlerini (ayarlama, temizlik, yağlama, parça değişimi) listeleyiniz"
            ],
            "Adım Adım Bakım İşlemleri": [
                "İşlem sırasını açık ve resimli olarak hazırlayınız",
                "Gerekli alet ve yedek parçaları belirtiniz",
                "Kontrol listelerini (checklist) oluşturunuz",
                "Fonksiyon testlerini ve işlem sonrası kontrolleri açıklayınız"
            ],
            "Teknik Veriler": [
                "Yağlama noktaları, yağ türleri ve miktarlarını belirtiniz",
                "Tork değerleri, ayar ölçüleri ve boşluk toleranslarını ekleyiniz",
                "Elektrik/Pnömatik/Hidrolik bağlantı basınçlarını tanımlayınız",
                "Sarf malzeme ömürlerini (filtre, kayış, conta) listeleyiniz"
            ],
            "Yedek Parça ve Sarf Malzemeleri": [
                "Orijinal yedek parça listesini parça numaralarıyla hazırlayınız",
                "Kritik yedeklerin stokta bulundurulma tavsiyelerini ekleyiniz",
                "Yanlış parça kullanımının risklerini açıklayınız"
            ],
            "Kayıt ve İzlenebilirlik": [
                "Bakım formu/tablosunu (tarih, sorumlu, yapılan işlemler) oluşturunuz",
                "Arıza kayıtları ve değiştirilen parça takip sistemini kurunuz",
                "Yasal gerekliliklere uygun izlenebilirlik sağlayınız"
            ],
            "Çevre ve Atık Yönetimi": [
                "Kullanılmış yağ, filtre, akü atıklarının bertaraf yöntemlerini belirtiniz",
                "Çevreye zarar vermeyecek imha ve geri dönüşüm talimatlarını ekleyiniz",
                "Atık yönetimi prosedürlerini ve çevresel uygunluk kriterlerini tanımlayınız"
            ],
            "Ekler": [
                "Resimli şema ve diyagramları ekleyiniz",
                "Arıza teşhis tablosunu (sorun-neden-çözüm) hazırlayınız",
                "İletişim bilgilerini (üretici, servis sağlayıcı) güncel tutunuz"
            ]
        }
        
        for category, score_data in sorted_categories[:5]:
            if score_data["percentage"] < 70:
                actions.extend(category_actions.get(category, []))
        
        if scores["percentage"] < 50:
            actions.insert(0, "ÖNCE: Doküman yapısını ve formatını yeniden gözden geçiriniz")
        
        return actions

    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        recommendations = []
        total_percentage = scores["percentage"]
        
        if total_percentage >= 70:
            recommendations.append(f"✅ Bakım Talimatları GEÇERLİ (Toplam: %{total_percentage:.1f})")
        else:
            recommendations.append(f"❌ Bakım Talimatları GEÇERSİZ (Toplam: %{total_percentage:.1f})")
        
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            if category_score < 40:
                recommendations.append(f"🔴 {category} bölümü yetersiz (%{category_score:.1f})")
                missing_items = [name for name, result in results.items() if not result.found]
                if missing_items:
                    recommendations.append(f"   Eksik: {', '.join(missing_items[:3])}")
            elif category_score < 70:
                recommendations.append(f"🟡 {category} bölümü geliştirilmeli (%{category_score:.1f})")
            else:
                recommendations.append(f"🟢 {category} bölümü yeterli (%{category_score:.1f})")
        
        return recommendations

    def analyze_maintenance_report(self, file_path: str) -> Dict[str, Any]:
        logger.info("Bakım Talimatları analizi başlatılıyor...")
        
        if not os.path.exists(file_path):
            return {"error": f"Dosya bulunamadı: {file_path}"}
        
        text = self.extract_text_from_file(file_path)
        if not text:
            return {"error": "Dosyadan metin çıkarılamadı"}
        
        detected_lang = self.detect_language(text)
        
        if detected_lang != 'tr':
            logger.info(f"{detected_lang.upper()} dilinden Türkçe'ye çeviriliyor...")
            text = self.translate_to_turkish(text, detected_lang)
        
        if not self.validate_maintenance_document(text):
            return {
                "error": "YANLIŞ DOKÜMAN: Bu dosya bakım talimatları dökümanı değil!",
                "document_type": "UNKNOWN",
                "suggestion": "Lütfen geçerli bir bakım talimatları dökümanı yükleyiniz."
            }
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category)
        
        scores = self.calculate_scores(analysis_results)
        extracted_values = self.extract_specific_values(text)
        recommendations = self.generate_recommendations(analysis_results, scores)
        improvement_actions = self.generate_improvement_actions(analysis_results, scores)
        
        final_status = "PASS" if scores["percentage"] >= 70 else "FAIL"
        
        return {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgisi": {
                "file_path": file_path,
                "file_type": os.path.splitext(file_path)[1].lower(),
                "detected_language": detected_lang
            },
            "cikarilan_degerler": extracted_values,
            "kategori_analizleri": analysis_results,
            "puanlama": scores,
            "oneriler": recommendations,
            "iyilestirme_eylemleri": improvement_actions,
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["percentage"],
                "durum": final_status,
                "rapor_tipi": "BAKIM_TALIMATLARI"
            }
        }

# ============================================
# HELPER FUNCTIONS (3-STAGE VALIDATION)
# ============================================
def validate_document_server(text):
    critical_terms = [
        ["bakım", "maintenance", "servis", "service", "onarım", "repair", "periyodik", "periodic"],
        ["makine", "machine", "ekipman", "equipment", "cihaz", "device", "sistem", "system"],
        ["prosedür", "procedure", "talimat", "instruction", "adım", "step", "kontrol", "check"],
        ["güvenlik", "safety", "uyarı", "warning", "dikkat", "caution", "tehlike", "danger"]
    ]
    category_found = [any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category) for category in critical_terms]
    valid = sum(category_found)
    logger.info(f"Validasyon: {valid}/4 kritik kategori")
    return valid >= 2

def check_strong_keywords_first_pages(filepath):
    strong_keywords = ["bakım", "bakim", "bakım talimatı", "bakim talimati", "maintenance", "MAINTENANCE", "servis", "service", "onarım", "repair", "prosedür", "procedure"]
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        found = [kw for kw in strong_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa: {len(found)} özgü kelime")
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"OCR hatası: {e}")
        return False

def check_excluded_keywords_first_pages(filepath):
    excluded = [
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm", "enclosure", "wrp-", "light curtain", "contactors", "controller",
        "espe", "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219", "teknik resim", "tasarım",
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide", "kılavuzu",
        "loto", "lvd", "TOPRAKLAMA SÜREKLİLİK", "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        "uygunluk", "beyan", "muayene", "conformity", "declaration",
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        "montaj", "assembly", "topraklama direnci", "grounding", "earthing", "60204", "topraklama", "TOPRAKLAMA DİRENCİ",
        "titreşim", "vibration", "mekanik", "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti"
    ]
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=200, first_page=1, last_page=1)
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        found = [kw for kw in excluded if re.search(rf"\b{kw.lower()}\b", all_text)]
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"Excluded OCR hatası: {e}")
        return False

def get_conclusion_message(status, percentage):
    if status == "PASS":
        return f"Bakım talimatları yüksek kalitede ve standartlara uygun (%{percentage:.0f})"
    return f"Bakım talimatları yetersiz, kapsamlı revizyon gerekli (%{percentage:.0f})"

def get_main_issues(analysis_result):
    issues = []
    for category, score_data in analysis_result['puanlama']['category_scores'].items():
        if score_data['percentage'] < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    if not issues and analysis_result['puanlama']['total_score'] < 50:
        issues = ["Genel makine bilgileri eksik", "Güvenlik önlemleri yetersiz", "Bakım türleri tanımlanmamış", "Adım adım talimatlar eksik"]
    return issues[:4]

def map_language_code(lang_code):
    lang_mapping = {'tr': 'turkish', 'en': 'english', 'de': 'german', 'fr': 'french', 'es': 'spanish', 'it': 'italian'}
    return lang_mapping.get(lang_code, 'turkish')

# ============================================
# FLASK SERVICE
# ============================================
app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads_bakim'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/bakimtalimatlari-report', methods=['POST'])
def analyze_maintenance_report():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        analyzer = MaintenanceReportAnalyzer()
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # 3 AŞAMALI KONTROL
        if file_ext == '.pdf':
            logger.info("Aşama 1: Bakım özgü kelime kontrolü...")
            if check_strong_keywords_first_pages(filepath):
                logger.info("✅ Aşama 1 geçti")
            else:
                logger.info("Aşama 2: Excluded kelime kontrolü...")
                if check_excluded_keywords_first_pages(filepath):
                    logger.info("❌ Excluded kelimeler bulundu")
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    return jsonify({'error': 'Invalid document type', 'message': 'Yüklediğiniz dosya bakım talimatları değil!'}), 400
                else:
                    logger.info("Aşama 3: Tam doküman kontrolü...")
                    try:
                        with open(filepath, 'rb') as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
                        if not text or len(text.strip()) < 50 or not validate_document_server(text):
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({'error': 'Invalid document type', 'message': 'Yüklediğiniz dosya bakım talimatları değil!'}), 400
                    except Exception as e:
                        logger.error(f"Aşama 3 hatası: {e}")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({'error': 'Analysis failed'}), 500
                    
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
                return jsonify({'error': 'Invalid document type', 'message': 'Yüklediğiniz dosya bakım talimatları değil!'}), 400

        logger.info(f"Bakım analizi yapılıyor: {filename}")
        analysis_result = analyzer.analyze_maintenance_report(filepath)
        
        try:
            os.remove(filepath)
        except:
            pass

        if 'error' in analysis_result:
            return jsonify({'error': 'Analysis failed', 'message': analysis_result['error']}), 400

        overall_percentage = analysis_result['puanlama']['percentage']
        status = "PASS" if overall_percentage >= 70 else "FAIL"
        
        response_data = {
            'analysis_date': analysis_result.get('analiz_tarihi'),
            'analysis_id': f"bakim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'category_scores': {},
            'extracted_values': analysis_result.get('cikarilan_degerler', {}),
            'file_type': 'BAKIM_TALIMATLARI',
            'filename': filename,
            'language_info': {'detected_language': map_language_code(analysis_result['dosya_bilgisi']['detected_language'])},
            'overall_score': {
                'percentage': round(overall_percentage, 2),
                'total_points': analysis_result['puanlama']['total_score'],
                'max_points': 100,
                'status': status,
                'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ'
            },
            'recommendations': analysis_result.get('oneriler', []),
            'improvement_actions': analysis_result.get('iyilestirme_eylemleri', []),
            'summary': {
                'is_valid': status == "PASS",
                'conclusion': get_conclusion_message(status, overall_percentage),
                'main_issues': [] if status == "PASS" else get_main_issues(analysis_result)
            }
        }
        
        for category, score_data in analysis_result['puanlama']['category_scores'].items():
            response_data['category_scores'][category] = {
                'score': score_data['normalized'],
                'max_score': score_data['max_weight'],
                'percentage': score_data['percentage'],
                'status': 'PASS' if score_data['percentage'] >= 70 else 'FAIL'
            }

        return jsonify({'success': True, 'message': 'Bakım Talimatları başarıyla analiz edildi', 'analysis_service': 'bakim_talimatları', 'data': response_data})

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Bakım Talimatları Analyzer API', 'version': '1.0.0', 'tesseract_available': tesseract_available, 'report_type': 'BAKIM_TALIMATLARI'})

@app.route('/', methods=['GET'])
def index():
    return jsonify({'service': 'Bakım Talimatları Analyzer API', 'version': '1.0.0', 'endpoints': {'POST /api/bakimtalimatlari-report': 'Bakım talimatları analizi', 'GET /api/health': 'Health check'}})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8014))
    logger.info(f"🚀 Bakım Talimatları Servisi - Port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)