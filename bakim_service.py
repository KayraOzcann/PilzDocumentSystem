# ============================================
# BAKIM TALİMATLARI ANALİZ SERVİSİ
# Standalone Service - Azure App Service Ready
# Port: 8014
# ============================================

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
# ANALİZ SINIFI - DATA CLASSES
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
# ANALİZ SINIFI - MAIN ANALYZER
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
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Dosya formatına göre metin çıkarma"""
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
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"DOCX metin çıkarma hatası: {e}")
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """TXT dosyasından metin çıkarma"""
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
            "makine_adi": "Bulunamadı",
            "makine_modeli": "Bulunamadı",
            "bakim_turu": "Bulunamadı"
        }
        
        machine_patterns = [
            r"(?:MAKİNE\s*ADI|MACHINE\s*NAME)\s*[:=]\s*([A-ZÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ\s-]{3,50})"
        ]
        
        for pattern in machine_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["makine_adi"] = match.group(1).strip()
                break
        
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        
        total_percentage = scores["percentage"]
        
        if total_percentage >= 70:
            recommendations.append(f"✅ Bakım Talimatları GEÇERLİ (Toplam: %{total_percentage:.1f})")
        else:
            recommendations.append(f"❌ Bakım Talimatları GEÇERSİZ (Toplam: %{total_percentage:.1f})")
        
        return recommendations

    def analyze_maintenance_report(self, file_path: str) -> Dict[str, Any]:
        """Ana Bakım Talimatları analiz fonksiyonu"""
        logger.info("Bakım Talimatları analizi başlatılıyor...")
        
        if not os.path.exists(file_path):
            return {"error": f"Dosya bulunamadı: {file_path}"}
        
        text = self.extract_text_from_file(file_path)
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
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["percentage"],
                "durum": final_status,
                "rapor_tipi": "BAKIM_TALIMATLARI"
            }
        }
    
# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_document_server(text):
    """Server kodunda doküman validasyonu - Bakım Talimatları için"""
    critical_terms = [
        ["bakım", "maintenance", "servis", "service", "onarım", "repair", "periyodik", "periodic"],
        ["makine", "machine", "ekipman", "equipment", "cihaz", "device", "sistem", "system"],
        ["prosedür", "procedure", "talimat", "instruction", "adım", "step", "kontrol", "check"],
        ["güvenlik", "safety", "uyarı", "warning", "dikkat", "caution", "tehlike", "danger"]
    ]
    
    category_found = [
        any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category)
        for category in critical_terms
    ]
    
    valid_categories = sum(category_found)
    logger.info(f"Doküman validasyonu: {valid_categories}/4 kritik kategori bulundu")
    
    return valid_categories >= 2


def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada özgü kelimeleri OCR ile ara"""
    strong_keywords = [
        "bakım", "bakim",
        "bakım talimatı", "bakim talimati",
        "maintenance", "MAINTENANCE",
        "servis",
        "service",
        "onarım",
        "repair",
        "prosedür",
        "procedure"
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
        logger.info(f"İlk sayfa kontrol: {len(found_keywords)} özgü kelime")
        return len(found_keywords) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa kontrol hatası: {e}")
        return False


def check_excluded_keywords_first_pages(filepath):
    """İlk 1-2 sayfada istenmeyen rapor türlerinin kelimelerini ara"""
    excluded_keywords = [
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm", "enclosure", "wrp-", "light curtain", "contactors", "controller",
        "espe",
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219", "teknik resim", "tasarım",
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide", "kılavuzu",
        "loto",
        "lvd", "TOPRAKLAMA SÜREKLİLİK", "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        "uygunluk", "beyan", "muayene", "conformity", "declaration",
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        "montaj", "assembly",
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama", "TOPRAKLAMA DİRENCİ",
        "titreşim", "vibration", "mekanik",
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
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
        
        found_excluded = [kw for kw in excluded_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa excluded kontrol: {len(found_excluded)} istenmeyen kelime")
        return len(found_excluded) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa excluded kontrol hatası: {e}")
        return False


def get_conclusion_message(status, percentage):
    """Sonuç mesajını döndür"""
    if status == "PASS":
        return f"Bakım talimatları yüksek kalitede ve standartlara uygun (%{percentage:.0f})"
    else:
        return f"Bakım talimatları yetersiz, kapsamlı revizyon gerekli (%{percentage:.0f})"


def get_main_issues(analysis_result):
    """Ana sorunları listele"""
    issues = []
    
    for category, score_data in analysis_result['puanlama']['category_scores'].items():
        if score_data['percentage'] < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    if not issues and analysis_result['puanlama']['total_score'] < 50:
        issues = [
            "Genel makine bilgileri eksik",
            "Güvenlik önlemleri yetersiz",
            "Bakım türleri tanımlanmamış",
            "Adım adım talimatlar eksik"
        ]
    
    return issues[:4]


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
    """Bakım Talimatları analiz endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        analyzer = MaintenanceReportAnalyzer()
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
            'language_info': {
                'detected_language': map_language_code(analysis_result['dosya_bilgisi']['detected_language'])
            },
            'overall_score': {
                'percentage': round(overall_percentage, 2),
                'total_points': analysis_result['puanlama']['total_score'],
                'max_points': 100,
                'status': status,
                'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ'
            },
            'recommendations': analysis_result.get('oneriler', []),
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

        return jsonify({
            'success': True,
            'message': 'Bakım Talimatları başarıyla analiz edildi',
            'analysis_service': 'bakim_talimatları',
            'data': response_data
        })

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': 'Server error', 'message': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'Bakım Talimatları Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'report_type': 'BAKIM_TALIMATLARI'
    })


@app.route('/', methods=['GET'])
def index():
    """Index page"""
    return jsonify({
        'service': 'Bakım Talimatları Analyzer API',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/bakimtalimatlari-report': 'Bakım talimatları analizi',
            'GET /api/health': 'Health check',
            'GET /': 'API info'
        }
    })


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Bakım Talimatları Analiz Servisi")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8014))
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)