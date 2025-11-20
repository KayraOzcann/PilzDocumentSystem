# ============================================
# HRC KUVVET-BASINÇ ÖLÇÜM RAPORU ANALİZ SERVİSİ
# Standalone Service - Azure App Service Ready
# Port: 8013
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
class HRCAnalysisResult:
    """HRC analiz sonucu veri sınıfı"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

# ============================================
# ANALİZ SINIFI - MAIN ANALYZER
# ============================================
class HRCReportAnalyzer:
    """HRC Kuvvet-Basınç Ölçüm Raporu analiz sınıfı"""
    
    def __init__(self):
        logger.info("HRC Kuvvet-Basınç Ölçüm Raporu analiz sistemi başlatılıyor...")
        logger.info("✅ OCR sistemi aktif - Tüm dosyalar OCR ile işlenmektedir")
        
        self.criteria_weights = {
            "Genel Bilgiler": 10,
            "Test Koşulları ve Senaryo Tanımı": 10,
            "Ölçüm Noktaları ve Metodoloji": 15,
            "Kuvvet ve Basınç Ölçüm Sonuçları": 25,
            "Sınır Değerlerle Karşılaştırma": 20,
            "Risk Değerlendirmesi ve Sonuç": 10,
            "Öneriler ve Önlemler": 5,
            "Ekler ve Kalibrasyon Belgeleri": 5
        }
        
        self.criteria_details = {
            "Genel Bilgiler": {
                "test_tarihi": {"pattern": r"(?i)(?:test|ölçüm|analiz|measurement|analysis|tarih|date)[\s\W]*[:=]?\s*(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4})|(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{4})(?:\s*(?:tarih|date))?", "weight": 2},
                "test_yapan_kurum": {"pattern": r"(?i)(?:test\s*yapan|tested\s*by|ölçüm\s*yapan|measured\s*by|kurum|institution|organization|şirket|company|firma|sorumlu|responsible)[\s\W]*[:=]?\s*([A-ZÇĞİÖŞÜa-züçğıöşü][A-Za-züçğıöşüÇĞİÖŞÜ\s\.\&\-]{3,50})", "weight": 2},
                "robot_modeli_seri": {"pattern": r"(?i)(?:robot\s*model|robot\s*tip|model|seri\s*no|serial\s*number|s\/n|robot\s*type|tip|robot\s*id)[\s\W]*[:=]?\s*([A-Z0-9\-]+[A-Z0-9\-\s]*)|([A-Z]{2,4}[\-\s]*[0-9]{1,5}[A-Z0-9]*)", "weight": 3},
                "uygulama_istasyon": {"pattern": r"(?i)(?:uygulama|application|istasyon|station|workstation|iş\s*istasyon|work\s*station|test\s*setup|test\s*düzen|çalışma\s*alan|work\s*area)[\s\W]*[:=]?\s*([A-ZÇĞİÖŞÜa-züçğıöşü][A-Za-züçğıöşüÇĞİÖŞÜ0-9\s\.\-]{3,50})", "weight": 3}
            },
            "Test Koşulları ve Senaryo Tanımı": {
                "robot_hiz_durum": {"pattern": r"(?i)(?:hız|speed|velocity|hızlı|durum|position|pose|duruş|stance|hareket|motion)[\s\W]*[:=]?\s*([0-9.,]+\s*(?:mm\/s|m\/s|%|derece|degree|°|rpm)|[A-ZÇĞİÖŞÜa-züçğıöşü][A-Za-züçğıöşüÇĞİÖŞÜ\s]{3,30})", "weight": 3},
                "calisma_modu": {"pattern": r"(?i)(?:çalışma\s*mod|work\s*mode|operation\s*mode|manuel|manual|otomatik|automatic|işbirlik|collaborative|cobot|hrc|interaction|mod|mode)", "weight": 2},
                "temas_bolgeleri": {"pattern": r"(?i)(?:temas\s*bölge|contact\s*area|contact\s*region|vücut\s*bölge|body\s*region|bölge|region|area|kol|arm|el|hand|gövde|body|torso|baş|head|bacak|leg|finger|parmak)", "weight": 3},
                "cevresel_sartlar": {"pattern": r"(?i)(?:sıcaklık|temperature|ortam|environment|çevre|ambient|celsius|nem|humidity|koşul|condition|şart)[\s\W]*[:=]?\s*([0-9.,]+\s*(?:°c|c|%|derece)|[A-ZÇĞİÖŞÜa-züçğıöşü][A-Za-züçğıöşüÇĞİÖŞÜ\s]{3,30})", "weight": 2}
            },
            "Ölçüm Noktaları ve Metodoloji": {
                "vucut_bolgeleri_iso15066": {"pattern": r"(?i)(?:iso\/ts\s*15066|iso\s*15066|ts\s*15066|vücut\s*bölge|body\s*region|tablo|table|kafa|head|yüz|face|boyun|neck|sırt|back|göğüs|chest|karın|abdomen|pelvis|kol|arm|dirsek|elbow|ön\s*kol|forearm|el|hand|parmak|finger|15066|skull|forehead|temple|masticatory|muscle)", "weight": 5},
                "olcum_yontemi": {"pattern": r"(?i)(?:ölçüm\s*yöntem|measurement\s*method|test\s*method|yöntem|method|statik|static|dinamik|dynamic|temas|contact|serbest\s*hareket|free\s*motion|engel|obstacle|barrier|prosedür|procedure|ölçüm\s*nokta|measurement\s*point|nokta\s*\d+|point\s*\d+|film|llw|max\s*pressure|mpa)", "weight": 5},
                "tekrar_sayisi_tutarlilik": {"pattern": r"(?i)(?:tekrar|repeat|iteration|test\s*sayı|test\s*count|n\s*=|n=|örnek\s*sayı|sample\s*count)[\s\W]*[:=]?\s*(\d{1,3})|(?:tutarlılık|consistency|repeatability|reproducibility|tekrarlama)|(?:a1|a2|b1|b2|c1|c2)[\s\W]*[=:]\s*([0-9.,]+)|(?:kuvvet\s*ölçüm|force\s*measurement)[\s\S]{0,50}(?:gelişim|development|progress)|(\d{1,2})\s*(?:nokta|point|ölçüm|measurement)", "weight": 5}
            },
            "Kuvvet ve Basınç Ölçüm Sonuçları": {
                "maksimum_temas_kuvveti": {"pattern": r"(?i)(?:maksimum\s*kuvvet|maximum\s*force|max\s*force|fmax|f\s*max|kuvvet\s*değer|force\s*value|en\s*büyük\s*kuvvet)[\s\W]*[:=]?\s*([0-9.,]+)\s*(?:n|newton|kn)|(?:f[\s\W]*max|fr[\s\W]*max|fs[\s\W]*max)[\s\W]*([0-9.,]+)|(\d{1,3})\s*(?=\s*\-?\s*\d{1,2}\s)|(?:maximum\s*permissible\s*kuvvet|maximum\s*permissible\s*force|izin\s*verilen\s*kuvvet)[\s\S]{0,50}([0-9.,]+)|(?:quasi[\-\s]*static|transient)[\s\S]{0,100}(?:kuvvet|force)[\s\S]{0,50}([0-9.,]+)", "weight": 7},
                "temas_suresi": {"pattern": r"(?i)(?:temas\s*süre|contact\s*time|contact\s*duration|süre|duration|time|t\s*=|t=|zaman|contact)[\s\W]*[:=]?\s*([0-9.,]+)\s*(?:ms|millisecond|s|second|saniye|msn|sn)|(\d{1,3})\s*(?:ms|s|saniye|sn)\b|(?:süre|duration|time)[\s\W]*(\d{1,3})|(?:transient|quasi[\-\s]*static)[\s\S]{0,50}(?:süre|time|duration)[\s\S]{0,30}([0-9.,]+)", "weight": 6},
                "basinc_degeri": {"pattern": r"(?i)(?:basınç\s*ts|pressure\s*ts|basınç|pressure|press|baskı)[\s\W]*[:=]?\s*([0-9.,]+)\s*(?:n\/cm²|n\/cm2|pa|kpa|mpa|bar|pascal)|(?:ps[\s\W]*max|p[\s\W]*max|basınç[\s\W]*max)[\s\W]*([0-9.,]+)|(\d{1,3})\s*(?:n\/cm²|n\/cm2|pa|bar)|(?:ts\s*15066)[\s\S]{0,50}(\d{1,3})\s*(?:n\/cm²|n\/cm2)|(?:maximum\s*permissible\s*basınç|maximum\s*permissible\s*pressure|izin\s*verilen\s*basınç)[\s\S]{0,50}([0-9.,]+)|(?:quasi[\-\s]*static|transient)[\s\S]{0,100}(?:basınç|pressure)[\s\S]{0,50}([0-9.,]+)", "weight": 6},
                "grafiksel_gosterim": {"pattern": r"(?i)(?:grafik|graph|chart|eğri|curve|kuvvet[\-\s]*zaman|force[\-\s]*time|plot|çizim|drawing|şekil\s*\d+|figure\s*\d+|diyagram|diagram|görsel|visual|resim\s*\d+|image\s*\d+|fig\s*\d+|şek\s*\d+|tablo\s*\d+|table\s*\d+|sonuç|result|ölçüm\s*sonuç|measurement\s*result)", "weight": 6}
            },
            "Sınır Değerlerle Karşılaştırma": {
                "iso15066_sinir_karsilastirma": {"pattern": r"(?i)(?:iso\/ts\s*15066|iso\s*15066|ts\s*15066|15066)[\s\S]*?(?:sınır|limit|threshold|eşik|karşılaştırma|comparison|compare|kıyas|standart|uygun|compliant)", "weight": 8},
                "vucut_bolgesi_limitleri": {"pattern": r"(?i)(?:izin\s*verilen|allowed|permitted|limit|sınır|maksimum\s*izin|maximum\s*allowed|kabul\s*edilebilir|acceptable)[\s\S]*?(?:kuvvet|force|basınç|pressure)", "weight": 6},
                "asim_risk_isaret": {"pattern": r"(?i)(?:aşım|exceed|over|fazla|limit\s*aş|limit\s*over|risk|tehlike|hazard|warning|uyarı|alert|güvenli\s*değil|not\s*safe|tehlikeli|dangerous)", "weight": 6}
            },
            "Risk Değerlendirmesi ve Sonuç": {
                "risk_seviye_analizi": {"pattern": r"(?i)(?:risk\s*analiz|risk\s*analysis|risk\s*assessment|risk\s*değerlendirme|risk|seviye|level|kategori|category|düşük|low|orta|medium|yüksek|high|değerlendirme|assessment)", "weight": 4},
                "risk_kabul_edilebilir": {"pattern": r"(?i)(?:kabul\s*edilebilir|acceptable|accept|uygun|suitable|güvenli|safe|güvenlik|safety|onay|approve|red|reject|kabul|uygunluk|compliance)", "weight": 3},
                "gereken_onlemler": {"pattern": r"(?i)(?:önlem|measure|action|tedbir|hız\s*sınır|speed\s*limit|güvenlik\s*sensör|safety\s*sensor|uç\s*efektör|end\s*effector|koruma|protection|emniyet|security)", "weight": 3}
            },
            "Öneriler ve Önlemler": {
                "emniyet_stratejisi": {"pattern": r"(?i)(?:emniyet\s*stratejisi|safety\s*strategy|strateji|güvenlik\s*stratejisi|hız\s*sınır|speed\s*limit|kuvvet\s*sınır|force\s*limit|yastıklama|padding|koruyucu|protective|eşik\s*değer|threshold\s*value|limit\s*değer|uygun\s*değer|yeşil|green|renk|color|renklendir|colored)", "weight": 2},
                "operatör_egitimi": {"pattern": r"(?i)(?:operatör|operator|eğitim|training|bilgilendirme|information|uyarı|warning|not|note|kullanıcı|user|personel|personnel|kabul\s*edilebilir|acceptable|uygun|appropriate|yeterli|sufficient)", "weight": 2},
                "periyodik_test": {"pattern": r"(?i)(?:periyodik|periodic|tekrarlayan|repeated|test\s*tekrar|test\s*repeat|düzenli|regular|planlı|scheduled|rutin|routine|bakım|maintenance|ölçüm\s*tablo|measurement\s*table|tablo|table|değerlendirme|evaluation)", "weight": 1}
            },
            "Ekler ve Kalibrasyon Belgeleri": {
                "kalibrasyon_sertifika": {"pattern": r"(?i)(?:kalibrasyon|calibration|sertifika|certificate|cert|belge|document|iso\s*17025|akreditasyon|accreditation|onay|approval|doğrulama|verification)", "weight": 2},
                "fotograf_video": {"pattern": r"(?i)(?:fotoğraf|photo|photograph|video|resim|image|picture|görsel|visual|kayıt|record|çekim|shot|dokümantasyon|documentation)", "weight": 2},
                "test_prosedur_referans": {"pattern": r"(?i)(?:prosedür|procedure|referans|reference|standart|standard|kılavuz|guide|manual|dokümantasyon|documentation|kaynak|source|metod|method)", "weight": 1}
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
        """Dosya formatına göre OCR ile metin çıkarma"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_with_ocr(file_path)
        elif file_ext in ['.docx', '.doc']:
            try:
                text = self.extract_text_from_docx(file_path)
                if len(text.strip()) > 50:
                    return text
                else:
                    return self.extract_text_with_ocr(file_path)
            except:
                return self.extract_text_with_ocr(file_path)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            logger.error(f"Desteklenmeyen dosya formatı: {file_ext}")
            return ""
    
    def extract_text_with_ocr(self, file_path: str) -> str:
        """OCR ile metin çıkarma"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                pages = pdf2image.convert_from_path(file_path, dpi=200)
                all_text = ""
                
                for i, page in enumerate(pages):
                    opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
                    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    
                    text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng')
                    all_text += text + "\n"
                    logger.info(f"PDF sayfa {i+1}/{len(pages)} OCR tamamlandı")
                
                return all_text.strip()
            else:
                return ""
                
        except Exception as e:
            logger.error(f"OCR hatası: {e}")
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
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, HRCAnalysisResult]:
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
            
            results[criterion_name] = HRCAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_count": len(matches) if matches else 0}
            )
        
        return results

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, HRCAnalysisResult]]) -> Dict[str, Any]:
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
        """HRC raporuna özgü spesifik değerleri çıkar"""
        values = {
            "robot_modeli": "Bulunamadı",
            "test_tarihi": "Bulunamadı",
            "olcum_cihazi": "Bulunamadı"
        }
        
        robot_patterns = [
            r"(?i)(?:Robot\s*tip[i]?|Robot\s*type|Robot\s*model)\s*[|\s]\s*([A-Z0-9\-]+(?:[A-Z0-9\-]*))(?=\s|$|\n)",
            r"([A-Z]{2,4}\-[0-9]{1,3}[A-Z]*)\b"
        ]
        
        for pattern in robot_patterns:
            match = re.search(pattern, text)
            if match:
                values["robot_modeli"] = match.group(1).strip()
                break
        
        date_patterns = [
            r"(?i)(?:Test\s*Tarih|Test\s*Date)\s*[:=]\s*(\d{1,2}[./]\d{1,2}[./]\d{2,4})",
            r"(\d{1,2}[./]\d{1,2}[./]\d{4})\s*(?:tarih|date)"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                values["test_tarihi"] = match.group(1).strip()
                break

        olcum_cihazi_patterns = [
            r"([A-Z]{3,8}\s+[a-z]{2,6}\s+[A-Z][a-z]+\s+[A-Z0-9]+)(?=\s+\d{4}|\s*\n|\s*$)"
        ]

        for pattern in olcum_cihazi_patterns:
            match = re.search(pattern, text)
            if match:
                found_value = match.group(1).strip()
                if 10 <= len(found_value) <= 25:
                    values["olcum_cihazi"] = found_value
                    break
                
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        
        total_percentage = scores["percentage"]
        
        if total_percentage >= 70:
            recommendations.append(f"✅ HRC Kuvvet-Basınç Raporu GEÇERLİ (Toplam: %{total_percentage:.1f})")
        else:
            recommendations.append(f"❌ HRC Kuvvet-Basınç Raporu GEÇERSİZ (Toplam: %{total_percentage:.1f})")
        
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 40:
                recommendations.append(f"🔴 {category} bölümü yetersiz (%{category_score:.1f})")
            elif category_score < 70:
                recommendations.append(f"🟡 {category} bölümü geliştirilmeli (%{category_score:.1f})")
            else:
                recommendations.append(f"🟢 {category} bölümü yeterli (%{category_score:.1f})")
        
        return recommendations

    def analyze_hrc_report(self, file_path: str) -> Dict[str, Any]:
        """Ana HRC analiz fonksiyonu"""
        logger.info("HRC Kuvvet-Basınç Ölçüm Raporu analizi başlatılıyor...")
        
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
                "rapor_tipi": "HRC_KUVVET_BASINC_RAPORU"
            }
        }

# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_document_server(text):
    """Server kodunda HRC doküman validasyonu"""
    critical_terms = [
        ["hrc", "collaborative", "işbirlik", "cobot", "kolaboratif", "human robot collaboration"],
        ["kuvvet", "force", "basınç", "pressure", "temas", "contact", "newton"],
        ["iso 15066", "iso/ts 15066", "ts 15066", "15066"],
        ["vücut", "body", "kol", "arm", "el", "hand", "baş", "head", "gövde", "torso", "boyun", "neck"]
    ]
    
    category_found = [
        any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category)
        for category in critical_terms
    ]
    
    valid_categories = sum(category_found)
    logger.info(f"HRC doküman validasyonu: {valid_categories}/4 kritik kategori bulundu")
    
    return valid_categories >= 4


def check_strong_keywords_first_pages(filepath):
    """İlk 1-2 sayfada HRC'ye özgü kelimeleri OCR ile ara"""
    strong_keywords = [
        "hrc",
        "cobot",
        "robot",
        "çarpışma",
        "collaborative",
        "kolaboratif",
        "sd conta"
    ]
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=2)
        
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 3 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        
        found_keywords = [kw for kw in strong_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        logger.info(f"İlk sayfa HRC kontrol: {len(found_keywords)} özgü kelime")
        return len(found_keywords) >= 2
        
    except Exception as e:
        logger.warning(f"İlk sayfa HRC kontrol hatası: {e}")
        return False


def check_excluded_keywords_first_pages(filepath):
    """İlk 1-2 sayfada istenmeyen rapor türlerinin kelimelerini ara"""
    excluded_keywords = [
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm", "enclosure", "wrp-", "light curtain", "contactors", "controller",
        "espe",
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219", "teknik resim", "tasarım",
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide", "kılavuzu",
        "loto",
        "lvd", "TOPRAKLAMA SÜREKLİLİK", "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        "montaj", "assembly",
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama", "TOPRAKLAMA DİRENCİ",
        "bakım", "maintenance", "servis", "service", "bakim", "MAINTENCE",
        "titreşim", "vibration", "mekanik",
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
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
        logger.info(f"İlk sayfa excluded kontrol: {len(found_excluded)} istenmeyen kelime")
        return len(found_excluded) >= 1
        
    except Exception as e:
        logger.warning(f"İlk sayfa excluded kontrol hatası: {e}")
        return False


def get_conclusion_message_hrc(status, percentage):
    """Sonuç mesajını döndür"""
    if status == "PASS":
        return f"HRC kuvvet-basınç raporu EN ISO 10218 ve ISO/TS 15066 standartlarına uygundur (%{percentage:.0f})"
    else:
        return f"HRC raporu standartlara uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"


def get_main_issues_hrc(analysis_result):
    """Ana sorunları listele"""
    issues = []
    
    for category, score_data in analysis_result['puanlama']['category_scores'].items():
        if score_data['percentage'] < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
    if not issues and analysis_result['puanlama']['total_score'] < 50:
        issues = [
            "Robot modeli ve seri numarası eksik",
            "Test koşulları ve senaryo tanımı eksik",
            "Kuvvet ve basınç ölçüm sonuçları eksik",
            "Risk değerlendirmesi yapılmamış"
        ]
    
    return issues[:4]


# ============================================
# FLASK SERVICE
# ============================================
app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads_hrc'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/hrc-report', methods=['POST'])
def analyze_hrc_report():
    """HRC Kuvvet-Basınç Ölçüm Raporu analiz endpoint"""
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
        
        analyzer = HRCReportAnalyzer()
        analysis_result = analyzer.analyze_hrc_report(filepath)
        
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
            'analysis_id': f"hrc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'category_scores': {},
            'extracted_values': analysis_result.get('cikarilan_degerler', {}),
            'file_type': 'HRC_KUVVET_BASINC_RAPORU',
            'filename': filename,
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
                'conclusion': get_conclusion_message_hrc(status, overall_percentage),
                'main_issues': [] if status == "PASS" else get_main_issues_hrc(analysis_result)
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
            'message': 'HRC Kuvvet-Basınç Raporu başarıyla analiz edildi',
            'analysis_service': 'hrc',
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
        'service': 'HRC Force-Pressure Report Analyzer API',
        'version': '1.0.0',
        'tesseract_available': tesseract_available,
        'report_type': 'HRC_KUVVET_BASINC_RAPORU'
    })


@app.route('/', methods=['GET'])
def index():
    """Index page"""
    return jsonify({
        'service': 'HRC Force-Pressure Report Analyzer API',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/hrc-report': 'HRC kuvvet-basınç raporu analizi',
            'GET /api/health': 'Health check',
            'GET /': 'API info'
        }
    })


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("HRC Kuvvet-Basınç Ölçüm Raporu Analiz Servisi")
    logger.info("=" * 60)
    
    port = int(os.environ.get('PORT', 8013))
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)