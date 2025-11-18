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
    from langdetect import detect
    LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ISGPeriyodikKontrolCriteria:
    """İSG Periyodik Kontrol kriterleri veri sınıfı"""
    tesis_ve_genel_bilgiler: Dict[str, Any]
    trafo_merkezi_kontrolu: Dict[str, Any]
    elektrik_guvenlik_kontrolu: Dict[str, Any]
    topraklama_sistemleri: Dict[str, Any]
    yangın_guvenlik_sistemleri: Dict[str, Any]
    is_guvenligi_malzemeleri: Dict[str, Any]

@dataclass
class ISGKontrolAnalysisResult:
    """İSG Kontrol analiz sonucu veri sınıfı"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

class ISGPeriyodikKontrolAnalyzer:
    """İSG Periyodik Kontrol Formu analiz sınıfı"""
    
    def __init__(self):
        logger.info("İSG Periyodik Kontrol analysis system starting...")
        
        self.criteria_weights = {
            "Genel Bilgiler ve Firma": 15,
            "Gürültü Ölçümleri": 20,
            "Aydınlatma Ölçümleri": 15,
            "Termal Konfor Ölçümleri": 15,
            "Hava Kalitesi Ölçümleri": 20,
            "Değerlendirme ve Sonuç": 15
        }
        
        self.criteria_details = {
            "Genel Bilgiler ve Firma": {
                "firma_adi": {"pattern": r"(?:firma.*adı|kuruluş.*adı|işletme.*adı|şirket.*adı|A\.Ş\.|LTD\.ŞTİ|TİC\.|SAN\.)", "weight": 4},
                "adres_bilgisi": {"pattern": r"(?:adres|bulvarı|caddesi|sokak|mahalle|il|şehir)", "weight": 3},
                "rapor_tarihi": {"pattern": r"(?:rapor.*tarih|ölçüm.*tarih|kontrol.*tarih|\d{1,2}[./\-]\d{1,2}[./\-]\d{4})", "weight": 3},
                "rapor_numarasi": {"pattern": r"(?:rapor.*no|deney.*rapor.*no|WA-\d+|LBR-\d+|\d{4}-\d+)", "weight": 2},
                "laboratuvar_bilgisi": {"pattern": r"(?:laboratuvar|akredite.*lab|test.*merkez|ölçüm.*kuruluş)", "weight": 3}
            },
            "Gürültü Ölçümleri": {
                "gurultu_seviyesi": {"pattern": r"(?:gürültü.*seviye|ses.*seviye|dBA|dB\(A\)|noise.*level|\d+.*dB)", "weight": 5},
                "gurultu_limiti": {"pattern": r"(?:gürültü.*limit|ses.*limit|85.*dB|90.*dB|80.*dB)", "weight": 3},
                "gurultu_olcum_nokta": {"pattern": r"(?:ölçüm.*nokta|gürültü.*ölçüm.*nokta|ses.*ölçüm)", "weight": 4},
                "gurultu_degerlendirme": {"pattern": r"(?:gürültü.*değerlendirme|ses.*değerlendirme|uygun|uygun.*değil)", "weight": 4},
                "gurultu_raporu": {"pattern": r"(?:gürültü.*ölçüm.*rapor|ses.*ölçüm.*rapor|noise.*measurement)", "weight": 4}
            },
            "Aydınlatma Ölçümleri": {
                "aydinlatma_seviyesi": {"pattern": r"(?:aydınlatma.*seviye|ışık.*seviye|lux|lüx|illumination|\d+.*lux)", "weight": 5},
                "aydinlatma_limiti": {"pattern": r"(?:aydınlatma.*limit|ışık.*limit|minimum.*lux|300.*lux|500.*lux)", "weight": 3},
                "aydinlatma_olcum": {"pattern": r"(?:aydınlatma.*ölçüm|ışık.*ölçüm|lux.*ölçüm)", "weight": 4},
                "aydinlatma_degerlendirme": {"pattern": r"(?:aydınlatma.*değerlendirme|ışık.*değerlendirme|uygun|yetersiz)", "weight": 4},
                "aydinlatma_raporu": {"pattern": r"(?:aydınlatma.*ölçüm.*rapor|ışık.*ölçüm.*rapor)", "weight": 4}
            },
            "Termal Konfor Ölçümleri": {
                "sicaklik_olcum": {"pattern": r"(?:sıcaklık.*ölçüm|temperature|°C|derece|termal.*konfor)", "weight": 4},
                "nem_olcum": {"pattern": r"(?:nem.*ölçüm|humidity|%.*nem|bağıl.*nem)", "weight": 4},
                "hava_hizi": {"pattern": r"(?:hava.*hızı|air.*velocity|m/s|air.*speed)", "weight": 3},
                "termal_konfor": {"pattern": r"(?:termal.*konfor|thermal.*comfort|iklim.*konfor)", "weight": 4}
            },
            "Hava Kalitesi Ölçümleri": {
                "co2_olcum": {"pattern": r"(?:CO2|karbon.*dioksit|carbon.*dioxide|karbondioksit|ppm.*CO2)", "weight": 4},
                "partikul_olcum": {"pattern": r"(?:partikül.*ölçüm|PM10|PM2.5|toz.*ölçüm|particulate)", "weight": 4},
                "formaldehit": {"pattern": r"(?:formaldehit|formaldehyde|HCHO)", "weight": 3},
                "kimyasal_olcum": {"pattern": r"(?:kimyasal.*ölçüm|chemical.*measurement|solvent|benzol|toluen)", "weight": 4},
                "hava_kalitesi": {"pattern": r"(?:hava.*kalite|air.*quality|iç.*hava|indoor.*air)", "weight": 5}
            },
            "Değerlendirme ve Sonuç": {
                "genel_degerlendirme": {"pattern": r"(?:genel.*değerlendirme|overall.*assessment|sonuç|conclusion)", "weight": 4},
                "uygunluk_durumu": {"pattern": r"(?:uygun|uygun.*değil|compliant|non.*compliant|kabul.*edilebilir)", "weight": 4},
                "oneri_tavsiye": {"pattern": r"(?:öneri|tavsiye|recommendation|suggestion|iyileştirme)", "weight": 3},
                "limit_karsilastirma": {"pattern": r"(?:limit.*değer|eşik.*değer|threshold|standart.*değer)", "weight": 4}
            }
        }
        
        # Onay durumu pattern'leri - OCR sonuçlarına göre güncellenmiş
        self.approval_patterns = {
            "uygun": r"(?:uygun|UYGUN|✓|√|✔|☑|☒|v|V|c|C|onaylandı|kabul|geçer|ok|OK|var|mevcut|tamam|yapıldı|kontrol.*edildi)",
            "uygun_degil": r"(?:uygun değil|UYGUN DEĞİL|degil|DEGIL|✗|✘|×|❌|x|X|red|yetersiz|eksik|yok|yapılmadı|kontrol.*edilmedi)",
            "not_var": r"(?:not|açıklama|dipnot|özel durum|NOT|gözlem|dikkat|uyarı)"
        }
    
    def is_isg_periyodik_kontrol_report(self, text: str) -> bool:
        """Metnin İSG periyodik kontrol raporu olup olmadığını kontrol et"""
        
        # İSG raporu işaretleri
        isg_indicators = [
            r"İSG.*ölçüm",
            r"iş.*hijyen.*ölçüm",
            r"ortam.*ölçüm.*rapor",
            r"gürültü.*ölçüm.*rapor",
            r"aydınlatma.*ölçüm.*rapor",
            r"termal.*konfor.*ölçüm",
            r"hava.*kalitesi.*ölçüm",
            r"partikül.*ölçüm",
            r"CO2.*ölçüm",
            r"formaldehit.*ölçüm",
            r"laboratuvar.*deney.*rapor",
            r"akredite.*laboratuvar",
            r"iş.*sağlığı.*güvenliği",
            r"WA-\d+.*rapor",
            r"deney.*raporu"
        ]
        
        # Elektrik/YG raporu işaretleri (bunlar varsa İSG raporu olmayabilir)
        elektrik_indicators = [
            r"YG.*kontrol",
            r"yüksek.*gerilim",
            r"trafo.*kontrol",
            r"ENH.*direk",
            r"elektrik.*tesis.*kontrol",
            r"branşman.*hattı",
            r"topraklama.*direnç",
            r"silikajel.*kontrol"
        ]
        
        isg_score = 0
        elektrik_score = 0
        
        for pattern in isg_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                isg_score += len(matches)
        
        for pattern in elektrik_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                elektrik_score += len(matches)
        
        logger.info(f"İSG indicators: {isg_score}, Elektrik indicators: {elektrik_score}")
        
        # Elektrik işaretleri çok daha fazlaysa İSG raporu değil
        if elektrik_score > 5 and isg_score < 3:
            return False
        
        # İSG işaretleri varsa İSG raporu
        if isg_score >= 2:
            return True
            
        return False
    
    def check_report_date_validity(self, olcum_tarihi: str) -> Dict[str, Any]:
        """Rapor tarihinin geçerliliğini kontrol et - 1 yıldan eski ise geçersiz"""
        result = {
            "is_valid": True,
            "days_old": 0,
            "message": "Tarih geçerli",
            "formatted_date": olcum_tarihi
        }
        
        if olcum_tarihi == "Bulunamadı":
            result["is_valid"] = False
            result["message"] = "Ölçüm tarihi bulunamadı"
            return result
        
        try:
            # Farklı tarih formatlarını dene
            date_formats = [
                "%d.%m.%Y",
                "%d/%m/%Y", 
                "%d-%m-%Y",
                "%Y-%m-%d",
                "%Y.%m.%d",
                "%Y/%m/%d"
            ]
            
            parsed_date = None
            for format_str in date_formats:
                try:
                    parsed_date = datetime.strptime(olcum_tarihi, format_str)
                    break
                except ValueError:
                    continue
            
            if parsed_date is None:
                result["is_valid"] = False
                result["message"] = f"Tarih formatı tanınmıyor: {olcum_tarihi}"
                return result
            
            # Bugünkü tarih ile karşılaştır
            today = datetime.now()
            days_difference = (today - parsed_date).days
            
            result["days_old"] = days_difference
            result["formatted_date"] = parsed_date.strftime("%d.%m.%Y")
            
            # 1 yıl = 365 gün kontrolü
            if days_difference > 365:
                result["is_valid"] = False
                result["message"] = f"Rapor tarihi 1 yıldan eski ({days_difference} gün önce)"
            elif days_difference < 0:
                result["is_valid"] = False
                result["message"] = "Rapor tarihi gelecekte - hatalı tarih"
            else:
                result["message"] = f"Rapor tarihi geçerli ({days_difference} gün önce)"
                
        except Exception as e:
            result["is_valid"] = False
            result["message"] = f"Tarih kontrolü hatası: {str(e)}"
        
        return result
            
    def detect_language(self, text: str) -> str:
        """Metin dilini tespit et"""
        if not LANGUAGE_DETECTION_AVAILABLE:
            return 'tr'
        
        try:
            sample_text = text[:500].strip()
            if not sample_text:
                return 'tr'
                
            detected_lang = detect(sample_text)
            logger.info(f"Detected language: {detected_lang}")
            return detected_lang if detected_lang in ['tr', 'en'] else 'tr'
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'tr'
        """Metin dilini tespit et"""
        if not LANGUAGE_DETECTION_AVAILABLE:
            return 'tr'
        
        try:
            sample_text = text[:500].strip()
            if not sample_text:
                return 'tr'
                
            detected_lang = detect(sample_text)
            logger.info(f"Detected language: {detected_lang}")
            return detected_lang if detected_lang in ['tr', 'en'] else 'tr'
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'tr'
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkar - PyPDF2 ve OCR ile"""
        pypdf_text = ""
        ocr_text = ""
        
        # Önce PyPDF2 ile dene
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pypdf_text += page_text + "\n"
                
                if len(pypdf_text.strip()) > 50:
                    logger.info("Text extracted using PyPDF2")
                    return pypdf_text.strip()
                
        except Exception as e:
            logger.error(f"PDF text extraction error: {e}")
        
        # PyPDF2 yeterli değilse OCR kullan
        try:
            logger.info("Insufficient text with PyPDF2, trying OCR...")
            images = convert_from_path(pdf_path, dpi=300)
            
            for i, image in enumerate(images):
                try:
                    text = pytesseract.image_to_string(image, lang='tur+eng')
                    text = re.sub(r'\s+', ' ', text)
                    ocr_text += text + "\n"
                    
                    logger.info(f"OCR extracted {len(text)} characters from page {i+1}")
                    
                except Exception as page_error:
                    logger.error(f"Page {i+1} OCR error: {page_error}")
                    continue
            
            logger.info(f"OCR total text length: {len(ocr_text)}")
            return ocr_text.strip() if ocr_text.strip() else pypdf_text.strip()
            
        except Exception as e:
            logger.error(f"OCR text extraction error: {e}")
            return pypdf_text.strip()
    
    def extract_approval_status(self, text: str, criteria_text: str) -> Dict[str, Any]:
        """Onay durumunu tespit et"""
        status = {
            "uygun": False,
            "uygun_degil": False,
            "not_var": False,
            "confidence": 0.0
        }
        
        # Kriter etrafındaki metin parçasını bul
        criteria_lower = criteria_text.lower()
        text_lower = text.lower()
        
        # Kriter bulunursa etrafındaki 200 karakter al
        for keyword in criteria_lower.split():
            if keyword in text_lower:
                pos = text_lower.find(keyword)
                if pos != -1:
                    start = max(0, pos - 100)
                    end = min(len(text), pos + 200)
                    context = text[start:end]
                    
                    # Onay pattern'lerini kontrol et
                    for pattern_name, pattern in self.approval_patterns.items():
                        if re.search(pattern, context, re.IGNORECASE):
                            status[pattern_name] = True
                            status["confidence"] = 0.8
                            break
                    break
        
        return status
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, ISGKontrolAnalysisResult]:
        """Kriterleri analiz et"""
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                content = f"Found: {str(matches[:3])}"
                found = True
                
                # Onay durumunu kontrol et
                approval_status = self.extract_approval_status(text, str(matches[0]) if matches else "")
                
                # Skoru hesapla - daha esnek sistem
                if approval_status["uygun_degil"]:
                    score = 0  # Açıkça uygun değil
                elif approval_status["uygun"]:
                    score = weight  # Açıkça uygun
                else:
                    # Belirsiz ama kriter mevcut - optimistik yaklaşım
                    score = int(weight * 0.8)  # %80 puan ver
                    
            else:
                content = "Not found"
                found = False
                score = 0
                approval_status = {"uygun": False, "uygun_degil": False, "not_var": False, "confidence": 0.0}
            
            results[criterion_name] = ISGKontrolAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={
                    "pattern_used": pattern,
                    "matches_count": len(matches) if matches else 0,
                    "approval_status": approval_status
                }
            )
        
        return results

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, ISGKontrolAnalysisResult]]) -> Dict[str, Any]:
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
        """İSG Kontrol formundan özel değerleri çıkar"""
        values = {
            "firma_adi": "Bulunamadı",
            "olcum_tarihi": "Bulunamadı",
            "rapor_numarasi": "Bulunamadı",
            "laboratuvar": "Bulunamadı",
            "adres": "Bulunamadı",
            "gurultu_seviye": "Bulunamadı",
            "aydinlatma_seviye": "Bulunamadı",
            "genel_degerlendirme": "Bulunamadı"
        }
        
        # Firma adı - İSG raporlarında yaygın pattern'lar
        firma_patterns = [
            r"(?:firma.*adı|kuruluş.*adı|işletme.*adı)\s*[:]*\s*([A-Za-zÇĞıİÖŞÜçğıöşü\s\.&\-]+)",
            r"([A-Za-zÇĞıİÖŞÜçğıöşü\s\.&\-]+(?:A\.Ş\.|LTD\.ŞTİ|TİC\.|SAN\.))",
            r"Adı\s*[:]*\s*([A-Za-zÇĞıİÖŞÜçğıöşü\s\.&\-]+)",
            r"COCA.*COLA.*İÇECEK.*A\.Ş\.|MAVİ.*BEYAZ.*LTD\.ŞTİ"
        ]
        for pattern in firma_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if "COCA" in match.group(0).upper():
                    values["firma_adi"] = "COCA COLA İÇECEK A.Ş."
                elif "MAVİ" in match.group(0).upper():
                    values["firma_adi"] = "MAVİ BEYAZ İŞ SAĞLIĞI LTD.ŞTİ"
                elif len(match.groups()) > 0:
                    firma_name = match.group(1).strip()
                    if len(firma_name) > 3:
                        values["firma_adi"] = firma_name
                break
        
        # Ölçüm tarihi
        date_patterns = [
            r"(?:ölçüm.*tarih|rapor.*tarih|gerçekleştirilen)\s*[:]*\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})"
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                values["olcum_tarihi"] = match.group(1).strip()
                break
        
        # Rapor numarası
        rapor_patterns = [
            r"(WA-\d{4}-\d+)",
            r"(?:rapor.*no|deney.*no)\s*[:]*\s*([A-Z]{2}-\d{4}-\d+)",
            r"(\d{4}-\d+.*nolu.*rapor)"
        ]
        for pattern in rapor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["rapor_numarasi"] = match.group(1).strip()
                break
        
        # Laboratuvar
        lab_patterns = [
            r"(ÇEVTEST.*LABORATUVARI|MAVİ.*BEYAZ.*LABORATUVARI)",
            r"(?:laboratuvar.*adı)\s*[:]*\s*([A-Za-zÇĞıİÖŞÜçğıöşü\s\.&\-]+)"
        ]
        for pattern in lab_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["laboratuvar"] = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                break
        
        # Gürültü seviyesi
        gurultu_patterns = [
            r"(\d+(?:[.,]\d+)?)\s*dB\(A\)",
            r"gürültü.*seviye\s*[:]*\s*(\d+(?:[.,]\d+)?)",
            r"(\d+(?:[.,]\d+)?)\s*dBA"
        ]
        for pattern in gurultu_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["gurultu_seviye"] = f"{match.group(1)} dB(A)"
                break
        
        # Aydınlatma seviyesi
        aydinlatma_patterns = [
            r"(\d+(?:[.,]\d+)?)\s*lux",
            r"aydınlatma.*seviye\s*[:]*\s*(\d+(?:[.,]\d+)?)",
            r"(\d+(?:[.,]\d+)?)\s*lüx"
        ]
        for pattern in aydinlatma_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["aydinlatma_seviye"] = f"{match.group(1)} lux"
                break
        
        # Genel değerlendirme
        degerlendirme_patterns = [
            r"GENEL.*DEĞERLENDİRME\s*[:]*\s*([A-Za-zÇĞıİÖŞÜçğıöşü\s]+)",
            r"SONUÇ\s*[:]*\s*([A-Za-zÇĞıİÖŞÜçğıöşü\s]+)",
            r"(?:uygun|uygun değil|kabul edilebilir|limit değerlerin altında)"
        ]
        for pattern in degerlendirme_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) > 0:
                    evaluation = match.group(1).strip()
                    if len(evaluation) > 2:
                        values["genel_degerlendirme"] = evaluation
                else:
                    values["genel_degerlendirme"] = match.group(0).strip()
                break
        
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """İSG Kontrol için öneriler oluştur"""
        recommendations = []
        
        total_percentage = scores["percentage"]
        
        if total_percentage >= 70:
            recommendations.append(f"✅ İSG Periyodik Kontrol GEÇERLİ (Toplam: %{total_percentage:.0f})")
        elif total_percentage >= 50:
            recommendations.append(f"🟡 İSG Periyodik Kontrol KOŞULLU (Toplam: %{total_percentage:.0f})")
        else:
            recommendations.append(f"❌ İSG Periyodik Kontrol YETERSİZ (Toplam: %{total_percentage:.0f})")
        
        for category, results in analysis_results.items():
            category_score = scores["category_scores"][category]["percentage"]
            
            if category_score < 50:
                recommendations.append(f"🔴 {category} bölümü yetersiz (%{category_score:.0f})")
            elif category_score < 70:
                recommendations.append(f"🟡 {category} bölümü geliştirilmeli (%{category_score:.0f})")
            else:
                recommendations.append(f"🟢 {category} bölümü yeterli (%{category_score:.0f})")
        
        if total_percentage < 70:
            recommendations.extend([
                "",
                "💡 İYİLEŞTİRME ÖNERİLERİ:",
                "- Eksik ölçüm kategorilerini tamamlayın",
                "- Gürültü ölçümlerini yeniden kontrol edin",
                "- Aydınlatma seviyelerini iyileştirin",
                "- Hava kalitesi ölçümlerini genişletin",
                "- Termal konfor koşullarını optimize edin"
            ])
        
        return recommendations

    def analyze_isg_kontrol(self, pdf_path: str) -> Dict[str, Any]:
        """Ana İSG Kontrol analiz fonksiyonu"""
        logger.info("İSG Periyodik Kontrol analysis starting...")
        
        if not os.path.exists(pdf_path):
            return {"error": f"PDF dosyası bulunamadı: {pdf_path}"}
        
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "PDF'den metin çıkarılamadı"}
        
        # Rapor türü kontrolü
        if not self.is_isg_periyodik_kontrol_report(text):
            return {
                "error": "Bu dosya İSG Periyodik Kontrol raporu değil",
                "suggestion": "Bu dosya elektrik/YG raporu veya başka bir rapor türü olabilir",
                "detected_type": "NON_ISG_REPORT"
            }
        
        detected_lang = self.detect_language(text)
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category)
        
        scores = self.calculate_scores(analysis_results)
        extracted_values = self.extract_specific_values(text)
        
        # Tarih kontrolü - kritik önem!
        date_check = self.check_report_date_validity(extracted_values.get("olcum_tarihi", "Bulunamadı"))
        
        # Normal puanları hesapla
        normal_recommendations = self.generate_recommendations(analysis_results, scores)
        normal_status = "PASS" if scores["percentage"] >= 70 else ("CONDITIONAL" if scores["percentage"] >= 50 else "FAIL")
        final_score = scores["total_score"]
        final_percentage = scores["percentage"]
        
        # Eğer tarih geçersizse sadece status'u FAIL yap, puanları değiştirme
        if not date_check["is_valid"]:
            final_status = "FAIL"  # Puanlar ne olursa olsun FAIL
            recommendations = [
                f"❌ RAPOR GEÇERSİZ: {date_check['message']}",
                "🔴 İSG raporları en fazla 1 yıl geçerlidir",
                "📅 Yeni bir İSG ölçüm raporu temin edilmelidir",
                "",
                "📊 PUANLAMA (Referans için):"
            ] + normal_recommendations
        else:
            recommendations = normal_recommendations
            final_status = normal_status
        
        report = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_info": {
                "pdf_path": pdf_path,
                "detected_language": detected_lang
            },
            "extracted_values": extracted_values,
            "date_validity": date_check,
            "category_analyses": analysis_results,
            "scoring": scores,
            "recommendations": recommendations,
            "summary": {
                "total_score": final_score,
                "percentage": final_percentage,
                "status": final_status,
                "report_type": "ISG_PERIYODIK_KONTROL"
            }
        }
        
        return report

def main():
    """Ana fonksiyon"""
    import sys
    
    analyzer = ISGPeriyodikKontrolAnalyzer()

    # Komut satırı argümanlarını kontrol et
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "periyodik kontrol raporları/İR 18-025 vestel beyaz eşya buzdolabı 1.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF dosyası bulunamadı: {pdf_path}")
        return
    
    print("🔍 İSG Periyodik Kontrol Analizi Başlatılıyor...")
    print("=" * 60)
    
    report = analyzer.analyze_isg_kontrol(pdf_path)
    
    if "error" in report:
        print(f"❌ Hata: {report['error']}")
        return
    
    print("\n📊 ANALİZ SONUÇLARI")
    print("=" * 60)
    
    print(f"📅 Analiz Tarihi: {report['analysis_date']}")
    print(f"🔍 Tespit Edilen Dil: {report['file_info']['detected_language'].upper()}")
    
    print(f"📋 Toplam Puan: {report['summary']['total_score']}/100")
    print(f"📈 Yüzde: %{report['summary']['percentage']:.0f}")
    print(f"🎯 Durum: {report['summary']['status']}")
    print(f"📄 Rapor Türü: {report['summary']['report_type']}")
    
    # Tarih kontrolü sonucunu göster
    date_check = report.get('date_validity', {})
    if date_check:
        if date_check['is_valid']:
            print(f"📅 Tarih Durumu: ✅ GEÇERLİ ({date_check['message']})")
        else:
            print(f"📅 Tarih Durumu: ❌ GEÇERSİZ ({date_check['message']})")
    
    print("\n📋 ÖNEMLİ ÇIKARILAN DEĞERLER")
    print("-" * 40)
    extracted_values = report['extracted_values']
    display_names = {
        "firma_adi": "Firma Adı",
        "olcum_tarihi": "Ölçüm Tarihi",
        "rapor_numarasi": "Rapor Numarası",
        "laboratuvar": "Laboratuvar",
        "adres": "Adres",
        "gurultu_seviye": "Gürültü Seviyesi",
        "aydinlatma_seviye": "Aydınlatma Seviyesi",
        "genel_degerlendirme": "Genel Değerlendirme"
    }
    
    for key, value in extracted_values.items():
        display_name = display_names.get(key, key.replace('_', ' ').title())
        print(f"{display_name}: {value}")
    
    print("\n📊 KATEGORİ PUANLARI")
    print("-" * 40)
    for category, score_data in report['scoring']['category_scores'].items():
        print(f"{category}: {score_data['normalized']}/{score_data['max_weight']} (%{score_data['percentage']:.0f})")
    
    print("\n💡 ÖNERİLER VE DEĞERLENDİRME")
    print("-" * 40)
    for recommendation in report['recommendations']:
        print(recommendation)
    
    print("\n📋 GENEL DEĞERLENDİRME")
    print("=" * 60)
    
    if report['summary']['percentage'] >= 70:
        print("✅ SONUÇ: GEÇERLİ")
        print(f"🌟 Toplam Başarı: %{report['summary']['percentage']:.0f}")
        print("📝 Değerlendirme: İSG Periyodik Kontrol raporu gerekli kriterleri sağlamaktadır.")
        
    elif report['summary']['percentage'] >= 50:
        print("🟡 SONUÇ: KOŞULLU")
        print(f"⚠️ Toplam Başarı: %{report['summary']['percentage']:.0f}")
        print("📝 Değerlendirme: İSG Kontrol raporu kabul edilebilir ancak bazı eksiklikler var.")
        
    else:
        print("❌ SONUÇ: YETERSİZ")
        print(f"⚠️ Toplam Başarı: %{report['summary']['percentage']:.0f}")
        print("📝 Değerlendirme: İSG Kontrol raporu minimum gereksinimleri karşılamıyor.")

if __name__ == "__main__":
    main()
