"""
Manuel/Kullanım Kılavuzu Analiz Servisi
========================================
Azure App Service için optimize edilmiş standalone servis

Endpoint: POST /api/manuel-report
Health Check: GET /api/health
"""

# ============================================
# IMPORTS
# ============================================
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

# ============================================
# LOGGING
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
    logger.warning("langdetect not available")

# ============================================
# DATA CLASSES
# ============================================
@dataclass
class ManualAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

# ============================================
# ANALYZER CLASS
# ============================================
class ManualReportAnalyzer:
    def __init__(self):
        logger.info("Kullanma Kılavuzu analiz sistemi başlatılıyor...")
        
        self.criteria_weights = {
            "Genel Bilgiler": 10,
            "Giriş ve Amaç": 5,
            "Güvenlik Bilgileri": 15,
            "Ürün Tanıtımı": 10,
            "Kurulum ve Montaj Bilgileri": 15,
            "Kullanım Talimatları": 20,
            "Bakım ve Temizlik": 10,
            "Arıza Giderme": 15
        }
        
        self.criteria_details = {
            "Genel Bilgiler": {
                "kilavuz_adi_kod": {"pattern": r"(?:Kılavuz|Manual|Guide|Kullan[ıi]m\s*K[ıi]lavuzu|User\s*Manual|Operating\s*Manual)", "weight": 5},
                "urun_modeli": {"pattern": r"(?:Ürün|Product|Model|Seri\s*No|Serial\s*Number|Part\s*Number)", "weight": 3},
                "revizyon_bilgisi": {"pattern": r"(?:Revizyon|Revision|Rev\.?|Version|v)\s*[:=]?\s*(\d+|[A-Z])", "weight": 2}
            },
            "Giriş ve Amaç": {
                "kilavuz_amaci": {"pattern": r"(?:Amaç|Purpose|Objective|Bu\s*k[ıi]lavuz|This\s*manual|Introduction|Giriş)", "weight": 3},
                "kapsam": {"pattern": r"(?:Kapsam|Scope|Coverage|Bu\s*dokuman|This\s*document)", "weight": 2}
            },
            "Güvenlik Bilgileri": {
                "genel_guvenlik": {"pattern": r"(?:Güvenlik|Safety|Güvenlik\s*Uyar[ıi]s[ıi]|Safety\s*Warning|UYARI|WARNING|DİKKAT|CAUTION)", "weight": 4},
                "tehlikeler": {"pattern": r"(?:Tehlike|Hazard|Risk|Tehlikeli|Dangerous|Yaralanma|Injury)", "weight": 4},
                "guvenlik_prosedur": {"pattern": r"(?:Prosedür|Procedure|Güvenlik\s*Prosedür|Safety\s*Procedure|Uyulmas[ıi]\s*gereken)", "weight": 3},
                "kkd_gerekliligi": {"pattern": r"(?:KKD|PPE|Personal\s*Protective|Koruyucu\s*Donanım|Protective\s*Equipment|Eldiven|Glove|Gözlük|Goggle|Koruyucu\s*Alet|Koruyucu\s*Ekipman|Safety\s*Equipment|Güvenlik\s*Ekipman|Koruyucu.*?kullan|Protective.*?use|Safety.*?wear|Güvenlik.*?giy|Emniy.*?alet|İş\s*güvenliği\s*ekipman)", "weight": 4}
            },
            "Ürün Tanıtımı": {
                "urun_tanimi": {"pattern": r"(?:Ürün\s*Tan[ıi]m[ıi]|Product\s*Description|Genel\s*Tan[ıi]m|General\s*Description)", "weight": 3},
                "teknik_ozellikler": {"pattern": r"(?:Teknik\s*Özellik|Technical\s*Specification|Specification|Özellik|Feature)", "weight": 3},
                "bilesenler": {"pattern": r"(?:Bileşen|Component|Parça|Part|Liste|List|İçerik|Content)", "weight": 2},
                "gorseller": {"pattern": r"(?:Görsel|Image|Resim|Picture|Şekil|Figure|Fotoğraf|Photo)", "weight": 2}
            },
            "Kurulum ve Montaj Bilgileri": {
                "kurulum_oncesi": {"pattern": r"(?:Kurulum\s*Öncesi|Before\s*Installation|Hazırl[ıi]k|Preparation|Ön\s*hazırl[ıi]k)", "weight": 4},
                "montaj_talimatlari": {"pattern": r"(?:Montaj|Installation|Assembly|Ad[ıi]m|Step|Talimat|Instruction)", "weight": 4},
                "gerekli_aletler": {"pattern": r"(?:Alet|Tool|Malzeme|Material|Gerekli|Required|Equipment)", "weight": 3},
                "kurulum_kontrolu": {"pattern": r"(?:Kontrol|Check|Test|Doğrula|Verify|Kurulum\s*Sonras[ıi]|After\s*Installation)", "weight": 4}
            },
            "Kullanım Talimatları": {
                "calistirma": {"pattern": r"(?:Çal[ıi]şt[ıi]rma|Start|Operation|Açma|Turn\s*On|Power\s*On)", "weight": 5},
                "kullanim_kilavuzu": {"pattern": r"(?:Kullan[ıi]m|Usage|Use|Operating|Ad[ıi]m\s*ad[ıi]m|Step\s*by\s*step)", "weight": 5},
                "calisma_modlari": {"pattern": r"(?:Mod|Mode|Ayar|Setting|Çal[ıi]şma\s*Mod|Operating\s*Mode)", "weight": 5},
                "kullanim_ipuclari": {"pattern": r"(?:İpucu|Tip|Öneri|Recommendation|Doğru\s*kullan[ıi]m|Proper\s*use)", "weight": 5}
            },
            "Bakım ve Temizlik": {
                "duzenli_bakim": {"pattern": r"(?:Bak[ıi]m|Maintenance|Düzenli|Regular|Periyodik|Periodic)", "weight": 3},
                "temizlik_yontemleri": {"pattern": r"(?:Temizlik|Cleaning|Temizle|Clean|Hijyen|Hygiene)", "weight": 3},
                "parca_degisimi": {"pattern": r"(?:Parça\s*Değiş|Part\s*Replace|Yedek\s*Parça|Spare\s*Part|Değiştir|Replace)", "weight": 4}
            },
            "Arıza Giderme": {
                "sorun_cozumleri": {"pattern": r"(?:Sorun|Problem|Ar[ıi]za|Fault|Troubleshoot|Çözüm|Solution)", "weight": 5},
                "hata_kodlari": {"pattern": r"(?:Hata\s*Kod|Error\s*Code|Kod|Code|Alarm|Uyar[ıi]\s*Lambas[ıi]|Warning\s*Light|Acil\s*Durum\s*Buton|Emergency\s*Button|Hata\s*Gösterg|Error\s*Indicator|Uyar[ıi]\s*Sistemi|Warning\s*System|K[ıi]z[ıi]l\s*[Iı]ş[ıi]k|Red\s*Light|Alarm\s*Sistemi|Hata\s*Durumu|Error\s*Status|Aktifleş|Activate)", "weight": 5},
                "teknik_destek": {"pattern": r"(?:Teknik\s*Destek|Technical\s*Support|Destek|Support|İletişim|Contact|Tel|Phone|Telefon|E-?mail|Mail|Fax|Website|Web|www\.|@|\.com|\.net|\.org|\+90|\+\d{1,3}|\d{3}[-.\s]\d{3}[-.\s]\d{4}|\(\d{3}\)\s*\d{3}[-.\s]\d{4}|0\d{3}\s*\d{3}\s*\d{2}\s*\d{2})", "weight": 3},
                "teknik_cizimler": {"pattern": r"(?:Çizim|Drawing|Şema|Scheme|Diyagram|Diagram|Plan)", "weight": 2}
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
        """Metni Türkçe'ye çevir - şimdilik devre dışı"""
        if source_lang != 'tr':
            logger.info(f"Tespit edilen dil: {source_lang.upper()} - Çeviri yapılmıyor, orijinal metin kullanılıyor")
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
            images = pdf2image.convert_from_path(pdf_path, dpi=300)
            
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
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, ManualAnalysisResult]:
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
                score = min(weight, len(matches) * (weight // 2))
                score = max(score, weight // 2)  # ✅ En az yarı puan
            else:
                content = "Bulunamadı"
                found = False
                score = 0
            
            results[criterion_name] = ManualAnalysisResult(
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
            "kilavuz_adi": "Bulunamadı",
            "urun_modeli": "Bulunamadı",
            "hazırlama_tarihi": "Bulunamadı",
            "hazirlayan": "Bulunamadı"
        }
        
        # Kılavuz adı
        manual_patterns = [
            r"(?:Kullan[ıi]m\s*K[ıi]lavuzu)\s*[:=]?\s*([^\n\r]+)",
            r"(?:User\s*Manual)\s*[:=]?\s*([^\n\r]+)",
            r"(?:Operating\s*Manual)\s*[:=]?\s*([^\n\r]+)",
            r"(Manual|K[ıi]lavuz|Guide)"
        ]
        
        for pattern in manual_patterns:
            manual_match = re.search(pattern, text, re.IGNORECASE)
            if manual_match:
                values["kilavuz_adi"] = manual_match.group(1).strip()[:50] if len(manual_match.groups()) > 0 else manual_match.group().strip()[:50]
                break
        
        # Ürün modeli
        product_patterns = [
            r"(?:Model)\s*[:=]?\s*([^\n\r]+)",
            r"(?:Product)\s*[:=]?\s*([^\n\r]+)",
            r"(?:Ürün)\s*[:=]?\s*([^\n\r]+)",
            r"(?:Part\s*Number)\s*[:=]?\s*([^\n\r]+)"
        ]
        
        for pattern in product_patterns:
            product_match = re.search(pattern, text, re.IGNORECASE)
            if product_match:
                values["urun_modeli"] = product_match.group(1).strip()[:50]
                break
        
        # Tarih
        date_patterns = [
            r"(?:Hazırlama|Prepared|Date|Tarih)\s*[:=]?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})",
            r"(\d{1,2}[./]\d{1,2}[./]\d{4})",
            r"(\d{4}[./]\d{1,2}[./]\d{1,2})"
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                values["hazırlama_tarihi"] = date_match.group(1)
                break
        
        # Hazırlayan
        author_patterns = [
            r"(?:Hazırlayan)\s*[:=]?\s*([^\n\r]+)",
            r"(?:Prepared\s*by)\s*[:=]?\s*([^\n\r]+)",
            r"(?:Author)\s*[:=]?\s*([^\n\r]+)"
        ]
        
        for pattern in author_patterns:
            author_match = re.search(pattern, text, re.IGNORECASE)
            if author_match:
                values["hazirlayan"] = author_match.group(1).strip()[:50]
                break
        
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        total_percentage = scores["percentage"]
        
        if total_percentage >= 70:
            recommendations.append(f"✅ Kullanma Kılavuzu GEÇERLİ (Toplam: %{total_percentage:.1f})")
        else:
            recommendations.append(f"❌ Kullanma Kılavuzu GEÇERSİZ (Toplam: %{total_percentage:.1f})")
        
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
        
        if total_percentage < 70:
            recommendations.extend([
                "",
                "💡 İYİLEŞTİRME ÖNERİLERİ:",
                "- Güvenlik uyarıları ve prosedürleri detaylandırılmalı",
                "- Kullanım talimatları adım adım açıklanmalı",
                "- Kurulum ve montaj bilgileri eksiksiz olmalı",
                "- Bakım ve arıza giderme bölümleri güçlendirilmeli",
                "- Teknik görseller ve şemalar eklenmeli"
            ])
        
        return recommendations

    def analyze_manual_report(self, pdf_path: str) -> Dict[str, Any]:
        """Ana Kullanma Kılavuzu analiz fonksiyonu"""
        logger.info("Kullanma Kılavuzu analizi başlatılıyor...")
        
        if not os.path.exists(pdf_path):
            return {"error": f"PDF dosyası bulunamadı: {pdf_path}"}
        
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"error": "PDF'den metin çıkarılamadı"}
        
        detected_lang = self.detect_language(text)
        
        if detected_lang != 'tr':
            logger.info(f"{detected_lang.upper()} dilinden Türkçe'ye çeviriliyor...")
            text = self.translate_to_turkish(text, detected_lang)
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category)
        
        scores = self.calculate_scores(analysis_results)
        extracted_values = self.extract_specific_values(text)
        recommendations = self.generate_recommendations(analysis_results, scores)
        
        final_status = "PASS" if scores["percentage"] >= 70 else "FAIL"
        
        return {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgisi": {"pdf_path": pdf_path, "detected_language": detected_lang},
            "cikarilan_degerler": extracted_values,
            "kategori_analizleri": analysis_results,
            "puanlama": scores,
            "oneriler": recommendations,
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["percentage"],
                "durum": final_status,
                "rapor_tipi": "KULLANMA_KILAVUZU"
            }
        }

# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_document_server(text):
    critical_terms = [
        ["kullanma", "kılavuz", "manual", "instruction", "talimat", "guide", "kılavuzu", "user manual"],
        ["güvenlik", "safety", "uyarı", "warning", "dikkat", "attention", "tehlike", "danger"],
        ["kurulum", "installation", "montaj", "assembly", "setup", "kurma", "takma", "yerleştirme"],
        ["kullanım", "operation", "işletim", "çalıştırma", "kullanma", "nasıl kullanılır", "how to use"],
        ["bakım", "maintenance", "temizlik", "cleaning", "arıza", "troubleshooting", "onarım", "repair"]
    ]
    category_found = [any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category) for category in critical_terms]
    valid = sum(category_found)
    logger.info(f"Manuel validasyon: {valid}/5 kategori")
    return valid >= 4

def check_strong_keywords_first_pages(filepath):
    strong_keywords = ["kullanma", "kılavuz", "manual", "instruction", "talimat", "guide", "kılavuzu"]
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
        logger.info(f"İlk sayfa: {len(found)} manuel kelime bulundu")
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"OCR hatası: {e}")
        return False

def check_excluded_keywords_first_pages(filepath):
    excluded = [
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219",
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        "loto",
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        "espe",
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        "montaj", "assembly",
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        "titreşim", "vibration", "mekanik",
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
        "topraklama direnci", "grounding", "earthing", "60204","topraklama", "TOPRAKLAMA DİRENCİ",
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
        found = [kw for kw in excluded if re.search(rf"\b{kw.lower()}\b", all_text)]
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"Excluded OCR hatası: {e}")
        return False

def get_conclusion_message_manuel(status, percentage):
    if status == "PASS":
        return f"Kullanım kılavuzu yeterli kriterleri sağlamaktadır (%{percentage:.0f})"
    return f"Kullanım kılavuzu yetersiz kriterlere sahip (%{percentage:.0f})"

def get_main_issues_manuel(report):
    issues = []
    if 'puanlama' in report and 'category_scores' in report['puanlama']:
        for category, score_data in report['puanlama']['category_scores'].items():
            if isinstance(score_data, dict) and score_data.get('percentage', 0) < 50:
                issues.append(f"{category} kategorisinde ciddi eksiklikler")
    return issues[:4]

def check_tesseract_installation():
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract OCR kurulu - Sürüm: {version}")
        return True, f"Tesseract v{version}"
    except Exception as e:
        logger.error(f"Tesseract OCR kurulu değil: {e}")
        return False, str(e)

tesseract_available, tesseract_info = check_tesseract_installation()

# ============================================
# FLASK APP
# ============================================
app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads_manuel'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/manuel-report', methods=['POST'])
def analyze_manuel_report():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"Manuel analizi başlatılıyor: {filename}")

            analyzer = ManualReportAnalyzer()
            file_ext = os.path.splitext(filepath)[1].lower()
            
            # 3 AŞAMALI KONTROL
            if file_ext == '.pdf':
                logger.info("Aşama 1: Kullanım Kılavuzu özgü kelime kontrolü...")
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
                        return jsonify({'error': 'Invalid document type', 'message': 'Bu dosya kullanım kılavuzu değil'}), 400
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
                                return jsonify({'error': 'Invalid document type', 'message': 'Yüklediğiniz dosya kullanım kılavuzu değil!'}), 400
                        except Exception as e:
                            logger.error(f"Aşama 3 hatası: {e}")
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({'error': 'Analysis failed'}), 500

            logger.info(f"Manuel analizi yapılıyor: {filename}")
            report = analyzer.analyze_manual_report(filepath)
            
            try:
                os.remove(filepath)
            except:
                pass

            if 'error' in report:
                return jsonify({'error': 'Analysis failed', 'message': report['error']}), 400

            overall_percentage = report.get('ozet', {}).get('yuzde', 0)
            status = "PASS" if overall_percentage >= 70 else "FAIL"
            
            response_data = {
                'analysis_date': report.get('analiz_tarihi'),
                'analysis_id': f"manuel_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'extracted_values': report.get('cikarilan_degerler', {}),
                'file_type': 'KULLANIM_KILAVUZU',
                'filename': filename,
                'language_info': {'detected_language': report.get('dosya_bilgisi', {}).get('detected_language', 'turkish')},
                'overall_score': {
                    'percentage': round(overall_percentage, 2),
                    'total_points': report.get('ozet', {}).get('toplam_puan', 0),
                    'max_points': 100,
                    'status': status,
                    'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ'
                },
                'recommendations': report.get('oneriler', []),
                'summary': {
                    'is_valid': status == "PASS",
                    'conclusion': get_conclusion_message_manuel(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_manuel(report)
                }
            }
            
            if 'puanlama' in report and 'category_scores' in report['puanlama']:
                for category, score_data in report['puanlama']['category_scores'].items():
                    if isinstance(score_data, dict):
                        response_data['category_scores'][category] = {
                            'score': score_data.get('normalized', 0),
                            'max_score': score_data.get('max_weight', 0),
                            'percentage': score_data.get('percentage', 0),
                            'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'FAIL'
                        }

            return jsonify({
                'success': True,
                'message': 'Manuel/Kullanım Kılavuzu başarıyla analiz edildi',
                'analysis_service': 'manuel_report',
                'data': response_data
            })

        except Exception as analysis_error:
            try:
                if 'filepath' in locals():
                    os.remove(filepath)
            except:
                pass
            logger.error(f"Analiz hatası: {str(analysis_error)}")
            return jsonify({'error': 'Analysis failed', 'message': str(analysis_error)}), 500

    except Exception as e:
        logger.error(f"API hatası: {str(e)}")
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Manual Report Analyzer API', 'version': '1.0.0', 'report_type': 'KULLANIM_KILAVUZU'})

@app.route('/', methods=['GET'])
def index():
    return jsonify({'service': 'Manual Report Analyzer API', 'version': '1.0.0', 'endpoints': {'POST /api/manuel-report': 'Manuel analizi', 'GET /api/health': 'Servis sağlık kontrolü'}})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8004))
    logger.info(f"🚀 Manuel Servisi - Port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)