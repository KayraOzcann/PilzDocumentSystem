"""
LOTO Prosedürü Analiz Servisi
==============================
Azure App Service için optimize edilmiş standalone servis
Database-driven configuration ile dinamik pattern yönetimi

Endpoint: POST /api/loto-report
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

from flask import Flask, request, jsonify, current_app
from werkzeug.utils import secure_filename

# ============================================
# DATABASE IMPORTS (YENİ)
# ============================================
from database import db, init_db
from db_loader import load_service_config

# ============================================
# CONFIG IMPORT (YENİ)
# ============================================
from config import Config

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
class LOTOAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

# ============================================
# ANALYZER CLASS
# ============================================
class LOTOReportAnalyzer:
    def __init__(self, app=None):
        logger.info("LOTO Prosedür Analiz Sistemi başlatılıyor...")
        
        # Flask app context varsa DB'den yükle, yoksa boş başlat
        if app:
            with app.app_context():
                try:
                    config = load_service_config('loto_report')
                    
                    # DB'den yüklenen veriler
                    self.criteria_weights = config.get('criteria_weights', {})
                    self.criteria_details = config.get('criteria_details', {})
                    self.pattern_definitions = config.get('pattern_definitions', {})
                    self.validation_keywords = config.get('validation_keywords', {})
                    self.category_actions = config.get('category_actions', {})
                    
                    logger.info(f"✅ Veritabanından yüklendi: {len(self.criteria_weights)} kategori")
                    
                except Exception as e:
                    logger.error(f"⚠️ Veritabanından yükleme başarısız: {e}")
                    logger.warning("⚠️ Fallback: Boş config kullanılıyor")
                    self.criteria_weights = {}
                    self.criteria_details = {}
                    self.pattern_definitions = {}
                    self.validation_keywords = {}
                    self.category_actions = {}
        else:
            # Flask app yoksa boş başlat (eski davranış)
            logger.warning("⚠️ Flask app context yok, boş config kullanılıyor")
            self.criteria_weights = {}
            self.criteria_details = {}
            self.pattern_definitions = {}
            self.validation_keywords = {}
            self.category_actions = {}
    
    def detect_language(self, text: str) -> str:
        if not LANGUAGE_DETECTION_AVAILABLE:
            return 'tr'
        try:
            return detect(text[:500].strip())
        except:
            return 'tr'
    
    def translate_to_turkish(self, text: str, source_lang: str) -> str:
        """Metni Türkçe'ye çevir - Temel İngilizce desteği"""
        if source_lang != 'tr' and source_lang == 'en':
            logger.info(f"İngilizce belgede temel terim çevirisi uygulanıyor...")
            
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
    
    def detect_document_type(self, text: str) -> str:
        """Belge türünü tespit et - DB'den gelen pattern'ler ile"""
        
        # DB'den pattern'leri al
        extract_values = self.pattern_definitions.get('extract_values', {})
        
      
        analysis_indicators = extract_values.get('analysis_indicators', [])
        procedure_indicators = extract_values.get('procedure_indicators', [])
        
        analysis_count = sum(1 for pattern in analysis_indicators 
                           if re.search(pattern, text, re.IGNORECASE))
        
        procedure_count = sum(1 for pattern in procedure_indicators 
                            if re.search(pattern, text, re.IGNORECASE))
        
        logger.info(f"Analiz göstergeleri: {analysis_count}, Prosedür göstergeleri: {procedure_count}")
        
        return "procedure_document" if procedure_count > analysis_count else "analysis_report"

    def analyze_criteria(self, text: str, category: str, document_type: str = "analysis_report") -> Dict[str, LOTOAnalysisResult]:
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
                
                if criterion_name in ["izolasyon_noktalari_tablo", "cihaz_durumu_kontrol"]:
                    score = weight
                else:
                    score = min(weight, len(matches) * (weight // 2))
                    score = max(score, weight // 2)
            else:
                content = "Bulunamadı"
                found = False
                score = 0
                
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
                details={"pattern_used": pattern, "matches_count": len(matches) if matches else 0, "document_type": document_type}
            )
        
        return results
    
    def handle_procedure_document_scoring(self, criterion_name: str, text: str, weight: int) -> int:
        """Prosedür dökümanı için özel puanlama"""
        procedure_adaptations = {
            "kabul_edilebilirlik": weight,
            "bulgular_yorumlar": weight // 2,
            "sonuc_tablolari": weight // 2,
            "oneriler": weight,
            "izolasyon_noktalari_tablo": weight if re.search(r"fig|figure|diagram|şema", text, re.IGNORECASE) else 0,
            "prosedur_detaylari": weight,
            "tavsiyeler": weight,
            "makine_tanimi": weight // 2 if re.search(r"line|hat|ekipman|equipment", text, re.IGNORECASE) else 0,
            "tesis_bilgileri": weight // 2 if re.search(r"zone|alan|facility", text, re.IGNORECASE) else 0,
            "uygunsuz_enerji_tablosu": weight if re.search(r"energy|enerji", text, re.IGNORECASE) else 0,
            "mevzuat_uygunlugu": weight // 2,
            "mevzuat_referanslari": weight // 2,
        }
        return procedure_adaptations.get(criterion_name, 0)

    def check_date_validity(self, text: str) -> Dict[str, Any]:
        """Rapor tarihini bul - DB'den gelen pattern'ler ile"""
        
        # DB'den date pattern'lerini al
        extract_values = self.pattern_definitions.get('extract_values', {})
        date_patterns = extract_values.get('date_patterns', [])
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(str(match[2])) == 4:
                        day, month, year = int(match[0]), int(match[1]), int(match[2])
                    else:
                        year, month, day = int(match[0]), int(match[1]), int(match[2])
                    
                    if 1 <= day <= 31 and 1 <= month <= 12 and 2020 <= year <= 2030:
                        report_date = datetime(year, month, day)
                        current_date = datetime.now()
                        date_diff = current_date - report_date
                        
                        return {
                            "found": True,
                            "report_date": report_date.strftime("%d.%m.%Y"),
                            "days_old": date_diff.days,
                            "is_valid": True,
                            "validity_reason": "Tarih bulundu"
                        }
                except:
                    continue
        
        return {
            "found": False,
            "report_date": "Bulunamadı",
            "days_old": 0,
            "is_valid": True,
            "validity_reason": "Rapor tarihi bulunamadı ama kabul edilebilir"
        }

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
        """Spesifik değerleri çıkar - DB'den gelen pattern'ler ile"""
        values = {
            "proje_adi": "Bulunamadı",
            "rapor_tarihi": "Bulunamadı",
            "hazirlayan_firma": "Bulunamadı",
            "kabul_durumu": "Bulunamadı"
        }
        
        # DB'den pattern'leri al
        extract_values = self.pattern_definitions.get('extract_values', {})
        
        # Eğer DB'den veri gelmediyse, boş liste kullan
        project_patterns = extract_values.get('proje_adi', [])
        date_patterns = extract_values.get('rapor_tarihi', [])
        company_patterns = extract_values.get('hazirlayan_firma', [])
        acceptance_patterns = extract_values.get('kabul_durumu', [])
        
        # PROJE ADI
        for pattern in project_patterns:
            project_match = re.search(pattern, text, re.IGNORECASE)
            if project_match:
                values["proje_adi"] = project_match.group(1).strip()[:50]
                break
        
        # RAPOR TARİHİ
        for pattern in date_patterns:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                values["rapor_tarihi"] = date_match.group(1)
                break
        
        # HAZIRLAYAN FİRMA
        for pattern in company_patterns:
            company_match = re.search(pattern, text, re.IGNORECASE)
            if company_match:
                if len(company_match.groups()) > 0:
                    values["hazirlayan_firma"] = company_match.group(1).strip()[:50]
                else:
                    values["hazirlayan_firma"] = company_match.group().strip()[:50]
                break
        
        # KABUL DURUMU
        for pattern in acceptance_patterns:
            acceptance_match = re.search(pattern, text, re.IGNORECASE)
            if acceptance_match:
                values["kabul_durumu"] = acceptance_match.group(1).upper()
                break
        
        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict, date_validity: Dict, document_type: str = "analysis_report") -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        
        if date_validity["found"]:
            recommendations.append(f"📅 Rapor tarihi: {date_validity['report_date']}")
        else:
            recommendations.append("📅 Rapor tarihi: Tespit edilemedi")
        
        total_percentage = scores["percentage"]
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
            min_threshold = 30 if document_type == "procedure_document" else 40
            good_threshold = 50 if document_type == "procedure_document" else 70
            
            if category_score < min_threshold:
                recommendations.append(f"🔴 {category} bölümü yetersiz (%{category_score:.1f})")
            elif category_score < good_threshold:
                recommendations.append(f"🟡 {category} bölümü geliştirilmeli (%{category_score:.1f})")
            else:
                recommendations.append(f"🟢 {category} bölümü yeterli (%{category_score:.1f})")
        
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
        
        document_type = self.detect_document_type(text)
        logger.info(f"Tespit edilen belge türü: {document_type}")
        
        date_validity = self.check_date_validity(text)
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category, document_type)
        
        # Mevzuat cross-check
        if ("Teknik Değerlendirme ve Sonuçlar" in analysis_results and 
            "mevzuat_uygunlugu" in analysis_results["Teknik Değerlendirme ve Sonuçlar"] and
            analysis_results["Teknik Değerlendirme ve Sonuçlar"]["mevzuat_uygunlugu"].found and
            "Dokümantasyon ve Referanslar" in analysis_results and
            "mevzuat_referanslari" in analysis_results["Dokümantasyon ve Referanslar"] and
            not analysis_results["Dokümantasyon ve Referanslar"]["mevzuat_referanslari"].found):
            
            mevzuat_ref = analysis_results["Dokümantasyon ve Referanslar"]["mevzuat_referanslari"]
            mevzuat_ref.found = True
            mevzuat_ref.content = "Teknik değerlendirmede mevzuat uygunluğu bulundu"
            mevzuat_ref.score = mevzuat_ref.max_score
        
        scores = self.calculate_scores(analysis_results)
        extracted_values = self.extract_specific_values(text)
        recommendations = self.generate_recommendations(analysis_results, scores, date_validity, document_type)
        
        pass_threshold = 50 if document_type == "procedure_document" else 70
        final_status = "PASS" if scores["percentage"] >= pass_threshold else "FAIL"
        
        return {
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

# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_document_server(text, validation_keywords):
    """Server kodunda doküman validasyonu - DB'den gelen keywords ile"""
    
    # DB'den gelen critical_terms
    critical_terms_data = validation_keywords.get('critical_terms', {})
    
    # Liste formatına dönüştür
    critical_terms = []
    for key, value in critical_terms_data.items():
        if key.startswith('category_') and isinstance(value, list):
            critical_terms.append(value)
    
    # Eğer DB'den veri gelmediyse, boş validasyon
    if not critical_terms:
        logger.warning("⚠️ Critical terms bulunamadı, validasyon atlanıyor")
        return True
    
    category_found = [any(re.search(rf"\b{term}\b", text, re.IGNORECASE) for term in category) for category in critical_terms]
    valid = sum(category_found)
    logger.info(f"LOTO validasyon: {valid}/{len(critical_terms)} kategori")
    return valid >= len(critical_terms) - 1  # En az (n-1) kategori

def check_strong_keywords_first_pages(filepath, validation_keywords):
    """İlk sayfada LOTO özgü kelimeleri OCR ile ara - DB'den"""
    
    # DB'den strong keywords al
    strong_keywords = validation_keywords.get('strong_keywords', [])
    
    # Eğ DB'den veri gelmediyse, validasyon atla
    if not strong_keywords:
        logger.warning("⚠️ Strong keywords bulunamadı, validasyon atlanıyor")
        return True
    
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
        logger.info(f"İlk sayfa: {len(found)} LOTO kelime bulundu")
        return len(found) >= 1
    except Exception as e:
        logger.warning(f"OCR hatası: {e}")
        return False

def check_excluded_keywords_first_pages(filepath, validation_keywords):
    """İlk 1-2 sayfada istenmeyen rapor türlerinin kelimelerini ara - DB'den"""
    
    # DB'den excluded keywords al
    excluded_keywords = validation_keywords.get('excluded_keywords', [])
    
    # Eğer DB'den veri gelmediyse, validasyon atla
    if not excluded_keywords:
        logger.warning("⚠️ Excluded keywords bulunamadı, validasyon atlanıyor")
        return False
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=2)
        all_text = ""
        for page in pages:
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        found = [kw for kw in excluded_keywords if re.search(rf"\b{kw.lower()}\b", all_text)]
        return len(found) >= 2
    except Exception as e:
        logger.warning(f"Excluded OCR hatası: {e}")
        return False

def get_conclusion_message_loto(status, percentage):
    if status == "PASS":
        return f"LOTO prosedürü OSHA standartlarına uygun (%{percentage:.0f})"
    return f"LOTO prosedürü standartlara uygun değil (%{percentage:.0f})"

def get_main_issues_loto(report):
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

# Database configuration (YENİ)
app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
app.config['SQLALCHEMY_ECHO'] = Config.SQLALCHEMY_ECHO

# Upload configuration
UPLOAD_FOLDER = 'temp_uploads_loto'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/loto-report', methods=['POST'])
def analyze_loto_report():
    """LOTO Prosedürü analiz API endpoint'i - 3 Aşamalı Validasyon"""
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
            logger.info(f"LOTO analizi başlatılıyor: {filename}")

            # Create analyzer instance with app context
            analyzer = LOTOReportAnalyzer(app=app)
            
            file_ext = os.path.splitext(filepath)[1].lower()
            
            # ÜÇ AŞAMALI LOTO KONTROLÜ
            logger.info(f"Üç aşamalı LOTO kontrolü başlatılıyor: {filename}")
            
            if file_ext == '.pdf':
                # PDF için üç aşamalı kontrol (OCR dahil)
                logger.info("Aşama 1: İlk sayfa LOTO özgü kelime kontrolü...")
                if check_strong_keywords_first_pages(filepath, analyzer.validation_keywords):
                    logger.info("✅ Aşama 1 geçti - LOTO özgü kelimeler bulundu")
                else:
                    logger.info("Aşama 2: İlk sayfa excluded kelime kontrolü...")
                    if check_excluded_keywords_first_pages(filepath, analyzer.validation_keywords):
                        logger.info("❌ Aşama 2'de excluded kelimeler bulundu - LOTO değil")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({
                            'error': 'Invalid document type',
                            'message': 'Bu dosya LOTO prosedürü değil (farklı rapor türü tespit edildi)',
                            'details': {
                                'filename': filename,
                                'document_type': 'OTHER_REPORT_TYPE',
                                'required_type': 'LOTO_PROSEDURU'
                            }
                        }), 400
                    else:
                        # AŞAMA 3: PyPDF2 ile tam doküman kontrolü
                        logger.info("Aşama 3: Tam doküman critical terms kontrolü...")
                        try:
                            with open(filepath, 'rb') as f:
                                pdf_reader = PyPDF2.PdfReader(f)
                                text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
                            
                            if not text or len(text.strip()) < 50:
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                                return jsonify({
                                    'error': 'Text extraction failed',
                                    'message': 'Dosyadan yeterli metin çıkarılamadı'
                                }), 400
                            
                            if not validate_document_server(text, analyzer.validation_keywords):
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                                return jsonify({
                                    'error': 'Invalid document type',
                                    'message': 'Yüklediğiniz dosya LOTO prosedürü değil! Lütfen geçerli bir LOTO prosedürü yükleyiniz.',
                                    'details': {
                                        'filename': filename,
                                        'document_type': 'NOT_LOTO_REPORT',
                                        'required_type': 'LOTO_PROSEDURU'
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

            # Buraya kadar geldiyse LOTO prosedürü, şimdi analizi yap
            logger.info(f"LOTO prosedürü doğrulandı, analiz başlatılıyor: {filename}")
            report = analyzer.analyze_loto_report(filepath)
            
            # Clean up
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
                'analysis_id': f"loto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'category_scores': {},
                'extracted_values': report.get('cikarilan_degerler', {}),
                'file_type': 'LOTO_PROSEDURU',
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
                    'conclusion': get_conclusion_message_loto(status, overall_percentage),
                    'main_issues': [] if status == "PASS" else get_main_issues_loto(report)
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
                'message': 'LOTO Prosedürü başarıyla analiz edildi',
                'analysis_service': 'loto_report',
                'service_description': 'LOTO Raporu Analizi',
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
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'LOTO Report Analyzer API',
        'version': '1.0.0',
        'upload_folder': UPLOAD_FOLDER,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'ocr_support': tesseract_available,
        'report_type': 'LOTO_PROSEDURU'
    })

@app.route('/', methods=['GET'])
def index():
    """Ana sayfa - API bilgileri"""
    return jsonify({
        'service': 'LOTO Report Analyzer API',
        'version': '1.0.0',
        'description': 'LOTO Prosedürlerini analiz eden REST API servisi',
        'endpoints': {
            'POST /api/loto-report': 'LOTO prosedür analizi',
            'GET /api/health': 'Servis sağlık kontrolü',
            'GET /': 'Bu bilgi sayfası'
        },
        'usage': {
            'upload_format': 'multipart/form-data',
            'file_field': 'file',
            'supported_types': list(ALLOWED_EXTENSIONS),
            'max_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        }
    })

# ============================================
# DATABASE INITIALIZATION
# ============================================
with app.app_context():
    db.init_app(app)

# ============================================
# APPLICATION ENTRY POINT (Azure-Friendly)
# ============================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("LOTO Prosedür Analiz Servisi")
    logger.info("=" * 60)
    logger.info(f"📁 Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"📊 Desteklenen formatlar: {', '.join(ALLOWED_EXTENSIONS)}")
    logger.info(f"📏 Maksimum dosya boyutu: {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB")
    logger.info("")
    logger.info("🔗 API Endpoints:")
    logger.info("  POST /api/loto-report - LOTO prosedür analizi")
    logger.info("  GET /api/health - Servis sağlık kontrolü")
    logger.info("  GET / - API bilgileri")
    logger.info("=" * 60)
    
    # Azure App Service PORT environment variable (default: 8005)
    port = int(os.environ.get('PORT', 8005))
    
    logger.info(f"🚀 Servis başlatılıyor - Port: {port}")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Production için debug=False
    )