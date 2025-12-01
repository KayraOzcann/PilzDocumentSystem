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
# DATABASE IMPORTS
# ============================================
from flask import current_app
from database import db, init_db
from db_loader import load_service_config
from config import Config

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    def __init__(self, app=None):
        if app:
            with app.app_context():
                try:
                    config = load_service_config('lvd_report')
                    
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
            logger.warning("⚠️ Flask app context yok, boş config kullanılıyor")
            self.criteria_weights = {}
            self.criteria_details = {}
            self.pattern_definitions = {}
            self.validation_keywords = {}
            self.category_actions = {}

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
        """Tarih bilgilerini çıkar - DB'den pattern'lerle"""
        
        # DB'den pattern'leri al
        check_date_patterns = self.pattern_definitions.get('check_date_validity', {})
        olcum_patterns = check_date_patterns.get('olcum_patterns', [])
        rapor_patterns = check_date_patterns.get('rapor_patterns', [])
        
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
            pattern = criterion_data.get("pattern", "")
            weight = criterion_data.get("weight", 0)
            reverse_logic = criterion_data.get("reverse_logic", False)
            
            if not pattern:
                logger.warning(f"Pattern bulunamadı: {criterion_name}")
                continue
            
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
                    # İkincil arama - daha genel pattern - DB'den
                    general_patterns_dict = self.pattern_definitions.get('general_patterns', {})
                    general_pattern_list = general_patterns_dict.get(criterion_name, [])
                    
                    if general_pattern_list:
                        for general_pattern in general_pattern_list:
                            general_matches = re.findall(general_pattern, text, re.IGNORECASE)
                            if general_matches:
                                content = f"Genel eşleşme bulundu: {general_matches[0]}"
                                found = True
                                score = weight // 2
                                break
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
        """Spesifik değerleri çıkarma - DB'den pattern'lerle"""
        values = {}
        
        # Dosya adından bilgileri çıkar
        if file_path:
            filename = os.path.basename(file_path)
            extract_patterns = self.pattern_definitions.get('extract_values', {})
            
            # Proje numarası
            proje_patterns = extract_patterns.get('proje_patterns', [])
            proje_no = "Bulunamadı"
            for pattern in proje_patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    proje_no = match.group(1)
                    break
            values["proje_no"] = proje_no
            
            # Rapor numarası
            rapor_patterns = extract_patterns.get('rapor_patterns', [])
            rapor_no = "Bulunamadı"
            for pattern in rapor_patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    rapor_no = match.group(1)
                    break
            values["rapor_numarasi"] = rapor_no
            
            # Müşteri bilgisi
            musteri_patterns = extract_patterns.get('musteri_patterns', [])
            musteri = "Bulunamadı"
            for pattern in musteri_patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    musteri = pattern
                    break
            values["musteri"] = musteri
            
            # Revizyon bilgisi
            revizyon_match = re.search(r'[vV](\d+)', filename)
            values["revizyon"] = f"v{revizyon_match.group(1)}" if revizyon_match else "v0"
        
        # Metinden değerleri çıkar - DB'den pattern'lerle
        value_patterns = self.pattern_definitions.get('value_patterns', {})
        
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
        
        # Ölçüm verilerini analiz et
        self.analyze_measurement_data(text, values)
        
        return values
    
    def analyze_measurement_data(self, text: str, values: Dict[str, Any]):
        """Ölçüm verilerini analiz et - DB'den pattern'lerle"""
        extract_patterns = self.pattern_definitions.get('extract_values', {})
        
        # RLO değerlerini topla
        rlo_patterns = extract_patterns.get('rlo_patterns', [])
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
        
        # Kesit bilgilerini analiz et
        kesit_patterns = extract_patterns.get('kesit_patterns', [])
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
        
        # Hat/bölge bilgileri - DB'den pattern
        hat_pattern_list = extract_patterns.get('hat_pattern', [])
        if hat_pattern_list:
            hat_matches = re.findall(hat_pattern_list[0], text, re.IGNORECASE)
            if hat_matches:
                unique_hats = list(set(hat_matches))
                values["makine_hatlari"] = ", ".join(unique_hats)
            else:
                values["makine_hatlari"] = "Bulunamadı"
        else:
            values["makine_hatlari"] = "Bulunamadı"
    
    def find_non_compliant_measurements(self, text: str, values: Dict[str, Any]):
        """Uygunsuz ölçümleri tespit et - DB'den pattern'lerle"""
        lines = text.split('\n')
        non_compliant = []
        
        extract_patterns = self.pattern_definitions.get('extract_values', {})
        high_rlo_patterns = extract_patterns.get('high_rlo_patterns', [])
        hat_patterns = extract_patterns.get('hat_patterns2', [])
        
        for i, line in enumerate(lines):
            sira_match = re.search(r'(\d+)\s', line)
            if sira_match:
                sira = sira_match.group(1)
                
                # Yüksek RLO değeri kontrolü
                for pattern in high_rlo_patterns:
                    high_rlo_match = re.search(pattern, line, re.IGNORECASE)
                    if high_rlo_match:
                        try:
                            rlo_value = float(str(high_rlo_match.group(1)).replace(',', '.'))
                            if rlo_value > 500:
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
                
                # D.Y. kontrolü
                if '*D.Y' in line or 'D.Y' in line or 'N/A' in line or 'N/A' in line:
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
        
        text = self.extract_text_from_pdf(file_path)
        if not text:
            return {"error": "Dosya okunamadı"}
        
        date_valid, olcum_tarihi, rapor_tarihi, date_message = self.check_date_validity(text, file_path)
        extracted_values = self.extract_specific_values(text, file_path)
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(text, category)
        
        scores = self.calculate_scores(analysis_results)
        
        final_status = "PASSED"
        if scores["overall_percentage"] < 70:
            final_status = "FAILED"
            fail_reason = f"Toplam puan yetersiz (%{scores['overall_percentage']:.1f} < 70)"
        else:
            fail_reason = None
        
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
        
        if scores["overall_percentage"] >= 70 and date_valid:
            recommendations.append("\n✅ RAPOR BAŞARILI")
        
        return recommendations

# ============================================================================
# HELPER FUNCTIONS
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

def validate_document_server(text, validation_keywords):
    critical_terms_data = validation_keywords.get('critical_terms', [])
    
    critical_terms = []
    for item in critical_terms_data:
        if isinstance(item, dict) and 'keywords' in item:
            critical_terms.append(item['keywords'])
        elif isinstance(item, list):
            critical_terms.append(item)
    
    if not critical_terms:
        logger.warning("⚠️ Critical terms bulunamadı")
        return True
    
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
    logger.info(f"Doküman validasyonu: {valid_categories}/{len(critical_terms)} kritik kategori bulundu")
    return valid_categories >= len(critical_terms) - 1

def check_strong_keywords_first_pages(filepath, validation_keywords):
    strong_keywords = validation_keywords.get('strong_keywords', [])
    
    if not strong_keywords:
        logger.warning("⚠️ Strong keywords bulunamadı")
        return True
    
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

def check_excluded_keywords_first_pages(filepath, validation_keywords):
    excluded_keywords = validation_keywords.get('excluded_keywords', [])
    
    if not excluded_keywords:
        logger.warning("⚠️ Excluded keywords bulunamadı")
        return False
    
    try:
        pages = pdf2image.convert_from_path(filepath, dpi=300, first_page=1, last_page=2)
        
        all_text = ""
        for i, page in enumerate(pages):
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6 -l tur+eng', timeout=15)
            all_text += text.lower() + " "
        
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
    if status == "PASS":
        return f"LVD topraklama raporu EN 60204-1 standardına uygun ve elektrik güvenlik gereksinimlerini karşılamaktadır (%{percentage:.0f})"
    elif status == "CONDITIONAL":
        return f"LVD topraklama raporu kabul edilebilir ancak bazı eksiklikler var (%{percentage:.0f})"
    else:
        return f"LVD topraklama raporu standartlara uygun değil, kapsamlı revizyon gerekli (%{percentage:.0f})"

def get_main_issues_lvd(report):
    issues = []
    for category, score_data in report['puanlama']['category_scores'].items():
        if isinstance(score_data, dict) and score_data.get('percentage', 0) < 50:
            issues.append(f"{category} kategorisinde ciddi eksiklikler")
    
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
# FLASK APP
# ============================================================================

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
app.config['SQLALCHEMY_ECHO'] = Config.SQLALCHEMY_ECHO

PORT = int(os.environ.get('PORT', 8007))
UPLOAD_FOLDER = 'temp_uploads_lvd'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/lvd-report', methods=['POST'])
def analyze_lvd_report():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        analyzer = GroundingContinuityReportAnalyzer(app=app)
        file_ext = os.path.splitext(filepath)[1].lower()

        if file_ext == '.pdf':
            logger.info("Aşama 1: LVD özgü kelime kontrolü...")
            if check_strong_keywords_first_pages(filepath, analyzer.validation_keywords):
                logger.info("✅ Aşama 1 geçti")
            else:
                logger.info("Aşama 2: Excluded kelime kontrolü...")
                if check_excluded_keywords_first_pages(filepath, analyzer.validation_keywords):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                    return jsonify({'error': 'Invalid document type', 'message': 'Bu dosya LVD raporu değil'}), 400
                else:
                    logger.info("Aşama 3: Tam doküman kontrolü...")
                    try:
                        with open(filepath, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
                        
                        if not text or len(text.strip()) < 50 or not validate_document_server(text, analyzer.validation_keywords):
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            return jsonify({'error': 'Invalid document type', 'message': 'Yüklediğiniz dosya LVD raporu değil!'}), 400
                    except Exception as e:
                        logger.error(f"Aşama 3 hatası: {e}")
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        return jsonify({'error': 'Analysis failed'}), 500

        logger.info(f"LVD raporu analizi yapılıyor: {filename}")
        report = analyzer.generate_detailed_report(filepath)
        
        try:
            os.remove(filepath)
        except:
            pass

        if 'error' in report:
            return jsonify({'error': 'Analysis failed', 'message': report['error']}), 400

        overall_percentage = report['ozet']['yuzde']
        status = "PASS" if overall_percentage >= 70 else "FAIL"
        
        response_data = {
            'analysis_date': report.get('analiz_tarihi'),
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
            'overall_score': {
                'percentage': round(overall_percentage, 2),
                'total_points': report['ozet']['toplam_puan'],
                'max_points': 100,
                'status': status,
                'status_tr': 'GEÇERLİ' if status == "PASS" else 'GEÇERSİZ'
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
                    'score': score_data.get('normalized', 0),
                    'max_score': score_data.get('max_weight', 0),
                    'percentage': score_data.get('percentage', 0),
                    'status': 'PASS' if score_data.get('percentage', 0) >= 70 else 'FAIL'
                }

        return jsonify({'success': True, 'message': 'LVD başarıyla analiz edildi', 'analysis_service': 'lvd_report', 'data': response_data})

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'LVD Report Analyzer API', 'version': '1.0.0'})

@app.route('/', methods=['GET'])
def index():
    return jsonify({'service': 'LVD Report Analyzer API', 'version': '1.0.0'})

with app.app_context():
    db.init_app(app)

if __name__ == '__main__':
    logger.info(f"🚀 LVD Servisi - Port: {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)