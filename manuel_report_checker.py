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
class ManualCriteria:
    """Kullanma Kılavuzu kriterleri veri sınıfı"""
    genel_bilgiler: Dict[str, Any]
    giris_amac: Dict[str, Any]
    guvenlik_bilgileri: Dict[str, Any]
    urun_tanitimi: Dict[str, Any]
    kurulum_montaj: Dict[str, Any]
    kullanim_talimatlari: Dict[str, Any]
    bakim_temizlik: Dict[str, Any]
    ariza_giderme: Dict[str, Any]

@dataclass
class ManualAnalysisResult:
    """Kullanma Kılavuzu analiz sonucu veri sınıfı"""
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

class ManualReportAnalyzer:
    """Kullanma Kılavuzu rapor analiz sınıfı"""
    
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
                score = max(score, weight // 2)
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
                details={
                    "pattern_used": pattern,
                    "matches_count": len(matches) if matches else 0
                }
            )
        
        return results

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, ManualAnalysisResult]]) -> Dict[str, Any]:
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
        
        report = {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgisi": {
                "pdf_path": pdf_path,
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
                "rapor_tipi": "KULLANMA_KILAVUZU"
            }
        }
        
        return report

def main():
    """Ana fonksiyon"""
    analyzer = ManualReportAnalyzer()

    pdf_path = "T240303_Indonesia_DANONE-Electrical Instruction manual_20240828_V1.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF dosyası bulunamadı: {pdf_path}")
        return
    
    print("📖 Kullanma Kılavuzu Analizi Başlatılıyor...")
    print("=" * 60)
    
    report = analyzer.analyze_manual_report(pdf_path)
    
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
    
    print("\n📋 ÖNEMLİ ÇIKARILAN DEĞERLER")
    print("-" * 40)
    for key, value in report['cikarilan_degerler'].items():
        display_name = {
            "kilavuz_adi": "Kılavuz Adı",
            "urun_modeli": "Ürün Modeli",
            "hazırlama_tarihi": "Hazırlama Tarihi",
            "hazirlayan": "Hazırlayan"
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
                if criterion_result.found:
                    print(f"  ✅ {criterion_name.replace('_', ' ').title()}: {criterion_result.score}/{criterion_result.max_score} puan")
                else:
                    print(f"  ❌ {criterion_name.replace('_', ' ').title()}: 0/{criterion_result.max_score} puan - BULUNAMADI")
    
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
        print("📝 Değerlendirme: Kullanma kılavuzu genel olarak yeterli kriterleri sağlamaktadır.")
    else:
        print("❌ SONUÇ: GEÇERSİZ")
        print(f"⚠️ Toplam Başarı: %{report['ozet']['yuzde']:.1f}")
        print("📝 Değerlendirme: Kullanma kılavuzu minimum gereklilikleri sağlamamaktadır.")
        
        print("\n⚠️ EKSİK GEREKLİLİKLER:")
        for category, results in report['kategori_analizleri'].items():
            missing_items = []
            for criterion, result in results.items():
                if not result.found:
                    missing_items.append(criterion)
            
            if missing_items:
                print(f"\n🔍 {category}:")
                for item in missing_items:
                    readable_name = {
                        "kilavuz_adi_kod": "Kılavuz Adı ve Kod",
                        "urun_modeli": "Ürün Modeli",
                        "hazırlama_tarihi": "Hazırlama Tarihi",
                        "hazirlayan_onaylayan": "Hazırlayan/Onaylayan",
                        "revizyon_bilgisi": "Revizyon Bilgisi",
                        "kilavuz_amaci": "Kılavuz Amacı",
                        "kapsam": "Kapsam",
                        "hedef_kullanici": "Hedef Kullanıcı",
                        "genel_guvenlik": "Genel Güvenlik",
                        "tehlikeler": "Tehlikeler",
                        "guvenlik_prosedur": "Güvenlik Prosedürü",
                        "kkd_gerekliligi": "KKD Gerekliliği",
                        "urun_tanimi": "Ürün Tanımı",
                        "teknik_ozellikler": "Teknik Özellikler",
                        "bilesenler": "Bileşenler",
                        "gorseller": "Görseller",
                        "kurulum_oncesi": "Kurulum Öncesi",
                        "montaj_talimatlari": "Montaj Talimatları",
                        "gerekli_aletler": "Gerekli Aletler",
                        "kurulum_kontrolu": "Kurulum Kontrolü",
                        "calistirma": "Çalıştırma",
                        "kullanim_kilavuzu": "Kullanım Kılavuzu",
                        "calisma_modlari": "Çalışma Modları",
                        "kullanim_ipuclari": "Kullanım İpuçları",
                        "duzenli_bakim": "Düzenli Bakım",
                        "temizlik_yontemleri": "Temizlik Yöntemleri",
                        "parca_degisimi": "Parça Değişimi",
                        "sorun_cozumleri": "Sorun Çözümleri",
                        "hata_kodlari": "Hata Kodları",
                        "teknik_destek": "Teknik Destek",
                        "teknik_cizimler": "Teknik Çizimler",
                        "baglanti_planlari": "Bağlantı Planları",
                        "yedek_parca_listesi": "Yedek Parça Listesi",
                        "garanti": "Garanti",
                        "yasal_uyarilar": "Yasal Uyarılar"
                    }.get(item, item.replace('_', ' ').title())
                    print(f"   ❌ {readable_name}")
        
        print("\n📌 YAPILMASI GEREKENLER:")
        print("1. Güvenlik bölümünü detaylandırın")
        print("2. Kullanım talimatlarını adım adım açıklayın")
        print("3. Kurulum ve montaj bilgilerini eksiksiz verin")
        print("4. Bakım ve arıza giderme bölümlerini güçlendirin")
        print("5. Teknik görseller ve şemalar ekleyin")
        print("6. Yasal uyarılar ve garanti bilgilerini belirtin")

if __name__ == "__main__":
    main()
