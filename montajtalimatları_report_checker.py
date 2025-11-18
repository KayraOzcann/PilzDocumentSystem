#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Report Checker (Kullanım Kılavuzu Analiz Sistemi)
Created for analyzing operating manuals from various companies
Supports both Turkish and English with OCR capabilities
"""

import re
import os
from datetime import datetime
from typing import Dict, List, Any
import PyPDF2
from dataclasses import dataclass
import logging
import pytesseract
from pdf2image import convert_from_path

try:
    from langdetect import detect
    LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
                "kkd_gerekliligi": {"pattern": r"(?:KKD|PPE|Personal\s*Protective|Koruyucu\s*Donanım|Protective\s*Equipment|Eldiven|Glove|Gözlük|Goggle|Koruyucu\s*Alet)", "weight": 4}
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
                "hata_kodlari": {"pattern": r"(?:Hata\s*Kod|Error\s*Code|Kod|Code|Alarm|Uyar[ıi]\s*Lambas[ıi]|Warning\s*Light)", "weight": 5},
                "teknik_destek": {"pattern": r"(?:Teknik\s*Destek|Technical\s*Support|Destek|Support|İletişim|Contact|Tel|Phone|E-?mail)", "weight": 3},
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
                content = f"Found: {str(matches[:3])}"
                found = True
                score = weight
            else:
                content = "Not found"
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
        """Puanlama hesaplama"""
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
        """Özel değerleri çıkar"""
        values = {
            "manual_name": "Bulunamadı",
            "product_model": "Bulunamadı",
            "revision_info": "Bulunamadı",
            "manufacturer": "Bulunamadı",
            "contact_info": "Bulunamadı",
            "safety_warnings_count": 0
        }
        
        # Manual name extraction
        manual_patterns = [
            r"(?:Kullan[ıi]m\s*K[ıi]lavuzu|User\s*Manual|Operating\s*Manual|Manual)",
            r"(?:Guide|K[ıi]lavuz|Handbook)"
        ]
        
        for pattern in manual_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["manual_name"] = match.group(0).strip()
                break
        
        # Product model
        model_patterns = [
            r"(?:Model|Product|Ürün)\s*(?:No|Number)?\s*[:\-]?\s*([A-Z0-9\-\.]{3,20})",
            r"(?:Type|Tip|Model)\s*[:\-]?\s*([A-Z0-9\-\.]{3,20})"
        ]
        
        for pattern in model_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values["product_model"] = match.group(1).strip()
                break
        
        # Safety warnings count
        safety_patterns = [
            r"(?:UYARI|WARNING|DİKKAT|CAUTION|Güvenlik)",
        ]
        
        safety_count = 0
        for pattern in safety_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            safety_count += len(matches)
        
        values["safety_warnings_count"] = safety_count
        
        return values
    
    def generate_recommendations(self, analysis_results: Dict[str, Dict[str, ManualAnalysisResult]], scores: Dict[str, Any]) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        
        for category, score_data in scores["category_scores"].items():
            if score_data["percentage"] < 50:
                recommendations.append(f"⚠️ {category} kategorisinde ciddi eksiklikler var (%{score_data['percentage']:.0f})")
            elif score_data["percentage"] < 70:
                recommendations.append(f"📝 {category} kategorisi geliştirilebilir (%{score_data['percentage']:.0f})")
        
        missing_critical = []
        for category, results in analysis_results.items():
            for criterion_name, result in results.items():
                if not result.found and result.max_score >= 4:
                    missing_critical.append(f"{category}: {criterion_name}")
        
        if missing_critical:
            recommendations.append("🔍 Eksik kritik kriterler:")
            for item in missing_critical[:5]:
                recommendations.append(f"  • {item}")
        
        total_percentage = scores["percentage"]
        if total_percentage >= 80:
            recommendations.append("✅ Kullanım kılavuzu yüksek kalitede ve standartlara uygun")
        elif total_percentage >= 70:
            recommendations.append("📋 Kullanım kılavuzu kabul edilebilir seviyede")
        else:
            recommendations.append("❌ Kullanım kılavuzu yetersiz, kapsamlı revizyon gerekli")
        
        return recommendations
    
    def analyze_manual(self, pdf_path: str) -> Dict[str, Any]:
        """Ana analiz fonksiyonu"""
        logger.info("Kullanım kılavuzu analizi başlıyor...")
        
        try:
            text = self.extract_text_from_pdf(pdf_path)
            
            if len(text.strip()) < 50:
                return {
                    "error": "PDF'den yeterli metin çıkarılamadı. Dosya bozuk olabilir veya sadece resim içeriyor olabilir.",
                    "text_length": len(text)
                }
            
            detected_language = self.detect_language(text)
            logger.info(f"Tespit edilen dil: {detected_language}")
            
            extracted_values = self.extract_specific_values(text)
            
            category_analyses = {}
            for category in self.criteria_weights.keys():
                category_analyses[category] = self.analyze_criteria(text, category)
            
            scoring = self.calculate_scores(category_analyses)
            recommendations = self.generate_recommendations(category_analyses, scoring)
            
            percentage = scoring["percentage"]
            if percentage >= 70:
                status = "PASS"
                status_tr = "GEÇERLİ"
            else:
                status = "FAIL"
                status_tr = "YETERSİZ"
            
            return {
                "analysis_date": datetime.now().isoformat(),
                "file_info": {
                    "filename": os.path.basename(pdf_path),
                    "text_length": len(text),
                    "detected_language": detected_language
                },
                "extracted_values": extracted_values,
                "category_analyses": category_analyses,
                "scoring": scoring,
                "recommendations": recommendations,
                "summary": {
                    "total_score": scoring["total_score"],
                    "percentage": percentage,
                    "status": status,
                    "status_tr": status_tr,
                    "report_type": "Kullanım Kılavuzu"
                }
            }
            
        except Exception as e:
            logger.error(f"Analiz hatası: {e}")
            return {
                "error": f"Analiz sırasında hata oluştu: {str(e)}",
                "analysis_date": datetime.now().isoformat()
            }

def print_analysis_report(report: Dict[str, Any]):
    """Analiz raporunu yazdır"""
    if "error" in report:
        print(f"❌ Hata: {report['error']}")
        return
    
    print("\n📊 KULLANIM KILAVUZU ANALİZ RAPORU")
    print("=" * 60)
    
    print(f"📅 Analiz Tarihi: {report['analysis_date']}")
    print(f"📁 Dosya: {report['file_info']['filename']}")
    print(f"🔍 Tespit Edilen Dil: {report['file_info']['detected_language'].upper()}")
    print(f"📄 Metin Uzunluğu: {report['file_info']['text_length']} karakter")
    
    print(f"\n📋 Toplam Puan: {report['summary']['total_score']}/100")
    print(f"📈 Yüzde: %{report['summary']['percentage']:.0f}")
    print(f"🎯 Durum: {report['summary']['status_tr']}")
    
    print("\n📋 ÇIKARILAN DEĞERLER")
    print("-" * 40)
    extracted_values = report['extracted_values']
    display_names = {
        "manual_name": "Kılavuz Adı",
        "product_model": "Ürün Modeli",
        "revision_info": "Revizyon Bilgisi",
        "manufacturer": "Üretici",
        "contact_info": "İletişim Bilgileri",
        "safety_warnings_count": "Güvenlik Uyarısı Sayısı"
    }
    
    for key, value in extracted_values.items():
        if key in display_names:
            print(f"{display_names[key]}: {value}")
    
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
        print("📝 Değerlendirme: Kullanım kılavuzu gerekli kriterleri sağlamaktadır.")
        
    else:
        print("❌ SONUÇ: YETERSİZ")
        print(f"⚠️ Toplam Başarı: %{report['summary']['percentage']:.0f}")
        print("📝 Değerlendirme: Kullanım kılavuzu minimum gereksinimleri karşılamıyor.")

def main():
    """Ana fonksiyon"""
    import sys
    
    default_pdf_path = "/Users/enesaktas/Documents/GitHub/pilzreportchecker/kullanım klavuzu raporlar/INSTALLATION AND SAFETY MANUAL-TH-06-15-1041.pdf"
    
    if len(sys.argv) == 1:

        pdf_path = default_pdf_path
        print(f"Varsayılan dosya kullanılıyor: {os.path.basename(pdf_path)}")
    elif len(sys.argv) == 2:
        pdf_path = sys.argv[1]
    else:
        print("Kullanım: python montajtalimatları_report_checker.py [pdf_dosyasi]")
        print(f"Varsayılan dosya: {os.path.basename(default_pdf_path)}")
        sys.exit(1)
    
    if not os.path.exists(pdf_path):
        print(f"❌ Dosya bulunamadı: {pdf_path}")
        if pdf_path != default_pdf_path:
            print(f"Varsayılan dosya deneniyor: {default_pdf_path}")
            if os.path.exists(default_pdf_path):
                pdf_path = default_pdf_path
            else:
                sys.exit(1)
        else:
            sys.exit(1)
    
    analyzer = ManualReportAnalyzer()
    report = analyzer.analyze_manual(pdf_path)
    print_analysis_report(report)

if __name__ == "__main__":
    main()
