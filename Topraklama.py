import re
import os
import json
from datetime import datetime
from typing import Dict, Any, List
import PyPDF2
from docx import Document
import logging
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from dataclasses import dataclass

# # ---------- OCR & Poppler ----------
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# poppler_path = r"C:\Users\nuvo_teknik_2\Desktop\poppler-24.08.0\Library\bin"
# os.environ["PATH"] += os.pathsep + poppler_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TopraklamaAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

class EN60204TopraklamaRaporuAnaliz:
    def __init__(self):
        logger.info("EN 60204-1 Topraklama Ölçüm Raporu Analiz Sistemi başlatılıyor...")

        self.criteria_weights = {
            "Şebeke Tipi (TN/TT/IT)": 15,
            "Ölçüm Sonuçları (R veya Zs değerleri)": 20,
            "Zs Hesabı ve Karşılaştırması (TN için)": 20,
            "Koruma Cihazı Bilgileri (MCB/RCD)": 10,
            "Ölçüm Metodu ve Koşulları": 8,
            "Ölçüm Cihazı ve Kalibrasyon": 7,
            "Yetkili Kişi / İmza-Kaşe": 2,
            "Tesis Bilgileri ve Ölçüm Tarihi": 3,
            "Sonuç ve Yorum": 15
        }

        self.criteria_details = {
            "Şebeke Tipi (TN/TT/IT)": {
                "sebeke_tipi": {"pattern": r"(TN-S|TN-C|TN-C-S|TT|IT)\s*sistem|Şebeke\s*Tipi.*?(TN|TT|IT)", "weight": 15}
            },
            "Ölçüm Sonuçları (R veya Zs değerleri)": {
                "topraklama_direnci": {"pattern": r"([0-9,.]+)\s*Ω|ohm|([0-9,.]+)\s*Ohm", "weight": 10},
                "zs_degeri": {"pattern": r"Zs\s*[:=]\s*([0-9,.]+)\s*Ω|Z_s\s*[:=]\s*([0-9,.]+)", "weight": 10}
            },
            "Zs Hesabı ve Karşılaştırması (TN için)": {
                "zs_karsilastirma": {"pattern": r"Zs\s*≤|izin\s*verilen\s*Zs|maksimum\s*Zs|Zs\s*karşılaştırma", "weight": 20}
            },
            "Koruma Cihazı Bilgileri (MCB/RCD)": {
                "koruma_cihazi": {"pattern": r"(MCB|RCD|sigorta|breaker|açma\s*cihazı|B\s*tip|C\s*tip|D\s*tip)", "weight": 10}
            },
            "Ölçüm Metodu ve Koşulları": {
                "olcum_metodu": {"pattern": r"(3\s*nokta|4\s*nokta|klemp|Wenner|ölçüm\s*yöntemi|metod)", "weight": 5},
                "olcum_kosullari": {"pattern": r"(sıcaklık|nem|zemin|hava|koşul|ortam)", "weight": 3}
            },
            "Ölçüm Cihazı ve Kalibrasyon": {
                "cihaz_bilgisi": {"pattern": r"(marka|model|seri\s*no|cihaz)", "weight": 3},
                "kalibrasyon": {"pattern": r"kalibrasyon|calibration|EN\s*60204-1.*18\.2\.1", "weight": 4}
            },
            "Yetkili Kişi / İmza-Kaşe": {
                "yetkili_kisi": {"pattern": r"(elektrik\s*mühendisi|uzman|yetkili|imza|kaşe)", "weight": 2}
            },
            "Tesis Bilgileri ve Ölçüm Tarihi": {
                "tesis_bilgisi": {"pattern": r"(tesis|işletme|sahne|lokasyon|yer)", "weight": 2},
                "olcum_tarihi": {"pattern": r"(\d{1,2}[./]\d{1,2}[./]\d{2,4})|tarih|date", "weight": 1}
            },
            "Sonuç ve Yorum": {
                "sonuc_yorum": {"pattern": r"(uygun|uygun\s*değil|geçerli|geçersiz|sonuç|yorum|kabul)", "weight": 15}
            }
        }

    def extract_text_with_ocr(self, file_path: str) -> str:
        """OCR ile metin çıkarımı (PDF veya görsel)"""
        try:
            images = convert_from_path(file_path, dpi=300)
            text = ""
            for i, image in enumerate(images):
                logger.info(f"OCR işleniyor: Sayfa {i+1}")
                text += pytesseract.image_to_string(image, lang='tur+eng')
            return text
        except Exception as e:
            logger.error(f"OCR hatası: {e}")
            return ""

    def extract_text_from_file(self, file_path: str) -> str:
        """Dosya türüne göre metin çıkarımı (OCR dahil)"""
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.pdf':
            # Önce normal PDF metin çıkarımı dene
            text = self.extract_text_from_pdf(file_path)
            if not text.strip():
                logger.info("PDF'de metin bulunamadı, OCR ile devam ediliyor...")
                text = self.extract_text_with_ocr(file_path)
            return text
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            return pytesseract.image_to_string(Image.open(file_path), lang='tur+eng')
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            logger.error(f"Desteklenmeyen dosya formatı: {file_ext}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkarımı"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        except Exception as e:
            logger.error(f"PDF metin çıkarma hatası: {e}")
            return ""

    def extract_text_from_docx(self, docx_path: str) -> str:
        """DOCX'den metin çıkarımı"""
        try:
            doc = Document(docx_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            logger.error(f"DOCX hatası: {e}")
            return ""

    def extract_text_from_txt(self, txt_path: str) -> str:
        """TXT'den metin çıkarımı"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""

    def analyze_criteria(self, text: str, category: str) -> Dict[str, TopraklamaAnalysisResult]:
        """Kriterleri analiz et"""
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)

            if matches:
                content = f"Bulunan: {matches[:3]}"
                found = True
                score = min(weight, len(matches) * (weight // 2)) if weight > 2 else min(weight, len(matches))
            else:
                content = "Bulunamadı"
                found = False
                score = 0

            results[criterion_name] = TopraklamaAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_count": len(matches) if matches else 0}
            )
        return results

    def calculate_scores(self, analysis_results: Dict[str, Dict[str, TopraklamaAnalysisResult]]) -> Dict[str, Any]:
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
            "sebeke_tipi": "Bulunamadı",
            "olcum_tarihi": "Bulunamadı",
            "topraklama_direnci": "Bulunamadı",
            "zs_degeri": "Bulunamadı",
            "olcum_cihazi": "Bulunamadı",
            "kalibrasyon_tarihi": "Bulunamadı",
            "sonuc": "Bulunamadı"
        }

        # Şebeke tipi
        match = re.search(r"(TN-S|TN-C|TN-C-S|TT|IT)\s*sistem|Şebeke\s*Tipi.*?(TN|TT|IT)", text, re.IGNORECASE)
        if match:
            values["sebeke_tipi"] = match.group(1).upper() if match.group(1) else match.group(2).upper()

        # Tarih
        match = re.search(r"(\d{1,2}[./]\d{1,2}[./]\d{2,4})", text)
        if match:
            values["olcum_tarihi"] = match.group(1)

        # Topraklama direnci
        match = re.search(r"([0-9,.]+)\s*Ω", text)
        if match:
            values["topraklama_direnci"] = match.group(1) + " Ω"

        # Zs
        match = re.search(r"Zs\s*[:=]\s*([0-9,.]+)\s*Ω", text, re.IGNORECASE)
        if match:
            values["zs_degeri"] = match.group(1) + " Ω"

        # Cihaz
        match = re.search(r"(Marka|Model)\s*[:=]\s*([A-Za-z0-9\- ]+)", text, re.IGNORECASE)
        if match:
            values["olcum_cihazi"] = match.group(2).strip()

        # Kalibrasyon
        match = re.search(r"Kalibrasyon.*?(\d{1,2}[./]\d{1,2}[./]\d{2,4})", text, re.IGNORECASE)
        if match:
            values["kalibrasyon_tarihi"] = match.group(1)

        # Sonuç
        if re.search(r"uygun\s*değil|geçersiz|kabul\s*edilmez", text, re.IGNORECASE):
            values["sonuc"] = "Uygun Değil"
        elif re.search(r"uygun|geçerli|kabul\s*edilebilir", text, re.IGNORECASE):
            values["sonuc"] = "Uygun"

        return values

    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        total_percentage = scores["percentage"]

        if total_percentage >= 70:
            recommendations.append(f"✅ EN 60204-1 Topraklama Raporu GEÇERLİ (Toplam: %{total_percentage:.1f})")
        else:
            recommendations.append(f"❌ EN 60204-1 Topraklama Raporu GEÇERSİZ (Toplam: %{total_percentage:.1f})")

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

    def generate_improvement_actions(self, analysis_results: Dict, scores: Dict) -> List[str]:
        """İyileştirme önerileri oluştur"""
        actions = []
        sorted_categories = sorted(scores["category_scores"].items(), key=lambda x: x[1]["percentage"])

        category_actions = {
            "Şebeke Tipi (TN/TT/IT)": [
                "Şebeke tipini (TN-S, TN-C, TN-C-S, TT, IT) mutlaka belirtiniz",
                "Sistem tipine göre ölçüm kriterlerini açıklayınız"
            ],
            "Ölçüm Sonuçları (R veya Zs değerleri)": [
                "Tüm ölçüm noktaları için ayrı ayrı değerler veriniz",
                "Topraklama direnci ve Zs değerlerini açıkça yazınız"
            ],
            "Zs Hesabı ve Karşılaştırması (TN için)": [
                "TN sistemlerde Zs hesabını yapınız",
                "İzin verilen maksimum Zs değeri ile karşılaştırınız"
            ],
            "Koruma Cihazı Bilgileri (MCB/RCD)": [
                "Kullanılan koruma cihazlarının tip ve değerlerini belirtiniz",
                "Açma karakteristiğini (B, C, D tipi) yazınız"
            ],
            "Ölçüm Metodu ve Koşulları": [
                "Ölçüm yöntemini (3 nokta, 4 nokta, klemp) açıklayınız",
                "Ölçüm koşullarını (sıcaklık, nem, zemin) belirtiniz"
            ],
            "Ölçüm Cihazı ve Kalibrasyon": [
                "Cihazın marka, model, seri numarasını yazınız",
                "Kalibrasyon belgesi ve tarihini ekleyiniz"
            ],
            "Yetkili Kişi / İmza-Kaşe": [
                "Ölçümü yapan yetkili kişinin bilgilerini ve imzasını ekleyiniz",
                "Kaşe ile resmiyet sağlayınız"
            ],
            "Tesis Bilgileri ve Ölçüm Tarihi": [
                "Tesis adı ve lokasyon bilgilerini belirtiniz",
                "Ölçüm tarihini açıkça yazınız"
            ],
            "Sonuç ve Yorum": [
                "Ölçüm sonuçlarının uygunluk durumunu değerlendiriniz",
                "Gerekli düzeltici faaliyetleri öneriniz"
            ]
        }

        for category, score_data in sorted_categories[:5]:
            if score_data["percentage"] < 70:
                actions.extend(category_actions.get(category, []))

        if scores["percentage"] < 50:
            actions.insert(0, "ÖNCE: Rapor formatını EN 60204-1 standardına uygun olarak yeniden düzenleyiniz")

        return actions

    def validate_document(self, text: str) -> bool:
        """Doküman validasyonu"""
        keywords = ["topraklama", "toprak", "earth", "ground", "EN 60204", "ölçüm", "measurement", "Zs", "TN", "TT", "IT"]
        found = sum(1 for kw in keywords if re.search(kw, text, re.IGNORECASE))
        return found >= 4

    def analyze_report(self, file_path: str) -> Dict[str, Any]:
        """Ana analiz fonksiyonu"""
        logger.info("EN 60204-1 Topraklama Ölçüm Raporu analizi başlatılıyor...")

        if not os.path.exists(file_path):
            return {"error": f"Dosya bulunamadı: {file_path}"}

        text = self.extract_text_from_file(file_path)
        if not text.strip():
            return {"error": "Dosyadan metin çıkarılamadı (OCR dahil)"}

        if not self.validate_document(text):
            return {
                "error": "YANLIŞ DOKÜMAN: Bu dosya EN 60204-1 topraklama ölçüm raporu değil!",
                "document_type": "UNKNOWN",
                "suggestion": "Lütfen geçerli bir topraklama ölçüm raporu yükleyiniz."
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
                "rapor_tipi": "EN60204_TOPRAKLAMA_OLCUM_RAPORU"
            }
        }

# Test için ana fonksiyon
def main():
    analyzer = EN60204TopraklamaRaporuAnaliz()
    
    # Test dosya yolu
    file_path = r"C:\Users\nuvo_teknik_2\Desktop\pilzkriterler\LA LORRAINE-TOPRAKLAMA RAPORU.PDF"
    
    if not os.path.exists(file_path):
        print(f"❌ Dosya bulunamadı: {file_path}")
        return
    
    print("🔌 EN 60204-1 Topraklama Ölçüm Raporu Analizi Başlatılıyor...")
    print("=" * 60)
    
    report = analyzer.analyze_report(file_path)
    
    if "error" in report:
        print(f"❌ Hata: {report['error']}")
        return
    
    print("\n📊 ANALİZ SONUÇLARI")
    print("=" * 60)
    
    print(f"📅 Analiz Tarihi: {report['analiz_tarihi']}")
    print(f"📄 Dosya Tipi: {report['dosya_bilgisi']['file_type'].upper()}")
    print(f"📋 Toplam Puan: {report['ozet']['toplam_puan']}/100")
    print(f"📈 Yüzde: %{report['ozet']['yuzde']}")
    print(f"🎯 Durum: {report['ozet']['durum']}")
    print(f"📄 Rapor Tipi: {report['ozet']['rapor_tipi']}")
    
    print("\n⚡ ÖNEMLİ ÇIKARILAN DEĞERLER")
    print("-" * 40)
    for key, value in report['cikarilan_degerler'].items():
        display_name = {
            "sebeke_tipi": "Şebeke Tipi",
            "olcum_tarihi": "Ölçüm Tarihi",
            "topraklama_direnci": "Topraklama Direnci",
            "zs_degeri": "Zs Değeri",
            "olcum_cihazi": "Ölçüm Cihazı",
            "kalibrasyon_tarihi": "Kalibrasyon Tarihi",
            "sonuc": "Sonuç"
        }.get(key, key.replace('_', ' ').title())
        print(f"{display_name}: {value}")
    
    print("\n📊 KATEGORİ PUANLARI")
    print("-" * 40)
    for category, score_data in report['puanlama']['category_scores'].items():
        print(f"{category}: {score_data['normalized']}/{score_data['max_weight']} (%{score_data['percentage']:.1f})")
    
    print("\n💡 ÖNERİLER VE DEĞERLENDİRME")
    print("-" * 40)
    for recommendation in report['oneriler']:
        print(recommendation)
    
    print("\n📋 GENEL DEĞERLENDİRME")
    print("=" * 60)
    
    if report['ozet']['yuzde'] >= 70:
        print("✅ SONUÇ: GEÇERLİ")
        print(f"🌟 Toplam Başarı: %{report['ozet']['yuzde']:.1f}")
        print("📝 Değerlendirme: Rapor EN 60204-1 standardına uygun olarak hazırlanmıştır.")
    else:
        print("❌ SONUÇ: GEÇERSİZ")
        print(f"⚠️ Toplam Başarı: %{report['ozet']['yuzde']:.1f}")
        print("📝 Değerlendirme: Rapor minimum gereklilikleri sağlamamaktadır.")
        
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
                        "sebeke_tipi": "Şebeke Tipi",
                        "topraklama_direnci": "Topraklama Direnci",
                        "zs_degeri": "Zs Değeri",
                        "zs_karsilastirma": "Zs Karşılaştırması",
                        "koruma_cihazi": "Koruma Cihazı Bilgileri",
                        "olcum_metodu": "Ölçüm Metodu",
                        "olcum_kosullari": "Ölçüm Koşulları",
                        "cihaz_bilgisi": "Cihaz Bilgisi",
                        "kalibrasyon": "Kalibrasyon",
                        "yetkili_kisi": "Yetkili Kişi",
                        "tesis_bilgisi": "Tesis Bilgisi",
                        "olcum_tarihi": "Ölçüm Tarihi",
                        "sonuc_yorum": "Sonuç ve Yorum"
                    }.get(item, item.replace('_', ' ').title())
                    print(f"   ❌ {readable_name}")
        
        print("\n📌 YAPILMASI GEREKENLER:")
        if "iyilestirme_eylemleri" in report:
            for i, action in enumerate(report['iyilestirme_eylemleri'], 1):
                print(f"{i}. {action}")

if __name__ == "__main__":
    main()