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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NoiseCriteria:
    rapor_kimlik_bilgileri: Dict[str, Any]
    olcum_ortam_ekipman: Dict[str, Any]
    olcum_cihazi_bilgileri: Dict[str, Any]
    olcum_metodolojisi: Dict[str, Any]
    olcum_sonuclari: Dict[str, Any]
    degerlendirme_yorum: Dict[str, Any]
    ekler_gorseller: Dict[str, Any]

@dataclass
class NoiseAnalysisResult:
    criteria_name: str
    found: bool
    content: str
    score: int
    max_score: int
    details: Dict[str, Any]

class NoiseReportAnalyzer:
    
    def __init__(self):
        self.criteria_weights = {
            "Rapor Kimlik Bilgileri": 15,
            "Ölçüm Yapılan Ortam ve Ekipman Bilgileri": 15,
            "Ölçüm Cihazı Bilgileri": 15,
            "Ölçüm Metodolojisi": 20,
            "Ölçüm Sonuçları": 20,
            "Değerlendirme ve Yorum": 10,
            "Ekler ve Görseller": 5
        }
        
        self.criteria_details = {
            "Rapor Kimlik Bilgileri": {
                "rapor_numarasi": {"pattern": r"(?:Rapor\s*No\s*[:=]\s*|Belge\s*Numarası\s*[:=]\s*|C\d{2}\.\d{3})", "weight": 3},
                "rapor_tarihi": {"pattern": r"(?:Rapor\s*)?Tarihi?\s*[:=]\s*(\d{2}[./]\d{2}[./]\d{4})", "weight": 2},
                "olcum_tarihi": {"pattern": r"(?:Ölçüm\s*Tarihi|İnceleme\s*Tarihi)\s*[:=]?\s*(\d{2}[./]\d{2}[./]\d{4})", "weight": 3},
                "hazirlayan_kurulus": {"pattern": r"(?:Pilz\s*Servisleri|Pilz\s*Emniyet|Hazırlayan)", "weight": 2},
                "olcum_yapan_uzman": {"pattern": r"(?:Yapan\s*[:=]\s*|Kaan\s*Karabağ|Savaş\s*Şahan)", "weight": 3},
                "uzman_imza": {"pattern": r"(?:İmza|Yetkilisi)", "weight": 2}
            },
            "Ölçüm Yapılan Ortam ve Ekipman Bilgileri": {
                "firma_adi": {"pattern": r"(?:FORD\s*OTOSAN|Ford|Otosan)", "weight": 3},
                "firma_adresi": {"pattern": r"(?:Denizevler\s*Mah|Gölcük/Kocaeli|Ali\s*Uçar\s*Cad)", "weight": 2},
                "ortam_tanimi": {"pattern": r"(?:Otomatik\s*Robotlu\s*Kaynak|Kaynak\s*Hattı|fabrika|atölye)", "weight": 3},
                "makine_adi": {"pattern": r"(?:8X9J\s*Otomatik|Robotlu\s*Kaynak|Kaynak\s*Hattı)", "weight": 3},
                "makine_konumu": {"pattern": r"(?:8X\d{2}\s*LH|8X\d{2}\s*RH|BÖLGESİ)", "weight": 2},
                "cevresel_kosullar": {"pattern": r"(?:Sıcaklık|Nem|Rüzgar|kapalı\s*ortam)", "weight": 2}
            },
            "Ölçüm Cihazı Bilgileri": {
                "cihaz_marka": {"pattern": r"(?:PCE\s*Gürültü|PCE)", "weight": 3},
                "cihaz_model": {"pattern": r"(?:PCE-322A|322A)", "weight": 3},
                "seri_numarasi": {"pattern": r"(?:Seri\s*Numarası\s*[:=]\s*|180914367)", "weight": 3},
                "kalibrasyon_tarihi": {"pattern": r"(?:Kalibrasyon\s*Tarihi\s*[:=]\s*|4\.10\.2020)", "weight": 3},
                "mikrofon_bilgileri": {"pattern": r"(?:mikrofon|aksesuar)", "weight": 2},
                "cihaz_ayarlari": {"pattern": r"(?:Hızlı|Yavaş|Sample\s*Rate|50ms|100ms)", "weight": 1}
            },
            "Ölçüm Metodolojisi": {
                "uygulanan_standart": {"pattern": r"(?:ISO\s*11201|ISO\s*9612|ISO\s*3744|EN\s*ISO\s*4871|EN\s*ISO\s*11200)", "weight": 5},
                "olcum_turu": {"pattern": r"(?:emission\s*sound\s*pressure|Time-averaged|LpA|LpC)", "weight": 3},
                "olcum_yukseklik": {"pattern": r"(?:yükseklik|height)", "weight": 2},
                "olcum_noktalari": {"pattern": r"(?:8X\d{2}\s*LH|8X\d{2}\s*RH|Ölçüm\s*Noktası)", "weight": 5},
                "olcum_suresi": {"pattern": r"(?:1\s*dakika|Ölçüm\s*Süresi)", "weight": 3},
                "arka_plan_gurultu": {"pattern": r"(?:arka\s*plan|background)", "weight": 2}
            },
            "Ölçüm Sonuçları": {
                "ses_basinc_seviyesi": {"pattern": r"(?:LpA\s*\(dBA\)|LpA\s*\(dBC\)|dB\(A\)|dB\(C\))", "weight": 5},
                "laeeq_degeri": {"pattern": r"(?:LAeq|L\s*peqT|Time-averaged)", "weight": 4},
                "lmax_lmin": {"pattern": r"(?:En\s*düşük\s*Değer|En\s*yüksek\s*Değer|Lmax|Lmin)", "weight": 3},
                "lcpeak_degeri": {"pattern": r"(?:LCpeak|LpC\s*peak|Peak\s*sound)", "weight": 3},
                "nokta_degerleri": {"pattern": r"(?:7[0-9],\d|9[0-9],\d)", "weight": 3},
                "maruziyet_suresi": {"pattern": r"(?:T\s*=|çalışma\s*süresi|8\s*saat)", "weight": 2}
            },
            "Değerlendirme ve Yorum": {
                "yasal_sinirlar": {"pattern": r"(?:85\s*dB|87\s*dB|yasal\s*sınır)", "weight": 3},
                "risk_degerlendirme": {"pattern": r"(?:risk\s*değerlendirme|maruziyet\s*risk)", "weight": 2},
                "onlemler": {"pattern": r"(?:kulaklık|izolasyon|perdeleme|önlem)", "weight": 3},
                "lex_8h": {"pattern": r"(?:LEX,8h|günlük\s*gürültü|8\s*saatlik)", "weight": 2}
            },
            "Ekler ve Görseller": {
                "ortam_krokisi": {"pattern": r"(?:kroki|çizim|plan)", "weight": 2},
                "fotograflar": {"pattern": r"(?:fotoğraf|görsel|resim)", "weight": 2},
                "kalibrasyon_sertifika": {"pattern": r"(?:kalibrasyon\s*sertifika|sertifika)", "weight": 1}
            }
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
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
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        try:
            doc = Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"DOCX okuma hatası: {e}")
            return ""
    
    def check_report_date_validity(self, text: str) -> Tuple[bool, str, str]:
        date_patterns = [
            r"(?:Ölçüm\s*Tarihi|İnceleme)\s*[:=]?\s*(\d{2}[./]\d{2}[./]\d{4})",
            r"(\d{2}[./]\d{2}[./]\d{4})"
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                date_str = matches[0]
                try:
                    date_str = date_str.replace('.', '/').replace('-', '/')
                    report_date = datetime.strptime(date_str, '%d/%m/%Y')
                    one_year_ago = datetime.now() - timedelta(days=365)
                    
                    is_valid = report_date >= one_year_ago
                    return is_valid, date_str, f"Rapor tarihi: {date_str} {'(GEÇERLİ)' if is_valid else '(GEÇERSİZ - 1 yıldan eski)'}"
                except ValueError:
                    continue
        
        return False, "", "Rapor tarihi bulunamadı"
    
    def analyze_criteria(self, text: str, category: str) -> Dict[str, NoiseAnalysisResult]:
        results = {}
        criteria = self.criteria_details.get(category, {})
        
        for criterion_name, criterion_data in criteria.items():
            pattern = criterion_data["pattern"]
            weight = criterion_data["weight"]
            
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                content = str(matches[0]) if len(matches) == 1 else str(matches)
                found = True
                score = weight
            else:
                general_patterns = {
                    "rapor_numarasi": r"(C\d+\.\d+|Rapor|Belge)",
                    "firma_adi": r"(Ford|Otosan)",
                    "cihaz_marka": r"(PCE|Gürültü)",
                    "ses_basinc_seviyesi": r"(dB|ses|gürültü)",
                    "yasal_sinirlar": r"(85|87|sınır|limit)"
                }
                
                general_pattern = general_patterns.get(criterion_name)
                if general_pattern:
                    general_matches = re.findall(general_pattern, text, re.IGNORECASE)
                    if general_matches:
                        content = f"Genel eşleşme bulundu: {general_matches[0]}"
                        found = True
                        score = weight // 2
                    else:
                        content = "Bulunamadı"
                        found = False
                        score = 0
                else:
                    content = "Bulunamadı"
                    found = False
                    score = 0
            
            results[criterion_name] = NoiseAnalysisResult(
                criteria_name=criterion_name,
                found=found,
                content=content,
                score=score,
                max_score=weight,
                details={"pattern_used": pattern, "matches_found": len(matches) if matches else 0}
            )
        
        return results
    
    def extract_specific_values(self, text: str) -> Dict[str, Any]:
        values = {}
        
        value_patterns = {
            "rapor_no": r"(?:Belge\s*Numarası\s*[:=]\s*|C\d{2}\.\d{3})",
            "olcum_tarihi": r"(?:İnceleme\s*Tarihi\s*[:=]?\s*)?(\d{2}[./]\d{2}[./]\d{4})",
            "firma_adi": r"(?:FORD\s*OTOSAN|Ford)",
            "makine_adi": r"(?:8X9J\s*Otomatik|Robotlu\s*Kaynak|Kaynak\s*Hattı)",
            "cihaz_marka": r"(?:PCE\s*Gürültü|PCE)",
            "cihaz_model": r"(?:PCE-322A|322A)",
            "seri_no": r"(?:Seri\s*Numarası\s*[:=]\s*|180914367)",
            "kalibrasyon_tarihi": r"(?:Kalibrasyon\s*Tarihi\s*[:=]\s*|4\.10\.2020)",
            "olcum_yapan": r"(?:Yapan\s*[:=]\s*|Kaan\s*Karabağ)",
            "yetkili": r"(?:Yetkilisi\s*[:=]\s*|Savaş\s*Şahan)",
            "min_deger": r"(?:En\s*düşük\s*Değer.*?(\d{2},\d))",
            "max_deger": r"(?:En\s*yüksek\s*Değer.*?(\d{2},\d))",
            "lpa_deger": r"(?:LpA\s*\(dBA\)\s*(\d{2},\d))",
            "lpc_deger": r"(?:LpA\s*\(dBC\)\s*(\d{2},\d))",
            "standart": r"(?:EN\s*ISO\s*4871|EN\s*ISO\s*11200|ISO\s*11201)",
            "olcum_suresi": r"(?:1\s*dakika|Ölçüm\s*Süresi)",
            "nokta_sayisi": r"(?:8X\d{2}.*?BÖLGESİ)",
            "cihaz_ayari": r"(?:Hızlı|Yavaş|Sample\s*Rate)",
            "proje_adi": r"(?:Proje\s*Adı\s*[:=]\s*|1\s*Adet\s*8X9J)"
        }
        
        for key, pattern in value_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                values[key] = matches[0].strip()
            else:
                values[key] = "Bulunamadı"
        
        return values
    
    def calculate_scores(self, analysis_results: Dict[str, Dict[str, NoiseAnalysisResult]]) -> Dict[str, Any]:
        category_scores = {}
        total_score = 0
        total_max_score = 100
        
        for category, results in analysis_results.items():
            category_max = self.criteria_weights[category]
            category_earned = sum(result.score for result in results.values())
            category_possible = sum(result.max_score for result in results.values())
            
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
    
    def generate_detailed_report(self, pdf_path: str, docx_path: str = None) -> Dict[str, Any]:
        logger.info("Gürültü ölçüm raporu analizi başlatılıyor...")
        
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return {"error": "PDF okunamadı"}
        
        date_valid, date_str, date_message = self.check_report_date_validity(pdf_text)
        extracted_values = self.extract_specific_values(pdf_text)
        
        analysis_results = {}
        for category in self.criteria_weights.keys():
            analysis_results[category] = self.analyze_criteria(pdf_text, category)
        
        scores = self.calculate_scores(analysis_results)
        recommendations = self.generate_recommendations(analysis_results, scores)
        
        report = {
            "analiz_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dosya_bilgileri": {
                "pdf_path": pdf_path,
                "docx_path": docx_path
            },
            "tarih_gecerliligi": {
                "gecerli": date_valid,
                "tarih": date_str,
                "mesaj": date_message
            },
            "cikarilan_degerler": extracted_values,
            "kategori_analizleri": analysis_results,
            "puanlama": scores,
            "oneriler": recommendations,
            "ozet": {
                "toplam_puan": scores["total_score"],
                "yuzde": scores["overall_percentage"],
                "durum": "GEÇERLİ" if scores["overall_percentage"] >= 70 else "YETERSİZ",
                "tarih_durumu": "GEÇERLİ" if date_valid else "GEÇERSİZ"
            }
        }
        
        return report
    
    def generate_recommendations(self, analysis_results: Dict, scores: Dict) -> List[str]:
        recommendations = []
        
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
            recommendations.append("- Rapor ISO 11201, ISO 9612 standartlarına tam uyumlu hale getirilmelidir")
            recommendations.append("- Eksik ölçüm bilgileri tamamlanmalıdır")
            recommendations.append("- Yasal sınırlarla karşılaştırma yapılmalıdır")
            recommendations.append("- Günlük gürültü maruziyet düzeyi (LEX,8h) hesaplanmalıdır")
        
        return recommendations
    
    def save_report_to_excel(self, report: Dict, output_path: str):
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            ozet_data = {
                'Kriter': ['Toplam Puan', 'Yüzde', 'Durum', 'Tarih Durumu'],
                'Değer': [
                    report['ozet']['toplam_puan'],
                    f"%{report['ozet']['yuzde']}",
                    report['ozet']['durum'],
                    report['ozet']['tarih_durumu']
                ]
            }
            pd.DataFrame(ozet_data).to_excel(writer, sheet_name='Özet', index=False)
            
            values_data = []
            for key, value in report['cikarilan_degerler'].items():
                values_data.append({'Kriter': key, 'Değer': value})
            pd.DataFrame(values_data).to_excel(writer, sheet_name='Çıkarılan Değerler', index=False)
            
            for category, results in report['kategori_analizleri'].items():
                category_data = []
                for criterion, result in results.items():
                    category_data.append({
                        'Kriter': criterion,
                        'Bulundu': result.found,
                        'İçerik': result.content,
                        'Puan': result.score,
                        'Max Puan': result.max_score
                    })
                
                sheet_name = category[:31]
                pd.DataFrame(category_data).to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Rapor Excel dosyası kaydedildi: {output_path}")
    
    def save_report_to_json(self, report: Dict, output_path: str):
        json_report = {}
        for key, value in report.items():
            if key == 'kategori_analizleri':
                json_report[key] = {}
                for category, results in value.items():
                    json_report[key][category] = {}
                    for criterion, result in results.items():
                        json_report[key][category][criterion] = asdict(result)
            else:
                json_report[key] = value
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Rapor JSON dosyası kaydedildi: {output_path}")

def main():
    analyzer = NoiseReportAnalyzer()
    
    pdf_path = "C19.082 P&G - Cobot Paletleme- Gürültü Ölçüm Raporu V0.pdf"
    docx_path = "Gürültü Ölçüm Raporu_Kriterleri_Puanlama.docx"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF dosyası bulunamadı: {pdf_path}")
        return
    
    print("🔍 Gürültü Ölçüm Raporu Analizi Başlatılıyor...")
    print("=" * 60)
    
    report = analyzer.generate_detailed_report(pdf_path, docx_path)
    
    if "error" in report:
        print(f"❌ Hata: {report['error']}")
        return
    
    print("\n📊 ANALİZ SONUÇLARI")
    print("=" * 60)
    
    print(f"📅 Analiz Tarihi: {report['analiz_tarihi']}")
    print(f"📋 Toplam Puan: {report['ozet']['toplam_puan']}/100")
    print(f"📈 Yüzde: %{report['ozet']['yuzde']}")
    print(f"🎯 Durum: {report['ozet']['durum']}")
    print(f"📆 Tarih Durumu: {report['ozet']['tarih_durumu']}")
    
    print(f"\n⚠️ Tarih Kontrolü: {report['tarih_gecerliligi']['mesaj']}")
    
    print("\n📋 ÖNEMLİ ÇIKARILAN DEĞERLER")
    print("-" * 40)
    important_values = ['rapor_no', 'olcum_tarihi', 'firma_adi', 'makine_adi', 'cihaz_marka', 
                       'cihaz_model', 'seri_no', 'kalibrasyon_tarihi', 'olcum_yapan', 'yetkili']
    
    for key in important_values:
        if key in report['cikarilan_degerler']:
            print(f"{key.replace('_', ' ').title()}: {report['cikarilan_degerler'][key]}")
    
    print("\n📊 KATEGORİ PUANLARI")
    print("-" * 40)
    for category, score_data in report['puanlama']['category_scores'].items():
        print(f"{category}: {score_data['normalized']}/{score_data['max_weight']} (%{score_data['percentage']:.1f})")
    
    print("\n💡 ÖNERİLER")
    print("-" * 40)
    for recommendation in report['oneriler']:
        print(recommendation)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f"Gurultu_Analiz_Raporu_{timestamp}.xlsx"
    json_path = f"Gurultu_Analiz_Raporu_{timestamp}.json"
    
    analyzer.save_report_to_excel(report, excel_path)
    analyzer.save_report_to_json(report, json_path)
    
    print(f"\n💾 Raporlar kaydedildi:")
    print(f"   📊 Excel: {excel_path}")
    print(f"   📄 JSON: {json_path}")

if __name__ == "__main__":
    main()