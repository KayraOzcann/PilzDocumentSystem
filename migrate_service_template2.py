"""
SERVİS VERİLERİNİ VERITABANINA AKTARMA TEMPLATE

KULLANIM:
1. "# TODO:" işaretli yerleri doldurun
2. Olmayan kısımlar için dictionary'leri boş bırakın: {}
3. python migrate_service_template.py çalıştırın
4. test_service_migration.py ile test edin
"""

from flask import Flask
from database import db, init_db
from models import DocumentType, CriteriaWeight, CriteriaDetail, PatternDefinition, ValidationKeyword, CategoryAction

app = Flask(__name__)
init_db(app)

# ============================================
# TODO: SERVİS BİLGİLERİNİ DOLDURUN
# ============================================

# Servis bilgileri
DOCUMENT_TYPE_CODE = 'lighting_report'  # TODO: Değiştir (örn: 'electric_circuit', 'espe_report')
DOCUMENT_TYPE_NAME = 'Aydınlatma Raporu Analizi'  # TODO: Değiştir
DOCUMENT_TYPE_DESCRIPTION = 'Aydınlatma Raporlarının Analizi'  # TODO: Değiştir
SERVICE_FILE = 'aydinlatma_service.py'  # TODO: Değiştir
ENDPOINT = '/api/aydinlatma-report'  # TODO: Değiştir
ICON = '💡'  # TODO: Değiştir
# ============================================
# TODO: CRITERIA WEIGHTS (Kategoriler + Puanlar)
# ============================================
# Format: {"Kategori Adı": puan}
# Örnek: {"Rapor Kimlik Bilgileri": 10, "Genel Bilgiler": 15}
# Eğer yoksa: {}

criteria_weights_data = {
            "Genel Rapor Bilgileri": 10,
            "Ölçüm Metodu ve Standart Referansları": 15,
            "Ölçüm Sonuç Tablosu": 25,
            "Uygunluk Değerlendirmesi": 20,
            "Görsel ve Teknik Dokümantasyon": 5,
            "Ölçüm Cihazı Bilgileri": 10,
            "Sonuç ve Öneriler": 15
}

# ============================================
# TODO: CRITERIA DETAILS (Alt Kriterler + Patternler)
# ============================================
# Format: {
#     "Kategori Adı": {
#         "kriter_adi": {"pattern": r"regex_pattern", "weight": 2},
#     }
# }
# Eğer yoksa: {}

criteria_details_data = {
            "Genel Rapor Bilgileri": {
                "proje_adi_numarasi": {"pattern": r"(?:Proje\s*Ad[ıi]|Project\s*Name|Proje\s*No|Project\s*Number|Proje\s*Kodu|Project\s*Code|İş\s*No|Job\s*Number|Sipariş\s*No|Order\s*Number)", "weight": 2},
                "olcum_rapor_tarihleri": {"pattern": r"(?:Ölçüm\s*Tarih|Measurement\s*Date|Rapor\s*Tarih|Report\s*Date|Test\s*Tarih|Test\s*Date|Analiz\s*Tarih|Analysis\s*Date|Değerlendirme\s*Tarih|\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}|\d{2,4}[\/\.\-]\d{1,2}[\/\.\-]\d{1,2})", "weight": 2},
                "tesis_bolge_alan": {"pattern": r"(?:Tesis|Facility|Fabrika|Factory|Ofis|Office|Bina|Building|İş\s*yeri|Workplace|Alan|Area|Bölge|Zone|Konum|Location|Adres|Address|Mekân|Space)", "weight": 1},
                "rapor_no_revizyon": {"pattern": r"(?:Rapor\s*No|Report\s*No|Rapor\s*Numaras[ıi]|Report\s*Number|Rev|Revizyon|Revision|Ver|Version|Sürüm|Döküman\s*No|Document\s*No)", "weight": 2},
                "olcumu_yapan_firma": {"pattern": r"(?:Ölçümü\s*Yapan|Measured\s*By|Test\s*Yapan|Tested\s*By|Firma|Company|Şirket|Corporation|Kurum|Institution|Organizasyon|Organization|Muayene\s*Kuruluş)", "weight": 1},
                "onay_imza": {"pattern": r"(?:Onay|Approval|İmza|Signature|Onaylayan|Approved\s*By|İmzalayan|Signed\s*By|Sorumlu|Responsible|Yetkili|Authorized|Mühür|Stamp|Seal)", "weight": 2}
            },
            "Ölçüm Metodu ve Standart Referansları": {
                "olcum_cihazi": {"pattern": r"(?:Lüksmetre|Luxmeter|Lux\s*Meter|Işık\s*Ölçer|Light\s*Meter|Işıklılık\s*Ölçer|Luminance\s*Meter|Fotometre|Photometer|Cihaz\s*Marka|Device\s*Brand|Model|Seri\s*No|Serial\s*Number)", "weight": 4},
                "kalibrasyon_bilgi": {"pattern": r"(?:Kalibrasyon|Calibration|Sertifika|Certificate|Akreditasyon|Accreditation|Geçerlilik|Validity|Son\s*Kalibrasyon|Last\s*Calibration|Kalibre|Calibrated)", "weight": 3},
                "olcum_yontemi": {"pattern": r"(?:Ölçüm\s*Yöntem|Measurement\s*Method|Test\s*Yöntem|Test\s*Method|Prosedür|Procedure|Metodoloji|Methodology|Ortlama|Average|Dört\s*Nokta|Four\s*Point|Grid|Izgara)", "weight": 4},
                "standartlar": {"pattern": r"(?:TS\s*EN\s*12464|ISO\s*8995|İş\s*Sağlığ[ıi]|Work\s*Safety|Occupational\s*Health|Standart|Standard|Norm|Specification|Tüzük|Regulation|Yönetmelik|Directive)", "weight": 4}
            },
            "Ölçüm Sonuç Tablosu": {
                "tablo_yapisi": {"pattern": r"(?:Tablo|Table|Liste|List|Sıra\s*No|Row\s*No|S[ıi]ra|Order|Numara|Number|Index|Çizelge|Chart|Matris|Matrix)", "weight": 5},
                "calisma_alani": {"pattern": r"(?:Çalışma\s*Alan[ıi]|Work\s*Area|İş\s*Alan[ıi]|Work\s*Zone|Bölge\s*Ad[ıi]|Area\s*Name|Konum|Location|Nokta|Point|Pozisyon|Position)", "weight": 4},
                "olculen_degerler": {"pattern": r"(?:Lüks|Lux|lx|Aydınlatma\s*Şiddet|Illumination|Light\s*Level|Işık\s*Seviye|Light\s*Intensity|Ölçülen|Measured|Mevcut|Current|Actual)", "weight": 8},
                "hedeflenen_degerler": {"pattern": r"(?:Hedeflenen|Target|İstenen|Desired|Gerekli|Required|Minimum|Standart|Standard|Önerilen|Recommended|Limit|Thresh)", "weight": 4},
                "uygunluk_durumu": {"pattern": r"(?:Uygun|Suitable|Conform|Uygunsuz|Not\s*Suitable|Non[\\-\\s]*Conform|Geçerli|Valid|Geçersiz|Invalid|PASS|FAIL|OK|NOK)", "weight": 4}
            },
            "Uygunluk Değerlendirmesi": {
                "toplu_degerlendirme": {"pattern": r"(?:Genel\s*Değerlendirme|General\s*Assessment|Toplu\s*Değerlendirme|Overall\s*Evaluation|Özet|Summary|Sonuç|Result|Analiz|Analysis)", "weight": 5},
                "limit_disi_degerler": {"pattern": r"(?:Limit\s*Dış[ıi]|Out\s*of\s*Limit|Standart\s*Dış[ıi]|Non[\\-\\s]*Standard|Uygunsuz|Non[\\-\\s]*Compliant|Eksik|Insufficient|Fazla|Excessive|Aş[ıi]r[ıi]|Over)", "weight": 5},
                "risk_belirtme": {"pattern": r"(?:Risk|Tehlike|Hazard|Göz\s*Yorgunluk|Eye\s*Fatigue|Verimlilik|Productivity|Güvenlik|Safety|Dikkat\s*Dağ[ıi]n[ıi]kl[ıi]|Distraction)", "weight": 5},
                "duzeltici_faaliyet": {"pattern": r"(?:Düzeltici\s*Faaliyet|Corrective\s*Action|İyileştirme|Improvement|Öneri|Recommendation|Çözüm|Solution|Aksiyon|Action|Tedbir|Measure)", "weight": 5}
            },
            "Görsel ve Teknik Dokümantasyon": {
                "alan_fotograflari": {"pattern": r"(?:Fotoğraf|Photo|Görsel|Visual|Resim|Picture|Image|Şekil|Figure|Çekim|Shot)", "weight": 1},
                "cihaz_fotograflari": {"pattern": r"(?:Cihaz\s*Fotoğraf|Device\s*Photo|Alet\s*Fotoğraf|Equipment\s*Photo|Ölçüm\s*Cihaz|Measurement\s*Device)", "weight": 1},
                "kroki_sema": {"pattern": r"(?:Kroki|Sketch|Şema|Schema|Plan|Layout|Çizim|Drawing|Diyagram|Diagram|Harita|Map|Yerleşim|Placement)", "weight": 1},
                "armatur_teknik": {"pattern": r"(?:Armatür|Fixture|Lamba|Lamp|LED|Fotometrik|Photometric|Lümen|Lumen|lm|Wat|Watt|W|Işık\s*Ak[ıi]s[ıi]|Luminous\s*Flux|Verim|Efficacy)", "weight": 2}
            },
            "Ölçüm Cihazı Bilgileri": {
                "cihaz_detay": {"pattern": r"(?:Marka|Brand|Model|Tip|Type|Seri|Serial|SN|S/N|Üretici|Manufacturer|Kalibrasyon|Calibration)", "weight": 5},
                "cihaz_ozellikleri": {"pattern": r"(?:Hassasiyet|Accuracy|Precision|Doğruluk|Range|Aralık|Ölçüm\s*Aral[ıi]ğ[ıi]|Measurement\s*Range|Çözünürlük|Resolution)", "weight": 3},
                "cihaz_durumu": {"pattern": r"(?:Durum|Status|Çalışır|Working|Aktif|Active|Geçerli|Valid|Uygun|Suitable|Kullan[ıi]labilir|Usable)", "weight": 2}
            },
            "Sonuç ve Öneriler": {
                "genel_uygunluk": {"pattern": r"(?:Genel\s*Sonuç|Overall\s*Result|Uygun|Suitable|Uygunsuz|Non[\\-\\s]*Suitable|Geçerli|Valid|Geçersiz|Invalid|PASS|FAIL|Başar[ıi]l[ıi]|Successful)", "weight": 4},
                "standart_atif": {"pattern": r"(?:Standart|Standard|TS\s*EN|ISO|Referans|Reference|Atıf|Citation|Uygunluk|Compliance|Conformity)", "weight": 3},
                "iyilestirme_onerileri": {"pattern": r"(?:İyileştirme|Improvement|Öneri|Recommendation|Aksiyon|Action|Tedbir|Measure|Çözüm|Solution|Gelişt|Develop)", "weight": 4},
                "tekrar_olcum": {"pattern": r"(?:Tekrar\s*Ölçüm|Re[\\-\\s]*Measurement|Periyot|Period|S[ıi]kl[ıi]k|Frequency|Süre|Duration|Kontrol|Control|İzleme|Monitoring)", "weight": 4}
            }
        }


# ============================================
# TODO: PATTERN DEFINITIONS (extract_specific_values)
# ============================================
# Format: {
#     "pattern_group_name": {
#         "field_name": [r"pattern1", r"pattern2", ...]
#     }
# }
# Eğer yoksa: {}
pattern_definitions_data = {
    
        "extract_values": {
            "rapor_numarasi": [r"(?i)(?:RAPOR\s*NO|REPORT\s*NO|RAPOR\s*NUMARAS[II])\s*[|\s]*\s*([A-Z0-9\-\/]{3,20})",
            r"(?i)(?:RAPOR\s*NO|REPORT\s*NO|RAPOR\s*NUMARAS[II]|REPORT\s*NUMBER|DÖKÜMAN\s*NO|DOCUMENT\s*NO)\s*[:=]?\s*([A-Z0-9\-\/]{3,20})",
            r"(?i)(?:TEST\s*NO|BELGE\s*NO|REFERANS\s*NO|REF\s*NO)\s*[:=]?\s*([A-Z0-9\-\/]{3,20})"],
            "proje_adi": [r"(?i)(?:PROJE\s*ADI|PROJECT\s*NAME|PROJE\s*TANIM|PROJECT\s*TITLE)?\s*[:=-]?\s*(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*REV\s*-?\d*\s*\d*\.*\s*BÖLÜM\s*)?[-\s]*(.*?(?:ÖLÇÜM|TEST|RAPOR|REPORT))",
            r"(?i)[-\s]*([A-ZÇĞİÖŞÜ0-9\s]{5,100}.*?(?:ÖLÇÜM|TEST|RAPOR|REPORT))",
            r"(?i)(?:PROJE\s*ADI|PROJECT\s*NAME|PROJE\s*TANIM|PROJECT\s*TITLE)\s*[:=-]?\s*[-\s]*(.*?(?:ÖLÇÜM|TEST|RAPOR|REPORT))",
            r"(?i)(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*REV\s*-?\d*\s*\d*\.*\s*BÖLÜM\s*)[-\s]*(.*?(?:ÖLÇÜM|TEST|RAPOR|REPORT))"],
            "olcum_tarihi": [r"(?i)(?:ÖLÇÜM\s*TARİH[İI]?\s*\/?\s*SAAT[İI]?)\s*[|\s]*\s*(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4})",
            r"(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4})\s*\/?\s*\d{1,2}:\d{1,2}[\-:]\d{1,2}:\d{1,2}",
            r"(?i)(?:ÖLÇÜM\s*TARİH[İI]?|MEASUREMENT\s*DATE|TEST\s*TARİH[İI]?|TEST\s*DATE)\s*[:=]\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})",
            r"(?i)(?:ÖLÇÜM.*?(?:TARİH|DATE).*?)\s*[|\s]*\s*(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4})",
            r"(?i)(?:ÖLÇÜM\s*YAPILDI[ĞG]I|MEASURED\s*ON)\s*[:=]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})",
            r"(?:TARİH[İI]?\s*\/?\s*SAAT[İI]?).*?(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4})",
            r"\b(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{4})\b"],
            "rapor_tarihi": [r"(?i)(?:RAPOR\s*TARİH[İI]?\s*\/?\s*SAAT[İI]?)\s*[|\s]*\s*(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4})",
            r"(?i)(?:RAPOR\s*TARİH[İI]?|REPORT\s*DATE|HAZIRLANMA\s*TARİH[İI]?|PREPARED\s*ON|DÜZENLEME\s*TARİH[İI]?)\s*[:=]\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})",
            r"(?i)(?:BELGE|DÖKÜMAN|DOCUMENT).*?TARİH[İI]?.*?(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4})"],
            "olcum_cihaz": [r"(?i)(?:LÜKSMETRE|LUXMETER|LUX\s*METER)\s*[:=]?\s*([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9\s\.\-]{2,50}(?=\s*(?:\d{4,}|Seri|No|Nolu|$)))",
            r"(?i)(?:MARKA|BRAND)\s*[:=]?\s*([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9\s\.\-]{2,30})(?:.*?(?:MODEL|TİP|TYPE)\s*[:=]?\s*([A-ZÇĞİÖŞÜ0-9\-]{1,30}))?",
            r"(?i)(?:CİHAZ|DEVICE|ALET|INSTRUMENT)\s*[:=]?\s*([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9\s\.\-]{2,50}(?=\s*(?:\d{4,}|Seri|No|Nolu|$)))",
            r"(?i)([A-ZÇĞİÖŞÜ0-9]+(?:\s+[A-ZÇĞİÖŞÜ0-9]+){0,3}\s+(?:LUXMETER|LUX\s*METER|LÜKSMETRE)(?=\s*(?:\d{4,}|Seri|No|Nolu|$)))",
            r"(?i)(?:IŞIK\s*ŞİDDETİ\s*ÖLÇÜM\s*CİHAZI)\s*([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9\s\.\-]{2,50}(?=\s*(?:\d{4,}|Seri|No|Nolu|$)))"],
            "tesis_adi": [r"(?i)(?:RAPOR\s*NO|REPORT\s*NO|RAPOR\s*NUMARAS[II])\s*[|\s]*\s*([A-Z0-9\-\/]{3,20})",
            r"(?i)(?:RAPOR\s*NO|REPORT\s*NO|RAPOR\s*NUMARAS[II]|REPORT\s*NUMBER|DÖKÜMAN\s*NO|DOCUMENT\s*NO)\s*[:=]?\s*([A-Z0-9\-\/]{3,20})",
            r"(?i)(?:TEST\s*NO|BELGE\s*NO|REFERANS\s*NO|REF\s*NO)\s*[:=]?\s*([A-Z0-9\-\/]{3,20})"],
            "genel_uygunluk": [r"(?i)\b(UYGUN|SUITABLE|CONFORM|GEÇERLİ|VALID|PASS)\b",
            r"(?i)\b(UYGUNSUZ|NOT\s*SUITABLE|NON[\\-\\s]*CONFORM|GEÇERSİZ|INVALID|FAIL)\b",
            r"(?i)(?:GENEL\s*SONUÇ|OVERALL\s*RESULT|SONUÇ|RESULT)\s*[:=]?\s*(UYGUN|UYGUNSUZ|SUITABLE|NOT\s*SUITABLE|PASS|FAIL|GEÇERLİ|GEÇERSİZ)"],
           
        },           
} 

# ============================================
# TODO: VALIDATION KEYWORDS - CRITICAL TERMS
# ============================================
# Format: {
#     "Kategori İsmi": ["kelime1", "kelime2", ...]
# }
# Eğer yoksa: {}

critical_terms = [  
        ["aydınlatma", "lighting", "illumination", "ışık", "lumen", "ışık şiddeti"],
        ["lux", "cd/m2", "candela", "luminance", "illuminance"],
        ["ts en 12464", "en 12464", "12464", "iso 8995", "cibse"],
        ["led", "fluorescent", "floresan", "armatur", "luminaire", "ballast"],
        ["genel aydınlatma", "general lighting", "task lighting", "görev aydınlatması", "accent lighting", "emergency lighting", "acil aydınlatma"]
]

# ============================================
# TODO: VALIDATION KEYWORDS - STRONG KEYWORDS
# ============================================
# Format: ["kelime1", "kelime2", ...]
# Eğer yoksa: []

strong_keywords = [
        "aydınlatma",
        "lighting",
        "illumination", 
        "lux",
        "lümen",
        "lumen",
        "ts en 12464",
        "en 12464",
        "ışık",
        "ışık şiddeti"
]

# ============================================
# TODO: VALIDATION KEYWORDS - EXCLUDED KEYWORDS
# ============================================
# Format: ["kelime1", "kelime2", ...]
# Eğer yoksa: []

excluded_keywords = [
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        "espe",
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219","teknik resim","tasarım",
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide","kılavuzu",
        "loto",
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        "uygunluk", "beyan", "muayene", "conformity", "declaration",
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        "montaj", "assembly",
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        "titreşim", "vibration", "mekanik",
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate"
]

# ============================================
# TODO: CATEGORY ACTIONS (İyileştirme Önerileri)
# ============================================
# Format: {
#     "Kategori Adı": [
#         "Öneri 1",
#         "Öneri 2",
#     ]
# }
# Eğer yoksa: {}

category_actions_data = {"Genel Rapor Bilgileri": [
                "Proje adı ve numarasını netleştiriniz",
                "Ölçüm ve rapor tarihlerini açıkça belirtiniz",
                "Tesis/bölge/alan bilgilerini detaylandırınız",
                "Rapor numarası ve revizyon bilgisini ekleyiniz",
                "Ölçümü yapan firma ve personel bilgilerini yazınız",
                "Raporu hazırlayanın onay/imzasını alınız"
            ],
            "Ölçüm Metodu ve Standart Referansları": [
                "Ölçüm cihazının marka, model ve seri numarasını belirtiniz",
                "Cihaz kalibrasyon bilgilerini ve sertifikalarını ekleyiniz",
                "Ölçüm yöntemini detaylandırınız (dört nokta ortalaması vb.)",
                "TS EN 12464-1, ISO 8995 gibi standart referanslarını ekleyiniz"
            ],
            "Ölçüm Sonuç Tablosu": [
                "Ölçüm sonuç tablosunu sıra no ile düzenleyiniz",
                "Çalışma alanı/bölge adlarını açıkça tanımlayınız",
                "Ölçülen aydınlatma şiddeti değerlerini (lüks) eksiksiz yazınız",
                "Hedeflenen aydınlatma şiddeti değerlerini belirtiniz",
                "Her nokta için uygunluk durumunu (UYGUN/UYGUN DEĞİL) belirtiniz"
            ],
            "Uygunluk Değerlendirmesi": [
                "Tüm ölçüm noktalarının genel değerlendirmesini yapınız",
                "Limit dışı değerlerin listesini çıkarınız",
                "Yetersiz/aşırı aydınlatmanın risklerini belirtiniz",
                "Somut düzeltici faaliyet önerileri sununuz"
            ],
            "Görsel ve Teknik Dokümantasyon": [
                "Ölçüm yapılan alan fotoğraflarını ekleyiniz",
                "Ölçüm cihazı fotoğraflarını çekiniz",
                "Ölçüm noktaları kroki veya şemasını hazırlayınız",
                "Armatür teknik belgelerini (fotometrik raporlar) ekleyiniz"
            ],
            "Ölçüm Cihazı Bilgileri": [
                "Cihaz marka, model, seri no detaylarını tam olarak yazınız",
                "Cihaz hassasiyet, doğruluk, ölçüm aralığı özelliklerini belirtiniz",
                "Cihazın çalışır durumda olduğunu teyit ediniz"
            ],
            "Sonuç ve Öneriler": [
                "Genel uygunluk sonucunu (UYGUN/UYGUNSUZ) açıkça belirtiniz",
                "İlgili standartlara atıf yapınız",
                "Somut iyileştirme önerilerini listeyiniz",
                "Tekrar ölçüm periyodu önerisinde bulununuz"
            ]}


# ============================================
# EKLEME FONKSİYONU (DOKUNMAYIN!)
# ============================================
def migrate_service():
    """Servisi veritabanına ekle"""
    
    print("=" * 80)
    print(f"{DOCUMENT_TYPE_NAME.upper()} - VERİTABANINA AKTARMA")
    print("=" * 80)
    
    with app.app_context():
        # 1. Document Type Ekle/Bul
        doc_type = DocumentType.query.filter_by(code=DOCUMENT_TYPE_CODE).first()
        
        if doc_type:
            print(f"\n⚠️  Document type zaten var: {DOCUMENT_TYPE_CODE}")
            print("Mevcut veriyi güncellemek için önce silin veya kodu değiştirin.")
            return
        
        doc_type = DocumentType(
            code=DOCUMENT_TYPE_CODE,
            name=DOCUMENT_TYPE_NAME,
            description=DOCUMENT_TYPE_DESCRIPTION,
            service_file=SERVICE_FILE,
            endpoint=ENDPOINT,
            icon=ICON
        )
        db.session.add(doc_type)
        db.session.commit()
        print(f"\n✅ Document Type eklendi: {doc_type.code}")
        
        # 2. Criteria Weights
        if criteria_weights_data:
            print(f"\n📊 Criteria Weights ekleniyor...")
            criteria_weights = {}
            for idx, (category_name, weight) in enumerate(criteria_weights_data.items(), 1):
                cw = CriteriaWeight(
                    document_type_id=doc_type.id,
                    category_name=category_name,
                    weight=weight,
                    display_order=idx
                )
                db.session.add(cw)
                db.session.flush()
                criteria_weights[category_name] = cw
                print(f"   ✅ {category_name}: {weight} puan")
            db.session.commit()
            print(f"✅ Toplam {len(criteria_weights)} kategori eklendi")
        else:
            print(f"\n⏭️  Criteria Weights yok, atlanıyor...")
            criteria_weights = {}
        
        # 3. Criteria Details
        if criteria_details_data:
            print(f"\n📋 Criteria Details ekleniyor...")
            detail_count = 0
            for category_name, criteria_dict in criteria_details_data.items():
                if category_name not in criteria_weights:
                    print(f"   ⚠️  '{category_name}' kategorisi bulunamadı, atlanıyor...")
                    continue
                
                cw = criteria_weights[category_name]
                for idx, (criterion_name, data) in enumerate(criteria_dict.items(), 1):
                    cd = CriteriaDetail(
                        criteria_weight_id=cw.id,
                        criterion_name=criterion_name,
                        pattern=data['pattern'],
                        weight=data['weight'],
                        display_order=idx
                    )
                    db.session.add(cd)
                    detail_count += 1
            db.session.commit()
            print(f"✅ {detail_count} criteria detail eklendi")
        else:
            print(f"\n⏭️  Criteria Details yok, atlanıyor...")
        
        # 4. Pattern Definitions
        if pattern_definitions_data:
            print(f"\n🔍 Pattern Definitions ekleniyor...")
            pattern_count = 0
            for pattern_group, fields_dict in pattern_definitions_data.items():
                for idx, (field_name, patterns_list) in enumerate(fields_dict.items(), 1):
                    pd = PatternDefinition(
                        document_type_id=doc_type.id,
                        pattern_group=pattern_group,
                        field_name=field_name,
                        patterns=patterns_list,
                        display_order=idx
                    )
                    db.session.add(pd)
                    pattern_count += 1
            db.session.commit()
            print(f"✅ {pattern_count} pattern definition eklendi")
        else:
            print(f"\n⏭️  Pattern Definitions yok, atlanıyor...")
        
        # 5. Validation Keywords - Critical Terms
        if critical_terms:
            print("🔑 Validation Keywords (Critical Terms) ekleniyor...")
            
            # ✅ FORMAT KONTROLÜ: Dictionary mi List mi?
            if isinstance(critical_terms, dict):
                # Format 1: Dictionary (Kategorili - Titreşim gibi)
                for category, keywords in critical_terms.items():
                    validation = ValidationKeyword(
                        document_type_id=doc_type.id,
                        keyword_type='critical_terms',
                        category=category,
                        keywords=keywords
                    )
                    db.session.add(validation)
                    print(f"   ✅ {category}: {len(keywords)} kelime")
            
            elif isinstance(critical_terms, list):
                # Format 2: List of Lists (Kategorisiz - Elektrik gibi)
                for idx, keyword_list in enumerate(critical_terms, 1):
                    validation = ValidationKeyword(
                        document_type_id=doc_type.id,
                        keyword_type='critical_terms',
                        category=f'category_{idx}',
                        keywords=keyword_list
                    )
                    db.session.add(validation)
                    print(f"   ✅ Kategori {idx}: {len(keyword_list)} kelime")
            
            db.session.commit()
            print(f"✅ Critical terms eklendi\n")
        else:
            print("⏭️ Critical terms boş, atlanıyor\n")
        
        # 6. Validation Keywords - Strong Keywords
        if strong_keywords:
            print(f"\n🔑 Validation Keywords (Strong Keywords) ekleniyor...")
            vk = ValidationKeyword(
                document_type_id=doc_type.id,
                keyword_type='strong_keywords',
                keywords=strong_keywords
            )
            db.session.add(vk)
            db.session.commit()
            print(f"✅ {len(strong_keywords)} strong keyword eklendi")
        else:
            print(f"\n⏭️  Strong Keywords yok, atlanıyor...")
        
        # 7. Validation Keywords - Excluded Keywords
        if excluded_keywords:
            print(f"\n🔑 Validation Keywords (Excluded Keywords) ekleniyor...")
            vk = ValidationKeyword(
                document_type_id=doc_type.id,
                keyword_type='excluded_keywords',
                keywords=excluded_keywords
            )
            db.session.add(vk)
            db.session.commit()
            print(f"✅ {len(excluded_keywords)} excluded keyword eklendi")
        else:
            print(f"\n⏭️  Excluded Keywords yok, atlanıyor...")
        
        # 8. Category Actions
        if category_actions_data:
            print(f"\n💡 Category Actions ekleniyor...")
            action_count = 0
            for category_name, actions_list in category_actions_data.items():
                if category_name not in criteria_weights:
                    print(f"   ⚠️  '{category_name}' kategorisi bulunamadı, atlanıyor...")
                    continue
                
                cw = criteria_weights[category_name]
                for idx, action_text in enumerate(actions_list, 1):
                    ca = CategoryAction(
                        criteria_weight_id=cw.id,
                        action_text=action_text,
                        display_order=idx
                    )
                    db.session.add(ca)
                    action_count += 1
            db.session.commit()
            print(f"✅ {action_count} category action eklendi")
        else:
            print(f"\n⏭️  Category Actions yok, atlanıyor...")
        
        print("\n" + "=" * 80)
        print("✅ SERVİS BAŞARIYLA EKLENDİ!")
        print("=" * 80)
        
        print(f"\n📊 Özet:")
        print(f"   - Document Type: {DOCUMENT_TYPE_CODE}")
        print(f"   - Criteria Weights: {len(criteria_weights)}")
        print(f"   - Criteria Details: {detail_count if criteria_details_data else 0}")
        print(f"   - Pattern Definitions: {pattern_count if pattern_definitions_data else 0}")
        print(f"   - Critical Terms: {len(critical_terms)}")
        print(f"   - Strong Keywords: {len(strong_keywords)}")
        print(f"   - Excluded Keywords: {len(excluded_keywords)}")
        print(f"   - Category Actions: {action_count if category_actions_data else 0}")
        print()


if __name__ == '__main__':
    migrate_service()
    
