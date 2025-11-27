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
DOCUMENT_TYPE_CODE = 'lvd_report'  # TODO: Değiştir (örn: 'electric_circuit', 'espe_report')
DOCUMENT_TYPE_NAME = 'LVD Raporu Analizi'  # TODO: Değiştir
DOCUMENT_TYPE_DESCRIPTION = 'LVD Raporlarının Analizi'  # TODO: Değiştir
SERVICE_FILE = 'lvd_service.py'  # TODO: Değiştir
ENDPOINT = '/api/lvd-report'  # TODO: Değiştir
ICON = '⚡'  # TODO: Değiştir
# ============================================
# TODO: CRITERIA WEIGHTS (Kategoriler + Puanlar)
# ============================================
# Format: {"Kategori Adı": puan}
# Örnek: {"Rapor Kimlik Bilgileri": 10, "Genel Bilgiler": 15}
# Eğer yoksa: {}

criteria_weights_data = {
            "Genel Rapor Bilgileri": 15,
            "Ölçüm Metodu ve Standart Referansları": 15,
            "Ölçüm Sonuç Tablosu": 25,
            "Uygunluk Değerlendirmesi": 20,
            "Görsel ve Teknik Dökümantasyon": 10,
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
                "proje_adi_numarasi": {"pattern": r"(?:Project\s*(?:Name|No)|Proje\s*(?:Ad[ıi]|No)|Report\s*Title|Document\s*Title|E\d{2}\.\d{3}|C\d{2}\.\d{3}|T\d{2,3}[-.]?\d{3,4})", "weight": 3},
                "olcum_tarihi": {"pattern": r"(?:Measurement\s*Date|Ölçüm\s*Tarihi|Test\s*Date|Date\s*of\s*(?:Test|Measurement)|Measured\s*on|Tested\s*on|\d{1,2}[./\-]\d{1,2}[./\-]\d{4})", "weight": 3},
                "rapor_tarihi": {"pattern": r"(?:Report\s*Date|Rapor\s*Tarihi|Issue\s*Date|Document\s*Date|Prepared\s*on|Created\s*on|Date|Tarih|\d{1,2}[./\-]\d{1,2}[./\-]\d{4})", "weight": 3},
                "tesis_bolge_hat": {"pattern": r"(?:Customer|Müşteri|Client|Facility|Tesis|Plant|Factory|Company|Firma|Toyota|DANONE|Ford|BOSCH)", "weight": 2},
                "rapor_numarasi": {"pattern": r"(?:Report\s*No|Rapor\s*No|Document\s*No|Belge\s*No|E\d{2}\.\d{3}|C\d{2}\.\d{3}|SM\s*\d+|MCC\d+)", "weight": 2},
                "revizyon": {"pattern": r"(?:Version|Revizyon|Rev\.?|v)\s*[:=]?\s*(\d+|[A-Z])", "weight": 1},
                "firma_personel": {"pattern": r"(?:Prepared\s*by|Hazırlayan|Performed\s*by|Ölçümü\s*Yapan|Consultant|Engineer|PILZ)", "weight": 1}
            },
            "Ölçüm Metodu ve Standart Referansları": {
                "olcum_cihazi": {"pattern": r"(?:Measuring\s*Instrument|Ölçüm\s*Cihaz[ıi]|Test\s*Equipment|Multimeter|Multimetre|Ohmmeter|Instrument|Equipment|Device|Tester|Fluke|Metrix|Chauvin|Megger|Hioki)", "weight": 6},
                "kalibrasyon": {"pattern": r"(?:Calibration|Kalibrasyon|Kalibre|Certificate|Sertifika|Cal\s*Date)", "weight": 4},
                "standartlar": {"pattern": r"(?:EN\s*60204[-\s]*1?|IEC\s*60364|Standard|Standart)", "weight": 5}
            },
            "Ölçüm Sonuç Tablosu": {
                "sira_numarasi": {"pattern": r"(?:S[ıi]ra\s*(?:No|Numaras[ıi])|^\s*\d+\s)", "weight": 3},
                "makine_hat_bolge": {"pattern": r"(?:8X45|8X50|8X9J|9J73|8X52|8X60|8X62|8X70)\s*(?:R[1-9])?\s*(?:Hatt[ıi]|Line|Zone|Bölge)", "weight": 3},
                "olcum_noktasi": {"pattern": r"(?:Robot\s*\d+\.\s*Eksen\s*Motoru|Kalemtraş|Lift\s*and\s*Shift|Motor|Ekipman|Equipment|Device)", "weight": 3},
                "rlo_degeri": {"pattern": r"(\d+[.,]?\d*)\s*(?:mΩ|mohm|ohm|Ω)", "weight": 5},
                "yuk_iletken_kesiti": {"pattern": r"(?:4x4|4x2[.,]5|4x6|4x10|Yük\s*İletken|Load\s*Conductor|PE\s*İletken|PE\s*Conductor)", "weight": 4},
                "referans_degeri": {"pattern": r"(?:500\s*mΩ|500\s*ohm|500\s*Ω|EN\s*60204|IEC\s*60364)", "weight": 3},
                "uygunluk_durumu": {"pattern": r"(?:UYGUN|OK|PASS|Compliant|Uygun)", "weight": 4},
                "kesit_uygunlugu": {"pattern": r"(?:UYGUN|OK|PASS|Compliant|Uygun)", "weight": 3}
            },
            "Uygunluk Değerlendirmesi": {
                "toplam_olcum_nokta": {"pattern": r"(?:222|220|200|Toplam.*\d+)", "weight": 5},
                "uygun_nokta_sayisi": {"pattern": r"(?:211|210|UYGUN)", "weight": 5},
                "uygunsuz_isaretleme": {"pattern": r"\*D\.Y", "weight": 5, "reverse_logic": True},
                "standart_referans_uygunluk": {"pattern": r"(?:500\s*mΩ|EN\s*60204)", "weight": 5}
            },
            "Görsel ve Teknik Dökümantasyon": {
                "cihaz_baglanti_fotografi": {"pattern": r"(?:Cihaz.*Fotoğraf|Bağlant[ıi].*Fotoğraf|Ölçüm.*Cihaz|Photo|Image|Figure|Resim|Görsel)", "weight": 10}
            },
            "Sonuç ve Öneriler": {
                "genel_uygunluk": {"pattern": r"(?:Genel\s*Uygunluk|Sonuç|UYGUN|UYGUNSUZ|Result|Conclusion|Compliant|Non-compliant)", "weight": 8},
                "standart_atif": {"pattern": r"(?:EN\s*60204|IEC\s*60364|Standart.*Atıf|Standart.*Referans|Standard.*Reference)", "weight": 7}
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
        "check_date_validity": {
            "olcum_patterns": [
            # Türkçe formatlar
            r"(?:Ölçüm\s*Tarihi|Test\s*Tarihi|Ölçüm\s*Yapıldığı\s*Tarih)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Ölçüm|Test).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4}).*?(?:ölçüm|test)",
            
            # İngilizce formatlar
            r"(?:Measurement\s*Date|Test\s*Date|Date\s*of\s*(?:Test|Measurement))\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Measured\s*on|Tested\s*on)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4}).*?(?:measurement|test)",
            
            # Genel formatlar
            r"(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})",
            r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
            r"(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})"
        ],

        # Rapor tarihi arama - çok kapsamlı pattern'lar
        "rapor_patterns" : [
            # Türkçe formatlar
            r"(?:Rapor\s*Tarihi|Belge\s*Tarihi|Hazırlanma\s*Tarihi)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Rapor|Belge).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Hazırlayan|Hazırlandı)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            
            # İngilizce formatlar
            r"(?:Report\s*Date|Document\s*Date|Issue\s*Date|Prepared\s*Date)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(?:Prepared\s*on|Issued\s*on|Created\s*on)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            
            # Genel formatlar
            r"(?:Date|Tarih)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            r"(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})",
            r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
            r"(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})"
            ]
        },
        "general_patterns": {
                        "proje_adi_numarasi": [r"(C\d+\.\d+|Proje|Project|SM\s*\d+)"],
                        "tesis_bolge_hat": [r"(Tesis|Makine|Hat|Bölge|Line)"],
                        "olcum_cihazi": [r"(Multimetre|Ohmmetre|Ölçüm|Cihaz)"],
                        "kalibrasyon": [r"(Kalibrasyon|Kalibre|Cert|Sertifika)"],
                        "standartlar": [r"(EN\s*60204|IEC\s*60364|Standard|Standart)"],
                        "rlo_degeri": [r"(\d+[.,]?\d*\s*(?:mΩ|mohm|ohm))"],
                        "uygunluk_durumu": [r"(UYGUN|OK|NOK|Uygun|Değil)"],
                        "risk_belirtme": [r"(Risk|Tehlike|Uygunsuz|Problem)"],
                        "genel_uygunluk": [r"(Sonuç|Result|Uygun|Geçer|Pass|Fail)"]
                    },
    
            "value_patterns": {
            # Proje adı/numarası için kapsamlı pattern'ler
            "proje_adi": [
                r"(?:Project\s*Name|Proje\s*Ad[ıi])\s*[:=]\s*([^\n\r]+)",
                r"(?:Project\s*No|Proje\s*No|Project\s*Number)\s*[:=]\s*([A-Z0-9.-]+)",
                r"(?:Report\s*Title|Rapor\s*Başl[ıi]ğ[ıi])\s*[:=]\s*([^\n\r]+)",
                r"(?:Document\s*Title|Belge\s*Başl[ıi]ğ[ıi])\s*[:=]\s*([^\n\r]+)",
                r"(LVD\s+[Öö]lç[üu]m[^,\n]*)",
                r"(Topraklama\s+S[üu]reklilik[^,\n]*)",
                r"(Grounding\s+Continuity[^,\n]*)",
                r"([A-Z][a-z]+\s*-\s*[A-Z][a-z]+.*?[Öö]lç[üu]m)",
                r"(E\d{2}\.\d{3}\s*-[^,\n]+)"
            ],
            
            # Rapor numarası için kapsamlı pattern'ler
            "rapor_numarasi": [
                r"(?:Report\s*No|Rapor\s*No|Report\s*Number)\s*[:=]\s*([A-Z0-9.-]+)",
                r"(?:Document\s*No|Belge\s*No)\s*[:=]\s*([A-Z0-9.-]+)",
                r"(E\d{2}\.\d{3})",
                r"(C\d{2}\.\d{3})",
                r"(T\d{2,3}[-.]?\d{3,4})",
                r"SM\s*(\d+)",
                r"MCC(\d+)",
                r"^\s*([A-Z]\d{2,3}[.-]\d{3,4})"
            ],
            
            # Ölçüm cihazı için çok kapsamlı pattern'ler
            "olcum_cihazi": [
                r"(?:Measuring\s*Instrument|Ölçüm\s*Cihaz[ıi]|Test\s*Equipment)\s*[:=]\s*([^\n\r]+)",
                r"(?:Multimeter|Multimetre|Ohmmeter|Ohmmetre)\s*[:=]?\s*([A-Z0-9\s.-]+)",
                r"(?:Instrument|Cihaz)\s*[:=]\s*([^\n\r]+)",
                r"(?:Equipment|Ekipman)\s*[:=]\s*([^\n\r]+)",
                r"(?:Device|Alet)\s*[:=]\s*([^\n\r]+)",
                r"(?:Tester|Test\s*Cihaz[ıi])\s*[:=]?\s*([A-Z0-9\s.-]+)",
                r"(Fluke\s*\d+[A-Z]*)",
                r"(Metrix\s*[A-Z0-9]+)",
                r"(Chauvin\s*Arnoux\s*[A-Z0-9]+)",
                r"(Megger\s*[A-Z0-9]+)",
                r"(Hioki\s*[A-Z0-9]+)",
                r"([A-Z][a-z]+\s*\d+[A-Z]*)",
                r"(MΩ\s*metre|mΩ\s*metre|Loop\s*Tester|Continuity\s*Tester)"
            ],
            
            # Tesis/müşteri bilgisi
            "tesis_adi": [
                r"(?:Customer|Müşteri|Client)\s*[:=]\s*([^\n\r]+)",
                r"(?:Facility|Tesis|Plant|Factory)\s*[:=]\s*([^\n\r]+)",
                r"(?:Company|Firma|Corporation)\s*[:=]\s*([^\n\r]+)",
                r"(Toyota[^\n\r]*)",
                r"(DANONE[^\n\r]*)",
                r"(Ford[^\n\r]*)",
                r"(BOSCH[^\n\r]*)",
                r"(?:8X45|8X50|8X9J|9J73)\s*(?:R1|R2|R3)?\s*Hatt[ıi]",
                r"([A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Factory|Plant|Facility))"
            ],
            
            # Firma/personel bilgisi
            "firma_personel": [
                r"(?:Prepared\s*by|Hazırlayan|Consultant)\s*[:=]\s*([^\n\r]+)",
                r"(?:Performed\s*by|Ölçümü\s*Yapan)\s*[:=]\s*([^\n\r]+)",
                r"(?:Company|Firma)\s*[:=]\s*([^\n\r]+)",
                r"(?:Engineer|Mühendis)\s*[:=]\s*([^\n\r]+)",
                r"(PILZ[^\n\r]*)",
                r"([A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Engineering|Mühendislik))"
            ],
            
            # Tarih pattern'leri - çok kapsamlı
            "olcum_tarihi": [
                # Türkçe formatlar
                r"(?:Ölçüm\s*Tarihi|Test\s*Tarihi|Ölçüm\s*Yapıldığı\s*Tarih)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Ölçüm|Test).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"Tarih\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                
                # İngilizce formatlar
                r"(?:Measurement\s*Date|Test\s*Date|Date\s*of\s*(?:Test|Measurement))\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Measured\s*on|Tested\s*on)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Date|When)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                
                # Çeşitli formatlar
                r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})\s*(?:tarihinde|on|at|de)",
                r"(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})",
                r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
                r"(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})",
                
                # Tablo içindeki tarihler
                r"(?:Measurement|Ölçüm).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            ],
            
            "rapor_tarihi": [
                # Türkçe formatlar
                r"(?:Rapor\s*Tarihi|Belge\s*Tarihi|Hazırlanma\s*Tarihi)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Rapor|Belge).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Hazırlayan|Hazırlandı)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                
                # İngilizce formatlar  
                r"(?:Report\s*Date|Document\s*Date|Issue\s*Date|Prepared\s*Date)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Prepared\s*on|Issued\s*on|Created\s*on)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(?:Report|Document).*?(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                
                # Çeşitli formatlar
                r"(?:Date|Tarih)\s*[:=]\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})",
                r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
                r"(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})",
                
                # Tablo başlığı veya footer'daki tarihler
                r"(?:Created|Issued|Published)\s*[:=]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
                r"(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})",
            ]
        },

        "extract_values": {
            "proje_patterns": [r'(C\d{2}\.\d{3})',
                r'(E\d{2}\.\d{3})',
                r'(T\d{2,3}[-\.]?\d{3,4})',
                r'(\d{4,6})',
                r'([A-Z]\d{2,3}[.-]\d{3,4})'],

            "rapor_patterns": [r'SM\s*(\d+)',
                r'MCC(\d+)',
                r'Report\s*No[\s:]*([A-Z0-9.-]+)',
                r'Rapor[\s:]*([A-Z0-9.-]+)'],

            "musteri_patterns": [r'Toyota',
                r'DANONE',
                r'Ford',
                r'BOSCH',
                r'P&G'],
            "rlo_patterns": [
            r"(\d+[.,]?\d*)\s*(?:mΩ|mohm|ohm|Ω)",
            r"(\d+)\s*(?:4x[2-9](?:[.,]\d+)?|4x4)\s*(?:[2-9](?:[.,]\d+)?|4)\s*500",
            r"(\d+)\s*(?:mΩ|mohm|ohm|Ω)"
            ],

            "kesit_patterns": [
            r"4x4",
            r"4x2[.,]5", 
            r"4x6",
            r"4x10",
            r"Yük\s*İletken",
            r"Load\s*Conductor",
            r"PE\s*İletken",
            r"PE\s*Conductor"
        ],

            "hat_pattern":[r"(8X45|8X50|8X9J|9J73|8X52|8X60|8X62|8X70)\s*(?:R[1-9])?\s*Hatt[ıi]"],

            "high_rlo_patterns": [
                    r'(\d{3,4})\s*(?:4x[2-9](?:[.,]\d+)?|4x4)\s*(?:[2-9](?:[.,]\d+)?|4)\s*500(\d+)\s*mΩ\s*<\s*500\s*mΩ',
                    r'(\d{3,4})\s*(?:mΩ|mohm|ohm|Ω)',
                    r'(\d{3,4})[.,]?\d*\s*(?:mΩ|mohm|ohm|Ω)'
                ],
            
            "hat_patterns2": [
                                    r'(8X\d+R?\d*)\s*(?:Hatt[ıi]|Line|Zone)?\s*(.*?)(?:\s+\d+)',
                                    r'(8X\d+R?\d*)\s*(.*?)(?:\s+\d+)',
                                    r'(Line\s*\d+|Zone\s*\d+)\s*(.*?)(?:\s+\d+)'
                                ]
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
         # Topraklama süreklilik temel terimleri
        ["süreklilik", "continuity", "iletkenler", "conductors", "bağlantı", "connection", "devamlılık"],
        
        # Kesit uygunluk terimleri
        ["kesit", "cross section", "iletken kesiti", "conductor cross section", "mm2", "mm²", "kesit uygunluğu"],
        
        # LVD ve standart terimleri
        ["lvd", "en 60204-1", "60204-1", "60204", "makine güvenliği", "machine safety"],
        
        # Ölçüm ve test terimleri
        ["ölçüm", "measurement", "test", "ohm", "direnç", "resistance", "milliohm", "mΩ"],
        
        # Elektrik güvenlik terimleri
        ["elektrik güvenliği", "electrical safety", "koruyucu iletken", "protective conductor", "pe", "protective earth"]
]
# ============================================
# TODO: VALIDATION KEYWORDS - STRONG KEYWORDS
# ============================================
# Format: ["kelime1", "kelime2", ...]
# Eğer yoksa: []

strong_keywords = [
        "lvd",
        "TOPRAKLAMA SÜREKLİLİK", 
        "topraklama süreklilik",
        "TOPRAKLAMA İLETKENLERİ", "TOPRAKLAMA ILETKENLERI",
        "topraklama iletkenleri",
        "topraklama sureklilik",
        "KESİT UYGUNLUĞU", "kesit uygunlugu", "kesit uygunluğu", "kesıt uygunluğu",
]

# ============================================
# TODO: VALIDATION KEYWORDS - EXCLUDED KEYWORDS
# ============================================
# Format: ["kelime1", "kelime2", ...]
# Eğer yoksa: []

excluded_keywords = [
        # Aydınlatma raporu
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
        
        # Hidrolik devre şeması
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219","teknik resim","tasarım",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # AT Uygunluk Beyanı (eski strong_keywords AT'den)
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        
        # Espe raporu  
        "espe",
        
        # Gürültü ölçüm raporu
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # Manuel/kullanma kılavuzu
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide","kılavuzu",
        
        # LOTO raporu
        "loto",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",

        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "TOPRAKLAMA DİRENCİ",
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

category_actions_data = {}


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