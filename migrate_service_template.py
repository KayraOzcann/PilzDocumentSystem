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
DOCUMENT_TYPE_CODE = 'electric_circuit'  # TODO: Değiştir (örn: 'electric_circuit', 'espe_report')
DOCUMENT_TYPE_NAME = 'Elektrik Devre Şeması Analizi'  # TODO: Değiştir
DOCUMENT_TYPE_DESCRIPTION = 'Elektrik devre şemalarının analizi'  # TODO: Değiştir
SERVICE_FILE = 'elektrik_service.py'  # TODO: Değiştir
ENDPOINT = '/api/elektrik-report'  # TODO: Değiştir
ICON = '🔌'  # TODO: Değiştir

# ============================================
# TODO: CRITERIA WEIGHTS (Kategoriler + Puanlar)
# ============================================
# Format: {"Kategori Adı": puan}
# Örnek: {"Rapor Kimlik Bilgileri": 10, "Genel Bilgiler": 15}
# Eğer yoksa: {}

criteria_weights_data = {
            "Semboller ve İşaretler": 30,
            "Bağlantı Hatları": 25,
            "Etiketleme ve Numara Sistemleri": 20,
            "Kontrol Panosu / Makine Otomasyon Öğeleri": 15,
            "Şematik Yerleşim": 10
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
    "Semboller ve İşaretler": {
                "direnc_sembol": {"pattern": r"(?i)(?:direnç|resistor|ohm|Ω|R\d+|[0-9]+[RKM][0-9]*|zigzag|potansiyometre|pot|trimmer|━+|─+)", "weight": 6},
                "kondansator_sembol": {"pattern": r"(?i)(?:kondansatör|capacitor|C\d+|[0-9]+[µnpF]+|paralel\s*çizgi|elektrolitik|seramik|\|\||═+|◇.*?\|\||◇.*?═+|⬧.*?\|\||⬧.*?═+|⬥.*?\|\||⬥.*?═+|<>.*?\|\||<>.*?═+|[\u25C7\u25C8\u25C6].*?(?:\|\||═+))", "weight": 6},
                "bobin_sembol": {"pattern": r"(?i)(?:bobin|inductor|L\d+|[0-9]+[mH]+|spiral|solenoid|trafo|transformatör|transformer|⤾|⟲|⥀)", "weight": 5},
                "diyot_sembol": {"pattern": r"(?i)(?:diyot|diode|D\d+|LED|zener|köprü|bridge|rectifier|doğrultucu|▶|►|⊳)", "weight": 5},
                "transistor_sembol": {"pattern": r"(?i)(?:transistör|transistor|Q\d+|NPN|PNP|FET|MOSFET|BJT|darlington|⊲|△)", "weight": 4},
                "toprak_sembol": {"pattern": r"(?i)(?:toprak|ground|earth|GND|⏚|⊥|chassis|şasi|PE|↧|⌁)", "weight": 2},
                "sigorta_sembol": {"pattern": r"(?i)(?:sigorta|fuse|F\d+|MCB|RCD|devre\s*kesici|circuit\s*breaker|termik|⚡|═+)", "weight": 2}
            },
            "Bağlantı Hatları": {
                "iletken_baglanti": {"pattern": r"(?i)(?:kablo|wire|cable|hat|line|bağlantı|connection|conductor|iletken|NYA|NYM|H0[57]|━+|─+)", "weight": 8},
                "kesisen_hatlar": {"pattern": r"(?i)(?:kesişen|crossing|köprü|bridge|junction|node|düğüm|bağlantı\s*noktası|●|⊏|⊐)", "weight": 6},
                "baglanti_noktalari": {"pattern": r"(?i)(?:bağlantı\s*noktası|connection\s*point|terminal|node|klemens|terminal\s*block|X\d+|●|○|◯|⊙)", "weight": 6},
                "elektriksel_yon": {"pattern": r"(?i)(?:yön|direction|ok|arrow|akış|flow|akım|current|→|←|↑|↓|⟶|⇾)", "weight": 5}
            },
            "Etiketleme ve Numara Sistemleri": {
                "bilesenlerin_etiketlenmesi": {"pattern": r"(?i)(?:[RCL]\d+|[QDT]\d+|[MKF]\d+|[UIC]\d+|[+-]V(?:cc|dd|ss)|[+-]?\d+V|S[0-9]|K[0-9])", "weight": 6},
                "elektriksel_degerler": {"pattern": r"(?i)(?:\d+(?:\.\d+)?.*?(?:[VvAaMmWwΩ]|volt|amp|watt|ohm|VA|kVA|mA|µA)|[~=]|\~|\∿)", "weight": 5},
                "klemens_numaralari": {"pattern": r"(?i)(?:klemens|terminal|X\d+|TB\d+|[0-9]+\.[0-9]+|L[123N]|PE|[UVWN]\d*)", "weight": 5},
                "kablo_etiketleri": {"pattern": r"(?i)(?:kablo|wire|H\d+|W\d+|[0-9]+[AWG]|NYA|NYM|H0[57]|[0-9xX]+mm²)", "weight": 4}
            },
            "Kontrol Panosu / Makine Otomasyon Öğeleri": {
                "plc_giris_cikis": {"pattern": r"(?i)(?:PLC|I[0-9]+|Q[0-9]+|DI|DO|AI|AO|input|output|giriş|çıkış|[0-9]+[VI][0-9]+)", "weight": 4},
                "kontaktor_rele": {"pattern": r"(?i)(?:kontaktör|contactor|röle|relay|K\d+|KM\d+|NO|NC|coil|bobin|⤾|⟲)", "weight": 4},
                "motor_starter": {"pattern": r"(?i)(?:motor|starter|M\d+|drive|sürücü|inverter|softstarter|DOL|VFD|⊏⊐|▭M)", "weight": 3},
                "buton_sensor": {"pattern": r"(?i)(?:buton|button|sensör|sensor|S\d+|B\d+|switch|anahtar|proximity|PNP|NPN|○|◯|⊙)", "weight": 2},
                "ac_dc_guc": {"pattern": r"(?i)(?:AC|DC|güç|power|[0-9]+[VvAa]|~|⎓|[1-3]~|\+|-|N|PE|L[123]|\∿|=)", "weight": 2}
            },
            "Şematik Yerleşim": {
                "bilgi_akisi": {"pattern": r"(?i)(?:giriş|input|çıkış|output|soldan|sağa|yukarı|aşağı|→|←|↑|↓|⟶|⇾)", "weight": 3},
                "mantikli_dizilim": {"pattern": r"(?i)(?:işleme|process|dönüşüm|transformation|kontrol|control|güç|power|▭|⊏⊐)", "weight": 3},
                "sayfa_basligi": {"pattern": r"(?i)(?:proje|project|tarih|date|çizim|drawing|revizyon|revision|ref|no)", "weight": 2},
                "cerceve_frame": {"pattern": r"(?i)(?:çerçeve|frame|başlık|title|numara|number|sayfa|page|sheet|▭|□)", "weight": 2}
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
    "proje_no": [
        r"(?:30292390|PROJE\s*NO|PROJECT\s*NO)"
    ],
    "sistem_tipi": [
        r"(?i)(?:elektrik\s*şeması|electric\s*circuit|electrical\s*diagram)"
    ],
    "tarih": [
        r"(\d{2}\.\d{2}\.\d{4})"
    ],
    "elektrik_paneli": [
        r"(?i)(?:ELEKTRİK\s*PANELİ|ELECTRICAL\s*PANEL|CONTROL\s*PANEL)"
    ],
    "voltaj": [
        r"(?i)(?:(\d+)\s*V|(\d+)\s*volt)"
    ],
    "akim": [
        r"(?i)(?:(\d+)\s*A|(\d+)\s*amp)"
    ],
    "guc": [
        r"(?i)(?:(\d+)\s*W|(\d+)\s*watt|(\d+)\s*kW)"
    ],
    "frekans": [
        r"(?i)(?:(\d+)\s*Hz|(\d+)\s*hertz)"
    ],
    "klemens_blogu": [
        r"(?i)(?:KLEMENS|TERMINAL|TB\d+|X\d+)"
    ]
}

# ============================================
# TODO: VALIDATION KEYWORDS - CRITICAL TERMS
# ============================================
# Format: {
#     "Kategori İsmi": ["kelime1", "kelime2", ...]
# }
# Eğer yoksa: {}

pattern_definitions_data = {
    "extract_values": {
        "proje_no": [r"(?:30292390|PROJE\s*NO|PROJECT\s*NO)"],
        "sistem_tipi": [r"(?i)(?:elektrik\s*şeması|electric\s*circuit|electrical\s*diagram)"],
        "tarih": [r"(\d{2}\.\d{2}\.\d{4})"],
        "elektrik_paneli": [r"(?i)(?:ELEKTRİK\s*PANELİ|ELECTRICAL\s*PANEL|CONTROL\s*PANEL)"],
        "voltaj": [r"(?i)(?:(\d+)\s*V|(\d+)\s*volt)"],
        "akim": [r"(?i)(?:(\d+)\s*A|(\d+)\s*amp)"],
        "guc": [r"(?i)(?:(\d+)\s*W|(\d+)\s*watt|(\d+)\s*kW)"],
        "frekans": [r"(?i)(?:(\d+)\s*Hz|(\d+)\s*hertz)"],
        "klemens_blogu": [r"(?i)(?:KLEMENS|TERMINAL|TB\d+|X\d+)"]
    }
}   

# ============================================
# TODO: VALIDATION KEYWORDS - STRONG KEYWORDS
# ============================================
# Format: ["kelime1", "kelime2", ...]
# Eğer yoksa: []

strong_keywords = [
    "elektrik",
    "circuit",
    "electrical",
    "voltage",
    "amper",
    "ohm",
    "enclosure",
    "wrp-",
    "light curtain",
    "contactors",
    "controller"
]

# ============================================
# TODO: VALIDATION KEYWORDS - EXCLUDED KEYWORDS
# ============================================
# Format: ["kelime1", "kelime2", ...]
# Eğer yoksa: []

excluded_keywords = [
    # Topraklama raporu (eski strong_keywords)
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        
        # Aydınlatma raporu
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
        
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Espe raporu  
        "espe",
        
        # Hidrolik devre şeması
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219","teknik resim","tasarım",
        
        # Gürültü ölçüm raporu
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # Manuel/kullanma kılavuzu
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide","kılavuzu",
        
        # LOTO raporu
        "loto",
        
        # LVD raporu
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        
        # AT tip muayene (AT uygunluk beyanı)
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # Bakım talimatları
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        
        # Mekanik titreşim raporu
        "titreşim", "vibration", "mekanik",
        
        # AT tip inceleme sertifikası
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
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
            print(f"\n🔑 Validation Keywords (Critical Terms) ekleniyor...")
            for category, keywords in critical_terms.items():
                vk = ValidationKeyword(
                    document_type_id=doc_type.id,
                    keyword_type='critical_terms',
                    category=category,
                    keywords=keywords
                )
                db.session.add(vk)
            db.session.commit()
            print(f"✅ {len(critical_terms)} critical term kategorisi eklendi")
        else:
            print(f"\n⏭️  Critical Terms yok, atlanıyor...")
        
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