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
DOCUMENT_TYPE_CODE = 'pneumatic_circuit'  # TODO: Değiştir (örn: 'electric_circuit', 'espe_report')
DOCUMENT_TYPE_NAME = 'Pnömatik Devre Şeması Analizi'  # TODO: Değiştir
DOCUMENT_TYPE_DESCRIPTION = 'Pnömatik Devre Şeması Raporlarının Analizi'  # TODO: Değiştir
SERVICE_FILE = 'pnomatic_service.py'  # TODO: Değiştir
ENDPOINT = '/api/pnomatic-control'  # TODO: Değiştir
ICON = '💨'  # TODO: Değiştir
# ============================================
# TODO: CRITERIA WEIGHTS (Kategoriler + Puanlar)
# ============================================
# Format: {"Kategori Adı": puan}
# Örnek: {"Rapor Kimlik Bilgileri": 10, "Genel Bilgiler": 15}
# Eğer yoksa: {}

criteria_weights_data = {
            "Temel Sistem Bileşenleri": 25,
            "Pnömatik Semboller ve Vana Sistemleri": 30,
            "Akış Yönü ve Bağlantı Hatları": 20,
            "Sistem Bilgileri ve Teknik Parametreler": 15,
            "Dokümantasyon ve Standart Uygunluk": 10
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
            "Temel Sistem Bileşenleri": {
                "hava_kaynagi_ve_hazirlama": {
                    "pattern": r"(?i)(?:pressure\s*source|hava\s*kaynağı|basınçlı\s*hava|air\s*supply|P\s*=\s*\d+.*?bar|kompresör|pneumatic|pnömatik|bar)",
                    "weight": 5,
                    "description": "Hava kaynağı ve hazırlama ünitesi"
                },
                "filtre_regulator_lubrikator": {
                    "pattern": r"(?i)(?:FRL|filtre|filter|regulator|regülatör|lubricator|yağlayıcı|kondisyoner|hava\s*hazırlama)",
                    "weight": 5,
                    "description": "Hava hazırlama grubu (FRL)"
                },
                "basinc_gosterge_sensoru": {
                    "pattern": r"(?i)(?:manometre|pressure.*?gauge|gösterge|indicator|PI|PT|PS|basınç.*?sensör|ölçüm|pressure|basınç)",
                    "weight": 4,
                    "description": "Basınç gösterge ve sensörleri"
                },
                "susturucu_egzoz": {
                    "pattern": r"(?i)(?:susturucu|muffler|exhaust|egzoz|silencer|tahliye|vent|boşaltım)",
                    "weight": 4,
                    "description": "Susturucu ve egzoz elemanları"
                },
                "genel_pnomatik_sistem": {
                    "pattern": r"(?i)(?:pneumatic|pnömatik|circuit|devre|diagram|diyagram|şema|sistem|air|hava)",
                    "weight": 7,
                    "description": "Genel pnömatik sistem varlığı"
                }
            },
            "Pnömatik Semboller ve Vana Sistemleri": {
                "silindir_actuator": {
                    "pattern": r"(?i)(?:silindir|cylinder|piston|actuator|çift.*?etkili|tek.*?etkili|double.*?acting|single.*?acting|C\d+|MGF|CYL|SIL)",
                    "weight": 8,
                    "description": "Silindir ve aktüatör sembolleri"
                },
                "yon_kontrol_vanalar": {
                    "pattern": r"(?i)(?:VUVG|Y\d+|V\d+|valf|valve|5/2|4/2|3/2|2/2|yön.*?kontrol|directional.*?control|solenoid|vana|kontrol)",
                    "weight": 10,
                    "description": "Yön kontrol vanaları"
                },
                "hiz_kontrol_vanalar": {
                    "pattern": r"(?i)(?:hız.*?kontrol|speed.*?control|flow.*?control|akış.*?kontrol|throttle|kısıcı|flow|akış)",
                    "weight": 6,
                    "description": "Hız kontrol vanaları"
                },
                "basinc_kontrol_vanalar": {
                    "pattern": r"(?i)(?:basınç.*?kontrol|pressure.*?control|relief|emniyet|PRV|basınç.*?azaltıcı|pressure)",
                    "weight": 6,
                    "description": "Basınç kontrol vanaları"
                }
            },
            "Akış Yönü ve Bağlantı Hatları": {
                "hava_besleme_hatlari": {
                    "pattern": r"(?i)(?:besleme|supply.*?line|hava.*?hattı|ana.*?hat|pressure.*?line|P|feed|input|giriş)",
                    "weight": 5,
                    "description": "Hava besleme hatları"
                },
                "calisma_hatlari": {
                    "pattern": r"(?i)(?:A|B|çalışma.*?hattı|working.*?line|port|output|çıkış)",
                    "weight": 5,
                    "description": "Çalışma hatları (A, B portları)"
                },
                "egzoz_tahliye_hatlari": {
                    "pattern": r"(?i)(?:R|S|EA|EB|egzoz|exhaust|tahliye|drain|vent|return|boşaltım)",
                    "weight": 3,
                    "description": "Egzoz ve tahliye hatları"
                },
                "yon_oklari_akim_gosterimi": {
                    "pattern": r"(?i)(?:→|←|↑|↓|⇒|⇐|⇑|⇓|yön|direction|ok|arrow|akış|flow|hat|line)",
                    "weight": 4,
                    "description": "Yön okları ve akış gösterimi"
                },
                "baglanti_hatlari": {
                    "pattern": r"(?i)(?:bağlantı|connection|hat|line|pipe|boru|tube|hose)",
                    "weight": 3,
                    "description": "Genel bağlantı hatları"
                }
            },
            "Sistem Bilgileri ve Teknik Parametreler": {
                "calisma_basinci": {
                    "pattern": r"(?i)(?:P\s*=\s*\d+(?:\.\d+)?.*?bar|\d+(?:\.\d+)?.*?bar|çalışma.*?basınç|working.*?pressure|4-6.*?bar|basınç|pressure|\d+\s*bar)",
                    "weight": 4,
                    "description": "Çalışma basıncı değerleri"
                },
                "hava_tuketimi": {
                    "pattern": r"(?i)(?:Q\s*=\s*\d+.*?l/min|\d+.*?l/min|hava.*?tüketim|air.*?consumption|flow.*?rate|tüketim|consumption|l/min|flow)",
                    "weight": 3,
                    "description": "Hava tüketimi değerleri"
                },
                "strok_boyutlari": {
                    "pattern": r"(?i)(?:strok|stroke|s\s*=\s*\d+.*?mm|\d+.*?mm|boyut|dimension|mesafe|size|mm|cm)",
                    "weight": 3,
                    "description": "Strok ve boyut bilgileri"
                },
                "vana_tipleri_ozellikler": {
                    "pattern": r"(?i)(?:normalde.*?kapalı|normalde.*?açık|NC|NO|spring.*?return|yay.*?geri|5/2|4/2|3/2|2/2)",
                    "weight": 3,
                    "description": "Vana tipleri ve özellikleri"
                },
                "teknik_parametreler": {
                    "pattern": r"(?i)(?:teknik|technical|parametre|parameter|özellik|specification|spec|sistem|system)",
                    "weight": 2,
                    "description": "Genel teknik parametre varlığı"
                }
            },
            "Dokümantasyon ve Standart Uygunluk": {
                "sembol_standartlari": {
                    "pattern": r"(?i)(?:ISO.*?1219|DIN.*?ISO|pnömatik.*?sembol|pneumatic.*?symbol|standart|standard|ISO|DIN)",
                    "weight": 2,
                    "description": "Sembol standartları"
                },
                "cizim_bilgileri": {
                    "pattern": r"(?i)(?:çizim.*?tarih|drawing.*?date|tasarım.*?tarih|design.*?date|\d{2}.\d{2}.\d{4}|\d{2}/\d{2}/\d{4}|created|designed|tarih|date)",
                    "weight": 2,
                    "description": "Çizim bilgileri ve tarihler"
                },
                "proje_bilgileri": {
                    "pattern": r"(?i)(?:proje.*?adı|project.*?name|sistem.*?adı|description|açıklama)",
                    "weight": 2,
                    "description": "Proje bilgileri"
                },
                "firma_logo_imza": {
                    "pattern": r"(?i)(?:ACT|festo|FESTO|onay.*?tarih|approval|kontrol.*?tarih|check|created.*?by|checked.*?by|approved|firma|company)",
                    "weight": 2,
                    "description": "Firma logosu ve imza bilgileri"
                },
                "dokumantasyon_genel": {
                    "pattern": r"(?i)(?:revision|rev|circuit.*?number|customer|müşteri|sheet|size|boyut|diagram|diyagram|şema|schema)",
                    "weight": 2,
                    "description": "Genel dokümantasyon varlığı"
                }
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
            "visual_templates": {
            "pneumatic_cylinders": ["cylinder", "piston", "MGF", "silindir"],
            "pneumatic_valves": ["VUVG", "valve", "valf", "Y01", "Y02"],
            "pneumatic_frl": ["FRL", "filtre", "regulator", "MS6"],
            "pneumatic_connections": ["connection", "T", "junction", "bağlantı"],
            "flow_arrows": ["arrow", "ok", "→", "←"],
            "pressure_gauges": ["gauge", "manometer", "PI", "pressure"]
        },
        
        "extract_values": {
            "calisma_basinci": [r"(\d+(?:\.\d+)?[-\s]*\d*)\s*bar"],
            "vana_sayisi": [r"(?:VUVG|Y\d+|V\d+|SV\d+)", r"(?:5/2|4/2|3/2|2/2)", r"(?:valf|valve|vana)"],
            "silindir_sayisi": [r"(?:C\d+|CYL\d*|SIL\d*)", r"(?:silindir|cylinder|piston)", r"(?:MGF|actuator)"],
            "rl_mevcut": [r"(?:FRL|MS6-LFR|FILTRE|REGULATOR)"],
            "created_by": [r"Created\s+by[:\s]*([A-ZÇĞÜŞİÖ\s]+(?:[A-ZÇĞÜŞİÖ\s]*[A-ZÇĞÜŞİÖ]))",
            r"Oluşturan[:\s]*([A-ZÇĞÜŞİÖ\s]+(?:[A-ZÇĞÜŞİÖ\s]*[A-ZÇĞÜŞİÖ]))",
            r"Tasarlayan[:\s]*([A-ZÇĞÜŞİÖ\s]+(?:[A-ZÇĞÜŞİÖ\s]*[A-ZÇĞÜŞİÖ]))",
            r"Designer[:\s]*([A-ZÇĞÜŞİÖ\s]+(?:[A-ZÇĞÜŞİÖ\s]*[A-ZÇĞÜŞİÖ]))"],
            "checked_by": [r"Checked\s+by[:\s]*([A-ZÇĞÜŞİÖ\s]+(?:[A-ZÇĞÜŞİÖ\s]*[A-ZÇĞÜŞİÖ]))",
            r"Kontrol\s+eden[:\s]*([A-ZÇĞÜŞİÖ\s]+(?:[A-ZÇĞÜŞİÖ\s]*[A-ZÇĞÜŞİÖ]))"],
            "creation_date": [r"(\d{1,2}[./]\d{1,2}[./]\d{4})", r"(\d{4}-\d{1,2}-\d{1,2})"],
            "circuit_number": [r"Circuit\s+Number[:\s]*([A-Z0-9\-\.]+)",
            r"Devre\s+Numarası[:\s]*([A-Z0-9\-\.]+)"],
            "description": [r"Description[:\s]*([A-ZÇĞÜŞİÖ0-9\s]+(?:PNOMATİK|PNEUMATIC|DİYAGRAM|DIAGRAM)[A-ZÇĞÜŞİÖ0-9\s]*)",
            r"(PRES\s+PNOMATİK\s+DİYAGRAM\s*\d*)"],

        }

} 

# ============================================
# TODO: VALIDATION KEYWORDS - CRITICAL TERMS
# ============================================
# Format: {
#     "Kategori İsmi": ["kelime1", "kelime2", ...]
# }
# Eğer yoksa: {}

critical_terms = [  
        ["pnömatik", "pnomatik", "pneumatic", "hava", "air", "basınçlı hava", "compressed air"],
        ["silindir", "cylinder", "valf", "valve", "vana", "frl", "lubricator", "regulator", "filter"],
        ["basınç", "pressure", "psi", "bar", "debi", "flow", "cfm", "l/min"],
        ["kontrol", "control", "yön kontrol", "directional control", "hız kontrol", "speed control"],
        ["iso 5599", "5599", "iso 1219", "sembol", "symbol", "bağlantı", "connection", "port"]
]

# ============================================
# TODO: VALIDATION KEYWORDS - STRONG KEYWORDS
# ============================================
# Format: ["kelime1", "kelime2", ...]
# Eğer yoksa: []

strong_keywords = [
    "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve"
]

# ============================================
# TODO: VALIDATION KEYWORDS - EXCLUDED KEYWORDS
# ============================================
# Format: ["kelime1", "kelime2", ...]
# Eğer yoksa: []

excluded_keywords = [
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219",
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        "espe",
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide", "kılavuzu",
        "loto",
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        "montaj", "assembly",
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        "bakım", "maintenance", "servis", "service","bakim","MAINTENCE",
        "titreşim", "vibration","TİTREŞİM",
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