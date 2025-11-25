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
DOCUMENT_TYPE_CODE = 'at_declaration'  # TODO: Değiştir (örn: 'electric_circuit', 'espe_report')
DOCUMENT_TYPE_NAME = 'AT Tip Muayene Analizi'  # TODO: Değiştir
DOCUMENT_TYPE_DESCRIPTION = 'AT Tip Muayene raporlarının analizi'  # TODO: Değiştir
SERVICE_FILE = 'at_declaration_service.py'  # TODO: Değiştir
ENDPOINT = '/api/at-declaration'  # TODO: Değiştir
ICON = '🔍'  # TODO: Değiştir
# ============================================
# TODO: CRITERIA WEIGHTS (Kategoriler + Puanlar)
# ============================================
# Format: {"Kategori Adı": puan}
# Örnek: {"Rapor Kimlik Bilgileri": 10, "Genel Bilgiler": 15}
# Eğer yoksa: {}

criteria_weights_data = {
            "Kritik Bilgiler": 60,
            "Zorunlu Teknik Bilgiler": 25,
            "Standartlar ve Belgeler": 15
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
            "Kritik Bilgiler": {
                "uretici_adi": {
                    "pattern": r"(?:biz\s+burada\s+beyan\s+ederiz\s+ki[;:\s]*([^,\n]+))|(?:üretici|manufacturer|imalatçı|company|şirket|firma|unvan|we|manufactured by|sibernetik|pilz|tarafımızdan|üretici\s+firma)[\s:]*([A-Za-zÇŞİĞÜÖıçşığüö\s\.\-&]{8,100})|(?:karaca\s+mekatronik)",
                    "weight": 15,
                    "critical": True,
                    "description": "Üretici veya yetkili temsilcinin adı"
                },
                "uretici_adres": {
                    "pattern": r"(?:adres|address|cd\.\s*no|street|road|mahallesi|caddesi|sokak)[\s:]*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{15,200})|(?:demirci[^,\n]*nilüfer[^,\n]*bursa)|(?:cork[^,\n]*ireland)",
                    "weight": 15,
                    "critical": True,
                    "description": "Üretici veya yetkili temsilcinin adresi"
                },
                "makine_tanimi": {
                    "pattern": r"(?:makinenin tanıtımı|tanım|machine|makine|model|tip|type|description)[\s:]*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\-\.]{5,100})|(?:ecotorq|kafa|baga|çakma|knee pad|punching|vibr)",
                    "weight": 15,
                    "critical": True,
                    "description": "Makine tanımı (tip, model, seri)"
                },
                "direktif_atif": {
                    "pattern": r"(?:2006/42|2006\/42|makine direktif|machine directive|machinery directive|EC|AT|directive|european directive|ab direktif)",
                    "weight": 10,
                    "critical": True,
                    "description": "2006/42/EC Direktif atfı"
                },
                "yetkili_imza": {
                    "pattern": r"(?:yetkili\s+imza|authorized|authorised|imza|signature|beyan yetkilisi|responsible|müdür|manager|director|managing director|şahiner|mcauliffe|genel müdür|beyan eden|sorumlu|name|adı|surname|soyadı|ünvan|position|title|başkan|president|chief|şef|general\s+manager|general\s+maneger|karaca|eşref)",
                    "weight": 5,
                    "critical": True,
                    "description": "Yetkili kişi imzası ve unvanı"
                }
            },
            "Zorunlu Teknik Bilgiler": {
                "uretim_yili": {
                    "pattern": r"(?:üretim|imal|manufacturing|production)[\s\w]*(?:yılı|year|date)[\s:]*([0-9]{4})|([0-9]{4})[\s]*(?:yılı|year)|february\s*([0-9]{4})|([0-9]{4})",
                    "weight": 5,
                    "critical": False,
                    "description": "Üretim yılı"
                },
                "seri_no": {
                    "pattern": r"(?:seri|serial|s/n|sn)[\s\w]*(?:no|number)[\s:]*([A-Za-z0-9\-]{2,20})|serial number[\s:]*([A-Za-z0-9\-]{2,20})",
                    "weight": 5,
                    "critical": False,
                    "description": "Seri numarası"
                },
                "beyan_ifadesi": {
                    "pattern": r"(?:beyan|declaration|conform|uygun|comply|uygunluk|conformity|declare|conformity with)",
                    "weight": 5,
                    "critical": False,
                    "description": "Uygunluk beyan ifadesi"
                },
                "tarih_yer": {
                    "pattern": r"(?:tarih|date|yer|place)[\s:]*([0-9]{1,2}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{2,4})|([0-9]{1,2}\s*february\s*[0-9]{4})|cork\s*ireland\s*([0-9]{1,2}\s*february\s*[0-9]{4})",
                    "weight": 5,
                    "critical": False,
                    "description": "Beyan tarihi ve yeri"
                },
                "diger_direktifler": {
                    "pattern": r"(?:2014/30|2014/35|EMC|LVD|alçak gerilim|low voltage|elektromanyetik|electromagnetic|european directive)",
                    "weight": 5,
                    "critical": False,
                    "description": "Diğer direktifler (EMC, LVD vb.)"
                }
            },
            "Standartlar ve Belgeler": {
                "uyumlu_standartlar": {
                    "pattern": r"(?:EN|ISO|IEC)[\s]*[0-9]{3,5}[\-:]*[0-9]*[:\-]*[0-9]*",
                    "weight": 8,
                    "critical": False,
                    "description": "Uygulanmış uyumlaştırılmış standartlar"
                },
                "teknik_dosya": {
                    "pattern": r"(?:teknik dosya|technical file|documentation|dokümantasyon)",
                    "weight": 4,
                    "critical": False,
                    "description": "Teknik dosya sorumlusu"
                },
                "onaylanmis_kurulus": {
                    "pattern": r"(?:onaylanmış kuruluş|notified body|tip inceleme|type examination|belge|certificate)",
                    "weight": 3,
                    "critical": False,
                    "description": "Onaylanmış kuruluş bilgileri"
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
        "extract_values": {
            "manufacturer_name":[
            r"(?:biz\s+burada\s+beyan\s+ederiz\s+ki[;:\s]*)([^,\n]+)",
            r"(?:we\s+)([A-Za-z\s&\.]+?)(?:\s+declare|\s+industrial)",
            r"(?:manufactured by|üretici|manufacturer)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö\s\.\-&]{5,100})",
            r"(sibernetik\s+makina\s*&?\s*otomasyon[^,\n]*)",
            r"(pilz\s+ireland\s+industrial\s+automation)",
            r"(suzhou\s+keber\s+technology\s+co)",
            r"([A-ZÜÇĞIÖŞ][a-züçğıöş]+(?:\s+[A-ZÜÇĞIÖŞ][a-züçğıöş]+)*\s+(?:makina|technology|industrial|automation|şirket|company))"],

            "manufacturer_address":[r"(demirci[^,\n]*cd\.[^,\n]*no[^,\n]*nilüfer[^,\n]*bursa)",
            r"(cork\s+business\s*&?\s*technology\s+park[^,]*model\s+farm\s+road[^,]*cork[^,]*ireland)",
            r"(no\.\s*[0-9]+[^,]*suzhou[^,]*jiangsu[^,]*)",
            r"(?:address|adres)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\.\-/,&]{20,200})",
            r"([A-ZÜÇĞIÖŞ][a-züçğıöş]+(?:\s+[A-Za-züçğıöş]+)*\s+(?:cd\.|caddesi|street|road)[^,\n]{10,100})",
            r"([^,\n]*(?:mahallesi|caddesi|sokak|street|road|park)[^,\n]{10,100})"],
                
            "machine_description": [r"(?:makinenin tanıtımı ve sınıfı|tanım|description)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\-\.]{5,100})",
            r"(fo\s*[0-9]+\.?[0-9]*lt?\s+ecotorq\s+kafa\s+baga\s+çakma)",
            r"(v[0-9]+b\s+knee\s+pad\s+punching\s+machine)",
            r"(vibratory\s+surface\s+finishing\s+machine)",
            r"(?:makine|machine|model|equipment)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö0-9\s\-\.]{8,80})"],

            "production_year":[r"([0-9]{4})",
            r"february\s+([0-9]{4})",
            r"(?:üretim|imal|year)\s*[:\-]?\s*([0-9]{4})"],

            "declaration_date":[r"(?:tarih|date)\s*[:\-]?\s*([0-9]{1,2}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{2,4})",
            r"([0-9]{1,2}[\.\/\-][0-9]{1,2}[\.\/\-][0-9]{2,4})"],

            "authorized_person":[r"(?:beyan yetkilisi|authorized|yetkili|name)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö\s]{5,50})",
            r"(?:adı soyadı|name)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö\s]{5,50})"],

            "position":[r"(?:ünvan|position|görevi|title)\s*[:\-]?\s*([A-Za-zÇŞİĞÜÖıçşığüö\s]{5,50})",
            r"(?:müdür|manager|director|president|başkan)"]

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
        # AT/EC temel terimleri
        ["AT TİP", "at tip", "ec type", "uygunluk", "beyan", "muayene", "conformity", "declaration"],
        
        # Sertifika ve belgelendirme terimleri
        ["SERTİFİKA", "sertifika", "certificate", "belge", "document", "onay", "approval"],
        
        # Makine direktifi ve standart terimleri
        ["2006/42/EC", "direktif", "directive", "makine", "machine", "standart", "standard"],
        
        # Üretici ve yetkili terimleri
        ["üretici", "manufacturer", "yetkili", "authorized", "imza", "signature", "sorumlu", "responsible"],
        
        # Muayene ve kontrol terimleri
        ["muayene", "inspection", "kontrol", "control", "test", "değerlendirme", "assessment", "onaylanmış kuruluş"]
]

# ============================================
# TODO: VALIDATION KEYWORDS - STRONG KEYWORDS
# ============================================
# Format: ["kelime1", "kelime2", ...]
# Eğer yoksa: []

strong_keywords = [
       "uygunluk", "beyan", "declaration","muayene","declare"
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
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219", "teknik resim", "tasarım",
        
        # Pnömatik devre şeması
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",

        # Gürültü ölçüm raporu
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        
        # İSG periyodik kontrol
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        
        # HRC raporu
        "hrc", "cobot", "robot", "çarpışma", "collaborative", "kolaboratif", "sd conta",
        
        # Elektrik devre şeması
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm","enclosure","wrp-","light curtain","contactors","controller",
        
        # Espe raporu  
        "espe",
        
        # Manuel/kullanma kılavuzu
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide","kılavuzu",
        
        # LOTO raporu
        "loto",
        
        # LVD raporu
        "lvd", "TOPRAKLAMA SÜREKLİLİK",  "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        
        # Montaj talimatları
        "montaj", "assembly",
        
        # EN 60204-1 topraklama raporu
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama","TOPRAKLAMA DİRENCİ",
        
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