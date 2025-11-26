"""
SERVİS MIGRATION TEST SCRIPTI - OTOMATİK KONTROL

KULLANIM:
1. migrate_service_template.py'deki SERVICE_CODE ile aynı olmalı
2. Migration sonrası bu scripti çalıştırın
3. Eksik/fazla yüklemeleri göreceksiniz
"""

from flask import Flask
from database import db, init_db
from models import DocumentType, CriteriaWeight, CriteriaDetail, PatternDefinition, ValidationKeyword, CategoryAction

app = Flask(__name__)
init_db(app)

# ============================================
# TODO: AYNI SERVİS KODUNU BURAYA DA YAZIN
# ============================================
SERVICE_CODE = 'pneumatic_circuit'  # TODO: migrate_service_template.py ile aynı olmalı!

print("\n" + "=" * 80)
print(f"{SERVICE_CODE.upper()} - MİGRATION KONTROL")
print("=" * 80)

with app.app_context():
    # Document Type Bul
    doc_type = DocumentType.query.filter_by(code=SERVICE_CODE).first()
    
    if not doc_type:
        print(f"\n❌ SERVİS BULUNAMADI: {SERVICE_CODE}")
        print("Migration yapılmamış! Önce migrate_service_template.py çalıştırın!")
        exit(1)
    
    print(f"\n✅ Servis Bulundu")
    print(f"   - Adı: {doc_type.name}")
    print(f"   - Code: {doc_type.code}")
    print(f"   - Dosya: {doc_type.service_file}")
    print(f"   - Endpoint: {doc_type.endpoint}")
    print(f"   - Icon: {doc_type.icon}")
    
    print("\n" + "=" * 80)
    print("YÜKLEME KONTROL RAPORU")
    print("=" * 80)
    
    issues = []
    warnings = []
    
    # ============================================
    # 1. CRITERIA WEIGHTS KONTROL
    # ============================================
    criteria_weights = CriteriaWeight.query.filter_by(document_type_id=doc_type.id).all()
    print(f"\n📊 1. Criteria Weights")
    
    if len(criteria_weights) == 0:
        print(f"   ❌ HİÇ KATEGORİ YÜKLENMEMİŞ!")
        issues.append("Criteria Weights boş")
    elif len(criteria_weights) < 3:
        print(f"   ⚠️  Sadece {len(criteria_weights)} kategori var (Az olabilir)")
        warnings.append(f"Criteria Weights: {len(criteria_weights)} (normalden az)")
    else:
        print(f"   ✅ {len(criteria_weights)} kategori yüklendi")
    
    for cw in criteria_weights:
        print(f"      • {cw.category_name}: {cw.weight} puan")
    
    # ============================================
    # 2. CRITERIA DETAILS KONTROL
    # ============================================
    print(f"\n📋 2. Criteria Details")
    total_details = 0
    empty_categories = []
    
    for cw in criteria_weights:
        details = CriteriaDetail.query.filter_by(criteria_weight_id=cw.id).all()
        total_details += len(details)
        if len(details) == 0:
            empty_categories.append(cw.category_name)
    
    if total_details == 0:
        print(f"   ❌ HİÇ KRİTER YÜKLENMEMİŞ!")
        issues.append("Criteria Details boş")
    elif empty_categories:
        print(f"   ⚠️  {total_details} kriter yüklendi AMA bazı kategoriler boş:")
        for cat in empty_categories:
            print(f"      ⚠️  '{cat}' kategorisinde kriter YOK")
        warnings.append(f"{len(empty_categories)} kategori boş")
    else:
        print(f"   ✅ {total_details} kriter yüklendi")
    
    # ============================================
    # 3. PATTERN DEFINITIONS KONTROL
    # ============================================
    print(f"\n🔍 3. Pattern Definitions (extract_specific_values)")
    patterns = PatternDefinition.query.filter_by(document_type_id=doc_type.id).all()

    if len(patterns) == 0:
        print(f"   ⚠️  Pattern yok (Bazı servislerde olmayabilir)")
        warnings.append("Pattern Definitions yok")
    else:
        # Pattern gruplarına göre grupla
        pattern_groups = {}
        for pattern in patterns:
            if pattern.pattern_group not in pattern_groups:
                pattern_groups[pattern.pattern_group] = []
            pattern_groups[pattern.pattern_group].append(pattern)
        
        total_patterns = sum(len(p.patterns) for p in patterns)
        print(f"   ✅ {len(pattern_groups)} pattern grubu, {len(patterns)} field, {total_patterns} pattern yüklendi")
        
        # Her grubu göster
        for group_name, group_patterns in pattern_groups.items():
            group_pattern_count = sum(len(p.patterns) for p in group_patterns)
            print(f"      📁 {group_name}: {len(group_patterns)} field, {group_pattern_count} pattern")
            
            # Her field'ı göster
            for pattern in group_patterns:
                print(f"         └─ {pattern.field_name}: {len(pattern.patterns)} pattern")
                # İlk pattern'in önizlemesi (çok uzunsa kısalt)
                if pattern.patterns:
                    first_pattern = pattern.patterns[0]
                    if len(first_pattern) > 60:
                        print(f"            └─ {first_pattern[:60]}...")
                    else:
                        print(f"            └─ {first_pattern}")
        
    # ============================================
    # 4. CRITICAL TERMS KONTROL
    # ============================================
    print(f"\n🔑 4. Critical Terms (Validasyon)")
    critical_terms = ValidationKeyword.query.filter_by(
        document_type_id=doc_type.id,
        keyword_type='critical_terms'
    ).all()
    
    if len(critical_terms) == 0:
        print(f"   ❌ CRITICAL TERMS YOK! (Döküman validasyonu çalışmaz)")
        issues.append("Critical Terms yok - Validasyon çalışmayacak")
    else:
        print(f"   ✅ {len(critical_terms)} kategori yüklendi")
        total_keywords = 0
        for vk in critical_terms:
            keyword_count = len(vk.keywords)
            total_keywords += keyword_count
            print(f"      • {vk.category}: {keyword_count} kelime")
        
        if total_keywords < 10:
            print(f"   ⚠️  Toplam {total_keywords} kelime (Az olabilir)")
            warnings.append(f"Critical Terms: {total_keywords} kelime (az)")
    
    # ============================================
    # 5. STRONG KEYWORDS KONTROL
    # ============================================
    print(f"\n🔑 5. Strong Keywords (İlk Sayfa OCR)")
    strong = ValidationKeyword.query.filter_by(
        document_type_id=doc_type.id,
        keyword_type='strong_keywords'
    ).first()
    
    if not strong:
        print(f"   ⚠️  Strong Keywords YOK (OCR validasyonu atlanacak)")
        warnings.append("Strong Keywords yok")
    else:
        keyword_count = len(strong.keywords)
        if keyword_count < 5:
            print(f"   ⚠️  Sadece {keyword_count} kelime var (Az olabilir)")
            warnings.append(f"Strong Keywords: {keyword_count} (az)")
        else:
            print(f"   ✅ {keyword_count} kelime yüklendi")
        print(f"      İlk 10: {strong.keywords[:10]}")
    
    # ============================================
    # 6. EXCLUDED KEYWORDS KONTROL
    # ============================================
    print(f"\n🔑 6. Excluded Keywords (Yanlış Döküman Tespiti)")
    excluded = ValidationKeyword.query.filter_by(
        document_type_id=doc_type.id,
        keyword_type='excluded_keywords'
    ).first()
    
    if not excluded:
        print(f"   ⚠️  Excluded Keywords YOK (Yanlış döküman kontrolü atlanacak)")
        warnings.append("Excluded Keywords yok")
    else:
        keyword_count = len(excluded.keywords)
        if keyword_count < 10:
            print(f"   ⚠️  Sadece {keyword_count} kelime var (Az olabilir)")
            warnings.append(f"Excluded Keywords: {keyword_count} (az)")
        else:
            print(f"   ✅ {keyword_count} kelime yüklendi")
        print(f"      İlk 10: {excluded.keywords[:10]}")
    
    # ============================================
    # 7. CATEGORY ACTIONS KONTROL
    # ============================================
    print(f"\n💡 7. Category Actions (İyileştirme Önerileri)")
    total_actions = 0
    categories_with_actions = []
    
    for cw in criteria_weights:
        actions = CategoryAction.query.filter_by(criteria_weight_id=cw.id).all()
        if len(actions) > 0:
            total_actions += len(actions)
            categories_with_actions.append(cw.category_name)
    
    if total_actions == 0:
        print(f"   ⚠️  Category Actions YOK (Bazı servislerde olmayabilir)")
        warnings.append("Category Actions yok")
    else:
        print(f"   ✅ {total_actions} öneri yüklendi ({len(categories_with_actions)} kategori)")
        for cat in categories_with_actions[:5]:  # İlk 5 kategori
            print(f"      • {cat}")
    
    # ============================================
    # SONUÇ RAPORU
    # ============================================
    print("\n" + "=" * 80)
    print("SONUÇ RAPORU")
    print("=" * 80)
    
    print(f"\n📊 Yükleme İstatistikleri:")
    print(f"   - Criteria Weights: {len(criteria_weights)}")
    print(f"   - Criteria Details: {total_details}")
    print(f"   - Pattern Definitions: {len(patterns)}")
    print(f"   - Critical Terms Kategorileri: {len(critical_terms)}")
    print(f"   - Strong Keywords: {'VAR' if strong else 'YOK'} ({len(strong.keywords) if strong else 0})")
    print(f"   - Excluded Keywords: {'VAR' if excluded else 'YOK'} ({len(excluded.keywords) if excluded else 0})")
    print(f"   - Category Actions: {total_actions}")
    
    print(f"\n🚨 Kritik Sorunlar: {len(issues)}")
    if issues:
        for issue in issues:
            print(f"   ❌ {issue}")
    else:
        print(f"   ✅ Kritik sorun yok")
    
    print(f"\n⚠️  Uyarılar: {len(warnings)}")
    if warnings:
        for warning in warnings:
            print(f"   ⚠️  {warning}")
    else:
        print(f"   ✅ Uyarı yok")
    
    # GENEL DURUM
    print("\n" + "=" * 80)
    if len(issues) > 0:
        print("❌ MİGRATION BAŞARISIZ - Kritik sorunlar var!")
        print("=" * 80)
        exit(1)
    elif len(warnings) > 3:
        print("⚠️  MİGRATION TAMAMLANDI AMA ÇOK UYARI VAR")
        print("=" * 80)
    else:
        print("✅ MİGRATION BAŞARILI!")
        print("=" * 80)
    print()