"""
PATTERN GÜNCELLEME TEMPLATE
============================
Herhangi bir servis için pattern_definitions tablosunu günceller

KULLANIM:
1. DOCUMENT_TYPE_CODE'u ayarlayın (örn: 'noise_report', 'vibration_report')
2. UPDATED_PATTERNS dictionary'sine güncellenecek pattern'leri ekleyin
3. python update_patterns_template.py çalıştırın

ÖNEMLİ:
- Sadece UPDATED_PATTERNS içindeki field'lar güncellenir
- Diğer pattern'lere dokunulmaz
- Eğer field yoksa yeni eklenir
"""

from flask import Flask
from database import db, init_db
from models import DocumentType, PatternDefinition

app = Flask(__name__)
init_db(app)

# ============================================
# TODO: SERVİS KODUNU AYARLAYIN
# ============================================
DOCUMENT_TYPE_CODE = 'isg_periodic_control'  # TODO: Değiştirin (örn: 'vibration_report', 'electric_circuit')

# ============================================
# TODO: GÜNCELLENECEK PATTERN'LERİ TANIMLAYIN
# ============================================
# Format:
# {
#     "pattern_group_name": {
#         "field_name": [pattern1, pattern2, ...]
#     }
# }

UPDATED_PATTERNS = {
    "approval_patterns": {
            "uygun": [r"(?:uygun|UYGUN|✓|√|✔|☑|☒|v|V|c|C|onaylandı|kabul|geçer|ok|OK|var|mevcut|tamam|yapıldı|kontrol.*edildi)"],
            "uygun_degil": [r"(?:uygun değil|UYGUN DEĞİL|degil|DEGIL|✗|✘|×|❌|x|X|red|yetersiz|eksik|yok|yapılmadı|kontrol.*edilmedi)"],
            "not_var": [r"(?:not|açıklama|dipnot|özel durum|NOT|gözlem|dikkat|uyarı)"]
    }
}


# ============================================
# GÜNCELLEME FONKSİYONU (DOKUNMAYIN!)
# ============================================
def update_patterns():
    """Pattern'leri güncelle veya ekle"""
    
    print("=" * 80)
    print(f"PATTERN GÜNCELLEME - {DOCUMENT_TYPE_CODE.upper()}")
    print("=" * 80)
    
    try:
        with app.app_context():
            # 1. Document type kontrolü
            doc_type = DocumentType.query.filter_by(code=DOCUMENT_TYPE_CODE).first()
            
            if not doc_type:
                print(f"\n❌ HATA: '{DOCUMENT_TYPE_CODE}' document type bulunamadı!")
                print(f"   Veritabanında böyle bir servis yok.")
                print(f"   Önce migration template'i çalıştırın veya kodu kontrol edin.")
                return False
            
            print(f"\n✅ Servis bulundu:")
            print(f"   📋 Code: {doc_type.code}")
            print(f"   📝 Name: {doc_type.name}")
            print(f"   🆔 ID: {doc_type.id}")
            
            if not UPDATED_PATTERNS:
                print(f"\n⚠️  UYARI: UPDATED_PATTERNS boş! Güncellenecek bir şey yok.")
                return False
            
            # 2. Her pattern group için işlem yap
            updated_count = 0
            added_count = 0
            
            for pattern_group, fields_dict in UPDATED_PATTERNS.items():
                print(f"\n📂 Pattern Group: {pattern_group}")
                
                for field_name, new_patterns in fields_dict.items():
                    # Mevcut pattern'i bul
                    existing = PatternDefinition.query.filter_by(
                        document_type_id=doc_type.id,
                        pattern_group=pattern_group,
                        field_name=field_name
                    ).first()
                    
                    if existing:
                        # Güncelle
                        old_patterns = existing.patterns.copy() if existing.patterns else []
                        existing.patterns = new_patterns
                        
                        print(f"   🔄 Güncelleniyor: {field_name}")
                        print(f"      📜 Eski pattern sayısı: {len(old_patterns)}")
                        print(f"      🆕 Yeni pattern sayısı: {len(new_patterns)}")
                        
                        # Pattern değişikliğini göster
                        if old_patterns != new_patterns:
                            print(f"      ✏️  Pattern değişti:")
                            for i, (old, new) in enumerate(zip(old_patterns, new_patterns), 1):
                                if old != new:
                                    print(f"         Pattern {i}:")
                                    print(f"            Eski: {old[:60]}...")
                                    print(f"            Yeni: {new[:60]}...")
                        
                        updated_count += 1
                    else:
                        # Yoksa yeni ekle - ONAY İSTE
                        print(f"   ⚠️  UYARI: {field_name} mevcut değil!")
                        print(f"      📝 Eklenecek pattern sayısı: {len(new_patterns)}")
                        print(f"      📋 Pattern'ler:")
                        for i, p in enumerate(new_patterns, 1):
                            print(f"         {i}. {p[:80]}...")
                        
                        response = input(f"\n   ❓ Bu field'ı eklemek istiyor musunuz? (e/h): ").strip().lower()
                        
                        if response in ['e', 'evet', 'yes', 'y']:
                            new_pd = PatternDefinition(
                                document_type_id=doc_type.id,
                                pattern_group=pattern_group,
                                field_name=field_name,
                                patterns=new_patterns,
                                display_order=1
                            )
                            db.session.add(new_pd)
                            
                            print(f"   ✅ Eklendi: {field_name}")
                            added_count += 1
                        else:
                            print(f"   ⏭️  Atlandı: {field_name}")
                            print(f"      Bu field eklenmeyecek.")
            
            # 3. Kaydet
            db.session.commit()
            
            print("\n" + "=" * 80)
            print("✅ GÜNCELLEME BAŞARILI!")
            print("=" * 80)
            print(f"\n📊 Özet:")
            print(f"   🔄 Güncellenen: {updated_count}")
            print(f"   ➕ Yeni eklenen: {added_count}")
            print(f"   📦 Toplam işlem: {updated_count + added_count}")
            print(f"\n🔍 Servisi yeniden başlatın ve test edin.")
            
            return True
            
    except Exception as e:
        print(f"\n❌ HATA OLUŞTU:")
        print(f"   {type(e).__name__}: {str(e)}")
        print(f"\n📋 Olası nedenler:")
        print(f"   1. Veritabanı bağlantısı yok")
        print(f"   2. Tablo yapısı yanlış")
        print(f"   3. Pattern formatı hatalı")
        return False


if __name__ == '__main__':
    success = update_patterns()
    
    if not success:
        print(f"\n⚠️  Güncelleme yapılamadı!")
        print(f"   Yukarıdaki hata mesajlarını kontrol edin.")
        exit(1)
    else:
        print(f"\n✅ İşlem tamamlandı!")
        exit(0)