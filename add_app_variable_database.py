"""
Add app_variable_name Column Migration
=======================================
document_types tablosuna app_variable_name kolonu ekler ve mevcut verileri günceller.

Kullanım:
    python add_app_variable_name.py
"""

from flask import Flask
from database import db, init_db
from models import DocumentType
from sqlalchemy import text

# Flask app oluştur
app = Flask(__name__)
init_db(app)

# Service mapping - hangi code hangi app variable'ını kullanıyor
SERVICE_APP_MAPPING = {
    'vibration_report': 'titresim_app',
    'electric_circuit': 'elektrik_app',
    'espe_report': 'espe_app',
    'noise_report': 'gurultu_app',
    'manuel_report': 'manuel_app',
    'loto_report': 'loto_app',
    'at_declaration': 'at_declaration_app',
    'lvd_report': 'lvd_app',
    'lighting_report': 'aydinlatma_app',
    'isg_periodic_control': 'isg_app',
    'pneumatic_circuit': 'pnomatic_app',
    'hydraulic_circuit': 'hidrolik_app',
    'assembly_instructions': 'montaj_app',
    'hrc_report': 'hrc_app',
    'maintenance_instructions': 'bakim_app',
    'at_type_report': 'at_tip_app',
    'grounding_report': 'topraklama_app'
}


def add_column():
    """app_variable_name kolonunu ekle"""
    
    with app.app_context():
        print("=" * 70)
        print("🔧 APP VARIABLE NAME COLUMN MIGRATION")
        print("=" * 70)
        
        try:
            # Kolon var mı kontrol et
            result = db.session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='document_types' 
                AND column_name='app_variable_name';
            """))
            
            if result.fetchone():
                print("\n✅ app_variable_name kolonu zaten mevcut!")
            else:
                # Kolonu ekle
                db.session.execute(text("""
                    ALTER TABLE document_types 
                    ADD COLUMN app_variable_name VARCHAR(100);
                """))
                db.session.commit()
                print("\n✅ app_variable_name kolonu eklendi!")
                
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Kolon ekleme hatası: {str(e)}")
            return False
        
        return True


def update_existing_data():
    """Mevcut kayıtları güncelle"""
    
    with app.app_context():
        print("\n" + "=" * 70)
        print("📝 MEVCUT VERİLERİ GÜNCELLEME")
        print("=" * 70 + "\n")
        
        success_count = 0
        error_count = 0
        
        for code, app_var in SERVICE_APP_MAPPING.items():
            try:
                doc_type = DocumentType.query.filter_by(code=code).first()
                
                if doc_type:
                    doc_type.app_variable_name = app_var
                    success_count += 1
                    print(f"✅ {code} → {app_var}")
                else:
                    print(f"⚠️  {code} bulunamadı (database'de yok)")
                    
            except Exception as e:
                error_count += 1
                print(f"❌ {code} güncellenemedi: {str(e)}")
        
        # Commit
        try:
            db.session.commit()
            print("\n" + "=" * 70)
            print(f"✅ Güncelleme başarılı: {success_count} kayıt")
            print(f"❌ Hata: {error_count} kayıt")
            print("=" * 70)
            
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Commit hatası: {str(e)}")


def verify_migration():
    """Migration'ı doğrula"""
    
    with app.app_context():
        print("\n" + "=" * 70)
        print("🔍 DOĞRULAMA")
        print("=" * 70 + "\n")
        
        doc_types = DocumentType.query.filter_by(is_active=True).order_by(DocumentType.code).all()
        
        print(f"Toplam aktif document type: {len(doc_types)}\n")
        
        missing_count = 0
        for dt in doc_types:
            if dt.app_variable_name:
                print(f"✅ {dt.code:<25} → {dt.app_variable_name}")
            else:
                print(f"⚠️  {dt.code:<25} → (BOŞ)")
                missing_count += 1
        
        print("\n" + "=" * 70)
        if missing_count == 0:
            print("✅ Tüm kayıtlar tamam!")
        else:
            print(f"⚠️  {missing_count} kayıtta app_variable_name eksik!")
        print("=" * 70)


if __name__ == '__main__':
    print("\n🚀 App Variable Name Migration Script\n")
    
    # 1. Kolonu ekle
    if add_column():
        # 2. Verileri güncelle
        update_existing_data()
        
        # 3. Doğrula
        verify_migration()
    
    print("\n✅ Migration tamamlandı!\n")