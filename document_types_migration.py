"""
Document Types Migration Script
================================
index.html ve main_api_gateway.py'den document type bilgilerini alıp
PostgreSQL document_types tablosuna aktarır.

Kullanım:
    python migrate_document_types.py
"""

from flask import Flask
from database import db, init_db, create_tables
from models import DocumentType
from datetime import datetime

# Flask app oluştur
app = Flask(__name__)

# Database'i başlat
init_db(app)

# Document types verileri (index.html + main_api_gateway.py birleşimi)
DOCUMENT_TYPES_DATA = [
    {
        'code': 'electric_circuit',
        'name': 'Elektrik Devre Şeması Analizi',
        'description': 'Elektrik devre şemalarının güvenlik ve uyumluluk analizi',
        'service_file': 'elektrik_service.py',
        'endpoint': '/api/elektrik-report',
        'icon': '🔌'
    },
    {
        'code': 'espe_report',
        'name': 'ESPE Raporu Analizi',
        'description': 'ESPE (Elektro-Sensitif Koruma Ekipmanı) rapor analizi',
        'service_file': 'espe_service.py',
        'endpoint': '/api/espe-report',
        'icon': '📋'
    },
    {
        'code': 'noise_report',
        'name': 'Gürültü Ölçüm Raporu Analizi',
        'description': 'İş yeri gürültü ölçüm raporlarının analizi',
        'service_file': 'gurultu_service.py',
        'endpoint': '/api/noise-report',
        'icon': '🔊'
    },
    {
        'code': 'manuel_report',
        'name': 'Manuel/Kullanım Kılavuzu Analizi',
        'description': 'Makine kullanım kılavuzlarının güvenlik analizi',
        'service_file': 'manuel_service.py',
        'endpoint': '/api/manuel-report',
        'icon': '📖'
    },
    {
        'code': 'loto_report',
        'name': 'LOTO Raporu Analizi',
        'description': 'Lockout/Tagout prosedürlerinin analizi',
        'service_file': 'loto_service.py',
        'endpoint': '/api/loto-report',
        'icon': '🔒'
    },
    {
        'code': 'lvd_report',
        'name': 'LVD Raporu Analizi',
        'description': 'Alçak Gerilim Direktifi uyumluluk raporu analizi',
        'service_file': 'lvd_service.py',
        'endpoint': '/api/lvd-report',
        'icon': '⚡'
    },
    {
        'code': 'at_declaration',
        'name': 'AT Tip Muayene Analizi',
        'description': 'AT Uygunluk Beyanı belgesi analizi',
        'service_file': 'at_declaration_service.py',
        'endpoint': '/api/at-declaration',
        'icon': '🔍'
    },
    {
        'code': 'isg_periodic_control',
        'name': 'İSG Periyodik Kontrol Analizi',
        'description': 'İş Sağlığı ve Güvenliği periyodik kontrol raporu analizi',
        'service_file': 'isg_service.py',
        'endpoint': '/api/isg-control',
        'icon': '🛡️'
    },
    {
        'code': 'pneumatic_circuit',
        'name': 'Pnömatik Devre Şeması Analizi',
        'description': 'Pnömatik sistemlerin güvenlik ve uyumluluk analizi',
        'service_file': 'pnomatic_service.py',
        'endpoint': '/api/pnomatic-control',
        'icon': '💨'
    },
    {
        'code': 'hydraulic_circuit',
        'name': 'Hidrolik Devre Şeması Analizi',
        'description': 'Hidrolik sistemlerin güvenlik ve uyumluluk analizi',
        'service_file': 'hidrolik_service.py',
        'endpoint': '/api/hydraulic-control',
        'icon': '🔧'
    },
    {
        'code': 'assembly_instructions',
        'name': 'Montaj Talimatları Analizi',
        'description': 'Makine montaj talimatlarının güvenlik analizi',
        'service_file': 'montaj_service.py',
        'endpoint': '/api/assembly-instructions',
        'icon': '🔨'
    },
    {
        'code': 'grounding_report',
        'name': 'EN 60204-1 Topraklama Raporu Analizi',
        'description': 'Elektrik topraklama ve süreklilik ölçüm raporu analizi',
        'service_file': 'topraklama_service.py',
        'endpoint': '/api/topraklama-report',
        'icon': '🌍'
    },
    {
        'code': 'hrc_report',
        'name': 'HRC Kuvvet-Basınç Raporu Analizi',
        'description': 'Human Robot Collaboration kuvvet ve basınç ölçüm analizi',
        'service_file': 'hrc_service.py',
        'endpoint': '/api/hrc-report',
        'icon': '🤖'
    },
    {
        'code': 'maintenance_instructions',
        'name': 'Bakım Talimatları Analizi',
        'description': 'Makine bakım talimatlarının güvenlik analizi',
        'service_file': 'bakim_service.py',
        'endpoint': '/api/bakimtalimatlari-report',
        'icon': '🔧'
    },
    {
        'code': 'vibration_report',
        'name': 'Mekanik Titreşim Raporu Analizi',
        'description': 'İş yeri mekanik titreşim ölçüm raporu analizi',
        'service_file': 'titresim_service.py',
        'endpoint': '/api/titresim-report',
        'icon': '📳'
    },
    {
        'code': 'lighting_report',
        'name': 'Aydınlatma Raporu Analizi',
        'description': 'İş yeri aydınlatma ölçüm raporu analizi',
        'service_file': 'aydinlatma_service.py',
        'endpoint': '/api/aydinlatma-report',
        'icon': '💡'
    },
    {
        'code': 'at_type_report',
        'name': 'AT Tip İnceleme Sertifikası Analizi',
        'description': 'AT Tip İnceleme Sertifikası belgesi analizi',
        'service_file': 'at_tip_service.py',
        'endpoint': '/api/at-type-cert-report',
        'icon': '📜'
    }
]


def migrate_document_types():
    """Document types'ları database'e aktar"""
    
    with app.app_context():
        print("=" * 70)
        print("📋 DOCUMENT TYPES MIGRATION")
        print("=" * 70)
        
        # Önce mevcut document type sayısını kontrol et
        existing_count = DocumentType.query.count()
        print(f"\n📊 Mevcut document type sayısı: {existing_count}")
        
        if existing_count > 0:
            response = input("\n⚠️  Mevcut veriler var! Devam edilsin mi? (y/n): ")
            if response.lower() != 'y':
                print("❌ İşlem iptal edildi.")
                return
        
        print(f"\n🚀 {len(DOCUMENT_TYPES_DATA)} document type aktarılıyor...\n")
        
        success_count = 0
        update_count = 0
        error_count = 0
        
        for data in DOCUMENT_TYPES_DATA:
            try:
                # Mevcut kaydı kontrol et
                existing = DocumentType.query.filter_by(code=data['code']).first()
                
                if existing:
                    # Güncelle
                    existing.name = data['name']
                    existing.description = data['description']
                    existing.service_file = data['service_file']
                    existing.endpoint = data['endpoint']
                    existing.icon = data['icon']
                    existing.is_active = True
                    existing.updated_at = datetime.utcnow()
                    
                    update_count += 1
                    print(f"🔄 GÜNCELLENDİ: {data['code']}")
                else:
                    # Yeni ekle
                    doc_type = DocumentType(
                        code=data['code'],
                        name=data['name'],
                        description=data['description'],
                        service_file=data['service_file'],
                        endpoint=data['endpoint'],
                        icon=data['icon'],
                        is_active=True
                    )
                    db.session.add(doc_type)
                    
                    success_count += 1
                    print(f"✅ EKLENDİ: {data['code']}")
                
            except Exception as e:
                error_count += 1
                print(f"❌ HATA ({data['code']}): {str(e)}")
        
        # Değişiklikleri kaydet
        try:
            db.session.commit()
            print("\n" + "=" * 70)
            print("📊 ÖZET")
            print("=" * 70)
            print(f"✅ Yeni eklenen: {success_count}")
            print(f"🔄 Güncellenen: {update_count}")
            print(f"❌ Hata: {error_count}")
            print(f"📋 Toplam: {success_count + update_count}")
            print("=" * 70)
            print("\n✨ Migration tamamlandı!")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Database commit hatası: {str(e)}")


def verify_migration():
    """Migration'ı doğrula"""
    
    with app.app_context():
        print("\n" + "=" * 70)
        print("🔍 DOĞRULAMA")
        print("=" * 70)
        
        doc_types = DocumentType.query.filter_by(is_active=True).order_by(DocumentType.code).all()
        
        print(f"\n📊 Aktif document type sayısı: {len(doc_types)}\n")
        
        for dt in doc_types:
            print(f"{dt.icon} {dt.code}")
            print(f"   📝 İsim: {dt.name}")
            print(f"   📄 Dosya: {dt.service_file}")
            print(f"   🔗 Endpoint: {dt.endpoint}")
            print()


if __name__ == '__main__':
    print("\n🚀 Document Types Migration Script\n")
    
    # Migration'ı çalıştır
    migrate_document_types()
    
    # Doğrulama yap
    verify_migration()
    
    print("\n✅ İşlem tamamlandı!\n")