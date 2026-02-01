# PILZ Report Checker API Gateway

Bu proje, farklı türdeki raporları analiz etmek için birleştirilmiş bir API gateway sistemidir. Tek bir API üzerinden 18 farklı analiz servisine erişim sağlar.

## Özellikler

- **Tek API Noktası**: Tüm analiz servisleri tek bir API üzerinden erişilebilir
- **Otomatik Servis Yönetimi**: Ana API başlatıldığında tüm alt servisler otomatik olarak başlatılır
- **Akıllı Yönlendirme**: Belge tipine göre doğru analiz servisine otomatik yönlendirme
- **Sistem Durumu İzleme**: Tüm servislerin sağlık durumunu kontrol etme
- **Hata Yönetimi**: Kapsamlı hata yakalama ve kullanıcı dostu mesajlar

## Desteklenen Belge Türleri

| Belge Türü | Port | Açıklama |
|------------|------|----------|
| `electric_circuit` | 5002 | Elektrik Devre Şeması Analizi |
| `espe_report` | 5003 | ESPE Raporu Analizi |
| `hydraulic_report` | 5004 | Hidrolik Devre Şeması Analizi |
| `noise_report` | 5005 | Gürültü Ölçüm Raporu Analizi |
| `manuel_report` | 5006 | Manuel Raporu Analizi |
| `loto_report` | 5007 | LOTO Raporu Analizi |
| `lvd_report` | 5008 | LVD Raporu Analizi |
| `at_type_inspection` | 5009 | AT Tip Muayene Analizi |
| `isg_periyodik_kontrol` | 5010 | İSG Periyodik Kontrol Analizi |
| `pneumatic_circuit` | 5011 | Pnömatik Devre Şeması Analizi |
| `hydraulic_circuit` | 5012 | Hidrolik Devre Şeması Analizi |
| `assembly_instructions` | 5013 | Montaj Talimatları Analizi |
| `grounding_report` | 5014 | EN 60204-1 Topraklama Raporu Analizi |
| `hrc_report` | 5015 | HRC Kuvvet-Basınç Raporu Analizi |
| `maintenance_instructions` | 5016 | Bakım Talimatları Analizi |
| `vibration_report` | 5017 | Mekanik Titreşim Raporu Analizi |
| `lighting_report` | 5018 | Aydınlatma Raporu Analizi |
| `at_certificate` | 5019 | AT Tip İnceleme Sertifikası Analizi |

## Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.8+
- Virtual environment (venv)
- Tüm bağımlılıkların yüklü olması (`requirements.txt`)

### Hızlı Başlatma (Önerilen)

**Shell Script ile tek komutla başlatma:**
```bash
./start_api_gateway.sh
```

### Manuel Başlatma

1. **Virtual environment'ı aktif edin:**
   ```bash
   source venv/bin/activate
   ```

2. **Ana API Gateway'i başlatın:**
   ```bash
   python3 main_api_gateway.py
   ```

3. **Sistem hazır!** Ana API http://localhost:5001 adresinde çalışmaya başlar ve otomatik olarak tüm alt servisleri başlatır.

## Web Arayüzü

### 🌐 Grafik Kullanıcı Arayüzü

Tarayıcınızdan `http://localhost:5001` adresine giderek modern web arayüzünü kullanabilirsiniz:

**Özellikler:**
- 📁 **Sürükle & Bırak**: Dosyaları kolayca yükleyin
- 🎨 **Modern Tasarım**: Responsive ve kullanıcı dostu arayüz
- ⚡ **Gerçek Zamanlı**: Analiz durumunu canlı izleyin
- 📊 **Sonuç Görüntüleme**: JSON sonuçları düzenli formatta
- 🔄 **Hızlı Reset**: Tek tıkla yeni analiz başlatın

**Desteklenen Özellikler:**
- Dosya türü kontrolü (PDF, JPG, JPEG, PNG)
- Boyut kontrolü (32MB'ye kadar)
- 18 farklı rapor türü seçimi
- Hata ve başarı mesajları
- Loading animasyonları

## API Kullanımı

### Ana Endpoint'ler

#### 1. Web Arayüzü
```
GET http://localhost:5001/
```
Modern grafik kullanıcı arayüzü.

#### 2. API Bilgileri
```
GET http://localhost:5001/api/info
```
API hakkında genel bilgiler ve kullanım kılavuzu.

#### 3. Belge Analizi
```
POST http://localhost:5001/api/analyze
```
**Parametreler:**
- `file`: PDF dosyası (form-data)
- `document_type`: Analiz edilecek belge türü

**Örnek cURL:**
```bash
curl -X POST \
  -F "file=@document.pdf" \
  -F "document_type=electric_circuit" \
  http://localhost:5001/api/analyze
```

#### 4. Mevcut Servisler
```
GET http://localhost:5001/api/services
```
Tüm mevcut analiz servislerinin listesi.

#### 5. Sistem Durumu
```
GET http://localhost:5001/api/health
```
Tüm servislerin sağlık durumu kontrolü.

## Postman Test Kılavuzu

### Collection Oluşturma

1. **Yeni Collection oluşturun:** "PILZ Report Checker API"

2. **Environment değişkenleri:**
   - `base_url`: `http://localhost:5001`

### Test Senaryoları

#### Test 1: Sistem Durumu Kontrolü
```
GET {{base_url}}/api/health
```

#### Test 2: Servis Listesi
```
GET {{base_url}}/api/services
```

#### Test 3: Elektrik Devre Analizi
```
POST {{base_url}}/api/analyze
Form-data:
- file: [PDF dosyası]
- document_type: electric_circuit
```

#### Test 4: LOTO Raporu Analizi
```
POST {{base_url}}/api/analyze
Form-data:
- file: [PDF dosyası]
- document_type: loto_report
```

#### Test 5: Pnömatik Devre Analizi
```
POST {{base_url}}/api/analyze
Form-data:
- file: [PDF dosyası]
- document_type: pneumatic_circuit
```

### Örnek Yanıtlar

#### Başarılı Analiz Yanıtı:
```json
{
  "success": true,
  "analysis_service": "electric_circuit",
  "service_description": "Elektrik Devre Şeması Analizi",
  "results": {
    // Analiz sonuçları
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Hata Yanıtı:
```json
{
  "error": "Invalid document type",
  "message": "Document type 'invalid_type' is not supported",
  "available_types": ["electric_circuit", "loto_report", ...]
}
```

## Hata Çözümü

### Yaygın Hatalar

1. **Service Unavailable (503)**
   - Alt servis çalışmıyor olabilir
   - `/api/health` endpoint'ini kontrol edin

2. **Invalid Document Type**
   - Desteklenen belge türlerini `/api/services` endpoint'inden kontrol edin

3. **File Upload Error**
   - Desteklenen dosya türleri: PDF, JPG, JPEG, PNG
   - Maksimum dosya boyutu: 32MB

### Logları Kontrol Etme

Ana API'nin logları konsolda görüntülenir. Her servisin durumu ve hata mesajları burada takip edilebilir.

## Geliştirme Notları

- Ana API port 5001'de çalışır
- Alt servisler 5002-5019 portları arasında çalışır
- Tüm servisler otomatik olarak virtual environment içinde başlatılır
- Sistem kapatıldığında tüm alt servisler otomatik olarak temizlenir

## Yeni Eklenen Servisler (2025)

### 13. Topraklama Raporu Analizi
```bash
curl -X POST \
  -F "file=@topraklama_raporu.pdf" \
  -F "document_type=grounding_report" \
  http://localhost:5001/api/analyze
```
**Direct API:** `http://localhost:5014/api/topraklama-report`

### 14. HRC Kuvvet-Basınç Raporu Analizi
```bash
curl -X POST \
  -F "file=@hrc_raporu.pdf" \
  -F "document_type=hrc_report" \
  http://localhost:5001/api/analyze
```
**Direct API:** `http://localhost:5015/api/hrc-report`

### 15. Bakım Talimatları Analizi
```bash
curl -X POST \
  -F "file=@bakim_talimatlari.pdf" \
  -F "document_type=maintenance_instructions" \
  http://localhost:5001/api/analyze
```
**Direct API:** `http://localhost:5016/api/bakimtalimatlari-report`

### 16. Mekanik Titreşim Raporu Analizi
```bash
curl -X POST \
  -F "file=@titresim_raporu.pdf" \
  -F "document_type=vibration_report" \
  http://localhost:5001/api/analyze
```
**Direct API:** `http://localhost:5017/api/titresim-report`

### 17. Aydınlatma Raporu Analizi
```bash
curl -X POST \
  -F "file=@aydinlatma_raporu.pdf" \
  -F "document_type=lighting_report" \
  http://localhost:5001/api/analyze
```
**Direct API:** `http://localhost:5018/api/aydinlatma-report`

### 18. AT Tip İnceleme Sertifikası Analizi
```bash
curl -X POST \
  -F "file=@at_sertifikasi.pdf" \
  -F "document_type=at_certificate" \
  http://localhost:5001/api/analyze
```
**Direct API:** `http://localhost:5019/api/ati-inceleme-report`

## Güvenlik

- Dosyalar geçici olarak saklanır ve analiz sonrası silinir
- Dosya türü kontrolü yapılır (PDF, JPG, JPEG, PNG, DOCX, DOC, TXT)
- Maksimum dosya boyutu kontrolü (32MB)
- Her servis için ayrı temp klasörleri kullanılır

## Hızlı Başlatma

**Shell Script ile tek komutla başlatma:**
```bash
./start_api_gateway.sh
```

Bu script otomatik olarak:
- Virtual environment kontrolü yapar
- Gerekli temp klasörleri oluşturur
- Ana API Gateway'i başlatır
- Tüm 18 alt servisi arka planda başlatır
- Web arayüzünü `http://localhost:5001` adresinde sunar
- Dosya boyutu sınırı vardır (32MB)
- Timeout koruması mevcuttur (5 dakika)
