#!/bin/bash

# PILZ Report Checker API Gateway Başlatma Scripti

echo "🚀 PILZ Report Checker API Gateway başlatılıyor..."
echo "================================================="

# Virtual environment kontrolü
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment bulunamadı!"
    echo "📝 Önce virtual environment oluşturun:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Virtual environment aktif etme
echo "🔧 Virtual environment aktif ediliyor..."
source venv/bin/activate

# Gerekli dizinleri oluşturma
echo "📁 Gerekli dizinler oluşturuluyor..."
mkdir -p temp_uploads_main
mkdir -p temp_uploads_topraklama
mkdir -p temp_uploads_hrc
mkdir -p temp_uploads_maintenance
mkdir -p temp_uploads_titresim
mkdir -p temp_uploads_aydinlatma
mkdir -p temp_uploads_at_tip

# Ana API Gateway'i başlatma
echo "🌟 Ana API Gateway başlatılıyor..."
echo "📋 Mevcut servisler:"
echo "   - Electric Circuit Analysis (Port: 5002)"
echo "   - ESPE Report Analysis (Port: 5003)"
echo "   - Hydraulic Circuit Analysis (Port: 5004)"
echo "   - Noise Measurement Report Analysis (Port: 5005)"
echo "   - Manual Report Analysis (Port: 5006)"
echo "   - LOTO Report Analysis (Port: 5007)"
echo "   - LVD Report Analysis (Port: 5008)"
echo "   - AT Type Inspection (Port: 5009)"
echo "   - ISG Periodic Control (Port: 5010)"
echo "   - Pneumatic Circuit Analysis (Port: 5011)"
echo "   - Hydraulic Circuit Analysis (Port: 5012)"
echo "   - Assembly Instructions (Port: 5013)"
echo "   - Grounding Report Analysis (Port: 5014)"
echo "   - HRC Force-Pressure Report Analysis (Port: 5015)"
echo "   - Maintenance Instructions Analysis (Port: 5016)"
echo "   - Mechanical Vibration Report Analysis (Port: 5017)"
echo "   - Lighting Report Analysis (Port: 5018)"
echo "   - AT Type Certificate Analysis (Port: 5019)"
echo ""
echo "🌐 Web Arayüzü: http://localhost:5001"
echo "� API Gateway: http://localhost:5001/api/info"
echo "🔍 Servis Durumu: http://localhost:5001/api/health"
echo "📋 Mevcut Servisler: http://localhost:5001/api/services"
echo ""
echo "⏳ Sistem başlatılıyor, lütfen bekleyin..."
echo "📱 Web arayüzüne tarayıcınızdan http://localhost:5001 adresinden erişebilirsiniz"
echo "================================================="

python3 main_api_gateway.py
