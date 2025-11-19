
# ============================================
# IMPORTS - STANDARD LIBRARIES
# ============================================
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
import logging
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import time

# ============================================
# LOGGING CONFIGURATION
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# FILE BASE URL (External System)
# ============================================
FILE_BASE_URL = "https://safetyexpert.app/fileupload/Account_103/Machine_4879/"


# ============================================
# IMPORTS - SERVİS DOSYALARI
# ============================================
# TODO: Her yeni servis dosyası eklendiğinde buraya import ekle

# Titreşim servisi 
from titresim_service import app as titresim_app

# Elektrik servisi 
from elektrik_service import app as elektrik_app

# ESPE servisi
from espe_service import app as espe_app

# Gürültü servisi
from gurultu_service import app as gurultu_app

# Manuel/Kullanım Kılavuzu servisi 
from manuel_service import app as manuel_app

# LOTO servisi 
from loto_service import app as loto_app

#at-declaration servisi
from at_declaration_service import app as at_declaration_app

# LVD servisi
from lvd_service import app as lvd_app

# Aydınlatma servisi
from aydinlatma_service import app as aydinlatma_app

# İSG Periyodik Kontrol servisi
from isg_service import app as isg_app

# Pnömatik Devre Şeması servisi
from pnomatic_service import app as pnomatic_app

# Hidrolik servisi
from hidrolik_service import app as hidrolik_app

# Montaj servisi
from montaj_service import app as montaj_app

# HRC servisi
from hrc_service import app as hrc_app

# ============================================
# FLASK APP CONFIGURATION
# ============================================
app = Flask(__name__)

# Upload settings
UPLOAD_FOLDER = 'temp_uploads_main'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB


# ============================================
# DOCUMENT TYPE MAPPING
# ============================================
# TODO: Her yeni servis için buraya mapping ekle

DOCUMENT_HANDLERS = {
    # Titreşim 
    'vibration_report': {
        'app': titresim_app,
        'endpoint': '/api/titresim-report',
        'description': 'Mekanik Titreşim Ölçüm Raporu Analizi'
    },
    
    # Elektrik 
    'electric_circuit': {
        'app': elektrik_app,
        'endpoint': '/api/elektrik-report',
        'description': 'Elektrik Devre Şeması Analizi'
    },

    # ESPE 
    'espe_report': {
        'app': espe_app,
        'endpoint': '/api/espe-report',
        'description': 'ESPE Raporu Analizi'
    },
    
    # Gürültü
    'noise_report': {
        'app': gurultu_app,
        'endpoint': '/api/noise-report',
        'description': 'Gürültü Ölçüm Raporu Analizi'
    },
    
    # Manuel 
    'manuel_report': {
        'app': manuel_app,
        'endpoint': '/api/manuel-report',
        'description': 'Manuel/Kullanım Kılavuzu Analizi'
    },

    # LOTO 
    'loto_report': {
        'app': loto_app,
        'endpoint': '/api/loto-report',
        'description': 'LOTO Prosedürü Analizi'
    },

    # AT Declaration
    'at_declaration': {
        'app': at_declaration_app,
        'endpoint': '/api/at-declaration',
        'description': 'AT Declaration Belgesi Analizi'
    },

    # LVD
    'lvd_report': {
        'app': lvd_app,
        'endpoint': '/api/lvd-report',
        'description': 'LVD Topraklama Süreklilik Raporu Analizi'
    },

    # Aydınlatma
    'lighting_report': {
        'app': aydinlatma_app,
        'endpoint': '/api/aydinlatma-report',
        'description': 'Aydınlatma Ölçüm Raporu Analizi'
    },

    # İSG Periyodik Kontrol
    'isg_periodic_control': {
        'app': isg_app,
        'endpoint': '/api/isg-control',
        'description': 'İSG Periyodik Kontrol Raporu Analizi'
    },

    # Pnömatik Devre Şeması
    'pneumatic_circuit': {
        'app': pnomatic_app,
        'endpoint': '/api/pnomatic-control',
        'description': 'Pnömatik Devre Şeması Analizi'
    },

    # Hidrolik Devre Şeması
    'hydraulic_circuit': {
        'app': hidrolik_app,
        'endpoint': '/api/hidrolik-control',
        'description': 'Hidrolik Devre Şeması Analizi'
    },

    # Montaj Talimatları
    'assembly_instructions': {
        'app': montaj_app,
        'endpoint': '/api/assembly-instructions',
        'description': 'Montaj Talimatları Analizi'
    },

    # HRC Kuvvet-Basınç 
    'hrc_report': {
        'app': hrc_app,
        'endpoint': '/api/hrc-report',
        'description': 'HRC Kuvvet-Basınç Raporu Analizi'
    },

    # ... DİĞER 14 SERVİS BURAYA EKLENECEK
    # Manuel, LOTO, LVD, AT Type, İSG, Pnömatik, Hidrolik, Montaj,
    # Topraklama, HRC, Bakım, Aydınlatma, AT Sertifika
}


# ============================================
# HELPER FUNCTIONS
# ============================================
def allowed_file(filename):
    """Dosya uzantısı kontrolü"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_temp_files():
    """1 saatten eski temp dosyalarını sil"""
    try:
        temp_folder = UPLOAD_FOLDER
        current_time = time.time()
        max_age = 60 * 60  # 1 saat (production)
        # max_age = 2 * 60  # 2 dakika (test)
        
        if not os.path.exists(temp_folder):
            return
        
        deleted_count = 0
        for filename in os.listdir(temp_folder):
            filepath = os.path.join(temp_folder, filename)
            
            if os.path.isfile(filepath):
                try:
                    file_modified_time = os.path.getmtime(filepath)
                    file_age = current_time - file_modified_time
                    
                    if file_age > max_age:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.info(f"✓ Eski temp dosyası silindi: {filename}")
                        
                except Exception as e:
                    logger.error(f"Dosya işlenirken hata {filename}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Toplam {deleted_count} eski temp dosyası temizlendi")
            
    except Exception as e:
        logger.error(f"Temp dosyaları temizlenirken hata: {e}")


# ============================================
# MAIN ANALYSIS ENDPOINT
# ============================================
@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    """
    Ana analiz endpoint'i - Basitleştirilmiş Azure versiyonu
    
    Diğer ekip bu endpoint'i kullanıyor:
    POST /api/analyze
    {
        "file": <binary>,
        "document_type": "vibration_report",
        "initial_comment": "...",  # opsiyonel
        "comment_author": "..."    # opsiyonel
    }
    """
    try:
        # 1. DOSYA KONTROLÜ
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please provide a file in the request'
            }), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a file to upload'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Only PDF, JPG, JPEG, and PNG files are allowed'
            }), 400

        # 2. DOCUMENT TYPE KONTROLÜ
        document_type = request.form.get('document_type')
        
        if not document_type:
            return jsonify({
                'error': 'No document type specified',
                'message': 'Please specify the document_type parameter',
                'available_types': list(DOCUMENT_HANDLERS.keys())
            }), 400

        if document_type not in DOCUMENT_HANDLERS:
            return jsonify({
                'error': 'Invalid document type',
                'message': f'Document type "{document_type}" is not supported',
                'available_types': list(DOCUMENT_HANDLERS.keys())
            }), 400

        # 3. OPSIYONEL PARAMETRELERİ AL
        custom_document_url = request.form.get('document_url', None)
        initial_comment = request.form.get('initial_comment', '').strip()
        comment_author = request.form.get('comment_author', '').strip()
        
        logger.info(f"Analiz isteği - Tip: {document_type}, Dosya: {file.filename}")
        if custom_document_url:
            logger.info(f"Custom document_url: {custom_document_url}")
        if initial_comment:
            logger.info(f"İlk yorum var - Yazar: {comment_author if comment_author else 'Anonim'}")

        # 4. DOSYAYI KAYDET
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 5. İLGİLİ SERVİS APP'İNE YÖNLENDİR
        handler_info = DOCUMENT_HANDLERS[document_type]
        service_app = handler_info['app']
        service_endpoint = handler_info['endpoint']
        
        logger.info(f"Servis çağrılıyor: {service_endpoint}")
        
        # Service app'in test client'ını kullan (internal routing)
        with service_app.test_client() as client:
            # Dosyayı ve diğer parametreleri servis'e gönder
            with open(filepath, 'rb') as f:
                response = client.post(
                    service_endpoint,
                    data={
                        'file': (f, filename),
                        **{key: value for key, value in request.form.items() if key != 'document_type'}
                    },
                    content_type='multipart/form-data'
                )
        
        # Servis'ten gelen cevabı al
        result = response.get_json()
        
        # 6. OTOMATIK TICKET OLUŞTUR
        if result and 'data' in result:
            try:
                ticket_data = {
                    'inspector_name': '',
                    'inspector_comment': '',
                    'document_name': result['data'].get('filename', filename),
                    'analysis_data': result['data']
                }
                
                tickets_dir = 'tickets'
                if not os.path.exists(tickets_dir):
                    os.makedirs(tickets_dir)
                
                tickets_file = os.path.join(tickets_dir, 'tickets.json')
                
                tickets = []
                if os.path.exists(tickets_file):
                    try:
                        with open(tickets_file, 'r', encoding='utf-8') as f:
                            tickets = json.load(f)
                    except:
                        tickets = []
                
                analysis_id = ticket_data['analysis_data'].get('analysis_id')
                existing_ticket = next((t for t in tickets if t.get('ticket_id') == analysis_id), None)
                
                if not existing_ticket:
                    ticket_no = f"ticket{len(tickets) + 1}"
                    
                    new_ticket = {
                        'ticket_no': ticket_no,
                        'ticket_id': analysis_id,
                        'document_name': ticket_data['analysis_data'].get('filename', 'Bilinmiyor'),
                        'document_type': ticket_data['analysis_data'].get('file_type', 'Bilinmiyor'),
                        'document_url': custom_document_url if custom_document_url else f"{FILE_BASE_URL}{ticket_data['analysis_data'].get('filename', filename)}",
                        'opening_date': datetime.now().isoformat(),
                        'last_updated': None,
                        'closing_date': None,
                        'status': 'İnceleniyor' if initial_comment else 'Kapalı',
                        'responsible': 'Savaş Bey',
                        'analysis_result': {
                            'overall_score': ticket_data['analysis_data'].get('overall_score', {}),
                            'category_scores': ticket_data['analysis_data'].get('category_scores', {}),
                            'extracted_values': ticket_data['analysis_data'].get('extracted_values', {}),
                            'recommendations': ticket_data['analysis_data'].get('recommendations', []),
                            'summary': ticket_data['analysis_data'].get('summary', '')
                        },
                        'comments': [
                            {
                                'comment_id': 'comment_1',
                                'author': comment_author if comment_author else 'Anonim Kullanıcı',
                                'text': initial_comment,
                                'timestamp': datetime.now().isoformat()
                            }
                        ] if initial_comment else [],
                        'inspector_comment': ''
                    }
                    
                    tickets.append(new_ticket)
                    
                    with open(tickets_file, 'w', encoding='utf-8') as f:
                        json.dump(tickets, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"Otomatik ticket oluşturuldu: {ticket_no} (ID: {analysis_id})")
                    
                    result['ticket_created'] = True
                    result['ticket_no'] = ticket_no
                    result['ticket_id'] = analysis_id
                else:
                    logger.info(f"Bu analiz için ticket zaten var: {existing_ticket.get('ticket_no')}")
                    result['ticket_created'] = False
                    result['ticket_no'] = existing_ticket.get('ticket_no')
                    result['ticket_id'] = existing_ticket.get('ticket_id')
                    
            except Exception as e:
                logger.error(f"Otomatik ticket oluşturma hatası: {str(e)}")
                result['ticket_error'] = str(e)
        
        # 7. SONUCU DÖNDÜR
        return jsonify(result), response.status_code

    except Exception as e:
        logger.error(f"Error in analyze_document: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


# ============================================
# TICKET MANAGEMENT ENDPOINTS
# ============================================
@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """Tüm tickets'ları döndür (filtreleme ile)"""
    try:
        tickets_file = os.path.join('tickets', 'tickets.json')
        
        if not os.path.exists(tickets_file):
            return jsonify([])
        
        with open(tickets_file, 'r', encoding='utf-8') as f:
            tickets = json.load(f)

        # Filtreleme parametreleri
        status_filter = request.args.get('status', '')
        date_from = request.args.get('date_from', '')
        date_to = request.args.get('date_to', '')
        responsible_filter = request.args.get('responsible', '').lower()
        
        # Filtrele
        if status_filter:
            tickets = [t for t in tickets if t.get('status') == status_filter]

        if date_from:
            date_from_obj = datetime.fromisoformat(date_from)
            tickets = [t for t in tickets 
                      if datetime.fromisoformat(t.get('opening_date', '')) >= date_from_obj]
        
        if date_to:
            date_to_obj = datetime.fromisoformat(date_to).replace(hour=23, minute=59, second=59)
            tickets = [t for t in tickets 
                      if datetime.fromisoformat(t.get('opening_date', '')) <= date_to_obj]
        
        if responsible_filter:
            tickets = [t for t in tickets 
                      if responsible_filter in t.get('responsible', '').lower()]
            
        # Sırala (en yeni önce)
        tickets.sort(key=lambda x: x.get('opening_date', ''), reverse=True)
        
        return jsonify(tickets)
        
    except Exception as e:
        logger.error(f"Tickets okunurken hata: {str(e)}")
        return jsonify([])


@app.route('/api/update-ticket-status', methods=['POST'])
def update_ticket_status():
    """Ticket statusünü güncelle (HYBRID: ticket_id veya ticket_no)"""
    try:
        data = request.get_json()
        
        ticket_id = data.get('ticket_id')
        ticket_no = data.get('ticket_no')
        new_status = data.get('status')
        
        if not ticket_id and not ticket_no:
            return jsonify({
                'success': False, 
                'message': 'ticket_id veya ticket_no gerekli'
            }), 400
        
        if not new_status:
            return jsonify({
                'success': False, 
                'message': 'status parametresi gerekli'
            }), 400
        
        tickets_file = os.path.join('tickets', 'tickets.json')
        
        if not os.path.exists(tickets_file):
            return jsonify({'success': False, 'message': 'Tickets bulunamadı'}), 404
        
        with open(tickets_file, 'r', encoding='utf-8') as f:
            tickets = json.load(f)
        
        # HYBRID ARAMA
        ticket = None
        if ticket_id:
            ticket = next((t for t in tickets if t.get('ticket_id') == ticket_id), None)
        
        if not ticket and ticket_no:
            ticket = next((t for t in tickets if t.get('ticket_no') == ticket_no), None)
        
        if not ticket:
            return jsonify({'success': False, 'message': 'Ticket bulunamadı'}), 404
        
        old_status = ticket.get('status')
        ticket['status'] = new_status
        
        # Kapalı statüsüne ÇEVRİLDİYSE
        if new_status == 'Kapalı' and old_status != 'Kapalı':
            ticket['closing_date'] = datetime.now().isoformat()
        
        # Kapalı'dan başka statüye GEÇİLDİYSE
        elif new_status != 'Kapalı' and old_status == 'Kapalı':
            ticket['closing_date'] = None
        
        ticket['last_updated'] = datetime.now().isoformat()
        
        with open(tickets_file, 'w', encoding='utf-8') as f:
            json.dump(tickets, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Ticket status güncellendi: {ticket.get('ticket_no')} -> {new_status}")
        
        return jsonify({'success': True, 'message': 'Status başarıyla güncellendi'})
        
    except Exception as e:
        logger.error(f"Status güncelleme hatası: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/add-comment', methods=['POST'])
def add_comment():
    """Ticket'a yeni yorum ekle (HYBRID: ticket_id veya ticket_no)"""
    try:
        data = request.get_json()
        
        ticket_id = data.get('ticket_id')
        ticket_no = data.get('ticket_no')
        author = data.get('author', '').strip()
        text = data.get('text', '').strip()
        
        if not ticket_id and not ticket_no:
            return jsonify({
                'success': False, 
                'message': 'ticket_id veya ticket_no gerekli'
            }), 400
        
        if not author or not text:
            return jsonify({
                'success': False, 
                'message': 'İsim ve yorum boş olamaz'
            }), 400
        
        tickets_file = os.path.join('tickets', 'tickets.json')
        
        if not os.path.exists(tickets_file):
            return jsonify({'success': False, 'message': 'Tickets bulunamadı'}), 404
        
        with open(tickets_file, 'r', encoding='utf-8') as f:
            tickets = json.load(f)
        
        # HYBRID ARAMA
        ticket = None
        if ticket_id:
            ticket = next((t for t in tickets if t.get('ticket_id') == ticket_id), None)
        
        if not ticket and ticket_no:
            ticket = next((t for t in tickets if t.get('ticket_no') == ticket_no), None)
        
        if not ticket:
            return jsonify({'success': False, 'message': 'Ticket bulunamadı'}), 404
        
        if 'comments' not in ticket:
            ticket['comments'] = []
        
        comment_id = f"comment_{len(ticket['comments']) + 1}"
        
        new_comment = {
            'comment_id': comment_id,
            'author': author,
            'text': text,
            'timestamp': datetime.now().isoformat()
        }
        
        ticket['comments'].append(new_comment)
        ticket['last_updated'] = datetime.now().isoformat()
        
        with open(tickets_file, 'w', encoding='utf-8') as f:
            json.dump(tickets, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Yorum eklendi: {ticket.get('ticket_no')} - {author}")
        
        return jsonify({
            'success': True,
            'message': 'Yorum başarıyla eklendi',
            'comment': new_comment
        })
        
    except Exception as e:
        logger.error(f"Yorum ekleme hatası: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/delete-ticket', methods=['POST'])
def delete_ticket():
    """Ticket'ı sil (HYBRID: ticket_id veya ticket_no)"""
    try:
        data = request.get_json()
        
        ticket_id = data.get('ticket_id')
        ticket_no = data.get('ticket_no')
        
        if not ticket_id and not ticket_no:
            return jsonify({
                'success': False, 
                'message': 'ticket_id veya ticket_no gerekli'
            }), 400
        
        tickets_file = os.path.join('tickets', 'tickets.json')
        
        if not os.path.exists(tickets_file):
            return jsonify({'success': False, 'message': 'Tickets bulunamadı'}), 404
        
        with open(tickets_file, 'r', encoding='utf-8') as f:
            tickets = json.load(f)
        
        # HYBRID ARAMA
        ticket_index = None
        
        if ticket_id:
            for i, t in enumerate(tickets):
                if t.get('ticket_id') == ticket_id:
                    ticket_index = i
                    break
        
        if ticket_index is None and ticket_no:
            for i, t in enumerate(tickets):
                if t.get('ticket_no') == ticket_no:
                    ticket_index = i
                    break
        
        if ticket_index is None:
            return jsonify({'success': False, 'message': 'Ticket bulunamadı'}), 404
        
        deleted_ticket = tickets.pop(ticket_index)
        
        with open(tickets_file, 'w', encoding='utf-8') as f:
            json.dump(tickets, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Ticket silindi: {deleted_ticket.get('ticket_no')}")
        
        return jsonify({
            'success': True,
            'message': 'Ticket başarıyla silindi'
        })
        
    except Exception as e:
        logger.error(f"Ticket silme hatası: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ============================================
# EVALUATION ENDPOINTS
# ============================================
@app.route('/api/save-evaluation', methods=['POST'])
def save_evaluation():
    """Analiz değerlendirmesini kaydet"""
    try:
        evaluation_data = request.get_json()
        
        if not evaluation_data:
            return jsonify({
                'success': False,
                'message': 'Değerlendirme verisi bulunamadı'
            }), 400
        
        evaluations_file = 'analysis_evaluations.json'
        
        evaluations = []
        if os.path.exists(evaluations_file):
            try:
                with open(evaluations_file, 'r', encoding='utf-8') as f:
                    evaluations = json.load(f)
            except Exception as e:
                logger.warning(f"Mevcut değerlendirmeler okunamadı: {e}")
                evaluations = []
        
        evaluations.append(evaluation_data)
        
        with open(evaluations_file, 'w', encoding='utf-8') as f:
            json.dump(evaluations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Değerlendirme kaydedildi: {evaluation_data.get('document_name', 'Unknown')}")
        
        return jsonify({
            'success': True,
            'message': 'Değerlendirme başarıyla kaydedildi',
            'evaluation_count': len(evaluations)
        })
        
    except Exception as e:
        logger.error(f"Değerlendirme kaydetme hatası: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Değerlendirme kaydedilemedi: {str(e)}'
        }), 500


@app.route('/api/evaluations', methods=['GET'])
def get_evaluations():
    """Tüm değerlendirmeleri döndür"""
    try:
        evaluations_file = 'analysis_evaluations.json'
        
        if not os.path.exists(evaluations_file):
            return jsonify([])
        
        with open(evaluations_file, 'r', encoding='utf-8') as f:
            evaluations = json.load(f)
        
        evaluations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify(evaluations)
        
    except Exception as e:
        logger.error(f"Değerlendirmeler okunurken hata: {str(e)}")
        return jsonify([])


# ============================================
# INFO & HEALTH ENDPOINTS
# ============================================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'PILZ Report Checker API Gateway',
        'version': '3.0.0-azure',
        'architecture': 'direct-import',
        'available_document_types': list(DOCUMENT_HANDLERS.keys()),
        'total_services': len(DOCUMENT_HANDLERS)
    })


@app.route('/api/info', methods=['GET'])
def api_info():
    """API documentation"""
    return jsonify({
        'service': 'PILZ Report Checker API Gateway',
        'version': '3.0.0-azure',
        'description': 'Unified API for document analysis services - Azure optimized',
        'architecture': 'Direct Import (No subprocess, no dynamic ports)',
        'endpoints': {
            'POST /api/analyze': 'Analyze any supported document type',
            'GET /api/tickets': 'List all tickets (with filters)',
            'POST /api/update-ticket-status': 'Update ticket status',
            'POST /api/add-comment': 'Add comment to ticket',
            'POST /api/delete-ticket': 'Delete ticket',
            'POST /api/save-evaluation': 'Save analysis evaluation',
            'GET /api/evaluations': 'Get all evaluations',
            'GET /api/health': 'Health check',
            'GET /api/info': 'This information',
            'GET /': 'Web interface'
        },
        'document_types': {
            doc_type: info['description'] 
            for doc_type, info in DOCUMENT_HANDLERS.items()
        },
        'usage': {
            'analyze_endpoint': '/api/analyze',
            'method': 'POST',
            'required_fields': ['file', 'document_type'],
            'optional_fields': ['document_url', 'initial_comment', 'comment_author'],
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'max_file_size': '32MB',
            'example_curl': 'curl -X POST -F "file=@document.pdf" -F "document_type=vibration_report" http://localhost:8000/api/analyze'
        }
    })


@app.route('/', methods=['GET'])
def index():
    """Main page with web interface"""
    return render_template('index.html')


# ============================================
# BACKGROUND SCHEDULER
# ============================================
scheduler = BackgroundScheduler()
scheduler.add_job(func=cleanup_old_temp_files, trigger="interval", minutes=20)
scheduler.start()

# Shutdown scheduler on exit
atexit.register(lambda: scheduler.shutdown())

logger.info("Background cleanup scheduler başlatıldı (20 dakikada bir çalışacak)")


# ============================================
# APPLICATION ENTRY POINT (Azure-Friendly)
# ============================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("PILZ Report Checker - Main API Gateway (Azure)")
    logger.info("=" * 60)
    logger.info(f"🔧 Mimari: Direct Import (No subprocess)")
    logger.info(f"📁 Upload klasörü: {UPLOAD_FOLDER}")
    logger.info(f"📊 Desteklenen formatlar: {', '.join(ALLOWED_EXTENSIONS)}")
    logger.info(f"📏 Maksimum dosya boyutu: {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB")
    logger.info("")
    logger.info(f"📋 Aktif Servisler ({len(DOCUMENT_HANDLERS)}):")
    for doc_type, info in DOCUMENT_HANDLERS.items():
        logger.info(f"  - {doc_type}: {info['description']}")
    logger.info("=" * 60)
    
    # Azure App Service PORT environment variable
    port = int(os.environ.get('PORT', 8000))
    
    logger.info(f"🚀 Gateway başlatılıyor - Port: {port}")
    logger.info("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )