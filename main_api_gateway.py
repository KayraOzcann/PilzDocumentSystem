from dotenv import load_dotenv

# ✅ .env dosyasını yükle (EN ÖNCE!)
load_dotenv()

# ============================================
# IMPORTS - STANDARD LIBRARIES
# ============================================
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime, timezone
import logging
import atexit
import time
import threading

from database import db, init_db
from models import DocumentType, Ticket, TicketComment, CriteriaWeight, CriteriaDetail, PatternDefinition, ValidationKeyword, CategoryAction
import anthropic

# ============================================
# LOGGING CONFIGURATION
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# FILE BASE URL (External System)
# ============================================
FILE_BASE_URL = "https://safetyexpert.app/fileupload/Account_103/Machine_4879/"

DOCUMENT_HANDLERS = {}


def load_document_handlers():
    """
    Database'den document type'ları ve service import'larını yükle
    """
    global DOCUMENT_HANDLERS
    
    try:
        from models import DocumentType
        
        doc_types = DocumentType.query.filter_by(is_active=True).all()
        
        # Dynamic service'i bir kere yükle
        dynamic_app = None
        
        for dt in doc_types:
            try:
                # Dynamic service kontrolü
                if dt.service_file == 'dynamic_service.py':
                    # Dynamic service'i lazy load yap
                    if dynamic_app is None:
                        logger.info("🔄 Dynamic service yükleniyor...")
                        dynamic_module = __import__('dynamic_service', fromlist=['app'])
                        dynamic_app = getattr(dynamic_module, 'app')
                        logger.info("✅ Dynamic service yüklendi")
                    
                    DOCUMENT_HANDLERS[dt.code] = {
                        'app': dynamic_app,
                        'endpoint': dt.endpoint,
                        'description': dt.description,
                        'is_dynamic': True  # Flag ekle
                    }
                    
                    logger.info(f"✅ Loaded (dynamic): {dt.code} → {dt.name}")
                    
                else:
                    # Normal servisler (eski)
                    module_name = dt.service_file.replace('.py', '')
                    module = __import__(module_name, fromlist=['app'])
                    service_app = getattr(module, 'app')
                    
                    DOCUMENT_HANDLERS[dt.code] = {
                        'app': service_app,
                        'endpoint': dt.endpoint,
                        'description': dt.description,
                        'is_dynamic': False
                    }
                    
                    logger.info(f"✅ Loaded: {dt.code} → {module_name}")
                
            except Exception as e:
                logger.error(f"❌ {dt.code} import hatası: {str(e)}")
        
        logger.info(f"📋 Total handlers loaded: {len(DOCUMENT_HANDLERS)}")
        
    except Exception as e:
        logger.error(f"❌ Document handlers yüklenemedi: {str(e)}")
        raise


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
# HELPER FUNCTIONS
# ============================================
def allowed_file(filename):
    """Dosya uzantısı kontrolü"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_ticket_no():
    """Yeni ticket numarası oluştur (sıralı)"""
    last_ticket = Ticket.query.order_by(Ticket.id.desc()).first()
    if last_ticket:
        # Son ticket_no'dan sayıyı çıkar (örn: "ticket5" -> 5)
        try:
            last_num = int(last_ticket.ticket_no.replace('ticket', ''))
            return f"ticket{last_num + 1}"
        except:
            return f"ticket{Ticket.query.count() + 1}"
    return "ticket1"


# ============================================
# MAIN ANALYSIS ENDPOINT
# ============================================
def analyze_document_async(file_data, filename, document_type, handler_info, custom_document_url, initial_comment, comment_author, ticket_id):
    """
    Arka planda analiz yapan fonksiyon (thread içinde çalışır)
    """
    try:
        logger.info(f"🔄 Async analiz başladı: {ticket_id}")
        
        # Dosyayı geçici olarak kaydet
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(file_data)
        
        # Service'e gönder
        service_app = handler_info['app']
        service_endpoint = handler_info['endpoint']
        is_dynamic = handler_info.get('is_dynamic', False)
        
        with service_app.test_client() as client:
            with open(filepath, 'rb') as file_to_send:
                form_data = {
                    'file': (file_to_send, filename),
                }
                
                if is_dynamic:
                    form_data['document_code'] = document_type
                
                response = client.post(
                    service_endpoint,
                    data=form_data,
                    content_type='multipart/form-data'
                )
        
        # Geçici dosyayı sil
        try:
            os.remove(filepath)
        except:
            pass
        
        result = response.get_json()
        
        # Ticket'ı güncelle
        with app.app_context():
            ticket = Ticket.query.filter_by(ticket_id=ticket_id).first()
            
            if ticket and result and 'data' in result:
                analysis_data = result['data']
                
                # Analiz sonucunu kaydet
                ticket.analysis_result = {
                    'overall_score': analysis_data.get('overall_score', {}),
                    'category_scores': analysis_data.get('category_scores', {}),
                    'extracted_values': analysis_data.get('extracted_values', {}),
                    'recommendations': analysis_data.get('recommendations', []),
                    'summary': analysis_data.get('summary', '')
                }
                
                # Status güncelle
                if initial_comment:
                    ticket.status = 'İnceleniyor'
                    # İlk yorumu ekle
                    first_comment = TicketComment(
                        ticket_id=ticket.id,
                        comment_id='comment_1',
                        author=comment_author if comment_author else 'Anonim Kullanıcı',
                        text=initial_comment,
                        timestamp=datetime.now()
                    )
                    db.session.add(first_comment)
                else:
                    ticket.status = 'Kapalı'
                    ticket.closing_date = datetime.now()
                
                ticket.last_updated = datetime.now()
                
                db.session.commit()
                
                logger.info(f"✅ Async analiz tamamlandı: {ticket_id} - Status: {ticket.status}")
            
    except Exception as e:
        logger.error(f"❌ Async analiz hatası ({ticket_id}): {str(e)}")
        
        # Hata durumunda ticket'ı güncelle
        try:
            with app.app_context():
                ticket = Ticket.query.filter_by(ticket_id=ticket_id).first()
                if ticket:
                    ticket.status = 'Hatalı'
                    ticket.analysis_result = {'error': str(e)}
                    ticket.last_updated = datetime.now()
                    db.session.commit()
        except:
            pass

@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    """
    Ana analiz endpoint'i - ASYNC (PostgreSQL ile)
    
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
        
        logger.info(f"Analiz isteği (ASYNC) - Tip: {document_type}, Dosya: {file.filename}")

        # 4. DOSYAYI MEMORY'YE AL
        filename = secure_filename(file.filename)
        file_data = file.read()  # Dosyayı memory'ye oku
        
        # 5. HANDLER BİLGİSİ
        handler_info = DOCUMENT_HANDLERS[document_type]

        # 6. TİCKET OLUŞTUR (İŞLENİYOR STATUS)
        analysis_id = f"{document_type}_{int(datetime.now().timestamp())}"
        ticket_no = generate_ticket_no()
        
        new_ticket = Ticket(
            ticket_no=ticket_no,
            ticket_id=analysis_id,
            document_name=filename,
            document_type=document_type,
            document_url=custom_document_url if custom_document_url else f"{FILE_BASE_URL}{filename}",
            opening_date=datetime.now(),
            status='İşleniyor',  # ASYNC: İşleniyor
            responsible='Savaş Bey',
            analysis_result={}  # Boş başlar
        )
        
        db.session.add(new_ticket)
        db.session.commit()
        
        logger.info(f"Ticket oluşturuldu (ASYNC): {ticket_no} (ID: {analysis_id})")
        
        # 7. ARKA PLANDA ANALİZ BAŞLAT (THREAD)
        thread = threading.Thread(
            target=analyze_document_async,
            args=(file_data, filename, document_type, handler_info, custom_document_url, initial_comment, comment_author, analysis_id)
        )
        thread.daemon = True
        thread.start()
        
        # 8. HEMEN RESPONSE DÖN
        return jsonify({
            'success': True,
            'message': 'Analiz başlatıldı',
            'ticket_id': analysis_id,
            'ticket_no': ticket_no,
            'status': 'İşleniyor'
        }), 200

    except Exception as e:
        logger.error(f"Error in analyze_document: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


# ============================================
# SERVICE INFORMATION ENDPOINT
# ============================================
@app.route('/api/services', methods=['GET'])
def get_services():
    """Mevcut tüm analiz servislerini listele"""
    try:
        services = []
        
        for service_name, config in DOCUMENT_HANDLERS.items():
            services.append({
                'service_name': service_name,
                'description': config['description'],
                'endpoint': config['endpoint'],
                'status': 'available'
            })
        
        return jsonify({
            'success': True,
            'services': services,
            'total_services': len(services)
        })
        
    except Exception as e:
        logger.error(f"Servisler listelenirken hata: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

    
# ============================================
# TICKET MANAGEMENT ENDPOINTS (PostgreSQL)
# ============================================
@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """Tüm tickets'ları döndür (filtreleme ile) - PostgreSQL"""
    try:
        # Base query
        query = Ticket.query
        
        # Filtreleme parametreleri
        ticket_id_filter = request.args.get('ticket_id', '')  # YENİ: ticket_id filtresi
        status_filter = request.args.get('status', '')
        date_from = request.args.get('date_from', '')
        date_to = request.args.get('date_to', '')
        responsible_filter = request.args.get('responsible', '')
        
        # Filtrele
        if ticket_id_filter:  # YENİ: ticket_id ile filtrele
            query = query.filter(Ticket.ticket_id == ticket_id_filter)
        
        if status_filter:
            query = query.filter(Ticket.status == status_filter)
        
        if date_from:
            date_from_obj = datetime.fromisoformat(date_from)
            query = query.filter(Ticket.opening_date >= date_from_obj)
        
        if date_to:
            date_to_obj = datetime.fromisoformat(date_to).replace(hour=23, minute=59, second=59)
            query = query.filter(Ticket.opening_date <= date_to_obj)
        
        if responsible_filter:
            query = query.filter(Ticket.responsible.ilike(f'%{responsible_filter}%'))
        
        # Sırala (en yeni önce)
        tickets = query.order_by(Ticket.opening_date.desc()).all()
        
        # Response formatı (frontend ile uyumlu)
        result = []
        for ticket in tickets:
            # Yorumları çek
            comments = TicketComment.query.filter_by(ticket_id=ticket.id).order_by(TicketComment.timestamp).all()
            
            result.append({
                'ticket_no': ticket.ticket_no,
                'ticket_id': ticket.ticket_id,
                'document_name': ticket.document_name,
                'document_type': ticket.document_type,
                'document_url': ticket.document_url,
                'opening_date': ticket.opening_date.isoformat(),
                'last_updated': ticket.last_updated.isoformat() if ticket.last_updated else None,
                'closing_date': ticket.closing_date.isoformat() if ticket.closing_date else None,
                'status': ticket.status,
                'responsible': ticket.responsible,
                'analysis_result': ticket.analysis_result,
                'comments': [
                    {
                        'comment_id': c.comment_id,
                        'author': c.author,
                        'text': c.text,
                        'timestamp': c.timestamp.isoformat()
                    }
                    for c in comments
                ]
            })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Tickets okunurken hata: {str(e)}")
        return jsonify([]), 500


@app.route('/api/update-ticket', methods=['POST'])
def update_ticket():
    """Ticket bilgilerini güncelle (status, responsible, vb.) - PostgreSQL"""
    try:
        data = request.get_json()
        
        # HYBRID: ticket_id veya ticket_no kabul et
        ticket_id = data.get('ticket_id')
        ticket_no = data.get('ticket_no')
        
        if not ticket_id and not ticket_no:
            return jsonify({
                'success': False, 
                'message': 'ticket_id veya ticket_no gerekli'
            }), 400
        
        # Güncellenebilir alanlar (opsiyonel)
        new_status = data.get('status')
        new_responsible = data.get('responsible')
        
        if not new_status and not new_responsible:
            return jsonify({
                'success': False, 
                'message': 'Güncellenecek alan belirtilmedi'
            }), 400
        
        # HYBRID ARAMA: Önce ticket_id, sonra ticket_no
        ticket = None
        if ticket_id:
            ticket = Ticket.query.filter_by(ticket_id=ticket_id).first()
        
        if not ticket and ticket_no:
            ticket = Ticket.query.filter_by(ticket_no=ticket_no).first()
        
        if not ticket:
            return jsonify({'success': False, 'message': 'Ticket bulunamadı'}), 404
        
        old_status = ticket.status
        
        # Status güncelleme
        if new_status:
            ticket.status = new_status
            
            # Kapalı statüsüne ÇEVRİLDİYSE kapanma tarihini ekle
            if new_status == 'Kapalı' and old_status != 'Kapalı':
                ticket.closing_date = datetime.now()
            
            # Kapalı'dan başka bir statüye GEÇİLDİYSE kapanma tarihini sil
            elif new_status != 'Kapalı' and old_status == 'Kapalı':
                ticket.closing_date = None
        
        # Sorumlu güncelleme
        if new_responsible:
            ticket.responsible = new_responsible
        
        # Son güncelleme tarihini ekle
        ticket.last_updated = datetime.now()
        
        db.session.commit()
        
        logger.info(f"Ticket güncellendi: {ticket.ticket_no} (ID: {ticket.ticket_id})")
        
        return jsonify({'success': True, 'message': 'Ticket başarıyla güncellendi'})
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Ticket güncelleme hatası: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/add-comment', methods=['POST'])
def add_comment():
    """Ticket'a yeni yorum ekle - PostgreSQL"""
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
        
        # HYBRID ARAMA
        ticket = None
        if ticket_id:
            ticket = Ticket.query.filter_by(ticket_id=ticket_id).first()
        
        if not ticket and ticket_no:
            ticket = Ticket.query.filter_by(ticket_no=ticket_no).first()
        
        if not ticket:
            return jsonify({'success': False, 'message': 'Ticket bulunamadı'}), 404
        
        # Mevcut yorum sayısını bul
        existing_comments_count = TicketComment.query.filter_by(ticket_id=ticket.id).count()
        comment_id = f"comment_{existing_comments_count + 1}"
        
        # Yeni yorum oluştur
        new_comment = TicketComment(
            ticket_id=ticket.id,
            comment_id=comment_id,
            author=author,
            text=text,
            timestamp=datetime.now()
        )
        
        db.session.add(new_comment)
        
        # Ticket'ın last_updated'ini güncelle
        ticket.last_updated = datetime.now()
        
        db.session.commit()
        
        logger.info(f"Yorum eklendi: {ticket.ticket_no} - {author}")
        
        return jsonify({
            'success': True,
            'message': 'Yorum başarıyla eklendi',
            'comment': {
                'comment_id': comment_id,
                'author': author,
                'text': text,
                'timestamp': new_comment.timestamp.isoformat()
            }
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Yorum ekleme hatası: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/delete-ticket', methods=['POST'])
def delete_ticket():
    """Ticket'ı sil - PostgreSQL (cascade ile yorumlar da silinir)"""
    try:
        data = request.get_json()
        
        ticket_id = data.get('ticket_id')
        ticket_no = data.get('ticket_no')
        
        if not ticket_id and not ticket_no:
            return jsonify({
                'success': False, 
                'message': 'ticket_id veya ticket_no gerekli'
            }), 400
        
        # HYBRID ARAMA
        ticket = None
        if ticket_id:
            ticket = Ticket.query.filter_by(ticket_id=ticket_id).first()
        
        if not ticket and ticket_no:
            ticket = Ticket.query.filter_by(ticket_no=ticket_no).first()
        
        if not ticket:
            return jsonify({'success': False, 'message': 'Ticket bulunamadı'}), 404
        
        ticket_no_deleted = ticket.ticket_no
        
        # Sil (cascade sayesinde yorumlar da silinir)
        db.session.delete(ticket)
        db.session.commit()
        
        logger.info(f"Ticket silindi: {ticket_no_deleted}")
        
        return jsonify({
            'success': True,
            'message': 'Ticket başarıyla silindi'
        })
        
    except Exception as e:
        db.session.rollback()
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
        'version': '3.0.0-azure-db',
        'architecture': 'database-driven',
        'available_document_types': list(DOCUMENT_HANDLERS.keys()),
        'total_services': len(DOCUMENT_HANDLERS)
    })


@app.route('/api/info', methods=['GET'])
def api_info():
    """API documentation"""
    return jsonify({
        'service': 'PILZ Report Checker API Gateway',
        'version': '3.0.0-azure-db',
        'description': 'Unified API for document analysis services - PostgreSQL powered',
        'architecture': 'Database-driven (PostgreSQL)',
        'endpoints': {
            'POST /api/analyze': 'Analyze any supported document type',
            'GET /api/tickets': 'List all tickets (with filters)',
            'POST /api/update-ticket': 'Update ticket status/responsible',
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
# DOCUMENT TYPES ENDPOINT
# ============================================
@app.route('/api/document-types', methods=['GET'])
def get_document_types():
    """Database'den aktif document type'ları döndür"""
    try:
        # Aktif document type'ları çek
        doc_types = DocumentType.query.filter_by(is_active=True).order_by(DocumentType.name).all()
        
        return jsonify({
            'success': True,
            'document_types': [
                {
                    'code': dt.code,
                    'name': dt.name,
                    'description': dt.description,
                    'icon': dt.icon,
                    'endpoint': dt.endpoint
                }
                for dt in doc_types
            ],
            'total': len(doc_types)
        })
        
    except Exception as e:
        logger.error(f"Document types hatası: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# ============================================
# DATABASE INITIALIZATION
# ============================================
# Database'i başlat
init_db(app)

# ============================================
# DYNAMIC DOCUMENT TYPE MANAGEMENT (YENİ)
# ============================================
@app.route('/api/create-dynamic-type', methods=['POST'])
def create_dynamic_type():
    """Yeni döküman türü oluştur - AI ile pattern generation"""
    try:
        data = request.get_json()
        
        # Validasyon
        if not data.get('name'):
            return jsonify({'success': False, 'message': 'Döküman adı gerekli'}), 400
        
        if not data.get('strong_keywords') or len(data['strong_keywords']) == 0:
            return jsonify({'success': False, 'message': 'En az 1 strong keyword gerekli'}), 400
        
        # Code oluştur (normalize)
        code = data['name'].lower().strip().replace(' ', '_').replace('ş', 's').replace('ğ', 'g').replace('ü', 'u').replace('ö', 'o').replace('ç', 'c').replace('ı', 'i')
        
        # Aynı code var mı kontrol
        existing = DocumentType.query.filter_by(code=code).first()
        if existing:
            return jsonify({'success': False, 'message': f'Bu döküman türü zaten var: {code}'}), 400
        
        logger.info(f"🚀 Yeni döküman türü oluşturuluyor: {data['name']} ({code})")
        logger.info(f"📊 AI pattern generation başlatılıyor...")
        
        # AI'ya gönder - Patterns oluştur
        ai_result = generate_patterns_with_ai(data)
        
        if not ai_result['success']:
            return jsonify({'success': False, 'message': f'AI pattern hatası: {ai_result["error"]}'}), 500
        
        logger.info(f"✅ AI patterns oluşturuldu")
        
        # Database'e kaydet
        doc_type = DocumentType(
            code=code,
            name=data['name'],
            description=data.get('description', f"{data['name']} analiz servisi"),
            service_file='dynamic_service.py',
            endpoint='/api/dynamic-report',
            icon=data.get('icon', '📄'),
            app_variable_name='dynamic_app',
            is_active=True,
            needs_ocr=data.get('needs_ocr', True)
        )
        
        db.session.add(doc_type)
        db.session.flush()
        
        # Criteria Weights ekle
        for idx, (category_name, weight) in enumerate(data['criteria_weights'].items(), 1):
            cw = CriteriaWeight(
                document_type_id=doc_type.id,
                category_name=category_name,
                weight=weight,
                display_order=idx
            )
            db.session.add(cw)
            db.session.flush()

            logger.info(f"📊 Kategori eklendi: {category_name}") 

            # 👇 CASE-INSENSITIVE EŞLEŞME
            # AI'dan gelen kategori adını bul (büyük/küçük harf fark etmez)
            matched_category = None
            for ai_category in ai_result['criteria_patterns'].keys():
                if ai_category.lower().replace('_', ' ') == category_name.lower().replace('_', ' '):
                    matched_category = ai_category
                    break
            
            # Criteria Details ekle (AI'dan gelen patterns ile)
            if matched_category:
                logger.info(f"   ✅ AI patterns bulundu: {len(ai_result['criteria_patterns'][matched_category])} kriter")
                for idx2, (criterion_name, criterion_data) in enumerate(ai_result['criteria_patterns'][matched_category].items(), 1):
                    logger.info(f"      - {criterion_name}: {criterion_data['weight']} puan")
                    cd = CriteriaDetail(
                        criteria_weight_id=cw.id,
                        criterion_name=criterion_name,
                        pattern=criterion_data['pattern'],
                        weight=criterion_data['weight'],
                        display_order=idx2
                    )
                    db.session.add(cd)
            else:
                logger.warning(f"   ⚠️  AI patterns BULUNAMADI: {category_name}")  
                logger.warning(f"   AI result keys: {ai_result['criteria_patterns'].keys()}")
                logger.warning(f"   AI keys: {list(ai_result['criteria_patterns'].keys())}")  
                
        # Pattern Definitions ekle (extract_values - AI'dan gelen)
        if ai_result.get('extract_patterns'):
            for idx, (field_name, patterns_list) in enumerate(ai_result['extract_patterns'].items(), 1):
                pd = PatternDefinition(
                    document_type_id=doc_type.id,
                    pattern_group='extract_values',
                    field_name=field_name,
                    patterns=patterns_list,
                    display_order=idx
                )
                db.session.add(pd)
        
        # Validation Keywords - Critical Terms (AI'dan gelen)
        if ai_result.get('critical_terms'):
            for idx, keywords_list in enumerate(ai_result['critical_terms'], 1):
                vk = ValidationKeyword(
                    document_type_id=doc_type.id,
                    keyword_type='critical_terms',
                    category=f'category_{idx}',
                    keywords=keywords_list
                )
                db.session.add(vk)
        
        # Strong Keywords
        vk_strong = ValidationKeyword(
            document_type_id=doc_type.id,
            keyword_type='strong_keywords',
            keywords=data['strong_keywords']
        )
        db.session.add(vk_strong)
        db.session.flush()

        # Excluded Keywords - Mevcut pool'u çek + yeni strong keywords ekle
        excluded_pool = get_excluded_keywords_pool(current_doc_type_id=doc_type.id) 
        
        
        vk_excluded = ValidationKeyword(
            document_type_id=doc_type.id,
            keyword_type='excluded_keywords',
            keywords=list(set(excluded_pool))  # Duplicate'leri kaldır
        )
        db.session.add(vk_excluded)
        
        db.session.commit()
        
        logger.info(f"✅ Döküman türü kaydedildi: {code}")
        logger.info(f"🔄 Document handlers yeniden yükleniyor...")
        
        # Handlers'ı yeniden yükle
        load_document_handlers()
        
        return jsonify({
            'success': True,
            'message': f'✅ {data["name"]} başarıyla oluşturuldu!',
            'code': code,
            'document_type': {
                'code': code,
                'name': data['name'],
                'icon': data.get('icon', '📄')
            }
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Create dynamic type hatası: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/delete-dynamic-type', methods=['POST'])
def delete_dynamic_type():
    """Dynamic döküman türünü sil - Cascade ile tüm ilişkili veriler silinir"""
    try:
        data = request.get_json()
        code = data.get('code')
        
        if not code:
            return jsonify({'success': False, 'message': 'Döküman kodu gerekli'}), 400
        
        # Döküman türünü bul
        doc_type = DocumentType.query.filter_by(code=code).first()
        
        if not doc_type:
            return jsonify({'success': False, 'message': f'Döküman türü bulunamadı: {code}'}), 404
        
        # Dynamic service mi kontrol et
        if doc_type.service_file != 'dynamic_service.py':
            return jsonify({'success': False, 'message': 'Sadece dynamic döküman türleri silinebilir'}), 400
        
        doc_name = doc_type.name
        
        logger.info(f"🗑️ Döküman türü siliniyor: {doc_name} ({code})")
        
        # Strong keywords'leri excluded pool'dan çıkar
        try:
            strong_vk = ValidationKeyword.query.filter_by(
                document_type_id=doc_type.id,
                keyword_type='strong_keywords'
            ).first()
            
            if strong_vk:
                logger.info(f"   📤 Strong keywords excluded pool'dan çıkarılıyor: {strong_vk.keywords}")
        except Exception as e:
            logger.warning(f"Strong keywords çıkarma hatası: {e}")
        
        # Cascade delete (ilişkili tüm kayıtlar silinir)
        db.session.delete(doc_type)
        db.session.commit()
        
        logger.info(f"✅ {doc_name} başarıyla silindi")
        logger.info(f"🔄 Document handlers yeniden yükleniyor...")
        
        # Handlers'ı yeniden yükle
        load_document_handlers()
        
        return jsonify({
            'success': True,
            'message': f'✅ {doc_name} başarıyla silindi!',
            'code': code
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Delete dynamic type hatası: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/get-dynamic-types', methods=['GET'])
def get_dynamic_types():
    """Tüm dynamic döküman türlerini listele"""
    try:
        # is_dynamic flag'i yoksa service_file='dynamic_service.py' olanları al
        dynamic_types = DocumentType.query.filter_by(
            service_file='dynamic_service.py',
            is_active=True
        ).order_by(DocumentType.name).all()
        
        return jsonify({
            'success': True,
            'types': [
                {
                    'code': dt.code,
                    'name': dt.name,
                    'icon': dt.icon,
                    'description': dt.description
                }
                for dt in dynamic_types
            ]
        })
        
    except Exception as e:
        logger.error(f"Get dynamic types hatası: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ============================================
# HELPER FUNCTIONS - AI & KEYWORDS
# ============================================
def generate_patterns_with_ai(data):
    try:
        import anthropic
        import os
        import json
        import re

        # API key kontrolü
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.error("❌ ANTHROPIC_API_KEY bulunamadı!")
            return {
                "success": False,
                "error": "API key tanımlanmamış. Lütfen sistem yöneticisiyle iletişime geçin."
            }
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Prompt hazırla
        prompt = f"""Sen bir regex pattern uzmanısın. Türkçe ve İngilizce dökümanlardan bilgi çıkarmak için regex pattern'leri oluşturuyorsun.

GÖREV 1 - CRITERIA DETAILS PATTERNS:

Kategoriler ve Kelimeler:
{format_criteria_for_ai(data['criteria_weights'], data['criteria_details'])}

PATTERN KURALLARI:
1. Her kelime için TEK bir UZUN regex pattern oluştur
2. (?i) SADECE pattern'in EN BAŞINDA kullan! Pattern ORTASINDA veya SONUNDA (?i) ASLA kullanma!
3. Non-capturing group kullan: (?:...)
4. Türkçe ve İngilizce alternatifleri içermeli
5. Boşluk toleranslı: \\s* veya \\s+

ÖRNEK UZUN PATTERN:
"(?i)(?:test\\s*yapan|tested\\s*by|ölçüm\\s*yapan|measured\\s*by|kurum|institution|organization|şirket|company|firma|sorumlu|responsible)[\\s\\W]*[:=]?\\s*([A-ZÇĞİÖŞÜa-züçğıöşü][A-Za-züçğıöşüÇĞİÖŞÜ\\s\\.\\&\\-]{{3,50}})"

DOĞRU YAZIM:
- "(?i)(?:yazar|author|writer)"
- "(?i)(?:rapor\\s*no|report\\s*number)\\s*[:=]?\\s*([A-Z0-9-]+)"

YANLIŞ YAZIM (YAPMA!):
- "(?i)yazar|author" ❌ (non-capturing group yok)
- "yazar|author" ❌ (case flag yok)

WEIGHT HESAPLAMA KURALLARI:  
Her kategori için weight'leri eşit dağıt:
- weight = kategori_toplam_puanı / kelime_sayısı
- Tam bölünmüyorsa kalanı son kelimeye ekle

Örnek:
Kategori: "genel bilgiler" (100 puan)
Kelimeler: ["firma_adi", "rapor_no", "tarih"] (3 kelime)
Hesaplama: 100 / 3 = 33.33
Weight dağılımı:
- firma_adi: 33
- rapor_no: 33
- tarih: 34 (kalan +1 puan)

Örnek 2:
Kategori: "teknik bilgiler" (50 puan)
Kelimeler: ["voltaj", "akim"] (2 kelime)
Hesaplama: 50 / 2 = 25
Weight dağılımı:
- voltaj: 25
- akim: 25

GÖREV 2 - EXTRACT VALUES PATTERNS:

Alanlar:
{format_extract_values_for_ai(data.get('extract_values', []))}

PATTERN KURALLARI:
Her alan için 2-4 FARKLI pattern oluştur:

KRİTİK UYARI - (?i) KULLANIMI:
- (?i) SADECE pattern'in EN BAŞINDA kullan!
- Pattern ORTASINDA veya SONUNDA (?i) ASLA kullanma!

1. ALAN ADINI GENİŞLET:
   - Türkçe + İngilizce alternatifleri
   - Yaygın varyasyonları ekle

2. BOŞLUK TOLERANSsI:
   - \\s* veya \\s+ veya [\\s\\W]* kullan
   - İki nokta/eşittir opsiyonel: [:=]?

3. DEĞER YAKALAMA (alan türüne göre):
   - TARİH: (\\d{{1,2}}[./\\-]\\d{{1,2}}[./\\-]\\d{{2,4}}) veya (\\d{{1,2}}\\s+\\d{{1,2}}\\s+\\d{{2,4}})
   - NUMARA/KOD: ([A-Z0-9\\-/]+) veya ([0-9]{{3,}})
   - İSİM/FİRMA: ([A-ZÇĞİÖŞÜ][A-Za-züçğıöşüÇĞİÖŞÜ\\s\\.\\&\\-]{{3,80}})
   - GENEL METİN: ([^\\n]{{5,100}})

4. FARKLI FORMAT VARYASYONLARI:
   - Alan adı ÖNCE, sonra değer: "(?i)(?:alan_adi)[\\s\\W]*[:=]?\\s*(değer_pattern)"
   - Değer ÖNCE, sonra alan adı: "(değer_pattern)\\s*(?:alan_adi)"
   - Boşluklu format: "(?i)(?:alan_adi)\\s+(değer_pattern)"

5. HER PATTERN:
   - (?i) ile başla (case-insensitive)
   - Non-capturing group: (?:...)
   - Capture group: (...) sadece değer için

ÖRNEK ARRAY PATTERN:
[
    "(?i)(?:test\\s*tarih|test\\s*date)\\s*[:=]\\s*(\\d{{1,2}}[./]\\d{{1,2}}[./]\\d{{2,4}})",
    "(\\d{{1,2}}[./]\\d{{1,2}}[./]\\d{{4}})\\s*(?:tarih|date)"
]

GÖREV 3 - CRITICAL TERMS:

Strong Keywords: {', '.join(data['strong_keywords'])}

Kurallar:
- Mevcut kelimeleri kullan
- Türkçe/İngilizce alternatifleri ekle  
- Kategoriler mantıklı olmalı
- 3-4 kategori oluştur

Örnek:
Input: ["termal", "konfor"]
Output: 
[
    ["termal", "thermal", "ısı", "heat", "sıcaklık", "temperature"],
    ["konfor", "comfort", "rahatlık"]
]

KRİTİK JSON KURALLARI:
1. Regex pattern'lerinde backslash DÖRT KERE: \\\\s \\\\d \\\\w \\\\S
2. String içinde çift tırnak varsa escape et: \\"
3. Yeni satır kullanma, tek satır JSON döndür
4. Türkçe karakterleri olduğu gibi bırak (escape etme)

JSON formatında döndür - ÖNEMLİ: Regex pattern'lerinde backslash'leri DÖRT KERE yaz (\\\\s \\\\d \\\\w):
{{
    "criteria_patterns": {{
        "kategori_ismi": {{
            "kelime_adi": {{
                "pattern": "(?i)(?:pattern_buraya)",
                "weight": 33 // ÖRNEK! Sen hesapla!
            }}
        }}
    }},
    "extract_patterns": {{
        "field_name": ["(?i)pattern1","(?i)pattern2"]
    }},
    "critical_terms": [
        ["kelime1", "kelime2"],
        ["kelime3", "kelime4"]
    ]
}}

ÖNEMLİ UYARILAR:
1. Kategori isimleri AYNEN şunlar olmalı: {list(data['criteria_weights'].keys())}
2. Weight'leri yukarıdaki "WEIGHT HESAPLAMA KURALLARI"na göre MUTLAKA HESAPLA!
3. JSON örneğindeki "33" sadece örnek, gerçek değerleri sen hesapla!  
4. JSON içinde backslash dört kere: \\\\s \\\\d \\\\w
"""
        
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=3000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Response'u parse et
        response_text = message.content[0].text
        
        # JSON'u çıkar (markdown varsa temizle)
        import json
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]

        # Fazla boşlukları temizle
        response_text = response_text.strip()

        # 👇 YENİ: Backslash kontrolü
        logger.info("📝 AI Response ilk 500 karakter:")
        logger.info(response_text[:500])  
        
        # JSON parse dene
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as je:
            logger.error(f"❌ JSON Parse Hatası: {je}")
            logger.error(f"Hatalı JSON (ilk 1000 karakter):")
            logger.error(response_text[:1000])
            
            # 👇 YENİ: Otomatik düzeltme dene
            logger.info("🔧 JSON düzeltme deneniyor...")
            
            # Tek backslash'leri double yap (sadece regex pattern'lerinde)
            fixed_text = re.sub(
                r'("pattern":\s*"[^"]*)(\\)([^\\"])',
                r'\1\\\\\3',
                response_text
            )
            
            try:
                result = json.loads(fixed_text)
                logger.info("✅ JSON otomatik düzeltildi!")
            except:
                return {
                    'success': False, 
                    'error': f'AI JSON parse hatası: {str(je)}\n\nİlk 500 karakter:\n{response_text[:500]}'
                }
        
        result['success'] = True
        return result
        
    except Exception as e:
        logger.error(f"AI pattern generation hatası: {str(e)}")
        return {'success': False, 'error': str(e)}


def format_criteria_for_ai(criteria_weights, criteria_details):
    """Criteria'ları AI için formatla"""
    text = ""
    for category, weight in criteria_weights.items():
        text += f"\n{category} (Puan: {weight}):\n"
        if category in criteria_details:
            for keyword in criteria_details[category]:
                text += f"  - {keyword}\n"
    return text


def format_extract_values_for_ai(extract_values):
    """Extract values'ları AI için formatla"""
    if not extract_values:
        return "Yok"
    return "\n".join([f"  - {field}" for field in extract_values])


def get_excluded_keywords_pool(current_doc_type_id=None):
    """Mevcut excluded keywords pool'unu döndür"""
    base_pool = [
        "hrc","cobot","robot","çarpışma","collaborative","kolaboratif","sd conta",
        "elektrik", "devre", "şema", "circuit", "electrical", "voltage", "amper", "ohm", "enclosure", "wrp-", "light curtain", "contactors", "controller",
        "espe",
        "hidrolik", "HİDROLİK", "hydraulic", "hidrolik yağ", "hydraulic oil", "iso 1219", "1219", "teknik resim",
        "gürültü", "noise", "ses", "sound", "decibel", "db", "akustik", "acoustic",
        "kullanma", "kılavuz", "manual", "instruction", "talimat", "guide", "kılavuzu",
        "loto",
        "lvd", "TOPRAKLAMA SÜREKLİLİK", "topraklama süreklilik", "TOPRAKLAMA İLETKENLERİ", "topraklama iletkenleri",
        "uygunluk", "beyan", "muayene", "conformity", "declaration", "declare",
        "isg", "periyodik", "kontrol", "periodic", "inspection", "denetim",
        "pnömatik", "pnomatik", "pneumatic", "lubricator", "inflate", "psi", "bar", "regis", "r102", "regulator", "dump valve", "oil",
        "montaj", "assembly",
        "topraklama direnci", "grounding", "earthing", "60204", "topraklama", "TOPRAKLAMA DİRENCİ",
        "bakım", "maintenance", "servis", "service", "bakim", "MAINTENCE",
        "titreşim", "vibration", "mekanik",
        "aydınlatma", "lighting", "illumination", "lux", "lümen", "lumen", "ts en 12464", "en 12464", "ışık", "ışık şiddeti",
        "AT TİP", "at tip", "ec type", "SERTİFİKA", "sertifika", "certificate",
    ]

    logger.info(f"📦 Base pool: {len(base_pool)} kelime")

    # ============================================
    # 2. DİĞER DYNAMIC SERVİSLERİN STRONG KEYWORDS'LERİ
    # ============================================
    try:
        # SADECE strong_keywords (critical_terms DEĞİL!)
        query = db.session.query(ValidationKeyword).join(DocumentType, ValidationKeyword.document_type_id == DocumentType.id).filter(ValidationKeyword.keyword_type == 'strong_keywords',DocumentType.service_file == 'dynamic_service.py')
        # Kendisini hariç tut
        if current_doc_type_id:
            query = query.filter(ValidationKeyword.document_type_id != current_doc_type_id)
        
        other_strong = query.all()
        
        # Diğer servislerin strong keywords'lerini ekle
        for vk in other_strong:
            base_pool.extend([kw.lower().strip() for kw in vk.keywords])
        
        logger.info(f"➕ Dynamic servisler: {len(other_strong)} servis, strong keywords eklendi")
        
    except Exception as e:
        logger.error(f"❌ Dynamic keywords hatası: {e}")
    
    # ============================================
    # 3. TEMİZLEME
    # ============================================
    # Unique + lowercase + boşluksuz
    final_pool = list(set([kw.lower().strip() for kw in base_pool if kw.strip()]))
    
    logger.info(f"✅ FINAL excluded pool: {len(final_pool)} unique kelime")
    
    return final_pool

with app.app_context():
    load_document_handlers()
    logger.info(f"✅ Gunicorn için {len(DOCUMENT_HANDLERS)} handler yüklendi")


# ============================================
# APPLICATION ENTRY POINT (Azure-Friendly)
# ============================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("PILZ Report Checker - Main API Gateway (PostgreSQL)")
    logger.info("=" * 60)

    logger.info(f"🔧 Mimari: Database-driven (PostgreSQL)")
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