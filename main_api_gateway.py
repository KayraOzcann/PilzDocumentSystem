from flask import Flask, request, jsonify, render_template
import os
import subprocess
import time
import requests
from werkzeug.utils import secure_filename
import logging
import threading
import signal
import sys
from typing import Dict, List, Optional
import json
from datetime import datetime
import queue
import uuid
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
FILE_BASE_URL = "https://safetyexpert.app/fileupload/Account_103/Machine_4879/"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads_main'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size


class RequestQueue:
    """Manages queued analysis requests when all ports are busy"""
    
    def __init__(self, max_queue_size: int = 10, max_wait_time: int = 300, server_manager=None):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.max_wait_time = max_wait_time  # 5 minutes max wait
        self.active_requests = {}  # {request_id: {'status': str, 'result': dict, 'timestamp': float}}
        self._lock = threading.Lock()
        self.server_manager = server_manager  # Reference to server manager for cleanup
        
        # Start queue processor thread
        self.processor_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processor_thread.start()
    
    def add_request(self, request_data: dict) -> str:
        """Add a request to the queue and return request ID"""
        request_id = str(uuid.uuid4())
        
        try:
            request_item = {
                'id': request_id,
                'data': request_data,
                'timestamp': time.time()
            }
            
            self.queue.put(request_item, block=False)
            
            with self._lock:
                self.active_requests[request_id] = {
                    'status': 'queued',
                    'result': None,
                    'timestamp': time.time()
                }
            
            logger.info(f"Request {request_id} added to queue. Queue size: {self.queue.qsize()}")
            return request_id
            
        except queue.Full:
            logger.warning("Request queue is full")
            return None
    
    def get_request_status(self, request_id: str) -> dict:
        """Get status of a queued request"""
        with self._lock:
            if request_id in self.active_requests:
                return self.active_requests[request_id]
            return {'status': 'not_found', 'result': None, 'timestamp': None}
    
    def _process_queue(self):
        """Background thread to process queued requests"""
        while True:
            try:
                # Wait for a request from queue
                request_item = self.queue.get(timeout=1)
                request_id = request_item['id']
                request_data = request_item['data']
                
                # Check if request is too old
                if time.time() - request_item['timestamp'] > self.max_wait_time:
                    with self._lock:
                        self.active_requests[request_id] = {
                            'status': 'timeout',
                            'result': {'error': 'Request timeout', 'message': 'Request waited too long in queue'},
                            'timestamp': time.time()
                        }
                    continue
                
                # Update status to processing
                with self._lock:
                    if request_id in self.active_requests:
                        self.active_requests[request_id]['status'] = 'processing'
                
                # Try to process the request
                try:
                    result = self._execute_analysis(request_data)
                    with self._lock:
                        self.active_requests[request_id] = {
                            'status': 'completed',
                            'result': result,
                            'timestamp': time.time()
                        }
                    
                    # Clean up server after queue processing completion
                    if 'service_port' in result and self.server_manager:
                        port = result['service_port']
                        logger.info(f"Queue analysis completed for request {request_id} on port {port}, scheduling cleanup...")
                        
                        def cleanup_queued_server():
                            import time
                            time.sleep(2)  # Wait before cleanup
                            self.server_manager.stop_server_by_port(port)
                            logger.info(f"Queued server on port {port} stopped and port released back to pool")
                        
                        cleanup_thread = threading.Thread(target=cleanup_queued_server, daemon=True)
                        cleanup_thread.start()
                        
                except Exception as e:
                    logger.error(f"Error processing queued request {request_id}: {e}")
                    with self._lock:
                        self.active_requests[request_id] = {
                            'status': 'error',
                            'result': {'error': 'Processing error', 'message': str(e)},
                            'timestamp': time.time()
                        }
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                time.sleep(1)
    
    def _execute_analysis(self, request_data: dict) -> dict:
        """Execute the actual analysis when port becomes available"""
        script_name = request_data['script_name']
        health_endpoint = request_data['health_endpoint']
        
        # Wait for available port with timeout
        max_attempts = 60  # 60 seconds
        for attempt in range(max_attempts):
            port = server_manager.get_or_start_server(script_name, health_endpoint)
            if port is not None:
                break
            time.sleep(1)
        
        if port is None:
            raise Exception("No available ports after waiting")
        
        # Process the file analysis
        return self._forward_to_analysis_server(port, request_data)
    
    def _forward_to_analysis_server(self, port: int, request_data: dict) -> dict:
        """Forward request to analysis server"""
        filepath = request_data['filepath']
        filename = request_data['filename']
        endpoint_path = request_data['endpoint_path']
        form_data = request_data['form_data']
        
        target_url = f"http://localhost:{port}{endpoint_path}"
        
        # Determine MIME type
        file_extension = filename.rsplit('.', 1)[1].lower()
        mime_types = {
            'pdf': 'application/pdf',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png'
        }
        mime_type = mime_types.get(file_extension, 'application/octet-stream')
        
        with open(filepath, 'rb') as f:
            files = {'file': (filename, f, mime_type)}
            
            response = requests.post(
                target_url,
                files=files,
                data=form_data,
                timeout=3600  # 60 minute timeout
            )
        
        # Keep file for issue reporting - will be auto-deleted after 1 hour
        # try:
        #     os.remove(filepath)
        # except:
        #     pass

        if response.status_code == 200:
            result = response.json()
            result['service_port'] = port
            return result
        else:
            raise Exception(f"Analysis service error: {response.status_code} - {response.text[:500]}")
    
    def cleanup_old_requests(self):
        """Clean up old completed requests"""
        current_time = time.time()
        with self._lock:
            old_requests = [
                req_id for req_id, req_info in self.active_requests.items()
                if current_time - req_info['timestamp'] > 3600  # 1 hour
            ]
            for req_id in old_requests:
                del self.active_requests[req_id]


class PortManager:
    """Manages dynamic port allocation for analysis servers"""
    
    def __init__(self):
        # Available ports for dynamic allocation (RunPod constraint: max 15 ports, 5001 is fixed)
        self.AVAILABLE_PORTS = [5002, 5003, 5004, 5005, 5006]
        self.used_ports = {}  # {port: {'script': str, 'process': subprocess, 'start_time': str, 'status': str}}
        self._lock = threading.Lock()
    
    def get_available_port(self) -> Optional[int]:
        """Get an available port from the pool"""
        with self._lock:
            for port in self.AVAILABLE_PORTS:
                if port not in self.used_ports:
                    return port
            return None
    
    def allocate_port(self, port: int, script: str, process: subprocess.Popen) -> bool:
        """Allocate a port to a script with process info"""
        with self._lock:
            if port in self.used_ports:
                return False
            
            self.used_ports[port] = {
                'script': script,
                'process': process,
                'start_time': datetime.now().isoformat(),
                'status': 'starting'
            }
            return True
    
    def release_port(self, port: int) -> bool:
        """Release a port back to the available pool"""
        with self._lock:
            if port in self.used_ports:
                del self.used_ports[port]
                logger.info(f"Released port {port} back to the pool")
                return True
            return False
    
    def is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        with self._lock:
            return port not in self.used_ports
    
    def update_port_status(self, port: int, status: str):
        """Update the status of a port"""
        with self._lock:
            if port in self.used_ports:
                self.used_ports[port]['status'] = status
    
    def get_port_info(self) -> Dict:
        """Get information about all ports"""
        with self._lock:
            return {
                'available_ports': [p for p in self.AVAILABLE_PORTS if p not in self.used_ports],
                'used_ports': dict(self.used_ports),
                'total_available': len(self.AVAILABLE_PORTS),
                'total_used': len(self.used_ports)
            }


class DynamicServerManager:
    """Manages dynamic server instances"""
    
    def __init__(self, port_manager: PortManager):
        self.port_manager = port_manager
        self.startup_timeout = 30
        self._cleanup_lock = threading.Lock()
    
    def start_server_with_env_port(self, script: str, port: int) -> Optional[subprocess.Popen]:
        """Start a server with environment variable port assignment"""
        try:
            # Set up environment with assigned port
            env = os.environ.copy()
            env['ASSIGNED_PORT'] = str(port)
            
            # Activate virtual environment and start the server
            cmd = f"source venv/bin/activate && python3 {script}"
            process = subprocess.Popen(
                cmd,
                shell=True,
                executable='/bin/zsh',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
                env=env
            )
            
            # Allocate the port in port manager
            if self.port_manager.allocate_port(port, script, process):
                logger.info(f"Started {script} on port {port} with PID {process.pid}")
                return process
            else:
                logger.error(f"Failed to allocate port {port} for {script}")
                process.terminate()
                return None
                
        except Exception as e:
            logger.error(f"Failed to start {script} on port {port}: {str(e)}")
            return None
    
    def wait_for_server_ready(self, port: int, health_endpoint: str = '/api/health', timeout: int = 30) -> bool:
        """Wait for a server to be ready and responsive"""
        start_time = time.time()
        health_url = f"http://localhost:{port}{health_endpoint}"
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    self.port_manager.update_port_status(port, 'running')
                    logger.info(f"Server on port {port} is healthy at {health_endpoint}")
                    return True
            except requests.exceptions.RequestException as e:
                logger.debug(f"Health check failed for port {port}: {e}")
                pass
            time.sleep(1)
        
        # Server failed to start, update status
        logger.error(f"Server on port {port} health check timeout after {timeout}s")
        self.port_manager.update_port_status(port, 'failed')
        return False
    
    def stop_server_by_port(self, port: int) -> bool:
        """Stop a server running on a specific port"""
        with self._cleanup_lock:
            port_info = self.port_manager.used_ports.get(port)
            if not port_info:
                return False
            
            process = port_info['process']
            try:
                # Kill the process group to ensure all child processes are terminated
                if process.poll() is None:  # Process is still running
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if not responding
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait(timeout=2)
                
                logger.info(f"Stopped server on port {port}")
                
            except (ProcessLookupError, OSError) as e:
                logger.warning(f"Process on port {port} already terminated: {e}")
            except Exception as e:
                logger.error(f"Error stopping server on port {port}: {e}")
                return False
            finally:
                # Always release the port
                self.port_manager.release_port(port)
            
            return True
    
    def cleanup_dead_servers(self):
        """Clean up any dead/zombie processes and release their ports"""
        with self._cleanup_lock:
            dead_ports = []
            
            for port, info in self.port_manager.used_ports.items():
                process = info['process']
                
                # Check if process is still alive
                try:
                    if process.poll() is not None:  # Process has terminated
                        dead_ports.append(port)
                        logger.info(f"Found dead process on port {port}, cleaning up")
                except:
                    dead_ports.append(port)
            
            # Clean up dead ports
            for port in dead_ports:
                self.port_manager.release_port(port)
    
    def get_or_start_server(self, script: str, health_endpoint: str = '/api/health') -> Optional[int]:
        """Get an existing server or start a new one for the script"""
        # Check if script is already running
        for port, info in self.port_manager.used_ports.items():
            if info['script'] == script and info['status'] == 'running':
                logger.info(f"Reusing existing server for {script} on port {port}")
                return port
        
        # Get available port
        port = self.port_manager.get_available_port()
        if port is None:
            logger.warning("No available ports for new server")
            return None
        
        # Start new server
        process = self.start_server_with_env_port(script, port)
        if process is None:
            return None
        
        # Wait for server to be ready with specific health endpoint
        if self.wait_for_server_ready(port, health_endpoint, self.startup_timeout):
            logger.info(f"Server {script} is ready on port {port}")
            return port
        else:
            logger.error(f"Server {script} failed to start on port {port}")
            self.stop_server_by_port(port)
            return None


# Initialize managers
port_manager = PortManager()
server_manager = DynamicServerManager(port_manager)
request_queue = RequestQueue(max_queue_size=20, max_wait_time=300, server_manager=server_manager)  # 20 requests, 5 min wait

# API Server Configuration - Now only contains script mapping, ports are dynamic
API_SERVERS = {
    'electric_circuit': {
        'script': 'server.py',
        'endpoint': '/api/elektrik-report',
        'health_endpoint': '/api/health',
        'description': 'Elektrik Devre Şeması Analizi'
    },
    'espe_report': {
        'script': 'server2.py',
        'endpoint': '/api/espe-report',
        'health_endpoint': '/api/espe-health',
        'description': 'ESPE Raporu Analizi'
    },
    'noise_report': {
        'script': 'server4.py',
        'endpoint': '/api/noise-report',
        'health_endpoint': '/api/health',
        'description': 'Gürültü Ölçüm Raporu Analizi'
    },
    'manuel_report': {
        'script': 'server5.py',
        'endpoint': '/api/manuel-report',
        'health_endpoint': '/api/manuel-health',
        'description': 'Manuel Raporu Analizi'
    },
    'loto_report': {
        'script': 'server6.py',
        'endpoint': '/api/loto-report',
        'health_endpoint': '/api/loto-health',
        'description': 'LOTO Raporu Analizi'
    },
    'lvd_report': {
        'script': 'server7.py',
        'endpoint': '/api/lvd-report',
        'health_endpoint': '/api/lvd-health',
        'description': 'LVD Raporu Analizi'
    },
    'at_type_inspection': {
        'script': 'server8.py',
        'endpoint': '/api/at-declaration',
        'health_endpoint': '/api/at-health',
        'description': 'AT Tip Muayene Analizi'
    },
    'isg_periyodik_kontrol': {
        'script': 'server9.py',
        'endpoint': '/api/isg-control',
        'health_endpoint': '/api/isg-health',
        'description': 'İSG Periyodik Kontrol Analizi'
    },
    'pneumatic_circuit': {
        'script': 'server10.py',
        'endpoint': '/api/pnomatic-control',
        'health_endpoint': '/api/pnomatic-health',
        'description': 'Pnömatik Devre Şeması Analizi'
    },
    'hydraulic_circuit': {
        'script': 'server11.py',
        'endpoint': '/api/hydraulic-control',
        'health_endpoint': '/api/health',
        'description': 'Hidrolik Devre Şeması Analizi'
    },
    'assembly_instructions': {
        'script': 'server12.py',
        'endpoint': '/api/assembly-instructions',
        'health_endpoint': '/api/health',
        'description': 'Montaj Talimatları Analizi'
    },
    'grounding_report': {
        'script': 'server13.py',
        'endpoint': '/api/topraklama-report',
        'health_endpoint': '/api/health',
        'description': 'EN 60204-1 Topraklama Raporu Analizi'
    },
    'hrc_report': {
        'script': 'server14.py',
        'endpoint': '/api/hrc-report',
        'health_endpoint': '/api/health',
        'description': 'HRC Kuvvet-Basınç Raporu Analizi'
    },
    'maintenance_instructions': {
        'script': 'server15.py',
        'endpoint': '/api/bakimtalimatlari-report',
        'health_endpoint': '/api/health',
        'description': 'Bakım Talimatları Analizi'
    },
    'vibration_report': {
        'script': 'server16.py',
        'endpoint': '/api/titresim-report',
        'health_endpoint': '/api/health',
        'description': 'Mekanik Titreşim Raporu Analizi'
    },
    'lighting_report': {
        'script': 'server17.py',
        'endpoint': '/api/aydinlatma-report',
        'health_endpoint': '/api/health',
        'description': 'Aydınlatma Raporu Analizi'
    },
    'at_certificate': {
        'script': 'server18.py',
        'endpoint': '/api/ati-inceleme-report',
        'health_endpoint': '/api/health',
        'description': 'AT Tip İnceleme Sertifikası Analizi'
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_all_servers():
    """Clean up all server processes"""
    logger.info("Cleaning up all server processes...")
    
    # Stop all servers managed by the dynamic server manager
    for port in list(port_manager.used_ports.keys()):
        server_manager.stop_server_by_port(port)
    
    # Cleanup any dead servers
    server_manager.cleanup_dead_servers()

def cleanup_old_temp_files():
    """1 saatten eski temp dosyalarını sil"""
    try:
        temp_folder = UPLOAD_FOLDER  # 'temp_uploads_main'
        current_time = time.time()
        max_age = 2 * 60 # 1 dakika test amaçlı (saniye cinsinden - 60 saniye)
        #60 * 60  # 1 saat (saniye cinsinden)
        
        if not os.path.exists(temp_folder):
            return
        
        deleted_count = 0
        for filename in os.listdir(temp_folder):
            filepath = os.path.join(temp_folder, filename)
            
            # Sadece dosyaları kontrol et
            if os.path.isfile(filepath):
                try:
                    file_modified_time = os.path.getmtime(filepath)
                    file_age = current_time - file_modified_time
                    
                    # DEBUG LOG
                    logger.info(f"Dosya: {filename}, Yaş: {file_age:.1f} saniye ({file_age/60:.1f} dakika)")
                    
                    if file_age > max_age:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.info(f"✓ Eski temp dosyası silindi: {filename}")
                    else:
                        logger.info(f"✗ Dosya henüz yeni: {filename} (bekliyor)")
                        
                except Exception as e:
                    logger.error(f"Dosya işlenirken hata {filename}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Toplam {deleted_count} eski temp dosyası temizlendi")
            
    except Exception as e:
        logger.error(f"Temp dosyaları temizlenirken hata: {e}")

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, cleaning up...")
    cleanup_all_servers()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.route('/api/services', methods=['GET'])
def get_services():
    """Get list of available analysis services with current port status"""
    services = []
    port_info = port_manager.get_port_info()
    
    for service_name, config in API_SERVERS.items():
        # Find if this service is currently running
        running_port = None
        for port, info in port_manager.used_ports.items():
            if info['script'] == config['script'] and info['status'] == 'running':
                running_port = port
                break
        
        services.append({
            'service_name': service_name,
            'description': config['description'],
            'script': config['script'],
            'endpoint': config['endpoint'],
            'status': 'running' if running_port else 'stopped',
            'current_port': running_port
        })
    
    return jsonify({
        'success': True,
        'services': services,
        'total_services': len(services),
        'port_manager_info': port_info
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    """Main endpoint to analyze documents with dynamic port allocation"""
    try:
        # Check if a file was sent in the request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please provide a PDF file in the request'
            }), 400

        # Check if document type was specified
        if 'document_type' not in request.form:
            return jsonify({
                'error': 'No document type specified',
                'message': 'Please specify the document_type parameter',
                'available_types': list(API_SERVERS.keys())
            }), 400

        file = request.files['file']
        document_type = request.form['document_type']
        # OPSIYONEL: Dış sistemden gelen document_url
        custom_document_url = request.form.get('document_url', None)
        logger.info(f"Custom document_url: {custom_document_url if custom_document_url else 'Yok (varsayılan kullanılacak)'}")

        # OPSIYONEL: İlk yorum parametreleri
        initial_comment = request.form.get('initial_comment', '').strip()
        comment_author = request.form.get('comment_author', '').strip()
        
        if initial_comment:
            logger.info(f"İlk yorum var - Yazar: {comment_author if comment_author else 'Anonim'}, Uzunluk: {len(initial_comment)} karakter")
        else:
            logger.info("İlk yorum yok - Ticket 'Kapalı' statusünde oluşacak")


        # Check if a file was selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a file to upload'
            }), 400

        # Check if the file is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': 'Only PDF, JPG, JPEG, and PNG files are allowed'
            }), 400

        # Check if document type is valid
        if document_type not in API_SERVERS:
            return jsonify({
                'error': 'Invalid document type',
                'message': f'Document type "{document_type}" is not supported',
                'available_types': list(API_SERVERS.keys())
            }), 400

        # Clean up any dead servers before starting
        server_manager.cleanup_dead_servers()

        # Get target server configuration
        target_server_config = API_SERVERS[document_type]
        script_name = target_server_config['script']
        endpoint_path = target_server_config['endpoint']
        health_endpoint = target_server_config['health_endpoint']

        logger.info(f"Starting analysis for {document_type} using {script_name}")

        # Get or start server for this analysis
        port = server_manager.get_or_start_server(script_name, health_endpoint)
        
        # If no port available, add to queue
        if port is None:
            # Save file temporarily for queue processing
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Prepare request data for queue
            request_data = {
                'script_name': script_name,
                'health_endpoint': health_endpoint,
                'endpoint_path': endpoint_path,
                'filepath': filepath,
                'filename': filename,
                'form_data': {key: value for key, value in request.form.items() if key != 'document_type'},
                'document_type': document_type,
                'target_server_config': target_server_config
            }
            
            # Add to queue
            request_id = request_queue.add_request(request_data)
            
            if request_id is None:
                # Queue is full
                # Keep file for issue reporting - will be auto-deleted after 1 hour
                # try:
                #     os.remove(filepath)
                # except:
                #     pass
                return jsonify({
                    'error': 'Service overloaded',
                    'message': 'All analysis servers are busy and the queue is full. Please try again later.',
                    'queue_status': 'full'
                }), 503
            
            # Return queue information
            return jsonify({
                'status': 'queued',
                'message': 'Your request has been queued for processing.',
                'request_id': request_id,
                'estimated_wait_time': f"{request_queue.queue.qsize() * 30} seconds",
                'queue_position': request_queue.queue.qsize(),
                'check_status_url': f'/api/status/{request_id}',
                'instructions': 'Use the request_id to check status and get results when ready.'
            }), 202  # Accepted

        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        target_url = f"http://localhost:{port}{endpoint_path}"
        logger.info(f"Forwarding {document_type} analysis to {target_url}")
        
        # Log file size for timeout estimation
        file_size = os.path.getsize(filepath)
        logger.info(f"Analyzing file: {filename} ({file_size} bytes) on port {port}")

        # Forward the request to the appropriate analysis server
        try:
            # Determine MIME type based on file extension
            file_extension = filename.rsplit('.', 1)[1].lower()
            mime_types = {
                'pdf': 'application/pdf',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'png': 'image/png'
            }
            mime_type = mime_types.get(file_extension, 'application/octet-stream')
            
            with open(filepath, 'rb') as f:
                files = {'file': (filename, f, mime_type)}
                
                # Add any additional form data if present
                data = {}
                for key, value in request.form.items():
                    if key != 'document_type':
                        data[key] = value
                
                response = requests.post(
                    target_url,
                    files=files,
                    data=data,
                    timeout=3600  
                )

            # Clean up temporary file
            # Keep file for issue reporting - will be auto-deleted after 1 hour
            # try:
            #     os.remove(filepath)
            # except:
            #     pass

            if response.status_code == 200:
                result = response.json()
                result['analysis_service'] = document_type
                result['service_description'] = target_server_config['description']
                result['service_port'] = port

                # OTOMATIK TICKET OLUŞTUR
                try:
                    # Ticket verisi hazırla
                    ticket_data = {
                        'inspector_name': '',  # Boş - API üzerinden geldi
                        'inspector_comment': '',  # Boş - otomatik "Kapalı" olacak
                        'document_name': result.get('data', {}).get('filename', filename),
                        'analysis_data': result.get('data', result)
                    }
                    
                    # Tickets klasörünü kontrol et
                    tickets_dir = 'tickets'
                    if not os.path.exists(tickets_dir):
                        os.makedirs(tickets_dir)
                    
                    tickets_file = os.path.join(tickets_dir, 'tickets.json')
                    
                    # Mevcut tickets'ları oku
                    tickets = []
                    if os.path.exists(tickets_file):
                        try:
                            with open(tickets_file, 'r', encoding='utf-8') as f:
                                tickets = json.load(f)
                        except:
                            tickets = []
                    
                    # Aynı analysis_id ile ticket var mı kontrol et
                    analysis_id = ticket_data['analysis_data'].get('analysis_id')
                    existing_ticket = next((t for t in tickets if t.get('ticket_id') == analysis_id), None)
                    
                    if not existing_ticket:
                        # Yeni ticket oluştur
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
                            'status': 'İnceleniyor' if initial_comment else 'Kapalı',  # İlk yorum varsa İnceleniyor,  # Yorum yok, otomatik kapalı
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
                        
                        # Kaydet
                        with open(tickets_file, 'w', encoding='utf-8') as f:
                            json.dump(tickets, f, ensure_ascii=False, indent=2)
                        
                        logger.info(f"Otomatik ticket oluşturuldu: {ticket_no} (ID: {analysis_id})")
                        
                        # Result'a ticket bilgisini ekle
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
                    # Ticket hatası analiz sonucunu etkilemesin
                    result['ticket_error'] = str(e)
                
                # IMPORTANT: Kill the server and release port after analysis completion
                logger.info(f"Analysis completed for {document_type} on port {port}, cleaning up server...")
                
                # Schedule cleanup in a separate thread to avoid blocking the response
                def cleanup_after_delay():
                    import time
                    time.sleep(2)  # Wait 2 seconds to ensure response is sent
                    server_manager.stop_server_by_port(port)
                    logger.info(f"Server on port {port} stopped and port released back to pool")
                
                cleanup_thread = threading.Thread(target=cleanup_after_delay, daemon=True)
                cleanup_thread.start()
                
                return jsonify(result)
            else:
                # Analysis failed, clean up the server and release port
                logger.error(f"Analysis service returned error {response.status_code} on port {port}: {response.text}")
                
                # Schedule cleanup for failed analysis
                def cleanup_failed_server():
                    import time
                    time.sleep(1)  # Brief wait
                    server_manager.stop_server_by_port(port)
                    logger.info(f"Failed server on port {port} stopped and port released back to pool")
                
                cleanup_thread = threading.Thread(target=cleanup_failed_server, daemon=True)
                cleanup_thread.start()
                
                return jsonify({
                    'error': 'Analysis service error',
                    'message': f'The analysis service returned an error: {response.status_code}',
                    'service_port': port,
                    'service_response': response.text if response.text else 'No response text'
                }), response.status_code

        except requests.exceptions.ConnectionError:
            # Server might have crashed, clean it up
            server_manager.stop_server_by_port(port)
            return jsonify({
                'error': 'Service unavailable',
                'message': f'The {document_type} analysis service is not responding. The service has been restarted.',
                'service_port': port
            }), 503

        except requests.exceptions.Timeout:
            # For timeout, clean up server and suggest async processing
            logger.error(f"Analysis timeout on port {port}, cleaning up server")
            
            def cleanup_timeout_server():
                import time
                time.sleep(1)
                server_manager.stop_server_by_port(port)
                logger.info(f"Timeout server on port {port} stopped and port released back to pool")
            
            cleanup_thread = threading.Thread(target=cleanup_timeout_server, daemon=True)
            cleanup_thread.start()
            
            return jsonify({
                'error': 'Analysis timeout',
                'message': 'The analysis took too long to complete (>60 minutes). This usually happens with very large or complex documents.',
                'suggestion': 'Try using smaller files or use the async analysis endpoint: POST /api/analyze-async',
                'service_port': port,
                'file_size': file_size if 'file_size' in locals() else 'unknown'
            }), 504

        except Exception as e:
            logger.error(f"Error forwarding request: {str(e)} on port {port}")
            
            # Clean up server on general errors
            def cleanup_error_server():
                import time
                time.sleep(1)  
                server_manager.stop_server_by_port(port)
                logger.info(f"Error server on port {port} stopped and port released back to pool")
            
            cleanup_thread = threading.Thread(target=cleanup_error_server, daemon=True)
            cleanup_thread.start()
            
            return jsonify({
                'error': 'Internal server error',
                'message': f'An error occurred while processing the request: {str(e)}',
                'service_port': port
            }), 500

    except Exception as e:
        logger.error(f"Error in analyze_document: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/analyze-async', methods=['POST'])
def analyze_document_async():
    """Asynchronous document analysis - always uses queue system"""
    try:
        # Validation (same as sync version)
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please provide a PDF file in the request'
            }), 400

        if 'document_type' not in request.form:
            return jsonify({
                'error': 'No document type specified',
                'message': 'Please specify the document_type parameter',
                'available_types': list(API_SERVERS.keys())
            }), 400

        file = request.files['file']
        document_type = request.form['document_type']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        if document_type not in API_SERVERS:
            return jsonify({
                'error': 'Invalid document type',
                'available_types': list(API_SERVERS.keys())
            }), 400

        # Get server configuration
        target_server_config = API_SERVERS[document_type]
        script_name = target_server_config['script']
        endpoint_path = target_server_config['endpoint']
        health_endpoint = target_server_config['health_endpoint']

        # Always use queue for async requests
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        file_size = os.path.getsize(filepath)
        logger.info(f"Async analysis queued: {filename} ({file_size} bytes)")

        # Prepare request data for queue
        request_data = {
            'script_name': script_name,
            'health_endpoint': health_endpoint,
            'endpoint_path': endpoint_path,
            'filepath': filepath,
            'filename': filename,
            'form_data': {key: value for key, value in request.form.items() if key != 'document_type'},
            'document_type': document_type,
            'target_server_config': target_server_config
        }

        # Add to queue
        request_id = request_queue.add_request(request_data)

        if request_id is None:
            # Keep file for issue reporting - will be auto-deleted after 1 hour
            # try:
            #     os.remove(filepath)
            # except:
            #     pass
            return jsonify({
                'error': 'Queue full',
                'message': 'Analysis queue is currently full. Please try again later.'
            }), 503

        return jsonify({
            'status': 'accepted',
            'message': 'Analysis request accepted and queued for processing.',
            'request_id': request_id,
            'file_size': file_size,
            'estimated_processing_time': f"{max(30, file_size // 100000)} seconds",
            'queue_position': request_queue.queue.qsize(),
            'check_status_url': f'/api/status/{request_id}',
            'poll_interval': '10 seconds recommended'
        }), 202

    except Exception as e:
        logger.error(f"Error in analyze_document_async: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check the health of the system and currently running services"""
    port_info = port_manager.get_port_info()
    health_status = {}
    overall_healthy = True
    
    # Check status of running services
    for port, info in port_manager.used_ports.items():
        service_name = None
        health_endpoint = '/api/health'  # default
        
        # Find service name and health endpoint from script
        for name, config in API_SERVERS.items():
            if config['script'] == info['script']:
                service_name = name
                health_endpoint = config['health_endpoint']
                break
        
        try:
            health_url = f"http://localhost:{port}{health_endpoint}"
            response = requests.get(health_url, timeout=5)
            health_status[service_name or info['script']] = {
                'status': 'healthy',
                'port': port,
                'script': info['script'],
                'health_endpoint': health_endpoint,
                'start_time': info['start_time'],
                'process_status': info['status']
            }
        except:
            health_status[service_name or info['script']] = {
                'status': 'unhealthy',
                'port': port,
                'script': info['script'],
                'health_endpoint': health_endpoint,
                'start_time': info['start_time'],
                'process_status': info['status']
            }
            overall_healthy = False
    
    return jsonify({
        'overall_status': 'healthy' if overall_healthy else 'degraded',
        'running_services': health_status,
        'port_manager': port_info,
        'gateway_status': 'running',
        'gateway_port': 5001
    })

@app.route('/', methods=['GET'])
def index():
    """Main page with web interfaces"""
    return render_template('index.html')

@app.route('/api/info', methods=['GET'])
def api_info():
    """API documentation endpoint"""
    return jsonify({
        'message': 'PILZ Report Checker API Gateway',
        'version': '2.0.0',
        'description': 'Unified API for document analysis services with dynamic port management',
        'endpoints': {
            'GET /api/services': 'List all available analysis services',
            'POST /api/analyze': 'Analyze a document synchronously (60min timeout)',
            'POST /api/analyze-async': 'Analyze a document asynchronously (no timeout)',
            'GET /api/status/<request_id>': 'Check status of queued analysis request',
            'GET /api/queue/status': 'Get overall queue status',
            'GET /api/health': 'Check health status of running services',
            'GET /api/info': 'API documentation',
            'POST /api/save-evaluation': 'Save analysis evaluation',
            'GET /api/ports': 'Get port allocation status',
            'POST /api/ports/cleanup': 'Cleanup dead servers and old requests',
            'GET /': 'Web interface'
        },
        'document_types': list(API_SERVERS.keys()),
        'port_management': {
            'gateway_port': 5001,
            'dynamic_port_pool': port_manager.AVAILABLE_PORTS,
            'current_allocations': port_manager.get_port_info()
        },
        'usage': {
            'analyze_endpoint': '/api/analyze',
            'method': 'POST',
            'required_fields': ['file (PDF/JPG/JPEG/PNG)', 'document_type'],
            'supported_formats': ['PDF', 'JPG', 'JPEG', 'PNG'],
            'max_file_size': '32MB',
            'example_curl': 'curl -X POST -F "file=@document.pdf" -F "document_type=electric_circuit" http://localhost:5001/api/analyze'
        }
    })

@app.route('/api/ports', methods=['GET'])
def get_port_status():
    """Get current port allocation status"""
    return jsonify(port_manager.get_port_info())

@app.route('/api/status/<request_id>', methods=['GET'])
def check_request_status(request_id):
    """Check the status of a queued analysis request"""
    try:
        status_info = request_queue.get_request_status(request_id)
        
        if status_info['status'] == 'not_found':
            return jsonify({
                'error': 'Request not found',
                'message': 'The requested analysis ID was not found or has expired.',
                'request_id': request_id
            }), 404
        
        response_data = {
            'request_id': request_id,
            'status': status_info['status'],
            'timestamp': status_info['timestamp']
        }
        
        if status_info['status'] == 'completed':
            response_data['result'] = status_info['result']
            response_data['message'] = 'Analysis completed successfully'
            
        elif status_info['status'] == 'processing':
            response_data['message'] = 'Your analysis is currently being processed'
            
        elif status_info['status'] == 'queued':
            queue_position = request_queue.queue.qsize()
            response_data['message'] = 'Your request is in the queue'
            response_data['queue_position'] = queue_position
            response_data['estimated_wait_time'] = f"{queue_position * 30} seconds"
            
        elif status_info['status'] == 'error':
            response_data['error'] = status_info['result']
            response_data['message'] = 'Analysis failed'
            
        elif status_info['status'] == 'timeout':
            response_data['error'] = status_info['result']
            response_data['message'] = 'Request timed out in queue'
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error checking request status: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/queue/status', methods=['GET'])
def get_queue_status():
    """Get overall queue status"""
    return jsonify({
        'queue_size': request_queue.queue.qsize(),
        'active_requests': len(request_queue.active_requests),
        'available_ports': port_manager.get_port_info()['available_ports'],
        'used_ports': len(port_manager.used_ports),
        'max_queue_size': request_queue.queue.maxsize,
        'max_wait_time': request_queue.max_wait_time
    })

@app.route('/api/ports/cleanup', methods=['POST'])
def cleanup_dead_ports():
    """Manually trigger cleanup of dead servers"""
    try:
        server_manager.cleanup_dead_servers()
        request_queue.cleanup_old_requests()  # Also cleanup old queue requests
        return jsonify({
            'success': True,
            'message': 'Dead servers and old requests cleaned up successfully',
            'port_status': port_manager.get_port_info(),
            'queue_status': {
                'queue_size': request_queue.queue.qsize(),
                'active_requests': len(request_queue.active_requests)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during cleanup: {str(e)}'
        }), 500

@app.route('/api/save-evaluation', methods=['POST'])
def save_evaluation():
    """Analiz değerlendirmesini JSON dosyasına kaydet"""
    try:
        evaluation_data = request.get_json()
        
        if not evaluation_data:
            return jsonify({
                'success': False,
                'message': 'Değerlendirme verisi bulunamadı'
            }), 400
        
        # JSON dosyası yolu
        evaluations_file = 'analysis_evaluations.json'
        
        # Mevcut değerlendirmeleri oku
        evaluations = []
        if os.path.exists(evaluations_file):
            try:
                import json
                with open(evaluations_file, 'r', encoding='utf-8') as f:
                    evaluations = json.load(f)
            except Exception as e:
                logger.warning(f"Mevcut değerlendirmeler okunamadı: {e}")
                evaluations = []
        
        # Yeni değerlendirmeyi ekle
        evaluations.append(evaluation_data)
        
        # JSON dosyasına kaydet
        import json
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
    """Tüm değerlendirmeleri JSON dosyasından oku"""
    try:
        evaluations_file = 'analysis_evaluations.json'
        
        if not os.path.exists(evaluations_file):
            return jsonify([])  # Boş array döndür
        
        # JSON dosyasını oku
        with open(evaluations_file, 'r', encoding='utf-8') as f:
            evaluations = json.load(f)
        
        # Tarihe göre sırala (en yeni önce)
        evaluations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify(evaluations)
        
    except Exception as e:
        logger.error(f"Değerlendirmeler okunurken hata: {str(e)}")
        return jsonify([])  # Hata durumunda boş array döndür

@app.route('/api/update-evaluation-notes', methods=['POST'])
def update_evaluation_notes():
    """Değerlendirme notlarını güncelle"""
    try:
        data = request.get_json()
        
        if not data or 'analysis_id' not in data:
            return jsonify({
                'success': False,
                'message': 'analysis_id gereklidir'
            }), 400
        
        analysis_id = data['analysis_id']
        new_notes = data.get('evaluation_notes', '')
        
        # JSON dosyası yolu
        evaluations_file = 'analysis_evaluations.json'
        
        if not os.path.exists(evaluations_file):
            return jsonify({
                'success': False,
                'message': 'Değerlendirme dosyası bulunamadı'
            }), 404
        
        # Mevcut değerlendirmeleri oku
        with open(evaluations_file, 'r', encoding='utf-8') as f:
            evaluations = json.load(f)
        
        # İlgili değerlendirmeyi bul ve güncelle
        updated = False
        for evaluation in evaluations:
            if evaluation.get('analysis_id') == analysis_id:
                evaluation['evaluation_notes'] = new_notes
                updated = True
                break
        
        if not updated:
            return jsonify({
                'success': False,
                'message': 'Değerlendirme bulunamadı'
            }), 404
        
        # JSON dosyasına kaydet
        with open(evaluations_file, 'w', encoding='utf-8') as f:
            json.dump(evaluations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Değerlendirme notları güncellendi: {analysis_id}")
        
        return jsonify({
            'success': True,
            'message': 'Notlar başarıyla güncellendi'
        })
        
    except Exception as e:
        logger.error(f"Not güncelleme hatası: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Not güncellenemedi: {str(e)}'
        }), 500

@app.route('/api/delete-evaluation', methods=['POST'])
def delete_evaluation():
    """Değerlendirmeyi sil"""
    try:
        data = request.get_json()
        
        if not data or 'analysis_id' not in data:
            return jsonify({
                'success': False,
                'message': 'analysis_id gereklidir'
            }), 400
        
        analysis_id = data['analysis_id']
        
        # JSON dosyası yolu
        evaluations_file = 'analysis_evaluations.json'
        
        if not os.path.exists(evaluations_file):
            return jsonify({
                'success': False,
                'message': 'Değerlendirme dosyası bulunamadı'
            }), 404
        
        # Mevcut değerlendirmeleri oku
        with open(evaluations_file, 'r', encoding='utf-8') as f:
            evaluations = json.load(f)
        
        # İlgili değerlendirmeyi bul ve sil
        original_count = len(evaluations)
        evaluations = [e for e in evaluations if e.get('analysis_id') != analysis_id]
        
        if len(evaluations) == original_count:
            return jsonify({
                'success': False,
                'message': 'Değerlendirme bulunamadı'
            }), 404
        
        # JSON dosyasına kaydet
        with open(evaluations_file, 'w', encoding='utf-8') as f:
            json.dump(evaluations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Değerlendirme silindi: {analysis_id}")
        
        return jsonify({
            'success': True,
            'message': 'Değerlendirme başarıyla silindi',
            'remaining_count': len(evaluations)
        })
        
    except Exception as e:
        logger.error(f"Değerlendirme silme hatası: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Değerlendirme silinemedi: {str(e)}'
        }), 500
    
@app.route('/api/report-issue', methods=['POST'])
def report_issue():
    """Sorun bildirimi kaydet ve ticket oluştur"""
    try:
        issue_data = request.get_json()
        
        if not issue_data:
            return jsonify({
                'success': False,
                'message': 'Sorun bildirimi verisi bulunamadı'
            }), 400
        
        # Gerekli alanları kontrol et
        required_fields = ['inspector_comment', 'analysis_data', 'document_name']
        for field in required_fields:
            if field not in issue_data:
                return jsonify({
                    'success': False,
                    'message': f'Eksik alan: {field}'
                }), 400
        
        # Tickets klasörünü oluştur
        tickets_dir = 'tickets'
        if not os.path.exists(tickets_dir):
            os.makedirs(tickets_dir)
            logger.info(f"Tickets klasörü oluşturuldu: {tickets_dir}")
        
        # Tickets JSON dosyası
        tickets_file = os.path.join('tickets', 'tickets.json')
        
        # Mevcut tickets'ları oku
        tickets = []
        if os.path.exists(tickets_file):
            try:
                with open(tickets_file, 'r', encoding='utf-8') as f:
                    tickets = json.load(f)
            except Exception as e:
                logger.warning(f"Mevcut tickets okunamadı: {e}")
                tickets = []
        
        # Yeni ticket numarası oluştur
        ticket_no = f"ticket{len(tickets) + 1}"
        
        # Dosyayı tickets klasörüne kopyala
        document_name = issue_data['document_name']
        source_file = os.path.join('temp_uploads_main', document_name)
        file_name = document_name  # Varsayılan değer

        if os.path.exists(source_file):
            file_name = os.path.basename(source_file)
            destination_file = os.path.join(tickets_dir, file_name)
            
            # Eğer aynı isimde dosya varsa, unique isim oluştur
            if os.path.exists(destination_file):
                base_name, ext = os.path.splitext(file_name)
                counter = 1
                while os.path.exists(destination_file):
                    file_name = f"{base_name}_{counter}{ext}"
                    destination_file = os.path.join(tickets_dir, file_name)
                    counter += 1
            
            # Dosyayı kopyala
            import shutil
            shutil.copy2(source_file, destination_file)
            document_path = destination_file
        else:
            document_path = "API üzerinden oluşturuldu - dosya yok"
        
        # Analysis data'dan bilgileri çıkar
        analysis_data = issue_data['analysis_data']
        
        # OPSIYONEL: analysis_data içinde custom_document_url varsa kullan
        custom_doc_url = analysis_data.get('custom_document_url', None)
        
        # Ticket verisini hazırla
        ticket_data = {
            'ticket_no': ticket_no,
            'ticket_id': analysis_data.get('analysis_id'),
            'document_name': analysis_data.get('filename', 'Bilinmiyor'),
            'document_type': analysis_data.get('file_type', 'Bilinmiyor'),
            #'document_path': document_path,
            'document_url': custom_doc_url if custom_doc_url else f"https://safetyexpert.app/fileupload/Account_103/Machine_4879/{analysis_data.get('filename', file_name)}",
            'opening_date': datetime.now().isoformat(),
            'last_updated': None,
            'closing_date': None,
            'status': 'Kapalı' if not issue_data['inspector_comment'].strip() else 'İnceleniyor',
            'responsible': 'Savaş Bey',
            'analysis_result': {
                'overall_score': analysis_data.get('overall_score', {}),
                'category_scores': analysis_data.get('category_scores', {}),
                'extracted_values': analysis_data.get('extracted_values', {}),
                'recommendations': analysis_data.get('recommendations', []),
                'summary': analysis_data.get('summary', '')
            },
            'inspector_comment': issue_data['inspector_comment']
        }

        # Eğer yorum varsa, comments array'ine ilk yorum olarak ekle
        if issue_data['inspector_comment'].strip():
            author_name = issue_data.get('inspector_name', '').strip()
            if not author_name:
                author_name = 'Kullanıcı'
            ticket_data['comments'] = [{
                'comment_id': 'comment_1',
                'author': author_name,  # Varsayılan yazar adı
                'text': issue_data['inspector_comment'],
                'timestamp': datetime.now().isoformat()
            }]
        else:
            ticket_data['comments'] = []
        
        # Yeni ticket'ı listeye ekle
        tickets.append(ticket_data)
        
        # JSON dosyasına kaydet
        with open(tickets_file, 'w', encoding='utf-8') as f:
            json.dump(tickets, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Ticket oluşturuldu: {ticket_no} - {ticket_data['document_name']}")
        
        return jsonify({
            'success': True,
            'message': 'Sorun bildirimi başarıyla kaydedildi',
            'ticket_no': ticket_no,
            'ticket_count': len(tickets)
        })
        
    except Exception as e:
        logger.error(f"Ticket oluşturma hatası: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Ticket oluşturulamadı: {str(e)}'
        }), 500
    
@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """Tüm tickets'ları JSON dosyasından oku"""
    try:
        tickets_file = os.path.join('tickets', 'tickets.json')
        
        if not os.path.exists(tickets_file):
            return jsonify([])  # Boş array döndür
        
        # JSON dosyasını oku
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

        # Tarih filtresi (açılma tarihi)
        if date_from:
            date_from_obj = datetime.fromisoformat(date_from)
            tickets = [t for t in tickets 
                      if datetime.fromisoformat(t.get('opening_date', '')) >= date_from_obj]
        
        if date_to:
            # Bitiş günü dahil olması için 23:59:59 ekle
            date_to_obj = datetime.fromisoformat(date_to).replace(hour=23, minute=59, second=59)
            tickets = [t for t in tickets 
                      if datetime.fromisoformat(t.get('opening_date', '')) <= date_to_obj]
        
        # Sorumlu filtresi
        if responsible_filter:
            tickets = [t for t in tickets 
              if responsible_filter in t.get('responsible', '').lower()]
            
        # Tarihe göre sırala (en yeni önce)
        tickets.sort(key=lambda x: x.get('opening_date', ''), reverse=True)
        
        return jsonify(tickets)
        
    except Exception as e:
        logger.error(f"Tickets okunurken hata: {str(e)}")
        return jsonify([])  # Hata durumunda boş array döndür

@app.route('/api/update-ticket-comment', methods=['POST'])
def update_ticket_comment():
    """Ticket'ın inspector_comment alanını güncelle (eski sistem uyumluluğu için)"""
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
        
        new_comment = data.get('inspector_comment', '')
        new_status = data.get('status', 'İnceleniyor')
        
        tickets_file = os.path.join('tickets', 'tickets.json')
        
        if not os.path.exists(tickets_file):
            return jsonify({'success': False, 'message': 'Tickets bulunamadı'}), 404
        
        with open(tickets_file, 'r', encoding='utf-8') as f:
            tickets = json.load(f)
        
        # HYBRID ARAMA: Önce ticket_id, sonra ticket_no
        ticket = None
        if ticket_id:
            ticket = next((t for t in tickets if t.get('ticket_id') == ticket_id), None)
        
        if not ticket and ticket_no:
            ticket = next((t for t in tickets if t.get('ticket_no') == ticket_no), None)
        
        if not ticket:
            return jsonify({'success': False, 'message': 'Ticket bulunamadı'}), 404
        
        ticket['inspector_comment'] = new_comment
        ticket['status'] = new_status
        ticket['last_updated'] = datetime.now().isoformat()
        
        # Kaydet
        with open(tickets_file, 'w', encoding='utf-8') as f:
            json.dump(tickets, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Ticket yorumu güncellendi: {ticket.get('ticket_no')} (ID: {ticket.get('ticket_id')})")
        
        return jsonify({
            'success': True,
            'message': 'Ticket başarıyla güncellendi'
        })
        
    except Exception as e:
        logger.error(f"Ticket güncelleme hatası: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/update-ticket', methods=['POST'])
def update_ticket():
    """Ticket bilgilerini güncelle (status, responsible, vb.)"""
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
        
        tickets_file = os.path.join('tickets', 'tickets.json')
        
        if not os.path.exists(tickets_file):
            return jsonify({'success': False, 'message': 'Tickets bulunamadı'}), 404
        
        with open(tickets_file, 'r', encoding='utf-8') as f:
            tickets = json.load(f)
        
        # HYBRID ARAMA: Önce ticket_id, sonra ticket_no
        ticket = None
        if ticket_id:
            ticket = next((t for t in tickets if t.get('ticket_id') == ticket_id), None)
        
        if not ticket and ticket_no:
            ticket = next((t for t in tickets if t.get('ticket_no') == ticket_no), None)
        
        if not ticket:
            return jsonify({'success': False, 'message': 'Ticket bulunamadı'}), 404
        
        old_status = ticket.get('status')
        
        # Status güncelleme
        if new_status:
            ticket['status'] = new_status
            
            # Kapalı statüsüne ÇEVRİLDİYSE kapanma tarihini ekle
            if new_status == 'Kapalı' and old_status != 'Kapalı':
                ticket['closing_date'] = datetime.now().isoformat()
            
            # Kapalı'dan başka bir statüye GEÇİLDİYSE kapanma tarihini sil
            elif new_status != 'Kapalı' and old_status == 'Kapalı':
                ticket['closing_date'] = None
        
        # Sorumlu güncelleme
        if new_responsible:
            ticket['responsible'] = new_responsible
        
        # Son güncelleme tarihini ekle
        ticket['last_updated'] = datetime.now().isoformat()
        
        with open(tickets_file, 'w', encoding='utf-8') as f:
            json.dump(tickets, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Ticket güncellendi: {ticket.get('ticket_no')} (ID: {ticket.get('ticket_id')})")
        
        return jsonify({'success': True, 'message': 'Ticket başarıyla güncellendi'})
        
    except Exception as e:
        logger.error(f"Ticket güncelleme hatası: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/add-comment', methods=['POST'])
def add_comment():
    """Ticket'a yeni yorum ekle"""
    try:
        data = request.get_json()
        
        # HYBRID: ticket_id veya ticket_no kabul et
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
        
        # HYBRID ARAMA: Önce ticket_id, sonra ticket_no
        ticket = None
        if ticket_id:
            ticket = next((t for t in tickets if t.get('ticket_id') == ticket_id), None)
        
        if not ticket and ticket_no:
            ticket = next((t for t in tickets if t.get('ticket_no') == ticket_no), None)
        
        if not ticket:
            return jsonify({'success': False, 'message': 'Ticket bulunamadı'}), 404
        
        # Comments array'i yoksa oluştur
        if 'comments' not in ticket:
            ticket['comments'] = []
        
        # Yeni comment_id oluştur
        comment_id = f"comment_{len(ticket['comments']) + 1}"
        
        # Yeni yorumu ekle
        new_comment = {
            'comment_id': comment_id,
            'author': author,
            'text': text,
            'timestamp': datetime.now().isoformat()
        }
        
        ticket['comments'].append(new_comment)
        ticket['last_updated'] = datetime.now().isoformat()
        
        # Kaydet
        with open(tickets_file, 'w', encoding='utf-8') as f:
            json.dump(tickets, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Yorum eklendi: {ticket.get('ticket_no')} (ID: {ticket.get('ticket_id')}) - {author}")
        
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
    """Ticket'ı sil"""
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
        
        tickets_file = os.path.join('tickets', 'tickets.json')
        
        if not os.path.exists(tickets_file):
            return jsonify({'success': False, 'message': 'Tickets bulunamadı'}), 404
        
        with open(tickets_file, 'r', encoding='utf-8') as f:
            tickets = json.load(f)
        
        # HYBRID ARAMA: Önce ticket_id, sonra ticket_no
        ticket_index = None
        ticket_info = None
        
        if ticket_id:
            for i, t in enumerate(tickets):
                if t.get('ticket_id') == ticket_id:
                    ticket_index = i
                    ticket_info = t
                    break
        
        if ticket_index is None and ticket_no:
            for i, t in enumerate(tickets):
                if t.get('ticket_no') == ticket_no:
                    ticket_index = i
                    ticket_info = t
                    break
        
        if ticket_index is None:
            return jsonify({'success': False, 'message': 'Ticket bulunamadı'}), 404
        
        # Ticket'ı sil
        deleted_ticket = tickets.pop(ticket_index)
        
        # Kaydet
        with open(tickets_file, 'w', encoding='utf-8') as f:
            json.dump(tickets, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Ticket silindi: {deleted_ticket.get('ticket_no')} (ID: {deleted_ticket.get('ticket_id')})")
        
        return jsonify({
            'success': True,
            'message': 'Ticket başarıyla silindi'
        })
        
    except Exception as e:
        logger.error(f"Ticket silme hatası: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def get_api_info():
    """Main API Gateway bilgilerini döndür"""
    return jsonify({
        'service': 'Pilz Report Analysis Gateway',
        'version': '2.0',
        'description': 'Unified API gateway for analyzing various types of reports',
        'available_endpoints': {
            '/': 'GET - Main web interface',
            '/api/analyze': 'POST - Analyze any supported document type',
            '/api/save-evaluation': 'POST - Save analysis evaluation',
            '/api/info': 'GET - This information',
            '/api/services': 'GET - List all available analysis services'
        },
        'document_types': list(API_SERVERS.keys()),
        'usage': {
            'required_fields': ['file (PDF/JPG/JPEG/PNG)', 'document_type'],
            'supported_formats': ['PDF', 'JPG', 'JPEG', 'PNG'],
            'max_file_size': '32MB',
            'example_curl': 'curl -X POST -F "file=@document.pdf" -F "document_type=electric_circuit" http://localhost:5001/api/analyze'
        }
    })

# Background scheduler for cleanup tasks
scheduler = BackgroundScheduler()
scheduler.add_job(func=cleanup_old_temp_files, trigger="interval", minutes=3)  # minutes = 20 Her 20 dakikada kontrol
scheduler.start()

# Shutdown scheduler on exit
atexit.register(lambda: scheduler.shutdown())

logger.info("Background cleanup scheduler başlatıldı (20 dakikada bir çalışacak)")


if __name__ == '__main__':
    logger.info("Starting main API gateway on port 5001")
    logger.info("Available document types: " + ", ".join(API_SERVERS.keys()))
    logger.info(f"Dynamic port pool: {port_manager.AVAILABLE_PORTS}")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)  # debug=False to avoid reloading issues
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        cleanup_all_servers()
