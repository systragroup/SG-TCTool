import os
import threading
import datetime
import logging

import uuid
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

from cv2 import VideoCapture, imread, imwrite

from utils import SessionManager, DataManager, Counter, Tracker, xlsxWriter, xlsxCompiler, StreetCountCompiler, Annotator

os.environ['install_ffmpeg_path'] = r"C:\ffmpeg\bin\ffmpeg.exe"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Line %(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# App configuration
app = Flask(__name__)

# Initialize session manager
session_manager = SessionManager()

app.config['CONTENTS'] = 'contents'

app.config['UPLOADS_FOLDER'] = os.path.join(app.config['CONTENTS'],'uploads')
app.config['MODELS_FOLDER'] = os.path.join(app.config['CONTENTS'],'models')
app.config['RESULTS_FOLDER'] = os.path.join(app.config['CONTENTS'],'results')
app.config['LOGS_FOLDER'] = os.path.join(app.config['CONTENTS'],'logs')

# Check directories for uploads, models, results, and logs
os.makedirs(os.path.join(app.root_path, app.config['UPLOADS_FOLDER']), exist_ok=True)
os.makedirs(os.path.join(app.root_path, os.path.join(app.config['UPLOADS_FOLDER'], 'compiler')), exist_ok=True)
os.makedirs(os.path.join(app.root_path, app.config['MODELS_FOLDER']), exist_ok=True)
os.makedirs(os.path.join(app.root_path, app.config['RESULTS_FOLDER']), exist_ok=True)
os.makedirs(os.path.join(app.root_path, os.path.join(app.config['RESULTS_FOLDER'], 'compiler')), exist_ok=True)
os.makedirs(os.path.join(app.root_path, app.config['LOGS_FOLDER']), exist_ok=True)

logging.info(f"--> logs in {os.path.join(app.root_path,app.config['LOGS_FOLDER'])}")

app.secret_key = "192b9bdd45ab9ed4d12e236c78afzb9a393ec15f71bbf5dc987d54727823bcbf"  #not used
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1000 MB


#Processing

def extract_first_frame(video_path, frame_path):
    cap = VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        imwrite(frame_path, frame)
        return True
    return None

def update_progress(session_id, step, percentage):
    if session_id not in session_manager.sessions[session_id]['progress']:
        session_manager.sessions[session_id]['progress'][session_id] = {}
    session_manager.sessions[session_id]['progress'][session_id][step] = percentage

def process_video_task(data_manager, session_id, paths):
    with app.app_context():
        try:
            start_time = datetime.datetime.now()
            session_dir = paths['session_dir']
            report_path = paths['report_path']
            for step in ['YOLO', 'Counting', 'Excel', 'Annotation']:
                update_progress(session_id, step, 0)

            # Initialize Tracker and Counter for multiple triplines
            tracker = Tracker(data_manager, progress_callback=lambda p: update_progress(session_id, 'YOLO', p))
            counter = Counter(data_manager, progress_callback=lambda p: update_progress(session_id, 'Counting', p))

            # Process video
            tracker.process_video(data_manager)
            update_progress(session_id, 'YOLO', 100)

            # Counting for multiple triplines
            update_progress(session_id, 'Counting', 0)
            counter.count(data_manager)
            update_progress(session_id, 'Counting', 100)

            # Export results
            update_progress(session_id, 'Excel', 0)
            writer = xlsxWriter(progress_callback=lambda p: update_progress(session_id, 'Excel', p))
            writer.write_to_excel(report_path, data_manager)
            update_progress(session_id, 'Excel', 100)

            # Perform annotation if export_video is True
            if data_manager.do_video_export:
                update_progress(session_id, 'Annotation', 0)
                annotated_video_path = paths['annotated_video_path']
                annotator = Annotator(data_manager, progress_callback=lambda p: update_progress(session_id, 'Annotation', p))
                annotator.write_annotated_video(annotated_video_path)
                if not os.path.exists(paths['ffmpeg_path']):
                    logger.warning(f"ffmpeg executable not found at {paths['ffmpeg_path']}")
                else :
                    paths['annotated_video_path'] = annotator.reformat_video(annotated_video_path, ffmpeg_path=paths['ffmpeg_path'], cleanup=True)
                update_progress(session_id, 'Annotation', 100)

            end_time = datetime.datetime.now()

        except Exception as e:
            update_progress(session_id, 'YOLO', -1)
            update_progress(session_id, 'Counting', -1)
            update_progress(session_id, 'Excel', -1)
            update_progress(session_id, 'Annotation', -1)
            session_manager.sessions[session_id]['results'][session_id] = {'error': f"Error processing video: {str(e)}"}
            logging.error(f"Error processing video: {str(e)}", exc_info=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pre_process/<session_id>', methods=['POST'])
def pre_process_video(session_id):
    # Check initialisation complete 
    initialize(session_id=session_id)
    # Store form data in session
    session_manager.sessions[session_id]['form_data'] = {
        'site_location': request.form.get('siteLocation'),
        'inference_tracker': request.form.get('inferenceTracker'),
        'export_video': request.form.get('exportVideo') == 'on',
        'start_date': request.form.get('startDate'),
        'start_time': request.form.get('startTime'),
        'triplines': request.form.get('triplines'),
        'directions': request.form.get('directions')
    }
    # Handle file uploads
    model_file = request.files.get('modelFile')
    
    if model_file:
        model_filename = secure_filename(model_file.filename)
        model_path = os.path.join(os.path.join(app.root_path, app.config['MODELS_FOLDER']), model_filename)
        model_file.save(model_path)
        session_manager.sessions[session_id]['model_path'] = model_path
        log_session(session_id)
        return jsonify({"status": "success"})
    
    return jsonify({'error': 'Missing files'}), 400

@app.route('/start_processing/<session_id>', methods=['POST'])
def start_processing(session_id):
    # Initialize DataManager with stored session data
    form_data = session_manager.sessions[session_id].get('form_data')
    data_manager = session_manager.sessions[session_id]['data_manager']
    data_manager.video_path = session_manager.sessions[session_id].get('video_path')
    data_manager.set_video_params(data_manager.video_path)
    data_manager.selected_model = session_manager.sessions[session_id].get('model_path')
    data_manager.set_names(data_manager.selected_model)
    
    # Set triplines from drawing stage
    tripline_data = request.get_json().get('triplines')
    data_manager.triplines = tripline_data
    data_manager.set_directions(request.get_json().get('directions'))
    
    # Set remaining parameters from stored form data
    data_manager.site_location = form_data['site_location']
    data_manager.inference_tracker = form_data['inference_tracker']
    data_manager.do_video_export = form_data['export_video']
    data_manager.set_start_datetime(form_data['start_date'], form_data['start_time'])
    
    # Define paths
    session_dir = os.path.join(app.root_path, app.config['RESULTS_FOLDER'], session_id)
    report_path = os.path.join(session_dir, 'report_'+ data_manager.site_location +'.xlsx')
    annotated_video_path = os.path.join(session_dir, 'annotated_'+ data_manager.site_location +'_video.mp4') if data_manager.do_video_export else None
    paths = {'session_dir' : session_dir, 'report_path' : report_path}
    if annotated_video_path:
        paths['annotated_video_path'] = annotated_video_path
        # Check for local ffmpeg path in environment variables
        paths['ffmpeg_path'] = os.getenv('install_ffmpeg_path', 'ffmpeg')
    
    # Start processing thread
    processing_thread = threading.Thread(target=process_video_task, 
                                      args=(data_manager, session_id, paths))
    processing_thread.start()

    response_paths = {key : os.path.basename(path) for key, path in paths.items()}

    return jsonify({"status": "Processing started", "session_id": session_id, "paths": response_paths})

def log_session(session_id):
    PROCESS_LOG_FILE = os.path.join(os.path.join(app.root_path, app.config['LOGS_FOLDER']), 'process_session_log.json')
    # Load existing log data
    if os.path.exists(PROCESS_LOG_FILE):
        with open(PROCESS_LOG_FILE, 'r') as f:
            session_log = json.load(f)
    else:
        session_log = {}

    # Add the new session data
    session_log[session_id] = { 
        'form_data': session_manager.sessions[session_id]['form_data'],
        'model_path': session_manager.sessions[session_id]['model_path'],
        'video_path': session_manager.sessions[session_id]['video_path'],
        'first_frame_filename': session_manager.sessions[session_id]['first_frame_filename'],
    }

    # Write the updated log back to the file
    with open(PROCESS_LOG_FILE, 'w') as f:
        json.dump(session_log, f, indent=4)

@app.route('/initialize', methods=['POST'])
def initialize(session_id = None):
    session_id = session_manager.create_session(session_id) # Will reinitialize if session_id is provided 

    session_dir = os.path.join(app.root_path, app.config['RESULTS_FOLDER'], session_id)
    os.makedirs(session_dir, exist_ok=True)

    video_file = request.files.get('videoFile')
    if (video_file):
        video_filename = secure_filename(video_file.filename)
        video_path = os.path.join(os.path.join(app.root_path, app.config['UPLOADS_FOLDER']), video_filename)
        video_file.save(video_path)
        first_frame_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_first_frame.jpg"
        frame_path = os.path.join(app.root_path, app.config['RESULTS_FOLDER'], session_id, first_frame_filename)
        success = extract_first_frame(video_path, frame_path)
        if success:
            frame_url = url_for('download_file', filename=first_frame_filename, session_id=session_id)

            session_manager.sessions[session_id]['video_path'] = video_path
            session_manager.sessions[session_id]['first_frame_filename'] = first_frame_filename

            return jsonify({'status': 'success', 'session_id': session_id , 'frame_url': frame_url})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to extract frame'}), 500
    else:
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOADS_FOLDER'], filename)

@app.route('/progress')
def progress_update():
    session_id = request.args.get('session_id')
    if session_id in session_manager.sessions[session_id]['progress']:
        return jsonify(session_manager.sessions[session_id]['progress'][session_id])
    else:
        return jsonify({'YOLO': -1, 'Counting': -1, 'Excel': -1, 'Annotation': -1})

@app.route('/results')
def get_results():
    session_id = request.args.get('session_id')
    if session_id in session_manager.sessions[session_id]['results']:
        return jsonify(session_manager.sessions[session_id]['results'][session_id])
    else:
        return jsonify({'error': 'Results not available yet'}), 202

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    directory = os.path.join(os.path.join(app.config['RESULTS_FOLDER']), session_id)
    return send_from_directory(directory, filename)

@app.route('/compile', methods=['GET', 'POST'])
def compile_reports():
    if request.method == 'POST':
        # Generate a unique session ID for this compile session
        session_id = str(uuid.uuid1())
        
        # Retrieve form data
        files = request.files.getlist('filePaths')
        output_filename = request.form.get('outputFilename')
        
        # Set default output filename if not provided
        if not output_filename.strip():
            output_filename = 'compiled_report.xlsx'
        elif not output_filename.endswith('.xlsx'):
            output_filename += '.xlsx'
        
        output_path = os.path.join(os.path.join(app.root_path, os.path.join(app.config['RESULTS_FOLDER'], 'compiler')), output_filename)
        
        # Initialize compile session data
        compile_data = {
            'session_id': session_id,
            'output_filename': output_filename,
            'timestamp': datetime.datetime.now(datetime.UTC),  # ISO 8601 format in UTC
            'status': 'initiated',
            'input_paths': [],
            'error_message': None
        }
        
        try:
            if not files:
                raise ValueError("No files selected for compilation.")
            file_paths = []
            for file in files:
                filename = secure_filename(file.filename)
                file_path = os.path.join(os.path.join(app.root_path, os.path.join(app.config['UPLOADS_FOLDER'], 'compiler')), filename)
                file.save(file_path)
                file_paths.append(file_path)
            compile_data['input_paths'].extend(file_paths)
            compiler = xlsxCompiler(file_paths=file_paths)
        
            
            # Perform compilation
            compiler.compile(output_path=output_path)
            compile_data['status'] = 'success'
            
            # Log the successful compile session
            log_compile_session(session_id, compile_data)
            
            # Send the compiled file as a download
            return send_from_directory(os.path.join(app.root_path,os.path.join(app.config['RESULTS_FOLDER'], 'compiler')), output_filename, as_attachment=True)
        
        except Exception as e:
            # Update compile_data with error details
            compile_data['status'] = 'error'
            compile_data['error_message'] = str(e)
            
            # Log the failed compile session
            log_compile_session(session_id, compile_data)
            
            # Return error response
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    return render_template('compile.html')

def log_compile_session(session_id, data):
    """
    Logs compile session details to compile_session_log.json.
    
    Args:
        session_id (str): Unique identifier for the compile session.
        data (dict): Dictionary containing compile session details.
    """
    COMPILE_LOG_FILE = os.path.join(os.path.join(app.root_path, app.config['LOGS_FOLDER']), 'compile_session_log.json')
    
    # Load existing log data
    if os.path.exists(COMPILE_LOG_FILE):
        with open(COMPILE_LOG_FILE, 'r') as f:
            compile_log = json.load(f)
    else:
        compile_log = {}
    
    # Add the new compile session data
    compile_log[session_id] = data
    
    # Write the updated log back to the file
    with open(COMPILE_LOG_FILE, 'w') as f:
        json.dump(compile_log, f, indent=4, default=str)

@app.route('/history', methods=['GET', 'POST'])
def history():
    # Paths to log files
    PROCESS_LOG_FILE = os.path.join(app.root_path, app.config['LOGS_FOLDER'], 'process_session_log.json')
    COMPILE_LOG_FILE = os.path.join(app.root_path, app.config['LOGS_FOLDER'], 'compile_session_log.json')

    # Load session data
    process_sessions = {}
    compile_sessions = {}

    if os.path.exists(PROCESS_LOG_FILE):
        with open(PROCESS_LOG_FILE, 'r') as f:
            process_sessions = json.load(f)
    if os.path.exists(COMPILE_LOG_FILE):
        with open(COMPILE_LOG_FILE, 'r') as f:
            compile_sessions = json.load(f)

    selected_type = None
    selected_session_id = None
    session_log = None
    triplines = None
    first_frame_path = None
    available_files = []

    if request.method == 'POST':
        selected_type = request.form.get('session_type')
        selected_session_id = request.form.get('session_id')

        # Get the session log based on selected type and session ID
        if selected_type == 'Counting' and selected_session_id in process_sessions:
            session_log = process_sessions[selected_session_id]
            # Load first frame
            first_frame_filename = session_log['first_frame_filename'] 
            first_frame_path = url_for('download_file', filename=first_frame_filename, session_id=selected_session_id)
            # Load triplines
            triplines = json.loads(session_log['form_data']['triplines'])
            print(triplines)
            # Determine available files for download
            session_dir = os.path.join(app.config['RESULTS_FOLDER'], selected_session_id)
            if os.path.exists(session_dir):
                for filename in os.listdir(session_dir):
                    available_files.append(filename)
        elif selected_type == 'Compiling' and selected_session_id in compile_sessions:
            session_log = compile_sessions[selected_session_id]
            # Compiled files are stored in RESULTS_FOLDER/compiler
            compiled_file = session_log.get('output_filename')
            if compiled_file:
                available_files.append(compiled_file)

    return render_template('history.html',
                           process_sessions=process_sessions,
                           compile_sessions=compile_sessions,
                           selected_type=selected_type,
                           first_frame_path=first_frame_path,
                           triplines=triplines,
                           selected_session_id=selected_session_id,
                           session_log=session_log,
                           available_files=available_files)

@app.route('/download_history_file/<session_type>/<session_id>/<filename>')
def download_history_file(session_type, session_id, filename):
    if session_type == 'Counting':
        directory = os.path.join(app.root_path, app.config['RESULTS_FOLDER'], session_id)
    elif session_type == 'Compiling':
        directory = os.path.join(app.root_path, app.config['RESULTS_FOLDER'], 'compiler')
    else:
        return "Invalid session type", 400
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/streetcount', methods=['GET', 'POST'])
def compile_streetcount():
    if request.method == 'POST':
        # Generate a unique session ID for this compile session
        session_id = str(uuid.uuid1())
        
        # Retrieve form data
        files = request.files.getlist('filePaths')
        site_location = request.form.get('siteLocation')
        timezone = request.form.get('timezone')
        # Convert timezone string to datetime timezone
        try:
            hours_offset = int(timezone.split(':')[0].replace('UTC', ''))
            minutes_offset = int(timezone.split(':')[1])
            timezone_offset = datetime.timezone(datetime.timedelta(hours=hours_offset, minutes=minutes_offset))
        except Exception as e:
            return jsonify({'status': 'error', 'message': 'Invalid timezone format'}), 400
        output_filename = request.form.get('outputFilename')
        
        # Set default output filename if not provided
        if not output_filename.strip():
            output_filename = 'compiled_report.xlsx'
        elif not output_filename.endswith('.xlsx'):
            output_filename += '.xlsx'
        
        output_path = os.path.join(os.path.join(app.root_path, os.path.join(app.config['RESULTS_FOLDER'], 'compiler')), output_filename)
        
        # Initialize compile session data
        compile_data = {
            'session_id': session_id,
            'output_filename': output_filename,
            'timestamp': datetime.datetime.now(datetime.UTC),  # ISO 8601 format in UTC
            'status': 'initiated',
            'input_paths': files,
            'site_location': site_location,
            'timezone': timezone,
            'error_message': None
        }
        try :
            compiler = StreetCountCompiler(file_paths=files, site_location=site_location, timezone=timezone_offset)
            
            # Perform compilation
            compiler.compile(output_path=output_path)
            compile_data['status'] = 'success'
            log_compile_session(session_id, compile_data)

            # Send the compiled file as a download
            return send_from_directory(os.path.join(app.root_path,os.path.join(app.config['RESULTS_FOLDER'], 'compiler')), output_filename, as_attachment=True)
        
        except Exception as e:
                # Update compile_data with error details
                compile_data['status'] = 'error'
                compile_data['error_message'] = str(e)
                
                # Log the failed compile session
                log_compile_session(session_id, compile_data)
                
                # Return error response
                return jsonify({'status': 'error', 'message': str(e)}), 500

    return render_template('streetcount.html')

# Run the app
if __name__ == "__main__" :
    app.run() #Only use for development. Debug should be False to prevent server restart at each change of .py files