import os
import uuid
import json
import logging
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session
from werkzeug.utils import secure_filename
from utils import DataManager, Counter, Tracker, xlsxWriter, xlsxCompiler, Annotator
from datetime import datetime

import os

app = Flask(__name__)
app.secret_key = "something"
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1000 MB

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
MODEL_FOLDER = os.path.join(app.root_path, 'models')
RESULTS_FOLDER = os.path.join(app.root_path, 'results')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Line %(lineno)d] - %(message)s')

logger = logging.getLogger(__name__)

data_manager = DataManager()

# Dictionaries to store progress and results per session
app.progress = {}
app.results = {}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def extract_first_frame(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        base_filename = secure_filename(os.path.basename(video_path))
        frame_filename = f"{os.path.splitext(base_filename)[0]}_first_frame.jpg"
        frame_path = os.path.join(UPLOAD_FOLDER, frame_filename)
        cv2.imwrite(frame_path, frame)
        return frame_filename  # Return only the filename
    return None

def update_progress(session_id, step, percentage):
    if session_id not in app.progress:
        app.progress[session_id] = {}
    app.progress[session_id][step] = percentage

def process_video_task(data_manager, session_id, paths):
    with app.app_context():
        try:
            start_time = datetime.now()
            session_dir = paths[0]
            report_path = paths[1]

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
                annotated_video_path = paths[2]
                annotator = Annotator(data_manager, progress_callback=lambda p: update_progress(session_id, 'Annotation', p))
                annotator.write_annotated_video(annotated_video_path)
                update_progress(session_id, 'Annotation', 100)

            end_time = datetime.now()

        except Exception as e:
            update_progress(session_id, 'YOLO', -1)
            update_progress(session_id, 'Counting', -1)
            update_progress(session_id, 'Excel', -1)
            update_progress(session_id, 'Annotation', -1)
            app.results[session_id] = {'error': f"Error processing video: {str(e)}"}
            logging.error(f"Error processing video: {str(e)}", exc_info=True)

@app.route('/process', methods=['POST'])
def process_video():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    
    # Store form data in session
    session['form_data'] = {
        'site_location': request.form.get('siteLocation'),
        'inference_tracker': request.form.get('inferenceTracker'),
        'export_video': request.form.get('exportVideo') == 'on',
        'start_date': request.form.get('startDate'),
        'start_time': request.form.get('startTime'),
        'triplines': request.form.get('triplines'),
        'directions': request.form.get('directions')
    }

    # Handle file uploads
    video_file = request.files.get('videoFile')
    model_file = request.files.get('modelFile')
    
    if video_file and model_file:
        video_filename = secure_filename(video_file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        video_file.save(video_path)
        
        # Extract first frame
        frame_path = extract_first_frame(video_path)
        session['video_path'] = video_path
        session['first_frame_path'] = frame_path
        
        model_filename = secure_filename(model_file.filename)
        model_path = os.path.join(MODEL_FOLDER, model_filename)
        model_file.save(model_path)
        session['model_path'] = model_path
        
        log_session(session['session_id'],session)
        return jsonify({"status": "success", "session_id": session_id})
    
    return jsonify({'error': 'Missing files'}), 400

@app.route('/start_processing/<session_id>', methods=['POST'])
def start_processing(session_id):
    # Initialize DataManager with stored session data
    form_data = session.get('form_data')
    data_manager = DataManager()
    data_manager.video_path = session.get('video_path')
    data_manager.set_video_params(data_manager.video_path)
    data_manager.selected_model = session.get('model_path')
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
    session_dir = os.path.join(RESULTS_FOLDER, session_id)
    report_path = os.path.join(session_dir, 'report.xlsx')
    annotated_video_path = os.path.join(session_dir, 'annotated_video.mp4') if data_manager.do_video_export else None
    os.makedirs(session_dir, exist_ok=True)
    paths = [session_dir, report_path]
    if annotated_video_path:
        paths.append(annotated_video_path)
    
    # Start processing thread
    processing_thread = threading.Thread(target=process_video_task, 
                                      args=(data_manager, session_id, paths))
    processing_thread.start()
    
    return jsonify({"status": "Processing started", "session_id": session_id})

def log_session(session_id, data):
    SESSION_LOG_FILE = os.path.join(RESULTS_FOLDER, 'session_log.json')
    # Load existing log data
    if os.path.exists(SESSION_LOG_FILE):
        with open(SESSION_LOG_FILE, 'r') as f:
            session_log = json.load(f)
    else:
        session_log = {}

    # Add the new session data
    session_log[session_id] = data

    # Write the updated log back to the file
    with open(SESSION_LOG_FILE, 'w') as f:
        json.dump(session_log, f, indent=4)

@app.route('/process_initial', methods=['POST'])
def process_initial():
    video_file = request.files.get('videoFile')
    if (video_file):
        video_filename = secure_filename(video_file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        video_file.save(video_path)

        # Extract first frame
        frame_filename = extract_first_frame(video_path)
        if frame_filename:
            frame_url = url_for('uploaded_file', filename=frame_filename)

            # Store the video and frame filenames in the session
            session['video_path'] = video_path
            session['frame_filename'] = frame_filename

            return jsonify({'status': 'success', 'frame_url': frame_url})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to extract frame'}), 500
    else:
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    logging.info(f"Serving file {filename} from {UPLOAD_FOLDER}")
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/progress')
def progress_update():
    session_id = request.args.get('session_id')
    if session_id in app.progress:
        return jsonify(app.progress[session_id])
    else:
        return jsonify({'YOLO': -1, 'Counting': -1, 'Excel': -1, 'Annotation': -1})

@app.route('/results')
def get_results():
    session_id = request.args.get('session_id')
    if session_id in app.results:
        return jsonify(app.results[session_id])
    else:
        return jsonify({'error': 'Results not available yet'}), 202

@app.route('/results/<session_id>/<filename>')
def download_file(session_id, filename):
    session_dir = os.path.join(RESULTS_FOLDER, session_id)
    return send_from_directory(session_dir, filename, as_attachment=True)

@app.route('/compile', methods=['GET', 'POST'])
def compile_reports():
    if request.method == 'POST':
        # Handle the form submission
        compile_option = request.form.get('compileOption')
        output_filename = request.form.get('outputFilename')
        if not output_filename.endswith('.xlsx'):
            output_filename += '.xlsx'
        
        output_path = os.path.join(RESULTS_FOLDER, output_filename)

        if compile_option == 'folder':
            # Handle folder compilation
            folder_path = request.form.get('folderPath')
            compiler = xlsxCompiler()
            compiler.compile_folder(folder_path, output_path)
        elif compile_option == 'files':
            # Handle specific files compilation
            file_paths = request.form.getlist('filePaths')
            compiler = xlsxCompiler()
            compiler.compile_files(file_paths, output_path)
        else:
            return jsonify({'error': 'Invalid compile option'}), 400

        return send_from_directory(RESULTS_FOLDER, output_filename, as_attachment=True)

    return render_template('compile.html')

if __name__ == '__main__':
    app.run(debug=True)
