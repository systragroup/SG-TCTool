import os
import threading
import datetime
import logging
from shutil import copy2, copytree
import subprocess
import onnxruntime as ort

import json
from cv2 import VideoCapture, imread, imwrite

from utils import DataManager, Counter, Tracker, xlsxWriter, xlsxCompiler, Annotator
import cv2

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Line %(lineno)d] - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

def dir_create():
    # Set up dirs :
    root_dir = 'pure_python_runs'
    models_dir = os.path.join(root_dir, 'models')
    uploads_dir = os.path.join(root_dir, 'uploads')
    content_dir_count = 0
    while os.path.exists(os.path.join(root_dir, str(content_dir_count))):
        content_dir_count += 1
    content_dir = os.path.join(root_dir, str(content_dir_count))
    os.makedirs(content_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)
    
    return models_dir, uploads_dir, content_dir

def extract_first_frame(video_path, frame_path):
    cap = VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        imwrite(frame_path, frame)
        return True
    return None

def pre_process(paths, video_path, model_path):
    ''' Save the video, model and first frame to the content directory. is a full copy
        '''
    #Check video, and copy if needed
    if not os.path.exists(os.path.join(paths['uploads_dir'], os.path.basename(video_path))) : #If it isn't already in the uploads dir
        if os.path.exists(video_path): #Copy
            video_path = copy2(video_path, paths['uploads_dir']) 
    else : video_path = os.path.join(paths['uploads_dir'], os.path.basename(video_path)) #Otherwise, just point to existing uploads dir copy

    #Check model, and copy if needed
    if not os.path.exists(os.path.join(paths['models_dir'], os.path.basename(model_path))) : #Check not already in models dir
        if os.path.exists(model_path) : #Check exists at source
            if os.path.isfile(model_path) : #If is file
                model_path = copy2(model_path, os.path.join(paths['models_dir'], os.path.basename(model_path)))
            elif os.path.isdir(model_path): #Else if folder
                model_path = copytree(model_path, os.path.join(paths['models_dir'], os.path.basename(model_path)))
    else : model_path = os.path.join(paths['models_dir'], os.path.basename(model_path)) #If already uploaded, just point to preexisting copy
    
    #Extract the first Frame
    frame_path = os.path.join(paths['content_dir'], 'first_frame.jpg')
    extract_first_frame(video_path, frame_path)

    # 
    report_path = os.path.join(paths['content_dir'], 'report.xlsx')
    
    # Create shortcuts in content_dir
    video_shortcut = os.path.join(paths['content_dir'], 'source_video_sc.mp4')
    model_shortcut = os.path.join(paths['content_dir'], 'inference_model_sc'+ os.path.splitext(model_path)[1])

    try:
        os.symlink(video_path, video_shortcut)
        os.symlink(model_path, model_shortcut)
        logger.info("Shortcuts created successfully.")
    except AttributeError:
        logger.error("Symbolic links are not supported on this system.")
    except OSError as e:
        logger.error(f"Failed to create symbolic links: {e}")
    return video_path, model_path, report_path, frame_path

def process_video_task(data_manager, paths):
    try:

        # Initialize Tracker and Counter for multiple triplines
        tracker = Tracker(data_manager)
        counter = Counter(data_manager)

        # Process video
        tracker.process_video(data_manager)

        # Counting for multiple triplines
        counter.count(data_manager)

        # Export results
        writer = xlsxWriter()
        writer.write_to_excel(paths['report_path'], data_manager)

        # Perform annotation if export_video is True
        if data_manager.do_video_export:
            paths['annotated_video_path'] = os.path.join(paths['content_dir'], 'annotated_video.mp4')
            annotator = Annotator(data_manager)
            annotator.write_annotated_video(paths['annotated_video_path'])
            paths['output_vid'] = annotator.reformat_video(paths['annotated_video_path'], ffmpeg_path=paths['ffmpeg_path'])

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)

def draw_triplines(first_frame_path):
    triplines = []
    drawing = False
    start_point = None
    img = cv2.imread(first_frame_path)

    if img is None:
        logging.error(f"Failed to load image from {first_frame_path}")
        return triplines

    def draw_line(event, x, y, flags, param):
        nonlocal drawing, start_point, img
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            logging.info(f"Tripline start point: {start_point}")
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_img = img.copy()
                cv2.line(temp_img, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow('Draw Triplines', temp_img)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            triplines.append({
                "start": {"x": start_point[0], "y": start_point[1]},
                "end": {"x": end_point[0], "y": end_point[1]}
            })
            cv2.line(img, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow('Draw Triplines', img)
            logger.info(f"Tripline end point: {end_point}")
            logger.info(f"Total triplines: {len(triplines)}")

    cv2.namedWindow('Draw Triplines')
    cv2.setMouseCallback('Draw Triplines', draw_line)

    instructions = '''Draw triplines by clicking and dragging the mouse - Press 'Enter' to finish, 'r' to redraw and 'esc' to cancel'''
    cv2.putText(img, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2, cv2.LINE_AA)

    while True:
        cv2.imshow('Draw Triplines', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            if len(triplines) == 0:
                logger.warning("No triplines drawn.")
            else:
                logger.info("Finished drawing triplines.")
            break
        elif key == ord('r'):
            triplines = []
            img = cv2.imread(first_frame_path)  # Reload image
            logger.info("Restarting tripline drawing.")
        elif key == 27:  # Esc key
            logger.info("Escape key pressed. Exiting process.")
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()
    return triplines

def log_setup(data_manager, paths):
    setup_data = {
    "video_path": data_manager.video_path,
    "model_path": data_manager.selected_model,
    "triplines": data_manager.triplines,
    "directions": data_manager.directions,
    "site_location": data_manager.site_location,
    "inference_tracker": data_manager.inference_tracker,
    "do_video_export": data_manager.do_video_export,
    "start_datetime": data_manager.start_datetime.isoformat()
}

    setup_file_path = os.path.join(paths['content_dir'], 'setup_data.json')
    with open(setup_file_path, 'w') as f:
        json.dump(setup_data, f, indent=4)

    logger.info(f"Setup data logged to {setup_file_path}")

def run(params):
    video_path = params['video_path']
    model_path = params['model_path']
    site_location = params['site_location']
    inference_tracker = params['inference_tracker'] # 2 are supported : `bytetrack.yaml` & `botsort.yaml` (BoT-SORT is slower)
    export_video = params['export_video']
    start_date = params['start_date'] # 'YYYY-MM-DD'
    start_time = params['start_time'] # 'HH:MM'
    ffmpeg_executable_path = params['ffmpeg_executable_path']
    

    global logger
    logger = setup_logging()
    paths = {}
    paths['models_dir'], paths['uploads_dir'], paths['content_dir'] = dir_create()
    paths['ffmpeg_path'] = ffmpeg_executable_path

    paths['video_path'], paths['model_path'], paths['report_path'], paths['first_frame_path'] = pre_process(paths, video_path, model_path) # Save the video, model and first frame to the content directory
    triplines = draw_triplines(paths['first_frame_path']) # Draw triplines on the first frame of the video
    directions = []
    
    if len(triplines) == 1:
        prompt = "Enter direction 1 : >"
        directions.append(input(prompt))
        prompt = "Enter direction 2 : >"
        directions.append(input(prompt))
    else :
        for count, tripline in enumerate(triplines):
            prompt = f"Enter the direction for tripline {count} : {tripline} >"
            directions.append(input(prompt))
    
    data_manager = DataManager()
    data_manager.video_path = paths['video_path']
    data_manager.set_video_params(data_manager.video_path) # Set the video parameters (fps, width, height)
    data_manager.selected_model = paths['model_path']
    data_manager.set_names(data_manager.selected_model) # Extract the names of the detection classes
    data_manager.triplines = triplines
    data_manager.directions = directions
    data_manager.site_location = site_location
    data_manager.inference_tracker = inference_tracker
    data_manager.do_video_export = export_video
    data_manager.set_start_datetime(start_date, start_time)

    log_setup(data_manager, paths=paths)

    process_video_task(data_manager, paths)

    compiler = xlsxCompiler(file_paths=[paths['report_path']])
    compiler.compile(output_path=os.path.join(paths['content_dir'],'totals.xlsx'))

if __name__ == '__main__':
    params = {}

    params['video_path'] = input(r"\path\to\your\vid>").strip().strip("'").strip('"')
    params['model_path'] = input(r"\path\to\your\model>").strip().strip("'").strip('"')
    params['site_location'] = input("Name of Location >").strip().strip("'").strip('"')
    params['inference_tracker'] = input("Tracker (2 are supported : `bytetrack.yaml` & `botsort.yaml` (BoT-SORT is slower)) >").strip().strip("'").strip('"')
    params['export_video'] = input("Do video export (True/False) >").strip().strip("'").strip('"') == "True"
    params['start_date'] = input("Date # 'YYYY-MM-DD' >") .strip().strip("'").strip('"')
    params['start_time'] = input("Time # 'HH:MM' >").strip().strip("'").strip('"')
    params['ffmpeg_executable_path'] = input("FFmpeg executable path >").strip().strip("'").strip('"')

    run(params)
    