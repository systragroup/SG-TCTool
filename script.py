import os
import threading
import datetime
import logging
from shutil import copy2
import subprocess

import json
from cv2 import VideoCapture, imread, imwrite

from utils import SessionManager, DataManager, Counter, Tracker, xlsxWriter, xlsxCompiler, StreetCountCompiler, Annotator
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
    # Save the video, model and first frame to the content directory. is a full copy
    if os.path.exists(video_path):
        video_path = copy2(video_path, paths['uploads_dir'])

        # extract_first_frame
        frame_path = os.path.join(paths['content_dir'], 'first_frame.jpg')
        extract_first_frame(video_path, frame_path)

    if os.path.exists(model_path):
        model_path = copy2(model_path, paths['models_dir'])

    report_path = os.path.join(paths['content_dir'], 'report.xslx')
    
    # Create shortcuts in content_dir
    video_shortcut = os.path.join(paths['content_dir'], 'source_video_sc.mp4')
    model_shortcut = os.path.join(paths['content_dir'], 'inference_model_sc'+ os.path.splitext(model_path)[1])

    try:
        os.symlink(video_shortcut, video_path)
        os.symlink(model_shortcut, model_path)
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


def main():
    video_path = r'/Users/amaurydufour/Desktop/SYSTRA/good-cut-shortest.mp4'
    model_path = r'/Users/amaurydufour/Desktop/SYSTRA/traffic_camera_us_v11n2.onnx'
    site_location = "Test Junction"
    inference_tracker = "bytetrack.yaml" # 2 are supported : `bytetrack.yaml` & `botsort.yaml` (BoT-SORT is slower)
    export_video = True
    start_date = "2025-01-20" # 'YYYY-MM-DD'
    start_time = "08:07" # 'HH:MM'
    ffmpeg_executable_path = r"C:\ffmpeg\bin\ffmpeg.exe"

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

    processing_thread = threading.Thread(target=process_video_task, 
                                      args=(data_manager, paths))
    processing_thread.start()

if __name__ == '__main__':
    main()