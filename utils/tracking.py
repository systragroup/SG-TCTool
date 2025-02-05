from collections import defaultdict
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from ultralytics import YOLO
import cv2
import numpy as np
from utils import DESC_WIDTH, DETECTION_MODEL_CONST

class Counter:
    """
    Handles counting objects crossing predefined triplines in video analysis.
    
    Analyzes tracks of detected objects and determines when they cross triplines,
    recording their class, direction, and confidence scores.
    """
    def __init__(self, data_manager, progress_callback=None):
        """
        Args:
            data_manager: DataManager instance containing video and tracking data
            progress_callback: Optional callback function to report progress
        """
        self.progress_callback = progress_callback
        self.triplines = data_manager.triplines  # Access multiple triplines
        self.directions = data_manager.directions

    def analyze_track(self, track_data):
        """Analyzes track history to determine most likely class"""
        # We use defaultdict to avoid explicitly initializing stats for new classes
        # This makes the code more concise and handles first appearances automatically
        class_stats = defaultdict(lambda: {
            'count': 0,
            'total_conf': 0.0,
            'max_consecutive': 0,
            'current_consecutive': 0,
            'last_class': None
        })
        
        # Track consecutive detections because stable classifications
        # are more reliable than sporadic ones
        for frame, _, conf, cls in track_data:
            stats = class_stats[cls]
            stats['count'] += 1
            stats['total_conf'] += conf
            
            # Track consecutive detections
            if stats['last_class'] == cls:
                stats['current_consecutive'] += 1
            else:
                stats['current_consecutive'] = 1
            stats['max_consecutive'] = max(stats['max_consecutive'], 
                                         stats['current_consecutive'])
            stats['last_class'] = cls

        # Combine different metrics for final class selection:
        # - Average confidence: How sure the model is
        # - Frequency: How often this class appears
        # - Consecutive detections: Stability of classification
        class_scores = {}
        for cls, stats in class_stats.items():
            avg_conf = stats['total_conf'] / stats['count']
            freq_score = stats['count'] / len(track_data)
            consec_score = stats['max_consecutive'] / len(track_data)
            
            # Combine scores (can be weighted differently)
            class_scores[cls] = (avg_conf + freq_score + consec_score) / 3

        # Get class with highest score
        final_class = max(class_scores.items(), key=lambda x: x[1])
        return {
            'class': final_class[0],
            'confidence': final_class[1],
            'stats': class_stats
        }

    def count(self, data_manager):
        """
        Processes all tracks to count objects crossing triplines.
        
        For each track, determines if and where it crosses triplines and records:
        - Frame number of crossing
        - Object class
        - Direction of crossing
        - Confidence scores
        
        Args:
            data_manager: DataManager instance containing tracking data
        """
        obj_count = 0
        total_objs = len(data_manager.TRACK_DATA)
        console_progress = tqdm(total=total_objs, 
                              desc=f'{"Counting crossings":<{DESC_WIDTH}}', 
                              unit="tracks", 
                              dynamic_ncols=True)
        
        with logging_redirect_tqdm():
             for track_id, data in data_manager.TRACK_DATA.items():
                # Analyze track once at the start
                track_analysis = self.analyze_track(data)
                data_manager.TRACK_ANALYSIS[track_id] = track_analysis # Store it for export
                
                for idx, tripline in enumerate(self.triplines):
                    for i in range(1, len(data)):
                        point_A = {'x': data[i - 1][1][0], 'y': data[i - 1][1][1]}
                        point_B = {'x': data[i][1][0], 'y': data[i][1][1]}
                        if self.intersect_tripline(tripline['start'], tripline['end'], point_A, point_B):
                            frame = data[i][0]
                            # Analyze entire track history to determine most likely class
                            track_analysis = self.analyze_track(data)
                            final_class = track_analysis['class']
                            final_conf = track_analysis['confidence']

                            if len(self.triplines) == 1:
                                direction = (self.directions[0] 
                                          if self.CP(tripline['start'], tripline['end'], 
                                                   point_A, point_B) > 0 
                                          else self.directions[1])
                            else:
                                direction = self.directions[idx]

                            # Store the tripline index, class, direction, and confidence
                            data_manager.CROSSED[track_id].append((
                                frame,
                                final_class,
                                direction,
                                idx,
                                final_conf,
                                track_analysis['stats']
                            ))
                            break
                console_progress.update(1)
                obj_count += 1 
        console_progress.close()

    def CP(self, START, END, A, B): #Cross Product (Positive means B is on left side of S-E, negative B is on the right and 0 is S-E and A-B colinear)
        # Visualise right-hand rule : index is Start-End(tripline), middle finger is A-B and thumb is CP. 
        return (B['x'] - A['x']) * (END['y'] - START['y']) - (B['y'] - A['y']) * (END['x'] - START['x'])

    def intersect_tripline(self, START, END, A, B):
        def ccw_point(P, Q, R): #Counter Clock wise order of points
            return (R['y'] - P['y']) * (Q['x'] - P['x']) > (Q['y'] - P['y']) * (R['x'] - P['x'])
        
        return ccw_point(START, A, B) != ccw_point(END, A, B) and ccw_point(START, END, A) != ccw_point(START, END, B)

class Tracker:
    """
    Handles object detection and tracking in video frames using YOLO.
    
    Processes video frames to detect and track objects, maintaining their
    position and class information throughout the video.
    """
    def __init__(self, data_manager, progress_callback=None, verbose=False):
        """
        Args:
            data_manager: DataManager instance containing video and model settings
            progress_callback: Optional callback function to report progress (for flask client calls)
            verbose: Boolean to control logging verbosity
        """
        self.progress_callback = progress_callback
        self.video_path = data_manager.video_path
        self.selected_model = data_manager.selected_model
        self.inference_tracker = data_manager.inference_tracker
        self.device_name = data_manager.device_name
        self.verbose = verbose
        if DETECTION_MODEL_CONST.ALLOW_RESIZE and data_manager.model_type ==".pt": #Only pt models support resizing
            self.image_size = [32 * (data_manager.width//32) + 32 * min (1,data_manager.width%32), 32 * (data_manager.height//32) + 32 * min (1,data_manager.height%32)] # Input size must be a multiple of max stride 32
        else : self.image_size = [640, 640]
        # Load YOLO model
        self.model = YOLO(self.selected_model, task='detect')

        self.current_frame = None
        self.current_frame_nb = 0

    def read_next_frame(self):
        self.success, self.current_frame = self.cap.read()

    def process_frame(self, data_manager):
        results = self.model.track(self.current_frame, imgsz=self.image_size, persist=True, verbose=self.verbose, tracker=self.inference_tracker, device=self.device_name, save=False, conf=DETECTION_MODEL_CONST.CONF_THRESHOLD, iou=DETECTION_MODEL_CONST.IOU_THRESHOLD, agnostic_nms=DETECTION_MODEL_CONST.AGNOSTIC_NMS)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id
        classes = results[0].boxes.cls
        confidences = results[0].boxes.conf

        track_inf = []
        if track_ids is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist() 
            for box, track_id, clss, confidence in zip(boxes, track_ids, classes, confidences):
                track_dat = data_manager.TRACK_DATA[track_id] #track_data is indexed by track_id : for a given object, see which frames it's been tracked on, where it is and what it is
                clss = int(clss)
                track_dat.append((int(self.current_frame_nb), box, confidence, clss))
                track_inf.append((track_id, len(track_dat)))

        data_manager.TRACK_INFO.append(track_inf) #TRACK_INFO is indexed by frame : for a given frame, see which objects are where, and how long they've been tracked

    def process_video(self, data_manager): 
        # Open video to process
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = data_manager.frame_count

        self.console_progress = tqdm(total=self.frame_count, desc=f"{'YOLO is working':<{DESC_WIDTH}}", unit="frames", dynamic_ncols=True)
        # Run inference and tracking
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        with logging_redirect_tqdm():
            while self.cap.isOpened():
                self.read_next_frame()
                if self.success:
                    self.process_frame(data_manager)
                    self.current_frame_nb += 1
                    self.console_progress.update(1)
                    current_frame += 1
                    # Update progress
                    if self.progress_callback:
                        progress_percentage = int((current_frame / total_frames) * 100)
                        self.progress_callback(progress_percentage)
                else:
                    break

        self.console_progress.close()
        self.cap.release()
        pass
