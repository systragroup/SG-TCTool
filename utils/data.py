import os
import datetime
import logging
from collections import defaultdict
import torch.cuda
import yaml
import cv2
from ultralytics import YOLO
import onnx

class DataManager:
    """
    Central data management class for the traffic counting application.
    
    Handles:
    - Video metadata and parameters
    - Model selection and configuration
    - Tracking data storage
    - Site and timing information
    - Export settings
    """
    def __init__(self):
        """Initialize data storage and default parameters."""
        self.names = None
        self.HOME = os.getcwd()
        self.video_path = None
        self.export_path = None
        self.selected_model = None
        self.inference_tracker = None
        self.site_location = None
        self.do_video_export = False
        self.start_datetime = None
        self.directions = None

        self.START, self.END = None, None

        self.CROSSED =  defaultdict(lambda: [])
        self.TRACK_DATA = defaultdict(lambda: [])
        self.TRACK_INFO = []
        self.TRACK_ANALYSIS = {}

        self.device_name = ""
        # Check if CUDA is available
        if torch.cuda.is_available(): #Else device_name is "" and torch will default to CPU
            self.device_name = torch.cuda.current_device()

        self.frame_count = 0
        self.fps = 30
        self.width = 0
        self.height = 0
        self.triplines = []  # Changed from single tripline to list of triplines

    def set_tripline(self):
        self.tripline = (self.START, self.END)

    def set_directions(self, direction_data):
        self.directions = [dir for dir in direction_data.values()]

    def set_start_datetime(self, start_date, start_time):
        self.start_datetime = datetime.datetime.strptime(f"{start_date} {start_time}:00", r"%Y-%m-%d %H:%M:%S")

    def set_video_params(self, video_path):
        """
        Extract and store video parameters from input file.
        
        Args:
            video_path: Path to input video file
        """
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def set_names(self, selected_model):
        """
        Load class names from model metadata.
        
        Supports:
        - YOLO .pt models
        - ONNX models
        - OpenVINO models
        
        Args:
            selected_model: Path to model file or directory
        """
        if os.path.isdir(selected_model): #check for an openvino model if model is a folder
            files = [f for f in os.listdir(selected_model) if os.path.isfile(os.path.join(selected_model, f))]
            if 'metadata.yaml' in files: 
                self.model_type = "openvino"
                yaml_path = os.path.join(selected_model, 'metadata.yaml')
                with open(yaml_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                self.names = metadata.get('names', {})
                logging.info("Class names loaded from yaml metadata.")
        else :
            _, extension = os.path.splitext(selected_model)
            self.model_type = extension
            if extension == ".pt": #YOLO returns pt model names in dict format
                class_names = YOLO(selected_model).model.names
                if class_names:
                    self.names = YOLO(selected_model).model.names
                    logging.info("Class names loaded from pt metadata.")
                else:
                    logging.warning("Class names not found in pt metadata.")
                    
            elif extension == ".onnx": #ONNX model names metadata is a string
                model = onnx.load(selected_model)
                metadata_props = {prop.key: prop.value for prop in model.metadata_props}
                class_names = metadata_props.get('names', None)
                if class_names:
                    items = class_names.strip('{}').split(',')
                    self.names =  {}
                    for item in items :
                        key, name = item.split(':')
                        key = int(key.strip())
                        name = name.strip().strip("'").strip('"')
                        self.names[key] = name
                    logging.info("Class names loaded from onnx metadata.")
                else:
                    logging.warning("Class names not found in onnx metadata.")
                    return []
            else :
                logging.warning("Unsupported model format : Class names not loaded from model metadata.")

    def set_site_location(self, site_location):
        self.site_location = site_location
