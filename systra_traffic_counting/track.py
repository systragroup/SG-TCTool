from collections import defaultdict

import cv2
from ultralytics import YOLO
from tqdm import tqdm
import onnx
import onnxruntime


class Tracker:
    """
    A class used to track objects in a video using YOLO and ByteTrack.
    Attributes
    ----------
    video_path : str
        Path to the video file to be processed.
    selected_model : str
        Path to the YOLO model file.
    inference_tracker : str
        Path to the ByteTrack configuration file.
    device_name : str
        Device to run the inference on (e.g., 'cpu' or 'cuda').
    verbose : bool
        Flag to enable verbose logging.
    model : YOLO
        Loaded YOLO model.
    TRACK_DATA : defaultdict
        Dictionary to store tracking data.
    TRACK_INFO : list
        List to store tracking information.
    current_frame : np.array
        Current frame being processed.
    current_frame_nb : int
        Current frame number.
    cap : cv2.VideoCapture
        Video capture object.
    frame_count : int
        Total number of frames in the video.
    progress : callable
        Progress bar function.
    Methods
    -------
    get_track_data():
        Returns the tracking data.
    get_crossed():
        Returns the crossed data.
    get_track_info():
        Returns the tracking information.
    read_next_frame():
        Reads the next frame from the video.
    process_frame():
        Processes the current frame for object tracking.
    process_video(progress_bar=None):
        Processes the entire video for object tracking.
    """
    def __init__(self, data_manager, verbose : bool = False):
        """
        Initializes the tracking system with the specified parameters.
        Args:
            video_path (str): Path to the video file to be processed.
            selected_model (str, optional): Path to the YOLO model file. Defaults to "yolov10n.pt".
            inference_tracker (str, optional): Path to the tracker configuration file. Defaults to "bytetrack.yaml".
            device_name (str, optional): Device to run the model on, e.g., "cpu" or "cuda". Defaults to "cpu".
            verbose (bool, optional): If True, enables verbose logging. Defaults to False.
        """
        self.video_path = data_manager.video_path
        self.selected_model = data_manager.selected_model
        self.inference_tracker = data_manager.inference_tracker
        self.device_name = data_manager.device_name
        self.verbose = verbose

        # Load YOLO model
        self.model = YOLO(self.selected_model)

        self.current_frame = None
        self.current_frame_nb = 0


    def get_track_data(self, data_manager):
        """
        Retrieve the track data.

        Returns:
            dict: The track data stored in the TRACK_DATA attribute.
        """
        return data_manager.TRACK_DATA

    def get_track_info(self, data_manager):
        """
        Retrieve the track information.

        Returns:
            list: A list containing the track information : TRACK_INFO[frame_nb] = [(track_id, len(track_at_frame_nb))].
        """
        return data_manager.TRACK_INFO

    def read_next_frame(self):
        """
        Reads the next frame from the video capture object.

        This method attempts to read the next frame from the video capture object
        (`self.cap`). It updates the `self.success` attribute to indicate whether
        the frame was successfully read, and the `self.current_frame` attribute
        with the frame data.

        Returns:
            None
        """
        self.success, self.current_frame = self.cap.read()
            
    def process_frame(self, data_manager):
        """
        Processes the current frame using the model's tracking functionality.
        This method performs the following steps:
        1. Tracks objects in the current frame using the model.
        2. Extracts bounding boxes, track IDs, classes, and confidence scores from the tracking results.
        3. Updates the tracking data for each detected object.
        4. Appends the tracking information to the TRACK_INFO list.
        Attributes:
            self.current_frame (ndarray): The current frame to be processed.
            self.verbose (bool): Flag to enable YOLO inference logging (will print for each frame).
            self.inference_tracker (str): The tracker to be used for inference.
            self.device_name (str): The device to be used for processing (e.g., 'cpu' or '0').
            self.TRACK_DATA (dict): Dictionary to store tracking data for each object.
            self.current_frame_nb (int): The current frame number.
            self.TRACK_INFO (list): List to store tracking information for all frames.
        Returns:
            None
        """
        results = self.model.track(self.current_frame, persist=True, verbose=self.verbose, tracker=self.inference_tracker, device=self.device_name, save=False)
        
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id
        classes = results[0].boxes.cls
        confidences = results[0].boxes.conf

        tracked = []
        if track_ids is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist() 
            for box, track_id, clss, confidence in zip(boxes, track_ids, classes, confidences):
                track = data_manager.TRACK_DATA[track_id]
                clss = int(clss)
                track.append((int(self.current_frame_nb), box, confidence, clss))
                tracked.append((track_id, len(track)))

        data_manager.TRACK_INFO.append(tracked)
    
    def process_video(self, data_manager, progress_var=None):
        self.progress = progress_var 
        
        # Open video to process
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = data_manager.frame_count

        self.console_progress = tqdm(total=self.frame_count, desc="YOLO is working", unit="frames")
        # Run inference and tracking
        while self.cap.isOpened():
            self.read_next_frame()
            if self.success:
                self.process_frame(data_manager)
                self.current_frame_nb += 1
                if self.progress is not None:
                    self.progress.set(self.current_frame_nb)
                    self.console_progress.update(1)
                    
            else:
                break

        self.console_progress.close()
        self.cap.release()

class altTracker:
    """
    A class used to track objects in a video using YOLO and ByteTrack.
    Attributes
    ----------
    video_path : str
        Path to the video file to be processed.
    selected_model : str
        Path to the YOLO model file.
    inference_tracker : str
        Path to the ByteTrack configuration file.
    device_name : str
        Device to run the inference on (e.g., 'cpu' or 'cuda').
    verbose : bool
        Flag to enable verbose logging.
    model : YOLO
        Loaded YOLO model.
    TRACK_DATA : defaultdict
        Dictionary to store tracking data.
    TRACK_INFO : list
        List to store tracking information.
    current_frame : np.array
        Current frame being processed.
    current_frame_nb : int
        Current frame number.
    cap : cv2.VideoCapture
        Video capture object.
    frame_count : int
        Total number of frames in the video.
    progress : callable
        Progress bar function.
    Methods
    -------
    get_track_data():
        Returns the tracking data.
    get_crossed():
        Returns the crossed data.
    get_track_info():
        Returns the tracking information.
    read_next_frame():
        Reads the next frame from the video.
    process_frame():
        Processes the current frame for object tracking.
    process_video(progress_bar=None):
        Processes the entire video for object tracking.
    """
    def __init__(self, data_manager, verbose : bool = False):
        """
        Initializes the tracking system with the specified parameters.
        Args:
            video_path (str): Path to the video file to be processed.
            selected_model (str, optional): Path to the YOLO model file. Defaults to "yolov10n.pt".
            inference_tracker (str, optional): Path to the tracker configuration file. Defaults to "bytetrack.yaml".
            device_name (str, optional): Device to run the model on, e.g., "cpu" or "cuda". Defaults to "cpu".
            verbose (bool, optional): If True, enables verbose logging. Defaults to False.
        """
        self.video_path = data_manager.video_path
        self.selected_model = data_manager.selected_model
        self.inference_tracker = data_manager.inference_tracker
        self.device_name = data_manager.device_name
        self.verbose = verbose

        # Load YOLO model
        self.model = YOLO(self.selected_model)

        self.current_frame = None
        self.current_frame_nb = 0


    def get_track_data(self, data_manager):
        """
        Retrieve the track data.

        Returns:
            dict: The track data stored in the TRACK_DATA attribute.
        """
        return data_manager.TRACK_DATA

    def get_track_info(self, data_manager):
        """
        Retrieve the track information.

        Returns:
            list: A list containing the track information : TRACK_INFO[frame_nb] = [(track_id, len(track_at_frame_nb))].
        """
        return data_manager.TRACK_INFO

    def read_next_frame(self):
        """
        Reads the next frame from the video capture object.

        This method attempts to read the next frame from the video capture object
        (`self.cap`). It updates the `self.success` attribute to indicate whether
        the frame was successfully read, and the `self.current_frame` attribute
        with the frame data.

        Returns:
            None
        """
        self.success, self.current_frame = self.cap.read()
            
    def process_frame(self, data_manager):
        """
        Processes the current frame using the model's tracking functionality.
        This method performs the following steps:
        1. Tracks objects in the current frame using the model.
        2. Extracts bounding boxes, track IDs, classes, and confidence scores from the tracking results.
        3. Updates the tracking data for each detected object.
        4. Appends the tracking information to the TRACK_INFO list.
        Attributes:
            self.current_frame (ndarray): The current frame to be processed.
            self.verbose (bool): Flag to enable YOLO inference logging (will print for each frame).
            self.inference_tracker (str): The tracker to be used for inference.
            self.device_name (str): The device to be used for processing (e.g., 'cpu' or '0').
            self.TRACK_DATA (dict): Dictionary to store tracking data for each object.
            self.current_frame_nb (int): The current frame number.
            self.TRACK_INFO (list): List to store tracking information for all frames.
        Returns:
            None
        """
        results = self.model.track(self.current_frame, persist=True, verbose=self.verbose, tracker=self.inference_tracker, device=self.device_name, save=False)
        
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id
        classes = results[0].boxes.cls
        confidences = results[0].boxes.conf

        tracked = []
        if track_ids is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist() 
            for box, track_id, clss, confidence in zip(boxes, track_ids, classes, confidences):
                track = data_manager.TRACK_DATA[track_id]
                clss = int(clss)
                track.append((int(self.current_frame_nb), box, confidence, clss))
                tracked.append((track_id, len(track)))

        data_manager.TRACK_INFO.append(tracked)
    
    def process_video(self, data_manager, progress_var=None):
        self.progress = progress_var 
        
        # Open video to process
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = data_manager.frame_count

        self.console_progress = tqdm(total=self.frame_count, desc="YOLO is working", unit="frames")
        # Run inference and tracking
        while self.cap.isOpened():
            self.read_next_frame()
            if self.success:
                self.process_frame(data_manager)
                self.current_frame_nb += 1
                if self.progress is not None:
                    self.progress.set(self.current_frame_nb)
                    self.console_progress.update(1)
                    
            else:
                break

        self.console_progress.close()
        self.cap.release()