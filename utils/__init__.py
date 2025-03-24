import cv2

# Width for console progress bars
DESC_WIDTH = 25

# Color schemes for visualization (RGB format)
CLASS_COLORS = {
    # Colors for different object classes
    0: (135, 110, 234),  # Purple
    1: (216, 183, 40),   # Yellow
    2: (197, 236, 116),  # Light green
    3: (243, 215, 140),  # Light yellow
    4: (238, 151, 57),   # Orange
    5: (215, 146, 255)   # Pink
}

TRIPLINE_COLORS = {
    # Colors for multiple triplines
    0: (7, 25, 206),     # Dark blue
    1: (51, 217, 165),   # Turquoise
    2: (206, 25, 7),     # Red
    3: (59, 219, 59),    # Green
    4: (215, 146, 255),  # Pink
    5: (199, 222, 125),  # Light green
}

class DETECTION_MODEL_CONST():
    '''Configuration constants for object detection models.'''
    
    def __init__(self):
        # Only YOLO .pt models support input resizing through their predictor
        # Other formats require preprocessing the input separately
        self.ALLOW_RESIZE = False
        
        # Lower confidence threshold = more detections but more false positives
        # 0.2 is a good balance for traffic counting where missing objects is worse
        # than occasional false positives
        self.CONF_THRESHOLD = 0.3
        
        # Higher IOU threshold = fewer boxes are merged into one single box
        # 0.6 works well for traffic as vehicles often partially overlap
        self.IOU_THRESHOLD = 0.6
        
        # Enable class-agnostic NMS because different classes can overlap
        # (e.g., car detected as both car and truck)
        self.AGNOSTIC_NMS = True

        # Track analysis score coefficients
        self.TRACK_SCORE_COEFFICIENTS = {
            'avg_conf' : 5,
            'freq_score' : 2,
            'consec_score' : 1,
        }

# Initialize detection constants
DETECTION_MODEL_CONST = DETECTION_MODEL_CONST()

from .session import SessionManager
from .data import DataManager
from .tracking import Counter, Tracker
from .export.xlsx import xlsxWriter, xlsxCompiler, StreetCountCompiler
from .export.video import Annotator

__all__ = [
    'SessionManager',
    'DataManager',
    'Counter',
    'Tracker',
    'xlsxWriter',
    'xlsxCompiler',
    'StreetCountCompiler',
    'Annotator',
]