import cv2

DESC_WIDTH = 25

CLASS_COLORS = {
    0: (135, 110, 234),  # #876eea
    1: (216, 183, 40),   # #28b7d8
    2: (197, 236, 116),  # #c5ec74
    3: (243, 215, 140),  # #f3d78c
    4: (238, 151, 57),   # #ee9739
    5: (215, 146, 255)   # #ff92d7
}

TRIPLINE_COLORS = {
    0: (7, 25, 206),     # #ce1907
    1: (51, 217, 165),   # #a5d933
    2: (206, 25, 7),     # #0482c8
    3: (59, 219, 59),    # #ffdb3b
    4: (215, 146, 255),  # #ff92d7
    5: (199, 222, 125),  # #7ddec7
}

class DETECTION_MODEL_CONST():
    def __init__(self):
        self.ALLOW_RESIZE = False # Allow image resizing (handled by the YOLO BasePredictor method, only works for yolo.pt models)
        self.CONF_THRESHOLD = 0.25 # Minimum confidence threshold for detections. Objects detected below  threshold are disregarded by predictor (0-1)
        self.IOU_THRESHOLD = 0.6 # Intersection Over Union threshold for Non-Maximum Suppression. Lower values result in fewer detections by eliminating overlapping boxes 
        self.AGNOSTIC_NMS = True # Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes.

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