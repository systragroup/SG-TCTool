# utils.py
import os
from ultralytics import YOLO
import datetime
import logging
from collections import defaultdict
from tqdm import tqdm
from openpyxl import Workbook
import torch.cuda
import cv2
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataManager:
    def __init__(self):
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

        self.CROSSED = {}
        self.TRACK_DATA = defaultdict(lambda: [])
        self.TRACK_INFO = []

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

    def get_directions(self, form_data):
        directions = []
        i = 1
        while True:
            direction = form_data.get(f'direction{i}')
            if not direction:
                break
            directions.append(direction)
            i += 1
        print("data_manager.directions : ", directions)
        return directions

    def set_video_params(self, video_path):
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def set_names(self, selected_model):
        _, extension = os.path.splitext(selected_model)
        if extension == ".pt": #YOLO returns pt model names in dict format
            from ultralytics import YOLO
            class_names = YOLO(selected_model).model.names
            if class_names:
                self.names = YOLO(selected_model).model.names
                logging.info("Class names loaded from pt metadata.")
            else:
                logging.warning("Class names not found in pt metadata.")
                
        elif extension == ".onnx": #ONNX model names metadata is a string
            import onnx
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

class Counter:
    def __init__(self, data_manager, progress_callback):
        self.progress_callback = progress_callback
        self.triplines = data_manager.triplines  # Access multiple triplines
        self.directions = data_manager.directions

    def count(self, data_manager):
        obj_count = 0
        total_objs = len(data_manager.TRACK_DATA)
        console_progress = tqdm(total=total_objs, desc="Counting crossings", unit="tracks")
        
        #For each tracked object, check if it has crossed each of the triplines
        for track_id, data in data_manager.TRACK_DATA.items(): #TRACK DATA[track_id] = [(frame_nb, box, confidence, clss)]   !!! > box = [x, y, w, h]
                for idx, tripline in enumerate(self.triplines):
                    for i in range(1, len(data)):
                        point_A, point_B = {'x' : data[i-1][1][0], 'y' : data[i-1][1][1]}, {'x' : data[i][1][0], 'y' : data[i][1][1]}
                        if self.intersect_tripline(tripline['start'], tripline['end'], point_A, point_B):
                            if len(self.triplines) == 1:
                                direction = self.directions[0] if self.CP(tripline[0], tripline[1], data[0][1], data[-1][1]) >= 0 else self.directions[1]
                            else:
                                direction = self.directions[idx]
                            data_manager.CROSSED[track_id] = [(data[i][0], data[i][3], direction)]
                            break
                console_progress.update(1)
                obj_count += 1 
                if self.progress_callback:
                    progress_percentage = int((obj_count / total_objs) * 100)
                    self.progress_callback(progress_percentage)
        console_progress.close()

    def CP(self, START, END, A, B): #Cross Product (Positive means B is on left side of S-E, negative B is on the right and 0 is S-E and A-B colinear)
        # Visualise right-hand rule : index is Start-End(tripline), middle finger is A-B and thumb is CP. 
        return (B['x'] - A['x']) * (END['y'] - START['y']) - (B['y'] - A['y']) * (END['x'] - START['x'])

    def intersect_tripline(self, START, END, A, B):
        def ccw_point(P, Q, R): #Counter Clock wise order of points
            return (R['y'] - P['y']) * (Q['x'] - P['x']) > (Q['y'] - P['y']) * (R['x'] - P['x'])
        
        return ccw_point(START, A, B) != ccw_point(END, A, B) and ccw_point(START, END, A) != ccw_point(START, END, B)

class Tracker:
    def __init__(self, data_manager, progress_callback=None, verbose=False):
        self.progress_callback = progress_callback
        self.video_path = data_manager.video_path
        self.selected_model = data_manager.selected_model
        self.inference_tracker = data_manager.inference_tracker
        self.device_name = data_manager.device_name
        self.verbose = verbose
        # Load YOLO model
        self.model = YOLO(self.selected_model, task='detect')
        
        self.current_frame = None
        self.current_frame_nb = 0

    def read_next_frame(self):
        self.success, self.current_frame = self.cap.read()

    def process_frame(self, data_manager):
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

        data_manager.TRACK_INFO.append(tracked) #TRACK_INFO is indexed by frame : for a given frame, see which objects are where, and how long they've been tracked

    def process_video(self, data_manager): 
        # Open video to process
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = data_manager.frame_count

        self.console_progress = tqdm(total=self.frame_count, desc="YOLO is working", unit="frames")
        # Run inference and tracking
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

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

class xlsxWriter:
    def __init__(self, progress_callback):
        self.progress_callback = progress_callback
        # Create a new workbook
        self.workbook = Workbook()
        # Create a new sheet
        self.sheet = self.workbook.active

    def __prepare_data(self, data_manager):
        self.write_data = []

        for obj_id, (frame, cls, direction) in data_manager.CROSSED.items():
            timestamp = frame / data_manager.fps
            crossing_time = data_manager.start_datetime + datetime.timedelta(seconds=timestamp)
            date = crossing_time.date()
            first_day_of_week = date - datetime.timedelta(days=date.weekday())
            actual_week_day = crossing_time.strftime("%A")
            time_of_crossing = crossing_time.time()
            interval_15_min = f"{crossing_time.hour}:{(crossing_time.minute // 15) * 15:02d}"
            interval_hr = f"{crossing_time.hour}:00"
            self.write_data.append([data_manager.site_location, date, first_day_of_week, actual_week_day, time_of_crossing, interval_15_min, interval_hr, direction, data_manager.names[cls]])

    def write_to_excel(self, export_path_excel, data_manager, progress_var = None):
        self.progress = progress_var  # Works with gradio tqdm progress bar
        self.console_progress = tqdm(total=len(data_manager.CROSSED), desc="Writing xlsx report", unit="objects")
        self.__prepare_data(data_manager)

        # Write the headers
        self.sheet.append(["Site", "Date", "First day of the Week", "Actual Week Day of crossing", "Time of crossing", "15 Min Interval of Crossing", "Hr interval of Crossing", "Direction", "Class"])

        # Write the data
        length, prog_count = len(self.write_data), 0
        for row in self.write_data:
            self.sheet.append(row)
            self.console_progress.update(1)
            prog_count += 1 
            if self.progress is not None : self.progress = prog_count// length
            if self.progress_callback:
                    progress_percentage = int((prog_count / length) * 100)
                    self.progress_callback(progress_percentage)

        # Check if same file exists and enumerate names if it does
        base, extension = os.path.splitext(export_path_excel)
        counter = 1
        new_export_path_excel = export_path_excel

        while os.path.exists(new_export_path_excel):
            new_export_path_excel = f"{base}_{counter}{extension}"
            counter += 1

        export_path_excel = new_export_path_excel

        # Save the workbook
        self.workbook.save(export_path_excel)
        print("--> Results written at : ", export_path_excel)

        self.console_progress.close()
        return export_path_excel
    
class xlsxCompiler:
    def __init__(self, folder_path=None, file_paths=None):
        self.folder_path = folder_path
        self.file_paths = file_paths if file_paths else []
        self.compiled_data = {}

    def read_files(self):
        if self.folder_path:
            files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith('.xlsx')]
        else:
            files = self.file_paths

        for file in files:
            self.extract_data(file)

    def extract_data(self, file_path):
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            for _, row in df.iterrows():
                site_location = row['Site']
                date = row['Date'].strftime('%x')
                vehicle_type = row['Class']
                direction = row['Direction']
                interval_15_min = row['15 Min Interval of Crossing']
                interval_hr = row['Hr interval of Crossing']

                key = (site_location, date, vehicle_type, direction, interval_15_min, interval_hr)
                if key not in self.compiled_data:
                    self.compiled_data[key] = 0
                self.compiled_data[key] += 1

    def write_compiled_data(self, output_path):
        workbook = Workbook()
        sheet = workbook.active

        headers = ['Site/Location', 'Date', 'Vehicle Type', 'Direction', '15 Min Interval', 'Hour Interval', 'Total Count']
        sheet.append(headers)

        for key, count in self.compiled_data.items():
            sheet.append(list(key) + [count])

        workbook.save(output_path)

    def compile(self, output_path):
        self.read_files()
        self.write_compiled_data(output_path)

class Annotator:
    def __init__(self, data_manager, progress_callback):
        self.progress_callback = progress_callback
        self.data_manager = data_manager
        self.START = data_manager.START
        self.END = data_manager.END

    def open_video(self):
        self.cap = cv2.VideoCapture(self.data_manager.video_path)
        self.frame_count = self.data_manager.frame_count
        self.START, self.END = self.data_manager.START, self.data_manager.END
        self.width, self.height = self.data_manager.width, self.data_manager.height
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def draw_box_on_frame(self, id : int, color : tuple[int,int,int], bbox : tuple[int,int,int,int], score : float, class_name : str):
        label = f"{class_name} - {id}: {score:0.2f}" # bbox label
        lbl_margin = 3 #label margin
        bbox = [int(value) for value in bbox]
        self.frame = cv2.rectangle(self.frame, (bbox[0]-bbox[2]//2,bbox[1]-bbox[3]//2),(bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2), color=color,thickness = 2) #Draw bbox
        label_size = cv2.getTextSize(label, # labelsize in pixels
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1, thickness=2)
        lbl_w, lbl_h = label_size[0] # label w and h
        lbl_w += 2* lbl_margin # add margins on both sides
        lbl_h += 2*lbl_margin
        self.frame = cv2.rectangle(self.frame, (bbox[0]-bbox[2]//2,bbox[1]-bbox[3]//2), # plot label background
                            (bbox[0]-bbox[2]//2+lbl_w,bbox[1]-bbox[3]//2-lbl_h),
                            color=color,
                            thickness=-1) # thickness=-1 means filled
        cv2.putText(self.frame, label, (bbox[0]-bbox[2]//2+lbl_margin,bbox[1]-bbox[3]//2-lbl_margin),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(255, 255, 255 ),
                    thickness=2)

    def write_annotated_video(self, export_path_mp4):
        self.frame_count = self.data_manager.frame_count
        self.console_progress = tqdm(total=self.frame_count, desc="Writing annotated video", unit="frames")
        
        COLORS = {
            0: (70, 130, 180),   # Car - Steel Blue
            1: (60, 179, 113),   # Van - Medium Sea Green
            2: (218, 165, 32),   # Bus - Goldenrod
            3: (138, 43, 226),   # Motorcycle - Blue Violet
            4: (255, 140, 0),    # Lorry - Dark Orange
            5: (128, 128, 128)   # Other - Gray
        }

        # Check if same file exists and enumerate names if it does
        base, extension = os.path.splitext(export_path_mp4)
        counter = 1
        new_export_path_mp4 = export_path_mp4

        while os.path.exists(new_export_path_mp4):
            new_export_path_mp4 = f"{base}_{counter}{extension}"
            counter += 1

        self.export_path = new_export_path_mp4
        
        self.frame_nb = 0
        counted = {}

        # Open video to process
        self.open_video()

        self.video_writer = cv2.VideoWriter(
            self.export_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.data_manager.fps,
            (self.width, self.height))

        while self.cap.isOpened():
            success, self.frame = self.cap.read()
            if success :
                # Draw the tracking lines and bounding boxes
                for track_id, track_length_at_frame in self.data_manager.TRACK_INFO[self.frame_nb]:
                    # First, see if objects has crossed
                    if track_id in self.data_manager.CROSSED.keys() and self.data_manager.CROSSED[track_id][0] == self.frame_nb:
                            counted[track_id] = self.data_manager.CROSSED[track_id][1]

                    cls = self.data_manager.TRACK_DATA[track_id][track_length_at_frame-1][3]
                    # Assign color based on class, default to green if counted
                    if track_id in counted.keys():
                        color = (0, 255, 0)  # Green for counted
                    else:
                        color = COLORS.get(cls, (255, 255, 255))  # Default to white

                    # Draw object track
                    points = np.array([[self.data_manager.TRACK_DATA[track_id][i][1][0], self.data_manager.TRACK_DATA[track_id][i][1][1]] for i in range(track_length_at_frame)]).astype(np.int32)
                    cv2.polylines(self.frame, [points], isClosed=False, color=color, thickness=2)
                    cv2.circle(self.frame, (points[-1][0], points[-1][1]), 5, color, -1)

                    # Draw box and label
                    if track_id in counted.keys(): color = (0, 255, 0)
                    self.draw_box_on_frame(track_id,
                            color,
                            self.data_manager.TRACK_DATA[track_id][track_length_at_frame-1][1],
                            self.data_manager.TRACK_DATA[track_id][track_length_at_frame-1][2],
                            self.data_manager.names[cls])


                # Draw the tripline on the frame
                cv2.line(self.frame, self.START, self.END, (0, 255, 0), 2)

                # Write the count of objects on each frame
                count_text_1 = f"{len(counted)}/{len(self.data_manager.CROSSED)} objects have crossed the line :"
                cv2.putText(self.frame, count_text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Add and display text lines for each of the detected classes
                class_lines = defaultdict(int)
                for obj in counted:
                    class_lines[int(self.data_manager.CROSSED[obj][1])] += 1

                line_y = 70
                for clss, count in class_lines.items():
                    class_text = f"{self.data_manager.names[int(clss)]}: {count}"
                    cv2.putText(self.frame, class_text, (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    line_y += 30

                # Add the model name in the bottom right corner
                model_name_text = f"Model: {os.path.basename(self.data_manager.selected_model)}"
                (model_text_w, model_text_h), _ = cv2.getTextSize(model_name_text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)
                model_text_x = self.width - model_text_w - 10 #10 px from right edge
                model_text_y = self.height - model_text_h - 5
                cv2.putText(self.frame, model_name_text, (model_text_x, model_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                
                # Write frame to video
                self.video_writer.write(self.frame)
                
                self.console_progress.update(1)
                self.frame_nb += 1
                # Update progress
                if self.progress_callback:
                    progress_percentage = int((self.frame_nb / self.frame_count) * 100)
                    self.progress_callback(progress_percentage)
            else:
                break
        self.console_progress.close()
        self.video_writer.release()
        self.cap.release()
        return self.export_path