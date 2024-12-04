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
import onnx
import onnxruntime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DESC_WIDTH = 25

CLASS_COLORS = {
    0: (134, 110, 135),  # Car -        #876eea
    1: (140, 215, 243),  # Van -        #f3d78c
    2: (116, 236, 197),  # Bus -        #c5ec74
    3: (40, 183, 216),   # Motorcycle - #28b7d8
    4: (57, 151, 238),   # Lorry -      #ee9739
    5: (186, 186, 186)   # Other -      #bababa
}

# Assign colors to each tripline
TRIPLINE_COLORS = {
    0: (7, 25, 206),     # #ce1907
    1: (51, 217, 165),   # #a5d933
    2: (206, 25, 7),     # #0482c8
    3: (59, 219, 59),    # #ffdb3b
    4: (215, 146, 255),  # #ff92d7
    5: (199, 222, 125),  # #7ddec7
}

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

        self.CROSSED =  defaultdict(lambda: [])
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

    def set_directions(self, direction_data):
        self.directions = [dir for dir in direction_data.values()]

    def set_start_datetime(self, start_date, start_time):
        self.start_datetime = datetime.datetime.strptime(f"{start_date} {start_time}:00", r"%Y-%m-%d %H:%M:%S")

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

class Counter:
    def __init__(self, data_manager, progress_callback):
        self.progress_callback = progress_callback
        self.triplines = data_manager.triplines  # Access multiple triplines
        self.directions = data_manager.directions

    def count(self, data_manager):
        obj_count = 0
        total_objs = len(data_manager.TRACK_DATA)
        console_progress = tqdm(total=total_objs, desc=f'{"Counting crossings":<{DESC_WIDTH}}', unit="tracks")
        for track_id, data in data_manager.TRACK_DATA.items():
            for idx, tripline in enumerate(self.triplines):
                for i in range(1, len(data)):
                    point_A = {'x': data[i - 1][1][0], 'y': data[i - 1][1][1]}
                    point_B = {'x': data[i][1][0], 'y': data[i][1][1]}
                    if self.intersect_tripline(tripline['start'], tripline['end'], point_A, point_B):
                        frame = data[i][0]
                        cls = data[i][3]
                        if len(self.triplines) == 1 : direction = self.directions[0] if self.CP(tripline['start'], tripline['end'], point_A, point_B) > 0 else self.directions[1]
                        else : direction = self.directions[idx]
                        # Store the tripline index
                        data_manager.CROSSED[track_id].append((frame, cls, direction, idx))
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
    def __init__(self, data_manager, progress_callback=None, verbose=False):
        self.progress_callback = progress_callback
        self.video_path = data_manager.video_path
        self.selected_model = data_manager.selected_model
        self.inference_tracker = data_manager.inference_tracker
        self.device_name = data_manager.device_name
        self.verbose = verbose
        if data_manager.model_type ==".pt": #Only pt models support resizing
            self.image_size = [32 * (data_manager.width//32) + 32 * min (1,data_manager.width%32), 32 * (data_manager.height//32) + 32 * min (1,data_manager.height%32)] # Input size must be a multiple of max stride 32
        else : self.image_size = [640, 640]
        # Load YOLO model
        self.model = YOLO(self.selected_model, task='detect')
        
        self.current_frame = None
        self.current_frame_nb = 0

    def read_next_frame(self):
        self.success, self.current_frame = self.cap.read()

    def process_frame(self, data_manager):
        results = self.model.track(self.current_frame, imgsz=self.image_size, persist=True, verbose=self.verbose, tracker=self.inference_tracker, device=self.device_name, save=False)
        
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

        self.console_progress = tqdm(total=self.frame_count, desc=f"{'YOLO is working':<{DESC_WIDTH}}", unit="frames")
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

        for obj_id, crossings in data_manager.CROSSED.items():
            for (frame, cls, direction, _) in crossings : 
                timestamp = frame / data_manager.fps
                crossing_time = data_manager.start_datetime + datetime.timedelta(seconds=timestamp)
                date = crossing_time.date()
                first_day_of_week = date - datetime.timedelta(days=date.weekday())
                actual_week_day = crossing_time.strftime("%A")
                time_of_crossing = crossing_time.time()
                interval_15_min = f"{crossing_time.hour}:{(crossing_time.minute // 15) * 15:02d}"
                interval_hr = f"{crossing_time.hour}:00"
                self.write_data.append([data_manager.site_location, date, first_day_of_week, actual_week_day, time_of_crossing, interval_15_min, interval_hr, direction, data_manager.names[cls], obj_id])

    def write_to_excel(self, export_path_excel, data_manager, progress_var = None):
        self.progress = progress_var  # Works with gradio tqdm progress bar
        self.__prepare_data(data_manager)
        # Write the headers
        self.sheet.append(["Site", "Date", "First day of Week", "Week day ", "Time of crossing", "15 Min Interval", "Hour Interval", "Direction", "Class", "ID"])

        # Write the data
        length, prog_count = len(self.write_data), 0
        self.console_progress = tqdm(total=length, desc=f'{"Writing xlsx report":<{DESC_WIDTH}}', unit="rows")
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

        self.console_progress.close()
        return export_path_excel
    
class xlsxCompiler:
    def __init__(self, folder_path=None, file_paths=None):
        self.folder_path = folder_path
        self.file_paths = file_paths if file_paths else []
        self.object_data = {} 
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
                obj_id = row['ID']
                site_location = row['Site']
                date = row['Date']
                vehicle_type = row['Class']
                direction = row['Direction']
                time = row['Time of crossing']
                interval_15_min = row['15 Min Interval']
                interval_hr = row['Hour Interval']

                if obj_id not in self.object_data:
                    self.object_data[obj_id] = {
                        'Site': site_location,
                        'Date': date,
                        'Vehicle Type': vehicle_type,
                        'Directions': set(),
                        'Time of last Crossing': time,
                        '15 Min Interval': interval_15_min,
                        'Hour Interval': interval_hr
                    }
                self.object_data[obj_id]['Directions'].add(direction)
                self.object_data[obj_id]['Time of Crossing'] = time
                self.object_data[obj_id]['15 Min Interval'] = interval_15_min
                self.object_data[obj_id]['Hour Interval'] = interval_hr

    def precompile(self):
        # Process object_data to concatenate directions
        for obj_id, data in self.object_data.items():
            site_location = data['Site']
            date = data['Date'].strftime('%Y-%m-%d')
            vehicle_type = data['Vehicle Type']
            directions = " - ".join(sorted(data['Directions']))
            interval_15_min = data['15 Min Interval']
            interval_hr = data['Hour Interval']
            time = data['Time of last Crossing']

            key = (site_location, date, vehicle_type, directions, interval_15_min, interval_hr)
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
        self.precompile()
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

    def draw_trajectory(self, points, traj_color, traj_thickness=2):
        cv2.polylines(self.frame, [points], isClosed=False, color=traj_color, thickness=traj_thickness)
        cv2.circle(self.frame, (points[-1][0], points[-1][1]), 5, traj_color, -1)
    
    def draw_box_on_frame(self, id : int, color : tuple[int,int,int], bbox : tuple[int,int,int,int], score : float, class_name : str):
        label = f"{class_name}|{id}: {score:0.2f}" # bbox label
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
        self.console_progress = tqdm(total=self.frame_count, desc=f'{"Writing annotated video":<{DESC_WIDTH}}', unit="frames")
        model_name_text = f"Model: {os.path.basename(self.data_manager.selected_model)}"

        # Check if same file exists and enumerate names if it does
        base, extension = os.path.splitext(export_path_mp4)
        counter = 1
        new_export_path_mp4 = export_path_mp4

        while os.path.exists(new_export_path_mp4):
            new_export_path_mp4 = f"{base}_{counter}{extension}"
            counter += 1

        self.export_path = new_export_path_mp4
        
        self.frame_nb = 0

        # Open video to process
        self.open_video()

        self.video_writer = cv2.VideoWriter(
            self.export_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.data_manager.fps,
            (self.width, self.height))

        # Build a dictionary of counted objects with tripline index
        counted = {}


        while self.cap.isOpened():
            success, self.frame = self.cap.read()
            if success:
                for track_id, track_length_at_frame in self.data_manager.TRACK_INFO[self.frame_nb]: # Get each object present on current frame
                    # Check if object crosses a tripline 
                    tripline_indexes = []
                    if track_id in self.data_manager.CROSSED.keys() :
                        tripline_indexes = [crossing[3] for crossing in self.data_manager.CROSSED[track_id]]
                        if self.data_manager.CROSSED[track_id][-1][0] == self.frame_nb:
                            # Store the clss for the object if it is it's last crossing (for class_lines)
                            counted[track_id] = self.data_manager.CROSSED[track_id][-1][1] 
                    # In all cases, get the class of the object
                    cls = self.data_manager.TRACK_DATA[track_id][track_length_at_frame-1][3] 
                    # Get corresponding class color for bounding box
                    class_color = CLASS_COLORS.get(cls, (255, 255, 255))  # Dflt 2 white

                    # Draw trajectories
                    points = np.array([[self.data_manager.TRACK_DATA[track_id][i][1][0], self.data_manager.TRACK_DATA[track_id][i][1][1]]
                                           for i in range(track_length_at_frame)]).astype(np.int32) 
                    if len(points) > 11: # Smoothen trajectories
                        kernel = np.ones(5) / 5.0  # Simple moving average kernel
                        points[5:-5, 0] = np.convolve(points[:, 0], kernel, mode='same')[5:-5]
                        points[5:-5, 1] = np.convolve(points[:, 1], kernel, mode='same')[5:-5]

                        if tripline_indexes != []: #Meaning it will/has cross(ed) a tripline
                            for cnt, trip_idx in enumerate(tripline_indexes):
                                trajectory_color = TRIPLINE_COLORS.get(trip_idx%len(TRIPLINE_COLORS))
                                # offset points for each tripline
                                offset_points = points + [3*cnt, 3*cnt]
                                self.draw_trajectory(offset_points, trajectory_color)

                        else:
                            trajectory_color = (128, 128, 128)  # Gray for uncounted tracks (or forgotten tracks)
                            self.draw_trajectory(points, trajectory_color)

                    # Draw bounding box with class color
                    self.draw_box_on_frame(
                        track_id,
                        class_color,
                        self.data_manager.TRACK_DATA[track_id][track_length_at_frame-1][1],
                        self.data_manager.TRACK_DATA[track_id][track_length_at_frame-1][2],
                        self.data_manager.names[cls]
                    )

                # Draw all triplines with their assigned colors
                for idx, tripline in enumerate(self.data_manager.triplines):
                    color = TRIPLINE_COLORS[idx%len(TRIPLINE_COLORS)]
                    cv2.line(
                        self.frame,
                        (int(tripline['start']['x']), int(tripline['start']['y'])),
                        (int(tripline['end']['x']), int(tripline['end']['y'])),
                        color=color,
                        thickness=2
                    )
                    # Optionally, label the tripline
                    cv2.putText(
                        self.frame,
                        f"{idx+1}",
                        (int(tripline['start']['x']), int(tripline['start']['y']) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        thickness=2
                    )

                # Write the count of objects on each frame
                count_text_1 = f"{len(counted)}/{len(self.data_manager.CROSSED)} objects :"
                cv2.putText(self.frame, count_text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Add and display text lines for each of the detected classes
                class_lines = defaultdict(int)
                for cls in counted.values(): # counted = {track_id : cls} for each object that has crossed it's last tripline at current frame
                    class_lines[int(cls)] += 1

                line_y = 70
                for clss, count in class_lines.items():
                    class_text = f"{self.data_manager.names[int(clss)]}: {count}"
                    cv2.putText(self.frame, class_text, (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 35, 210), 2)
                    line_y += 30

                # Add the model name in the bottom right corner
                (model_text_w, model_text_h), _ = cv2.getTextSize(model_name_text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)
                model_text_x = self.width - model_text_w - 10 #10 px from right edge
                model_text_y = self.height - model_text_h - 5
                cv2.putText(self.frame, model_name_text, (model_text_x, model_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 35, 210), 2)


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