from openpyxl import Workbook
from collections import defaultdict
from tkinter import filedialog
import os
import cv2
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import datetime
import torch.cuda
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InputGUI:
    class HintEntry(ttk.Entry):
        def __init__(self, master=None, hint="Enter text here", **kwargs):
            super().__init__(master, **kwargs)
            self.hint = hint
            self.default_fg = 'grey'
            self.user_fg = 'black'
            self.insert(0, self.hint)
            self.configure(foreground=self.default_fg)
            self.bind("<FocusIn>", self._clear_hint)
            self.bind("<FocusOut>", self._add_hint)

        def _clear_hint(self, event):
            if self.get() == self.hint:
                self.delete(0, tk.END)
                self.configure(foreground=self.user_fg)

        def _add_hint(self, event):
            if not self.get():
                self.insert(0, self.hint)
                self.configure(foreground=self.default_fg)

    def __init__(self, data_manager, top_root=None):
        self.data_manager = data_manager
        self.top_root = top_root
        logging.info("InputGUI initialized")


    def get_inputs(self):

        def lift_wdw(window):
            window.attributes('-topmost', True)
            window.attributes('-topmost', False)

        def browse_export_folder():
            folder_selected = filedialog.askdirectory()
            self.export_folder_var.set(folder_selected)
            lift_wdw(self.root)
            logging.info(f"Export folder selected: {folder_selected}")

        def browse_video_file():
            file_selected = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
            self.video_file_var.set(file_selected)
            self.data_manager.set_video_params(file_selected)
            self.tripline_button.config(state=tk.NORMAL)
            lift_wdw(self.root)
            logging.info(f"Video file selected: {file_selected}")

        def browse_model_file():
            file_selected = filedialog.askopenfilename(filetypes=[("Model files", "*.pt;*.onnx")])
            self.model_file_var.set(file_selected)
            self.data_manager.set_names(file_selected)
            names_label_short = self.data_manager.names.copy()
            if len(names_label_short) > 7 :
                names_label_short = ', '.join(str(names_label_short).split(',')[:6] + [' ...'] + str(names_label_short).split(',')[-1:])
            self.names_label.config(text=f"Detection classes : {names_label_short}")
            lift_wdw(self.root)
            logging.info(f"Model file selected: {file_selected}")
            
        def define_tripline():
            self.tripline_helper = TriplineHelper(self.video_file_var.get())
            start, end = self.tripline_helper.define()
            self.tripline_start_x_var.set(start[0])
            self.tripline_start_y_var.set(start[1])
            self.tripline_end_x_var.set(end[0])
            self.tripline_end_y_var.set(end[1])
            self.tripline_button.config(bootstyle="secondary-outline")
            self.tripline_label_var.set(f"{start} --> {end}")
            self.tripline_button_var.set("Redraw")
            lift_wdw(self.root)
            logging.info(f"Tripline defined: {start} --> {end}")

        def get_date():
            self.start_date_var.set(self.start_date_entry.entry.get())
            logging.info(f"Date selected: {self.start_date_var.get()}")

        def get_location():
            if self.site_location_var.get() == site_location_hint : 
                self.site_location_var.set(os.path.basename(os.path.splitext(self.video_file_var.get())[0]))
            logging.info(f"Site/Location set: {self.site_location_var.get()}")

        def continue_fn():
            get_date()
            get_location()
            self.data_manager.selected_model = self.model_file_var.get()
            self.data_manager.export_path = self.export_folder_var.get()
            self.data_manager.inference_tracker = self.tracker_var.get()
            self.data_manager.site_location = self.site_location_var.get()
            self.data_manager.START = (self.tripline_start_x_var.get(), self.tripline_start_y_var.get())
            self.data_manager.END = (self.tripline_end_x_var.get(), self.tripline_end_y_var.get())
            self.data_manager.set_tripline()
            self.data_manager.do_video_export = self.do_video_export_var.get()
            self.data_manager.start_datetime = datetime.datetime.strptime(f"{self.start_date_var.get()} {self.start_hour_var.get()}:{self.start_minute_var.get()}:{self.start_second_var.get()}", r"%x %H:%M:%S")
            self.data_manager.directions = (self.direction_1_var.get(), self.direction_2_var.get())
            self.data_manager.set_video_params(self.video_file_var.get())
            self.root.destroy()
            logging.info("Params saved and InputGUI window closed")

        if self.top_root is None : self.root = ttk.Window(themename="simplex")
        else : self.root = self.top_root
        self.root.title("Systra Traffic Counting App - Inputs & Parameters")

        self.export_folder_var = tk.StringVar()
        self.video_file_var = tk.StringVar()
        self.model_file_var = tk.StringVar()
        self.site_location_var = tk.StringVar()
        self.tracker_var = tk.StringVar(value="bytetrack.yaml")
        self.tripline_start_x_var = tk.IntVar()
        self.tripline_start_y_var = tk.IntVar()
        self.tripline_end_x_var = tk.IntVar()
        self.tripline_end_y_var = tk.IntVar()
        self.tripline_label_var = tk.StringVar(value="Tripline :")
        self.tripline_button_var = tk.StringVar(value="Define")
        self.do_video_export_var = tk.BooleanVar()
        self.start_date_var = tk.StringVar()
        self.start_hour_var = tk.StringVar(value="07")
        self.start_minute_var = tk.StringVar(value="00")
        self.start_second_var = tk.StringVar(value="00")
        self.direction_1_var = tk.StringVar(value="North")
        self.direction_2_var = tk.StringVar(value="South")

        self.input_frame = ttk.Labelframe(self.root, text="Input Files", padding=(10, 5), bootstyle="primary")
        self.input_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

        ttk.Label(self.input_frame, text="Video File:", width=12).grid(row=0, column=0, padx=10, pady=5, sticky="e")
        video_entry_hint = "Accepts .mp4 / .avi"
        InputGUI.HintEntry(self.input_frame, textvariable=self.video_file_var, width=50, hint=video_entry_hint).grid(row=0, column=1, padx=10, pady=5)
        ttk.Button(self.input_frame, text="Browse", command=browse_video_file, bootstyle="outline", width=12).grid(row=0, column=2, padx=10, pady=5)

        ttk.Label(self.input_frame, text="Model File:", width=12).grid(row=1, column=0, padx=10, pady=5, sticky='e')
        model_entry_hint = "Accepts .pt / .onnx"
        InputGUI.HintEntry(self.input_frame, textvariable=self.model_file_var, width=50, hint=model_entry_hint).grid(row=1, column=1, padx=10, pady=5)
        ttk.Button(self.input_frame, text="Browse", command=browse_model_file, bootstyle="outline", width=12).grid(row=1, column=2, padx=10, pady=5)
        self.names_label = ttk.Label(self.input_frame, text="", font=("TkSmallCaptionFont", 8))
        self.names_label.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky='ew')

        self.site_settings_frame = ttk.Labelframe(self.root, text="Site Settings", padding=(10, 5), bootstyle="primary")
        self.site_settings_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

        ttk.Label(self.site_settings_frame, text="Site Location:", width=12).grid(row=0, column=0, padx=10, pady=5, sticky='e')
        site_location_hint = "If none, defaults \nto video name"
        InputGUI.HintEntry(self.site_settings_frame, textvariable=self.site_location_var, width=50, hint=site_location_hint).grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(self.site_settings_frame, text="Filming date:", width=12).grid(row=1, column=0, padx=10, pady=5, sticky='e')

        self.time_date_middle_frame = ttk.Frame(self.site_settings_frame)
        self.time_date_middle_frame.grid(row=1, column=1, padx=10, pady=5, sticky='ew')

        self.start_date_entry = ttk.DateEntry(self.time_date_middle_frame)
        self.start_date_entry.pack(side="left")

        ttk.Label(self.time_date_middle_frame, text="Filming start time :").pack(side="right")

        self.time_input_frame = ttk.Frame(self.site_settings_frame, width=12)
        ttk.Entry(self.time_input_frame, textvariable=self.start_hour_var, width=2).pack(side='left')
        ttk.Label(self.time_input_frame, text=":").pack(side='left')
        ttk.Entry(self.time_input_frame, textvariable=self.start_minute_var, width=2).pack(side='left')
        ttk.Label(self.time_input_frame, text=":").pack(side='left')
        ttk.Entry(self.time_input_frame, textvariable=self.start_second_var, width=2).pack(side='left')
        self.time_input_frame.grid(row=1, column=2, columnspan=1, pady=5, padx=10, sticky="w")

        ttk.Label(self.site_settings_frame, text="Directions :", width=12).grid(row=2, column=0, padx=10, pady=5, sticky='e')
        self.directions_frame = ttk.Frame(self.site_settings_frame)
        ttk.Entry(self.directions_frame, textvariable=self.direction_1_var, width=6).pack(side='left')
        ttk.Label(self.directions_frame, text=" / ").pack(side='left')
        ttk.Entry(self.directions_frame, textvariable=self.direction_2_var, width=6).pack(side='left')
        ttk.Label(self.directions_frame, textvariable=self.tripline_label_var).pack(side='right')
        self.directions_frame.grid(row=2, column=1, padx=10, pady=5, sticky='ew')


        self.tripline_button = ttk.Button(self.site_settings_frame, textvariable=self.tripline_button_var, command=define_tripline, bootstyle="primary", width=12)
        self.tripline_button.grid(row=2, column=2, columnspan=3, pady=5, padx=10)
        self.tripline_button.config(state=tk.DISABLED)

        self.process_settings_frame = ttk.Labelframe(self.root, text="Processing Settings", padding=(10, 5), bootstyle="primary")
        self.process_settings_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

        ttk.Label(self.process_settings_frame, text="Export Folder :", width=12).grid(row=0, column=0, padx=10, pady=5, sticky='e')
        export_folder_entry_hint = "Excel report & video write location"
        InputGUI.HintEntry(self.process_settings_frame, textvariable=self.export_folder_var, width=50, hint=export_folder_entry_hint).grid(row=0, column=1, padx=10, pady=5)
        ttk.Button(self.process_settings_frame, text="Browse", command=browse_export_folder, bootstyle="outline", width=12).grid(row=0, column=2, padx=10, pady=5)
        ttk.Label(self.process_settings_frame, text="Tracker :", width=12).grid(row=1, column=0, padx=10, pady=5, sticky='e')
        self.menu_export_frame = ttk.Frame(self.process_settings_frame)
        self.menu_export_frame.grid(row=1, column=1, pady=5, sticky='ew', padx=10)
        self.tracker_menu = ttk.Menubutton(self.menu_export_frame, textvariable=self.tracker_var, bootstyle="outline")
        self.tracker_menu.menu = tk.Menu(self.tracker_menu, tearoff=0)
        self.tracker_menu["menu"] = self.tracker_menu.menu
        self.tracker_menu.menu.add_radiobutton(label="bytetrack.yaml", variable=self.tracker_var, value="bytetrack.yaml")
        self.tracker_menu.menu.add_radiobutton(label="botsort.yaml", variable=self.tracker_var, value="botsort.yaml")
        self.tracker_menu.pack(side="left")

        ttk.Label(self.menu_export_frame, text="Export Annotated Video:").pack(side="right")
        self.check_frame = ttk.Frame(self.process_settings_frame, width=12)
        ttk.Checkbutton(self.check_frame, variable=self.do_video_export_var, bootstyle="round-toggle").grid(row=0, column=2, padx=10, pady=5)
        self.check_frame.grid(row=1, column=2, columnspan=1, pady=5, sticky='w')

        self.continue_button = ttk.Button(self.root, text="Save Inputs & Parameters", command=continue_fn, bootstyle="success")
        self.continue_button.grid(row=11, column=0, columnspan=3, pady=10)
        self.continue_button.config(state=tk.DISABLED)

        self.device_info = ttk.Label(self.root, text=f"Working on device {self.data_manager.device_name} :\n {torch.cuda.get_device_name(self.data_manager.device_name) if self.data_manager.device_name != "" else "CPU"}", font=("TkSmallCaptionFont", 7), justify="center")
        self.device_info.grid(row=11, column=2, pady=10)
        def validate_fields(*args):
            if self.export_folder_var.get() and self.video_file_var.get() and self.model_file_var.get() and self.tracker_var.get() and self.tripline_start_x_var.get() and self.tripline_start_y_var.get() and self.tripline_end_x_var.get() and self.tripline_end_y_var.get() and self.start_date_entry.entry.get() and self.start_hour_var.get() and self.start_minute_var.get():
                if self.export_folder_var.get() != export_folder_entry_hint and self.video_file_var.get() != video_entry_hint and self.model_file_var.get() != model_entry_hint :
                    self.continue_button.config(state=tk.NORMAL)
            else:
                self.continue_button.config(state=tk.DISABLED)

        self.export_folder_var.trace_add("write", validate_fields)
        self.video_file_var.trace_add("write", validate_fields)
        self.model_file_var.trace_add("write", validate_fields)
        self.tracker_var.trace_add("write", validate_fields)
        self.tripline_start_x_var.trace_add("write", validate_fields)
        self.tripline_start_y_var.trace_add("write", validate_fields)
        self.tripline_end_x_var.trace_add("write", validate_fields)
        self.tripline_end_y_var.trace_add("write", validate_fields)
        self.start_date_var.trace_add("write", validate_fields)
        self.start_hour_var.trace_add("write", validate_fields)
        self.start_minute_var.trace_add("write", validate_fields)

        lift_wdw(self.root)
        logging.info("InputGUI window opened")
        if self.top_root is None : self.root.mainloop()

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

        self.__exported = False

    def set_tripline(self):
        self.tripline = (self.START, self.END)

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

    def export(self):
        import copy
        import json

        # Create a deep copy of the tracking and counting data for an eventual export later on
        data_copy = {
            "CROSSED": copy.deepcopy(self.CROSSED),
            "TRACK_DATA": {track_id : [item[0], item[1].tolist(), item[2].tolist(), item[3]] for track_id, item in self.TRACK_DATA.items()},
            "TRACK_INFO": copy.deepcopy(self.TRACK_INFO),
            "params" : {"START": self.START,
                         "END": self.END,
                         "video_path": self.video_path,
                         "selected_model": self.selected_model,
                         "inference_tracker": self.inference_tracker,
                         "site_location": self.site_location,
                         "start_datetime": self.start_datetime.isoformat() if self.start_datetime else None,
                         "directions": self.directions}
        }

        # Define the export directory path
        export_dir_path = os.path.join(self.export_path, "data_output")
        os.makedirs(export_dir_path, exist_ok=True)

        # Write each item to a separate JSON file
        for key, value in data_copy.items():
            export_file_path = os.path.join(export_dir_path, f"{key}.json")
            with open(export_file_path, "w") as f:
                json.dump(value, f, indent=4)
                logging.info(f"Data for {key} exported to {export_file_path}")

        logging.info(f"Tracking and counting data exported to {export_file_path}")
        self.__exported = True
        return export_file_path
        

    def clear(self, do_export_check=True) :
        if not(do_export_check) :
            self.__exported = True
        if self.__exported :
            self.CROSSED.clear()
            self.TRACK_DATA.clear()
            self.TRACK_INFO.clear()
            self.START, self.END = None, None
            self.video_path = None
            self.export_path = None
            self.selected_model = None
            self.inference_tracker = None
            self.site_location = None
            self.do_video_export = False
            self.start_datetime = None
            self.directions = None
            self.frame_count = 0
            self.fps = 30
            self.width = 0
            self.height = 0
        else : 
            logging.warning("Data not exported. Please export the data before clearing. You can disable this check with 'do_export_check=False'")

class xlsxWriter:
    def __init__(self):
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

class TriplineHelper:
    """
    A helper class to define a tripline on a video frame using mouse events.
    Attributes:
        video_path (str): Path to the video file.
        cap (cv2.VideoCapture): Video capture object.
        fps (float): Frames per second of the video.
        width (int): Width of the video frame.
        height (int): Height of the video frame.
        START (tuple): Starting point of the tripline.
        END (tuple): Ending point of the tripline.
    Methods:
        __mouse_callback(event, x, y, flags, param):
            Private method to handle mouse events and set the START and END points.
        define():
            Displays the first frame of the video and allows the user to define a tripline by clicking and dragging the mouse.
            Returns the START and END points of the tripline.
    """
    def __init__(self, video_path):
        """
        Initializes the video capture object with the given video path.

        Args:
            video_path (str): The path to the video file.

        Attributes:
            video_path (str): The path to the video file.
            cap (cv2.VideoCapture): The video capture object.
            fps (float): The frames per second of the video.
            width (int): The width of the video frames.
            height (int): The height of the video frames.
            START (None): Placeholder for the start time or frame.
            END (None): Placeholder for the end time or frame.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.width, self.height =  int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),  int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.START, self.END = None, None

    def __mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function to handle mouse events.
        This function is triggered by mouse events and updates the START and END 
        coordinates based on the type of event.
        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP).
            x (int): The x-coordinate of the mouse event.
            y (int): The y-coordinate of the mouse event.
            flags (int): Any relevant flags passed by OpenCV.
            param (any): Additional parameters passed by OpenCV.
        Updates:
            self.START (tuple): Coordinates (x, y) when the left mouse button is pressed down.
            self.END (tuple): Coordinates (x, y) when the left mouse button is released.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.START = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.END = (x, y)

    def define(self):
        """
        Captures a frame from the video stream and allows the user to define a line by clicking on the frame.
        
        The method displays the captured frame and waits for the user to press the SPACE key to continue.
        During this time, the user can define a line by clicking on the frame. The line is drawn between
        the start and end points defined by the user's clicks.

        Returns:
            tuple: A tuple containing the start and end points of the line defined by the user.
        """
        ret, frame = self.cap.read()
        window_title = f'Define tripline for : {os.path.basename(self.video_path)} '
        if ret:
            key = 0
            old_frame = frame.copy()
            new_frame = frame.copy()
            while not key == ord(' '):
                cv2.putText(new_frame, 'Press SPACE to continue', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(window_title, cv2.resize(old_frame, (self.width, self.height)))
                cv2.setMouseCallback(window_title, self.__mouse_callback)
                cv2.line(new_frame, self.START, self.END, (250, 155, 0), 5)
                old_frame = new_frame
                new_frame = frame.copy()
                key = cv2.waitKey(50)
            cv2.destroyAllWindows()
            self.cap.release()
        return (self.START, self.END)

if __name__ == "__main__":
    data_manager = DataManager()
    input_helper = InputGUI(data_manager)
    input_helper.get_inputs()