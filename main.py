import logging
import os
import time
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.widgets import Meter
import threading
import torch.cuda

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Loading Systra Traffic Counting App dependencies")
import systra_traffic_counting as stc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class STC_App():
    def __init__(self):
        # Initialize DataManager 
        self.DATA = stc.DataManager()
        
    def launch(self):
        # Initialize the GUI
        self.root = ttk.Window(themename="simplex")
        self.root.title("Systra Traffic Counting App")

        def set_inputs():
            self.input_start = time.time()
            self.root.attributes('-topmost', False)
            self.input_GUI_window = ttk.Toplevel(self.root)
            self.input_manager = stc.InputGUI(self.DATA, top_root=self.input_GUI_window)
            self.input_manager.get_inputs()
            self.input_GUI_window.wait_window()
            self.root.attributes('-topmost', True)
            self.root.attributes('-topmost', False)
            self.video_var.set(os.path.basename(self.DATA.video_path))
            self.model_var.set(os.path.basename(self.DATA.selected_model))
            self.site_var.set(os.path.basename(self.DATA.site_location))
            self.do_video_export_var.set(self.DATA.do_video_export)
            self.tracking_meter.textright.config(text="/ "+ str(self.DATA.frame_count) +" frames")
            self.tracking_meter.amounttotalvar.set(self.DATA.frame_count)
            self.annotate_meter.amounttotalvar.set(self.DATA.frame_count)
            if self.DATA.do_video_export:  
                self.annotate_meter.textright.config(text="/ "+ str(self.DATA.frame_count) +" frames")
            self.launch_button.config(state=tk.NORMAL)
            self.root.update_idletasks()
            self.input_end = time.time()

            # Load the YOLO/tracking, counting, annotation and export helpers
            self.load_start = time.time()
            self.tracker = stc.Tracker(self.DATA)
            self.counter = stc.Counter(self.DATA)
            self.annotator = stc.Annotator(self.DATA)
            self.excel = stc.xlsxWriter()
            self.load_end = time.time()

            logging.info("Inputs set and helpers loaded")

        def start_inputGUI():
            # Start the processing in a separate thread
            processing_thread = threading.Thread(target=set_inputs)
            processing_thread.start()
            logging.info("Started input GUI thread")

        def process_video():
            self.launch_button.config(state=tk.DISABLED)
            logging.info("Processing video started")
            
            # Do inference on the video
            self.process_start = time.time()
            self.tracker.process_video(self.DATA, progress_var=self.tracking_meter.amountusedvar)
            self.process_end = time.time()
            self.counting_meter.textright.config(text="/ "+ str(len(self.DATA.TRACK_DATA)) +" tracked obj")
            self.counting_meter.amounttotalvar.set(len(self.DATA.TRACK_DATA))
            self.root.update_idletasks()
            logging.info("Inference on video completed")

            # Count the cars from tracking DATA
            self.count_start = time.time()
            self.counter.count(self.DATA,  progress_var=self.counting_meter.amountusedvar)
            self.count_end = time.time()
            self.counting_meter.amountusedvar.set(self.counting_meter.amounttotalvar.get())
            self.xlsx_meter.textright.config(text="/ "+ str(len(self.DATA.CROSSED)) +" counted obj")
            self.xlsx_meter.amounttotalvar.set(len(self.DATA.CROSSED))
            self.root.update_idletasks()
            logging.info("Counting completed")

            # Export results in excel file
            self.export_start = time.time()
            os.makedirs(self.DATA.export_path, exist_ok=True)
            self.export_path_excel = os.path.join(self.DATA.export_path, f"counted_{self.DATA.site_location}.xlsx")
            self.export_path_excel = self.excel.write_to_excel(self.export_path_excel, self.DATA, progress_var=self.xlsx_meter.amountusedvar)
            self.export_end = time.time()
            self.xlsx_meter.amountusedvar.set(self.xlsx_meter.amounttotalvar.get())
            self.root.update_idletasks()
            logging.info("Export to Excel completed")

            # Export the annotated video
            self.annotate_start = time.time()
            if self.DATA.do_video_export : 
                self.export_path_mp4 = os.path.join(self.DATA.export_path, f"annotated_{self.DATA.site_location}.mp4")
                self.export_path_mp4 = self.annotator.write_annotated_video(self.export_path_mp4, self.DATA, progress_var=self.annotate_meter.amountusedvar)
                self.annotate_meter.amountusedvar.set(self.annotate_meter.amounttotalvar.get())
                logging.info("Annotated video export completed")
            else : logging.info("Annotated video export disabled")
            self.annotate_end = time.time()
            

            # Display the results
            results_label_text = f"{self.export_path_excel}"
            if self.DATA.do_video_export: results_label_text += f"\n{self.export_path_mp4}"
            self.results_label.config(text=results_label_text)

            self.view_xlsx_button.config(state=tk.NORMAL)
            if self.DATA.do_video_export: self.view_video_button.config(state=tk.NORMAL)
            self.open_results_folder_button.config(state=tk.NORMAL)
            self.export_track_jsons.config(state=tk.NORMAL)
            
            logging.info("Processing of video and exports ended")
            logging.info("Time Performance Stats:")
            total = 0.0
            logging.info(f"     Loading helpers : {self.load_end - self.load_start:.2f}s")
            total += self.load_end - self.load_start
            logging.info(f"     Infering on video : {self.process_end - self.process_start:.2f}s ({self.DATA.frame_count/(self.process_end - self.process_start):.2f}fps)")
            total += self.process_end - self.process_start
            logging.info(f"     Counting : {self.count_end - self.count_start:.2f}s")
            total += self.count_end - self.count_start
            logging.info(f"     Exporting results to Excel : {self.export_end - self.export_start:.2f}s")
            total += self.export_end - self.export_start
            if self.DATA.do_video_export:
                logging.info(f"     Exporting annotated video : {self.annotate_end - self.annotate_start:.2f}s  ({self.DATA.frame_count/(self.annotate_end - self.annotate_start):.2f}fps)")
                total += self.annotate_end - self.annotate_start
            logging.info(f"     Total execution time: {total:.2f}s")
            logging.info(f"Human input time (not included in total) was {self.input_end - self.input_start:.2f}s")

        def start_processing():
            # Start the processing in a separate thread
            processing_thread = threading.Thread(target=process_video)
            processing_thread.start()
            logging.info("Started video processing thread")
            # Refresh the window while the thread keeps running
            def refresh_window():
                while processing_thread.is_alive():
                    self.root.update_idletasks()
                    time.sleep(0.1)
            
            refresh_thread = threading.Thread(target=refresh_window)
            refresh_thread.start()
            logging.info("Started window refresh thread")

        def start_compiler():
            processing_thread = threading.Thread(target=launch_compiler)
            processing_thread.start()
            logging.info("Started compiler thread")

        def view_excel():
            os.startfile(self.export_path_excel)
            logging.info(f"Opened Excel file: {self.export_path_excel}")

        def view_video():
            os.startfile(self.export_path_mp4)
            logging.info(f"Opened video file: {self.export_path_mp4}")

        def open_results_folder():
            os.startfile(os.path.dirname(self.export_path_excel))
            logging.info(f"Opened results folder: {os.path.dirname(self.export_path_excel)}")
        
        def launch_compiler():
            self.compiler_GUI_window = ttk.Toplevel(self.root)
            self.compiler = stc.CompilerGUI(top_root=self.compiler_GUI_window)
            logging.info("Compiler GUI launched")
            self.compiler_GUI_window.wait_window()
            self.root.attributes('-topmost', True)
            self.root.attributes('-topmost', False)
            logging.info("Compiler GUI closed")
        
        def export_track_res() :
            self.DATA.export()

        # Inputs & Parameters
        self.video_var = tk.StringVar(value="Video not loaded yet")
        self.model_var = tk.StringVar(value="Model not loaded yet")
        self.site_var = tk.StringVar(value="Site not set yet")
        self.do_video_export_var = tk.BooleanVar()

        self.input_frame = ttk.Labelframe(self.root, text="Setup", padding=(10, 5), bootstyle="primary")
        self.input_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky='ew')
        ttk.Label(self.input_frame, text="Inputs & Parameters : ").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        ttk.Button(self.input_frame, text="Set Inputs", command=start_inputGUI, bootstyle="outline").grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.device_info = ttk.Label(self.input_frame,  bootstyle="info", text=f"Working on {torch.cuda.get_device_name(self.DATA.device_name) if self.DATA.device_name != "" else "CPU"}", font=("TkSmallCaptionFont", 7), justify="right")
        self.device_info.grid(row=0, column=5, pady=5, padx=10, sticky="e")
        ttk.Label(self.input_frame, text="Video : ", bootstyle="seco").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        ttk.Label(self.input_frame, textvariable=self.video_var, bootstyle="secondary", font=("tkSmallCaptionFont", 8 )).grid(row=1, column=1, padx=10, pady=5, sticky="w")
        ttk.Label(self.input_frame, text="Model : ").grid(row=1, column=2, padx=10, pady=5, sticky="e")
        ttk.Label(self.input_frame, textvariable=self.model_var, bootstyle="secondary", font=("tkSmallCaptionFont", 8 )).grid(row=1, column=3, padx=10, pady=5, sticky="w")
        ttk.Label(self.input_frame, text="Site / Location : ").grid(row=1, column=4, padx=10, pady=5, sticky="e")
        ttk.Label(self.input_frame, textvariable=self.site_var, bootstyle="secondary", font=("tkSmallCaptionFont", 8 )).grid(row=1, column=5, padx=10, pady=5, sticky="w")
        self.video_export_button = ttk.Checkbutton(self.input_frame, text="Export annotated video", variable=self.do_video_export_var)
        self.video_export_button.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        self.video_export_button.config(state=tk.DISABLED)

        # Processing of video
        self.process_frame = ttk.Labelframe(self.root, text="Processing", padding=(10, 5), bootstyle="primary")
        self.process_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky='ew')
        self.launch_button = ttk.Button(self.process_frame, text="Process Video", command=start_processing, bootstyle="success", width=12)
        self.launch_button.grid(row=0, column=0, padx=10, pady=5)
        self.launch_button.config(state=tk.DISABLED)

        # Display progress of processing
        self.tracking_meter = Meter(self.process_frame, metersize=170, amounttotal=1, amountused=0, metertype='semi', subtext='YOLO', textright="/ ? frames", textfont="-size 13 -weight bold", subtextfont="-size 8")
        self.tracking_meter.grid(row=1, column=0, padx=10, pady=5)

        self.counting_meter = Meter(self.process_frame, metersize=170, amounttotal=100, amountused=0, metertype='semi', subtext='Counting', textright="/ ? tracked obj", textfont="-size 13 -weight bold", subtextfont="-size 8")
        self.counting_meter.grid(row=1, column=1, padx=10, pady=5)

        self.xlsx_meter = Meter(self.process_frame, metersize=170, amounttotal=100, amountused=0, metertype='semi', subtext='Writing xlsx report', textright="/ ? counted obj", textfont="-size 13 -weight bold", subtextfont="-size 8")
        self.xlsx_meter.grid(row=1, column=2, padx=10, pady=5)

        if self.DATA.do_video_export:  
            self.annotate_meter = Meter(self.process_frame, metersize=170, amounttotal=100, amountused=0, metertype='semi', subtext="Writing annotated video", textright="/ ? frames", textfont="-size 13 -weight bold", subtextfont="-size 8")
        else:
            self.annotate_meter = Meter(self.process_frame, metersize=170, amounttotal=100, amountused=0, metertype='semi', subtext="Video export disabled", textright="", textfont="-size 13 -weight bold", subtextfont="-size 8")
        self.annotate_meter.grid(row=1, column=3, padx=10, pady=5)

        # Results
        self.result_frame = ttk.Labelframe(self.root, text="Results", padding=(10, 5), bootstyle="primary")
        self.result_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky='ew')
        ttk.Label(self.result_frame, text="Results : ").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.results_label = ttk.Label(self.result_frame, text="", font=("tkSmallCaptionFont", 8 ))
        self.results_label.grid(row=0, column=1, columnspan=2, padx=10, pady=5, sticky="ew")
        self.view_xlsx_button = ttk.Button(self.result_frame, text="View Excel", command=view_excel, bootstyle="outline", width=12)
        self.view_xlsx_button.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.view_video_button = ttk.Button(self.result_frame, text="View Video", command=view_video, bootstyle="outline", width=12)
        self.view_video_button.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.open_results_folder_button = ttk.Button(self.result_frame, text="Open Results Folder", command=open_results_folder, bootstyle="outline")
        self.open_results_folder_button.grid(row=1, column=2, padx=10, pady=5, sticky="ew")
        self.export_track_jsons = ttk.Button(self.result_frame, text="Export Tracking Results", command=export_track_res, bootstyle="outline")
        self.export_track_jsons.grid(row=1, column=3, padx=10, pady=5, sticky="ew")
        self.view_xlsx_button.config(state=tk.DISABLED)
        self.view_video_button.config(state=tk.DISABLED)
        self.open_results_folder_button.config(state=tk.DISABLED)
        self.export_track_jsons.config(state=tk.DISABLED)

        # Compile
        self.compile_frame = ttk.Labelframe(self.root, text="Compile", padding=(10, 5), bootstyle="info")
        self.compile_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky='ew')
        self.compile_button = ttk.Button(self.compile_frame, text="Launch Excel Compiler", command=start_compiler, bootstyle="outline-info")
        self.compile_button.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.root.mainloop()

if __name__ == "__main__":
    logging.info("Launching app")
    app = STC_App()
    app.launch()