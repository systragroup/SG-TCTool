import os
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from openpyxl import Workbook
import pandas as pd

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
            self.extract_data(file, os.path.basename(file))

    def extract_data(self, file_path, file_name):
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

class CompilerGUI:
    def __init__(self, top_root = None):
        if top_root is None : self.root= ttk.Window(themename="simplex")
        else : self.root = top_root
        self.root.title("Systra Traffic Counting App - Excel Compiler")
        
        self.folder_path = tk.StringVar()
        self.file_paths = []
        
        self.create_widgets()
        self.root.mainloop()

    def create_widgets(self):
        
        ttk.Label(self.root, text="Select Folder or Files:").grid(row=1, column=0, columnspan=3, pady=5, sticky="ew")
        
        ttk.Button(self.root, text="Select Folder", command=self.select_folder, bootstyle="outline").grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(self.root, text="Select Files", command=self.select_files, bootstyle="outline").grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        self.folder_label = ttk.Label(self.root, textvariable=self.folder_path, wraplength=300)
        self.folder_label.grid(row=3, column=0, columnspan=3, pady=5, sticky="ew")
        
        self.file_frame = ttk.Labelframe(self.root, text="Selected files", bootstyle="primary")
        self.file_frame.grid(row=4, column=0, columnspan=3, pady=5, padx=5, sticky="nsew")
    
        self.files_label = ttk.Label(self.file_frame, text="", wraplength=300)
        self.files_label.pack(padx=5, pady=5, fill="both", expand=True)
        
        self.compile_button = ttk.Button(self.root, text="Compile", command=self.compile, bootstyle="solid")
        self.compile_button.grid(row=5, column=0, columnspan=3, padx= 10, pady=10, sticky="ew")

        self.reveal_button = ttk.Button(self.root, text="Reveal in Explorer", command=self.reveal_file, bootstyle="outline")
        self.reveal_button.grid(row=6, column=0, padx=5, pady=5, sticky="ew")
        self.reveal_button.config(state=tk.DISABLED)

        self.open_button = ttk.Button(self.root, text="Open in Excel", command=self.open_file, bootstyle="outline")
        self.open_button.grid(row=6, column=1, padx=5, pady=5, sticky="ew")
        self.open_button.config(state=tk.DISABLED)

        self.close_button = ttk.Button(self.root, text="Close", command=self.root.destroy, bootstyle="outline")

        
    def change_compile_button(self):
        self.compile_button.config(bootstyle="success", text="Compiled ! Recompile ?")

    def compile(self):
        if not self.folder_path.get() and not self.file_paths:
            print("Error, please select a folder or files to compile.")
            return
        
        self.output_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if not self.output_path:
            return
        
        compiler = xlsxCompiler(folder_path=self.folder_path.get(), file_paths=self.file_paths)
        compiler.compile(self.output_path)
        self.change_compile_button()
        self.reveal_button.config(state=tk.NORMAL)
        self.open_button.config(state=tk.NORMAL)
        
    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)
            self.file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.xlsx')]
            file_base_names = (os.path.basename(file) for file in self.file_paths)
            self.files_label.config(text=" · "+"\n · ".join(file_base_names))
    
    def select_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Excel files", "*.xlsx")])
        if files:
            self.file_paths = files
            self.folder_path.set("")
            file_base_names = (os.path.basename(file) for file in files)
            self.files_label.config(text=" · "+"\n · ".join(file_base_names))
    
    def reveal_file(self):
        os.startfile(os.path.dirname(self.output_path))

    def open_file(self):
        os.startfile(self.output_path)
    

if __name__ == "__main__":
    app = CompilerGUI()
