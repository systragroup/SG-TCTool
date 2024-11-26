from ultralytics import YOLO
model = YOLO(r"C:\Users\adufour\OneDrive - SystraGroup\Desktop\GUI_and_compile_test102324\testing\traffic_camera_us_v11n2.pt") 
model.export(format="onnx")