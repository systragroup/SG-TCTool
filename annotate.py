import cv2, os
from collections import defaultdict
from numpy import array, int32
from tqdm import tqdm

class Annotator :
    def __init__(self, data_manager):
        pass
        
    def open_video(self, data_manager):
        self.cap = cv2.VideoCapture(data_manager.video_path)
 
        self.frame_count = data_manager.frame_count
        self.START, self.END = data_manager.START, data_manager.END
        self.width, self.height = data_manager.width, data_manager.height

    def draw_preview(self, GRID_SIZE : int = 100):
        self.GRID_SIZE = GRID_SIZE

        # Open the video file
        self.open_video()

        # Read first frame
        success, self.frame = self.cap.read()
        if not success:
            self.cap.release()
            raise Exception("Couldn't read frame from video.")
            quit()

        # Draw grid on frame
        for x in range(0, self.width, self.GRID_SIZE):
            cv2.line(self.frame, (x, 0), (x, self.height), (206, 206, 206), 1)
        for y in range(0, self.height, self.GRID_SIZE):
            cv2.line(self.frame, (0, y), (self.width, y), (206, 206, 206), 1)
        # Write axis names on frame
        cv2.putText(self.frame, "X", (self.width // 2, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (206, 206, 206), 2)
        cv2.putText(self.frame, "Y", (30, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (206, 206, 206), 2)

        # Number the axes
        for x in range(0, self.width, self.GRID_SIZE):
            cv2.putText(self.frame, str(x), (x,self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (206, 206, 206), 1)
        for y in range(0, self.height, self.GRID_SIZE):
            cv2.putText(self.frame, str(y), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (206, 206, 206), 1)

        # Draw the tripline on the frame
        cv2.line(self.frame, self.START, self.END, (0, 255, 0), 5)

        # Release the video capture object
        self.cap.release()

        return self.frame

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

    def write_annotated_video(self, export_path_mp4, data_manager, progress_var = None):
        self.frame_count = data_manager.frame_count
        self.progress = progress_var 
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
        self.open_video(data_manager)

        self.video_writer = cv2.VideoWriter(
            self.export_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            data_manager.fps,
            (self.width, self.height))

        while self.cap.isOpened():
            success, self.frame = self.cap.read()
            if success :
                # Draw the tracking lines and bounding boxes
                for track_id, track_length_at_frame in data_manager.TRACK_INFO[self.frame_nb]:
                    # First, see if objects has crossed
                    if track_id in data_manager.CROSSED.keys() and data_manager.CROSSED[track_id][0] == self.frame_nb:
                            counted[track_id] = data_manager.CROSSED[track_id][1]

                    cls = data_manager.TRACK_DATA[track_id][track_length_at_frame-1][3]
                    # Assign color based on class, default to green if counted
                    if track_id in counted.keys():
                        color = (0, 255, 0)  # Green for counted
                    else:
                        color = COLORS.get(cls, (255, 255, 255))  # Default to white

                    # Draw object track
                    points = array([[data_manager.TRACK_DATA[track_id][i][1][0], data_manager.TRACK_DATA[track_id][i][1][1]] for i in range(track_length_at_frame)]).astype(int32)
                    cv2.polylines(self.frame, [points], isClosed=False, color=color, thickness=2)
                    cv2.circle(self.frame, (points[-1][0], points[-1][1]), 5, color, -1)

                    # Draw box and label
                    if track_id in counted.keys(): color = (0, 255, 0)
                    self.draw_box_on_frame(track_id,
                            color,
                            data_manager.TRACK_DATA[track_id][track_length_at_frame-1][1],
                            data_manager.TRACK_DATA[track_id][track_length_at_frame-1][2],
                            data_manager.names[cls])


                # Draw the tripline on the frame
                cv2.line(self.frame, self.START, self.END, (0, 255, 0), 2)

                # Write the count of objects on each frame
                count_text_1 = f"{len(counted)}/{len(data_manager.CROSSED)} objects have crossed the line :"
                cv2.putText(self.frame, count_text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Add and display text lines for each of the detected classes
                class_lines = defaultdict(int)
                for obj in counted:
                    class_lines[int(data_manager.CROSSED[obj][1])] += 1

                line_y = 70
                for clss, count in class_lines.items():
                    class_text = f"{data_manager.names[int(clss)]}: {count}"
                    cv2.putText(self.frame, class_text, (10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    line_y += 30

                # Add the model name in the bottom right corner
                model_name_text = f"Model: {os.path.basename(data_manager.selected_model)}"
                (model_text_w, model_text_h), _ = cv2.getTextSize(model_name_text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)
                model_text_x = self.width - model_text_w - 10 #10 px from right edge
                model_text_y = self.height - model_text_h - 5
                cv2.putText(self.frame, model_name_text, (model_text_x, model_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                
                # Write frame to video
                self.video_writer.write(self.frame)
                
                self.console_progress.update(1)
                self.frame_nb += 1
                if self.progress is not None: self.progress.set(self.frame_nb)

            else:
                break
        self.console_progress.close()
        self.video_writer.release()
        self.cap.release()
        print("--> Video written at : ", self.export_path)
        return self.export_path