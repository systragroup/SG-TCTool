import cv2
import logging
import os
import numpy as np
import subprocess
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from collections import defaultdict
from utils import CLASS_COLORS, TRIPLINE_COLORS, DESC_WIDTH


class Annotator:
    def __init__(self, data_manager, progress_callback=None):
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
        # Get analyzed data
        track_analysis = self.data_manager.TRACK_ANALYSIS.get(id, None)
        if track_analysis:
            final_class = self.data_manager.names[track_analysis['class']]
            avg_conf = track_analysis['confidence']
            if final_class==class_name : 
                label = f"ID{id}-{final_class} ({avg_conf:.2f}/{score:.2f})"
            else : 
                label = f"ID{id}-{final_class}/{class_name} ({avg_conf:.2f}/{score:.2f})"
        else:
            label = f"ID{id}-{class_name} ({score:.2f})"

        lbl_margin = 3 #label margin
        bbox = [int(value) for value in bbox]
        self.frame = cv2.rectangle(self.frame, (bbox[0]-bbox[2]//2,bbox[1]-bbox[3]//2),(bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2), color=color,thickness = 2) #Draw bbox
        label_size = cv2.getTextSize(label, # labelsize in pixels
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.7, thickness=2)
        lbl_w, lbl_h = label_size[0] # label w and h
        lbl_w += 2* lbl_margin # add margins on both sides
        lbl_h += 2*lbl_margin
        self.frame = cv2.rectangle(self.frame, (bbox[0]-bbox[2]//2,bbox[1]-bbox[3]//2), # plot label background
                            (bbox[0]-bbox[2]//2+lbl_w,bbox[1]-bbox[3]//2-lbl_h),
                            color=color,
                            thickness=-1) # thickness=-1 means filled
        cv2.putText(self.frame, label, (bbox[0]-bbox[2]//2+lbl_margin,bbox[1]-bbox[3]//2-lbl_margin),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7, color=(255, 255, 255 ),
                    thickness=2)

    def write_annotated_video(self, export_path_mp4):
        self.frame_count = self.data_manager.frame_count
        self.console_progress = tqdm(total=self.frame_count, desc=f'{"Writing annotated video":<{DESC_WIDTH}}', unit="frames", dynamic_ncols=True)
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

        with logging_redirect_tqdm():
            while self.cap.isOpened():
                success, self.frame = self.cap.read()
                if success:
                    for track_id, track_length_at_frame in self.data_manager.TRACK_INFO[self.frame_nb]: # Get each object present on current frame
                        # Get analyzed class for color
                        if track_id in self.data_manager.TRACK_ANALYSIS:
                            cls = self.data_manager.TRACK_ANALYSIS[track_id]['class']
                        else:
                            cls = self.data_manager.TRACK_DATA[track_id][track_length_at_frame-1][3]
                            
                        class_color = CLASS_COLORS.get(cls%len(CLASS_COLORS))

                        # Check if object crosses a tripline 
                        tripline_indexes = []
                        if track_id in self.data_manager.CROSSED.keys() :
                            tripline_indexes = [x[3] for x in self.data_manager.CROSSED[track_id]]
                            if self.data_manager.CROSSED[track_id][-1][0] == self.frame_nb:
                                # Store the clss for the object if it is it's last crossing (for class_lines)
                                counted[track_id] = cls

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
    
    def reformat_video(self, input_path : str, ffmpeg_path='ffmpeg', cleanup=True):
        output_path = str(input_path).replace('.mp4', '_reformatted.mp4')

        command = [
            ffmpeg_path,
            '-i', input_path,
            '-c:v', 'libx264',        # Video codec
            '-preset', 'fast',        # Encoding speed/quality trade-off
            '-crf', '22',             # Constant Rate Factor (quality)
            '-an',                     # Disable audio
            '-movflags', '+faststart',# Enable streaming
            output_path
        ]

        try:
            logging.info(f"Reformatting video output")
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if cleanup:
                try:
                    os.remove(input_path)
                    os.rename(output_path, input_path)
                except Exception as e:
                    logging.warning(f"Could not delete or rename video: {str(e)}")
            else :
                try:
                    os.rename(input_path, input_path.replace('.mp4', '_old.mp4'))
                    os.rename(output_path, input_path)
                except Exception as e:
                    logging.warning(f"Could not rename temp video: {str(e)}")
            logging.info("Video reformatting completed successfully.")
            return output_path
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg error: {e.stderr.decode()}")
            return None