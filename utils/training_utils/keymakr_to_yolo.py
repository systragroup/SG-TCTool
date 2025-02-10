import json
import os
import shutil
import random
import logging
from pathlib import Path
import yaml
from ultralytics.data.utils import compress_one_image
from ultralytics.utils.downloads import zip_directory

def find_empty_frames(frames):
    ''' Expects the data with the zeroframe (metadata item at beginning of keymakr json removed) '''
    # Find frames with no objects annotated (may not be ideal if wish to prevent overtraining)
    empty_frames = []
    for i, frame in enumerate(frames):
        if len(frame.get('objects', [])) == 0:
            empty_frames.append(i)

    return empty_frames

def run(file_dirs, fps_extraction, data_split, ffmpeg = 'ffmpeg'):
    classes = []
    temp_dataset = 'temp'

    for dir in file_dirs:
        files = os.listdir(dir)
        json_file = next((f for f in files if f.endswith('.json')), None)
        if json_file:
            json_file = os.path.join(dir, json_file)

        with open(json_file) as f:
            data = json.load(f)
        data = list(data)
        zero_frame = data[0]

        for obj in zero_frame.get('objects', []):
            cls = obj.get('attributes', {'Vehicle_type' : 'Pedestrian'}).get('Vehicle_type')
            if cls not in classes :
                classes.append(cls)

    classes.sort()
    reverse_classes = {cls : i for i, cls in enumerate(classes)}
    classes = {i : cls for i, cls in enumerate(classes)}
    print(f'{classes=}')
    print(f'{reverse_classes=}')

    for dir in file_dirs:
        files = os.listdir(dir)
        json_file = next((f for f in files if f.endswith('.json')), None)
        if json_file:
            json_file = os.path.join(dir, json_file)

        with open(json_file) as f:
            data = json.load(f)
        zero_frame = data.pop(0)

        empty_frames = find_empty_frames([zero_frame] + data)
        if empty_frames:
            print(f"Skipping {len(empty_frames)} empty frames in {dir}")
        
        img_width = zero_frame.get('width', 0)
        img_height = zero_frame.get('height', 0)

        objects = {}
        for obj in zero_frame.get('objects', []):
            nm = obj.get('nm')
            cls = obj.get('attributes', {'Vehicle_type' : 'Pedestrian'}).get('Vehicle_type')
            objects[nm] = cls

        labels_dir = os.path.join(temp_dataset,os.path.basename(dir), 'labels')
        os.makedirs(labels_dir, exist_ok=False)

        for i, frame in enumerate(data):
            # Skip this one if in the empty frame list
            if i in empty_frames:
                continue

            frame_file = os.path.join(labels_dir, f'{i+1}.txt')
            with open(frame_file, 'w') as f:
                for obj in frame.get('objects', []):
                    nm = obj.get('nm')
                    cls = objects.get(nm, 'Unknown')
                    cls_id = reverse_classes.get(cls, reverse_classes['Pedestrian'])
                    x1 = obj.get('x1', 0)
                    y1 = obj.get('y1', 0)
                    x2 = obj.get('x2', 0)
                    y2 = obj.get('y2', 0)
                    width, height = x2 - x1, y2 - y1
                    x_center, y_center = (x2 + x1) / 2, (y2 + y1) / 2
                    norm_width, norm_x_center = width / img_width, x_center / img_width
                    norm_height, norm_y_center = height / img_height, y_center / img_height
                    f.write(f'{cls_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n')

        total_frames = len(data) - len(empty_frames)
        print(f'Processed {dir} labels into {labels_dir} : {total_frames} frames labeled, {len(objects)} unique objects found')

        video_file = next((f for f in files if f.endswith('.mp4')), None)
        if video_file:
            video_file = os.path.join(dir, video_file)
            images = os.path.join(temp_dataset,os.path.basename(dir), 'images')
            os.makedirs(images, exist_ok=False)
            
            os.system(f'{ffmpeg_path} -i "{video_file}" -vf "select=not(mod(n\,{fps_extraction}))" -vsync vfr {images}/%d.jpg -hide_banner -loglevel error')
            for empty_frame in empty_frames:
                empty_frame_path = os.path.join(images, f'{empty_frame}.jpg')
                if os.path.exists(empty_frame_path):
                    os.remove(empty_frame_path)
            print(f'Processed {dir} video into {images} : {len(os.listdir(images))} frames extracted from {video_file}')
            if total_frames != len(os.listdir(images)):
                logging.error(f'{total_frames} frames labeled but {len(os.listdir(images))} frames extracted from {video_file} ; check fps extraction count (currently set to {fps_extraction})\n Script will continue but this may cause issues')
    
    for f in Path(temp_dataset).rglob("*.jpg"):
        compress_one_image(f)
    print("All images compressed")

    # After processing all directories, organize into train/val/test splits
    dataset_root = "dataset"
    count = 0
    while os.path.exists(dataset_root):
        count += 1
        dataset_root = f"dataset_{count}"

    splits = ["train", "val", "test"]

    for split in splits:
        for subdir in ["images", "labels"]:
            os.makedirs(os.path.join(dataset_root, split, subdir), exist_ok=True)

    for dir in os.listdir(temp_dataset):
        src_images = os.path.join(temp_dataset, dir, "images")
        src_labels = os.path.join(temp_dataset, dir, "labels")
        
        image_files = sorted([f for f in os.listdir(src_images) if f.endswith('.jpg')])
        total_files = len(image_files)
        
        train_size = int(total_files * data_split[0] / 100)
        val_size = int(total_files * data_split[1] / 100)
        # test_size will be the remainder
        
        random.seed('Alto Zafferano')
        random.shuffle(image_files)
        
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]
        
        for files, split in zip([train_files, val_files, test_files], splits):
            for img_file in files:
                # Get corresponding label txt
                label_file = os.path.splitext(img_file)[0] + '.txt'
                
                shutil.copy2(
                    os.path.join(src_images, img_file),
                    os.path.join(dataset_root, split, "images", f"{dir}_{img_file}")
                )
                
                # Copy label
                shutil.copy2(
                    os.path.join(src_labels, label_file),
                    os.path.join(dataset_root, split, "labels", f"{dir}_{label_file}")
                )
        print(f"Processed {dir} into train({train_size})/val({val_size})/test({total_files-train_size-val_size}) split")

    shutil.rmtree(temp_dataset)
    print("Temporary dataset removed")

    print(f"Dataset created in {dataset_root} directory :") 
    print(f"    {len(os.listdir(os.path.join(dataset_root, splits[0], "images")))} train images")
    print(f"    {len(os.listdir(os.path.join(dataset_root, splits[1], "images")))} val images")
    print(f"    {len(os.listdir(os.path.join(dataset_root, splits[2], "images")))} test images")

    yaml_content = {
        'train': '../train/images',
        'val': '../val/images', 
        'test': '../test/images',
        'nc': len(classes),
        'names': classes
        }

    with open(f'{dataset_root}/data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Created dataset configuration file: {dataset_root}/data.yaml")

    zip_directory(dataset_root)
    print(f"Dataset compressed in {dataset_root}.zip")

if __name__ == '__main__':
    file_dirs = ['Indonesia', 'Malaysia', 'Phillipines', 'Singapore']
    fps_extraction = 10 # Extract 1 frame every x frames
    data_split = [80, 20, 0]
    ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'

    run(file_dirs, fps_extraction, data_split, ffmpeg=ffmpeg_path)