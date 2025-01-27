# Traffic Counting App

A Flask-based web app that processes uploaded video files to perform object detection, counting, and optional annotated video export. It uses YOLO (Ultralytics) for detection and can create Excel reports with the results.

## Overview

- **[`script.py`](script.py)** provides the core processing logic without any GUI code. It handles reading a video, detecting objects, tracking them, counting crossings, generating Excel reports, and optionally exporting annotated videos.  
- **[`app.py`](app.py)** uses Flask to provide a browser-based interface and asynchronous server handling. This allows the server to be deployed separately from the client machines accessing it.

## Installation

1. Clone the repository.  
2. Create and activate a Python virtual environment.

    ```bash
    python -m venv .venv
    ```

3. Install dependencies:  

    ```bash
    pip install -r requirements.txt
    ```

4. By default, the FFmpeg executable used is the one referenced by the `FFmepg PATH`. Uncomment and edit `os.environ['ffmpeg'] = "\path\to\the\exe"`.

## Usage

Development/testing only. Use a production server for actual deployment *(refer to [Flask - Deploying to Production](https://flask.palletsprojects.com/en/stable/deploying/))*

1. Run the Flask app:

    ```bash
    flask run
    ```

    - This starts a local development server on the default host/port (see *[Flask - Quickstart](https://flask.palletsprojects.com/en/stable/quickstart/#debug-mode)*).
2. Access the web interface (e.g., [127.0.0.1:5000](127.0.0.1:5000)) to upload video and model files ( `.pt`, `openvino` and `onnx` supported), draw triplines, set direction names, and process the video.

3. The backend logic and processing is handled in separate threads by main.py : multiple current processes can be handled at once (performance is however degraded)

## Notes

- The Flask server can be run on a separate machine so multiple users can access it for video processing tasks.
- All the graphical interactions (tripline drawing, form inputs, etc.) are handled by the web app.
- **[`script.py`](script.py)** can be imported and used as such :

    ```python
    from script import *

    params = {}

    params['video_path'] = "\path\to\your\vid"
    params['model_path'] = "\path\to\your\model"
    params['site_location'] = "Name of Location"
    params['inference_tracker'] = "tracker.yaml" # 2 are supported : `bytetrack.yaml` & `botsort.yaml` (BoT-SORT is slower)
    params['export_video'] = True # False 
    params['start_date'] = "2025-01-20" # 'YYYY-MM-DD'
    params['start_time'] = "12:12" # 'HH:MM'
    params['ffmpeg_executable_path'] = "ffmpeg"

    run(params)

    ```
