import cv2

def rescale_frame(frame, size=(640, 640)):
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

def show_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    rescaled_frame = rescale_frame(frame)
    cv2.imshow('Rescaled Frame', rescaled_frame)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    video_path = r"C:\Users\adufour\Downloads\M6 Motorway Traffic.mp4"  # Replace with your video file path
    show_first_frame(video_path)
    