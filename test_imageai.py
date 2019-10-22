from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "models/yolo.h5"))
detector.loadModel()

trackedObjects = []
objectIdCounter = 0

def forFrame(frame_number, output_array, output_count, detected_frame):
    global objectIdCounter

    if not trackedObjects:
        for detection in output_array:
            trackedObjects.append({
                "id": objectIdCounter,
                "class": detection.name,
                "probability": detection.percentage_probability,
                "bounds": detection.box_points
            })
            objectIdCounter += 1
    else:
        for detection in output_array:
            # Attempt to match with existing objects
            for trackedObject in trackedObjects:

            
        

detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "dataset/preferred.mp4"),
    frames_per_second=30,
    log_progress=True,
    per_frame_function=forFrame,
    save_detected_video=False,
    minimum_percentage_probability=30,
    return_detected_frame=True,
    frame_detection_interval = 5)
