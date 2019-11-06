from imageai.Detection import ObjectDetection
import os
import sys
import cv2
import json
import numpy as np
import subprocess

from dimensionfour.stages.base_stage import BaseStage

class DetectStage(BaseStage):
   def __init__(self, args):
      super().__init__(args)

      if not os.path.isfile(args.input):
         print("[DetectStage] Input file %s not found" % args.input)
         sys.exit(1)

      yoloPath = "./run_artifacts/models/yolo.h5"
      
      if not os.path.isfile(yoloPath):
         print("[DetectStage] %s does not exist. Downloading now." % yoloPath)
         os.makedirs(os.path.dirname(yoloPath), exist_ok=True)
         subprocess.call(["wget","-q","--show-progress","https://www.dropbox.com/s/q5bsuy7ltxoxbkr/yolo.h5?dl=1","-O",yoloPath])

      self.detections = []
      self.frameCounter = 0

      self.cap = cv2.VideoCapture(args.input)

      self.detector = ObjectDetection()
      self.detector.setModelTypeAsYOLOv3()
      self.detector.setModelPath(yoloPath)
      self.detector.loadModel()

   def execute(self):

      frames = []
      while True:

         # get frame from the video
         hasFrame, frame = self.cap.read()

         # Stop the stage if reached end of video
         if not hasFrame:
            print("[DetectStage] Done processing %d frame(s)." % (self.frameCounter - 1))
            self.cap.release()
            break

         # Detect on every x frames and save results
         if self.frameCounter % 5 == 0:
            print("[DetectStage] Frame %d: Detecting" % self.frameCounter)
            _, output_array = self.detector.detectObjectsFromImage(input_type="array", output_type="array", input_image=frame, minimum_percentage_probability=30)
            self.detections.append(output_array)
            frames.append(frame)
         else:
            print("[DetectStage] Frame %d: Skipping" % self.frameCounter)

         self.frameCounter += 1
      
      
      self.writeArtifact(self.detections, "DetectStage.out.json", cls=NpEncoder)

      print("[DetectStage] Calculating median frame")
      medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)   
      cv2.imwrite(self.getArtifactPath("background_model.jpg"), medianFrame)
      print("[DetectStage] Finished writing median frame")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)