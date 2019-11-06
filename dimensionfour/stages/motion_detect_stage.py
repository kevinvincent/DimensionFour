import os
import sys
import cv2
import json
import numpy as np
import subprocess

from dimensionfour.stages.base_stage import BaseStage

class MotionDetectStage(BaseStage):
   def __init__(self, args):
      super().__init__(args)

      self.detections = []
      self.frameCounter = 0

      self.cap = cv2.VideoCapture(args.input)
      self.backgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()


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

         print("[DetectStage] Frame %d: Detecting" % self.frameCounter)

         # Calculate difference
         difference = self.backgroundSubtractor.apply(frame)

         # Process
         kernel = np.ones((3,3),np.uint8)
         difference = cv2.erode(difference, kernel, iterations=3)
         difference = cv2.dilate(difference, kernel, iterations=3)

         # Find Countours
         raw_countours, _ = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

         # Filter out small contours, convert to Convex Hull
         # contours = []
         # if(len(raw_countours) > 2):
         #    a = np.array([cv2.contourArea(cont) for cont in raw_countours])
         #    median = np.percentile(a, 75)
         #    for x in raw_countours:
         #          if cv2.contourArea(x) > median:
         #             contours.append(x)
         # else:
         contours = raw_countours

         # Generate bboxes
         frameDetections = []
         for contour in contours:
            [x,y,w,h] = cv2.boundingRect(contour)
            frameDetections.append({
               "name": "all",
               "percentage_probability": 100,
               "box_points": [x, y, x+w, y+h]
            })
         self.detections.append(frameDetections)

         # Add to frames
         if self.frameCounter % 5 == 0:
            frames.append(frame)

         self.frameCounter += 1
      
      medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)   
      cv2.imwrite(self.getArtifactPath("background_model.jpg"), medianFrame)
      self.writeArtifact(self.detections, "MotionDetectStage.out.json", cls=NpEncoder)

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