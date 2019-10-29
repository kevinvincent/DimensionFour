import json
import os
import sys
import cv2
import time
import numpy as np

from stages.base_stage import BaseStage

class VisualizeStage(BaseStage):
   def __init__(self, args):
      super().__init__(args)

      if not os.path.isfile(args.input):
         print("Input video file ", args.input, " doesn't exist")
         sys.exit(1)

      self.cap = cv2.VideoCapture(args.input)

      # Create Video Writer
      self.vout = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('M','J','P','G'),
         30, (round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

   def execute(self):

      frameToDetections = self.readArtifact("FrameAssignStage.out.json")

      frameNum = 0
      while True:

         # get frame from the video
         hasFrame, frame = self.cap.read()

         # Stop the stage if reached end of video
         if not hasFrame:
            print("[VisualizeStage] Done processing %d frame(s)." % (frameNum - 1))
            break

         if str(frameNum) in frameToDetections:
            detections = frameToDetections[str(frameNum)]
            for detection in detections:
               self.drawPred(frame, "%d - %s" % (detection["id"], detection["name"]) , detection["bbox"][0], detection["bbox"][1], detection["bbox"][2], detection["bbox"][3])
         
         # cv2.imshow("Window", frame)
         self.vout.write(frame.astype(np.uint8))

         # key = cv2.waitKey(1) #pauses for 3 seconds before fetching next image
         # if key == 27:#if ESC is pressed, exit loop
         #    break

         frameNum += 1
      
      self.cap.release()
      self.vout.release()
      cv2.destroyAllWindows()

      
   def drawPred(self, frame, label, left, top, right, bottom):

      label = str(label)

      left = int(left)
      top = int(top)
      right = int(right)
      bottom = int(bottom)

      # Draw a bounding box.
      cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

      #Display the label at the top of the bounding box
      labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
      top = max(top, labelSize[1])
      cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
      cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

      