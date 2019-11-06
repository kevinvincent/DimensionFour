import json
import os
import sys
import tempfile
import shutil
import cv2
import numpy as np

from dimensionfour.stages.base_stage import BaseStage
from dimensionfour.lib.util import iou

class AssembleStage(BaseStage):
   def __init__(self, args):
      super().__init__(args)

   def execute(self):

      # Verify input files exist
      for inputPath in self.args.input:
         if not os.path.isfile(inputPath):
            print("[AssembleStage] Input file %s preprocess artifact not found" % inputPath)
            sys.exit(1)

      # Open up temporary working directory
      with tempfile.TemporaryDirectory(prefix="dimensionfour_") as tempDir:

         # Unpack each artifact
         for index, path in enumerate(self.args.input):
            unpackPath = os.path.join(tempDir, os.path.basename(path))
            shutil.unpack_archive(path, unpackPath, "zip")
            print("[AssembleStage] Unpacking %s to %s" % (path, unpackPath))
            
         # Calculate overall background model
         backgroundModels = []
         for index, path in enumerate(self.args.input):
            backgroundModels.append(cv2.imread(os.path.join(tempDir, os.path.basename(path), "background_model.jpg")))
         backgroundModel = np.median(backgroundModels, axis=0).astype(dtype=np.uint8) 

         print("[AssembleStage] Calculated overall background model from %d inputs" % len(backgroundModels))
         
         # Setup captures and load detections
         maxLength = 0
         inputs = []
         vout = None
         for index, path in enumerate(self.args.input):
            cap = cv2.VideoCapture(os.path.join(tempDir, os.path.basename(path), "video.mp4"))
            detections = json.load(open(os.path.join(tempDir, os.path.basename(path), "detections.json"),"r"))
            inputs.append({
               'cap': cap,
               'detections': detections,
               'name': os.path.basename(path).split(".")[0]
            })
            maxLength = max(maxLength, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            if vout == None:
               vout = cv2.VideoWriter(self.args.output, cv2.VideoWriter_fourcc('M','J','P','G'),
         30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


         print("[AssembleStage] Output video length: %d" % maxLength)

         # Overlay videos
         for frameNum in range(0, maxLength):
            
            compiledFrame = np.copy(backgroundModel)
            compiledDetections = []

            for input in inputs:

               # get frame from the video
               hasFrame, frame = input['cap'].read()
               if hasFrame:

                  allDetections = input['detections'].get(str(frameNum)) or []

                  # Filter detections
                  detections = []
                  if self.args.filter:
                     for detection in allDetections:
                        if detection['name'] in self.args.filter:
                           detections.append(detection)
                  else:
                     detections = allDetections

                  mask = np.zeros(compiledFrame.shape, np.uint8)

                  # Draw masks
                  for detection in detections:
                     (a, b, c, d) = detection['bbox']

                     overlap = False
                     for bbox in compiledDetections:
                        if iou(bbox, detection['bbox']) > 0:
                           overlap = True
                           break

                     if overlap:
                        cv2.rectangle(mask,(int(a),int(b)),(int(c),int(d)),(170,170,170), -1)
                     else:
                        cv2.rectangle(mask,(int(a),int(b)),(int(c),int(d)),(255,255,255), -1)
                        

                  # Expand and blur the masks
                  kernel = np.ones((7,7),np.uint8)
                  mask = cv2.dilate(mask, kernel, iterations=3)
                  mask = cv2.blur(mask, (10,10))

                  # Blend into frame
                  compiledFrame = self.alphaBlend(compiledFrame, frame, mask)

                  # Save masks
                  compiledDetections = compiledDetections + [detection['bbox'] for detection in detections]

                  # Draw Labels
                  for detection in detections:
                     (left, top, right, bottom) = detection['bbox']

                     label = input['name']

                     left = int(left)
                     top = int(top)
                     right = int(right)
                     bottom = int(bottom)

                     #Display the label at the top of the bounding box
                     labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 2)
                     top = max(top, labelSize[1])
                     cv2.putText(compiledFrame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 2)

            vout.write(compiledFrame.astype(np.uint8))

         vout.release()
         for input in inputs:
            input['cap'].release()
   
   def alphaBlend(self, img1, img2, mask):
      """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
      """
      if mask.ndim==3 and mask.shape[-1] == 3:
         alpha = mask/255.0
      else:
         alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
      blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
      return blended



               
            


      
