import json

from dimensionfour.stages.base_stage import BaseStage

class FrameAssignStage(BaseStage):
   def __init__(self, args):
      super().__init__(args)

   def execute(self):

      tracks = self.readArtifact("FilterMotionStage.out.json")

      frameToDetections = {}
      for i, track in enumerate(tracks):
         for detection in track:

            if detection["frame"] not in frameToDetections:
               frameToDetections[detection["frame"]] = []

            frameToDetections[detection["frame"]].append({
               'bbox': detection["bbox"],
               'name': detection["name"],
               'id': i
            })
      
      print("[FrameAssignStage] %d frame(s) assigned detections" % len(frameToDetections))
      
      self.writeArtifact(frameToDetections, "FrameAssignStage.out.json")