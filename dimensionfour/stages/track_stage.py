import json

from stages.base_stage import BaseStage
from lib.iou_tracker import track_iou


class TrackStage(BaseStage):
   def __init__(self, args):
      super().__init__(args)

   def execute(self):

      frames = self.readArtifact("DetectStage.out.json")

      # Format data for tracker
      dataFormatted = []
      for i, frame in enumerate(frames):
         detectionFormatted = []
         for detection in frame:
            bb = detection["box_points"]
            s = detection["percentage_probability"]
            name = detection["name"]
            detectionFormatted.append({'roi': [bb[1], bb[0], bb[3], bb[2]], 'score': s, 'centroid': [0.5*(bb[0] + bb[2]), 0.5*(bb[1] + bb[3])], 'name': name, 'frame': i*5})
         dataFormatted.append(detectionFormatted)

      # Run tracker
      tracker_output = track_iou(dataFormatted, .30, .30, 1, 4)
      # 1) Min Probability
      # 2) IOU threshold to be same object
      # 3) maximum frames a track remains pending before termination.
      # 4) minimum track length in frames.

      # Format back to our preferred format
      bbox_tracks = []
      for track in tracker_output:
         bbox_track = []
         for detection in track:
            roi = detection["roi"]
            frame = detection["frame"]
            name = detection["name"]
            bbox_track.append({
               'bbox': [roi[1],roi[0],roi[3],roi[2]],
               'frame': frame,
               'name': name
            })
         bbox_tracks.append(bbox_track)

      self.writeArtifact(bbox_tracks, "TrackStage.out.json")