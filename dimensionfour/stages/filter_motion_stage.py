import json
import math

from stages.base_stage import BaseStage

class FilterMotionStage(BaseStage):
   def __init__(self, args):
      super().__init__(args)

   def execute(self):

      tracks = self.readArtifact("TrackStage.out.json")

      eliminationCount = 0
      preserveCount = 0

      for i in range(len(tracks) - 1, -1, -1):
         track = tracks[i]
         
         first = track[0]['bbox']
         last = track[-1]['bbox']

         firstCentroid = [0.5*(first[0] + first[2]), 0.5*(first[1] + first[3])]
         lastCentroid = [0.5*(last[0] + last[2]), 0.5*(last[1] + last[3])]

         distance = math.sqrt( ((first[0]-last[0])**2)+((first[1]-last[1])**2) )

         if distance < 100:
            del tracks[i]
            eliminationCount += 1
         else:
            preserveCount += 1

      print("[FilterMotionStage] Eliminated %d, Preserved %d" % (eliminationCount, preserveCount))

      self.writeArtifact(tracks, "FilterMotionStage.out.json")