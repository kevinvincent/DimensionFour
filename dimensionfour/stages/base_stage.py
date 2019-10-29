import json
import math
import os

class BaseStage(object):
   def __init__(self, args):
      self.args = args

   def readArtifact(self, artifact):
      os.makedirs(os.path.join('./run_artifacts/', os.path.basename(self.args.input)), exist_ok = True)
      return json.load(open(os.path.join("./run_artifacts",os.path.basename(self.args.input),artifact), "r" ))

   def writeArtifact(self, data, artifact, cls=None):
      os.makedirs(os.path.join('./run_artifacts/', os.path.basename(self.args.input)), exist_ok = True)
      return json.dump(data, open(os.path.join("./run_artifacts",os.path.basename(self.args.input),artifact), "w" ), cls=cls)