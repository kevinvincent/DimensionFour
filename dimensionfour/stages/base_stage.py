import json
import math
import os

class BaseStage(object):
   def __init__(self, args):
      self.args = args

   def getWorkingPath(self):
      return os.path.join('./run_artifacts/', os.path.basename(self.args.input))

   def getArtifactPath(self, artifact):
      return os.path.join(self.getWorkingPath(), artifact)

   def readArtifact(self, artifact):
      return json.load(open(self.getArtifactPath(artifact), "r" ))

   def writeArtifact(self, data, artifact, cls=None):
      os.makedirs(self.getWorkingPath(), exist_ok = True)
      return json.dump(data, open(self.getArtifactPath(artifact), "w" ), cls=cls)