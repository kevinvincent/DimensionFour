import json
import os
import sys
import tempfile
import shutil

from dimensionfour.stages.base_stage import BaseStage

class PackageStage(BaseStage):
   def __init__(self, args):
      super().__init__(args)

      if not os.path.isfile(args.input):
         print("Input video file ", args.input, " doesn't exist")
         sys.exit(1)

   def execute(self):

      with tempfile.TemporaryDirectory(prefix="dimensionfour_") as tempDir:

         # Copy detections to artifact dir
         source = self.getArtifactPath("background_model.jpg")
         destination = os.path.join(tempDir, "background_model.jpg")
         shutil.copyfile(source, destination) 

         # Copy detections to artifact dir
         source = self.getArtifactPath("FrameAssignStage.out.json")
         destination = os.path.join(tempDir, "detections.json")
         shutil.copyfile(source, destination) 

         # Copy video to artifact dir
         source = self.args.input
         destination = os.path.join(tempDir, "video.mp4")
         shutil.copyfile(source, destination)

         # Make and save our archive to output
         shutil.make_archive(self.args.output + ".d4artifact", 'zip', tempDir)

         print("[PackageStage] Packaged as %s.d4artifact.zip" % self.args.output)



      
      
      
      