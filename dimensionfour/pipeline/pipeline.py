import os
import urllib.request
import time
import sys

class Pipeline(object):     
   def __init__(self, pipeline, args):
      self.pipeline = pipeline
      self.args = args
      self.start_time = time.time()

   def run(self):
      for i in range(self.args.start, len(self.pipeline)):
         print("[Pipeline] Starting %s" % self.pipeline[i].__name__)
         step = self.pipeline[i](self.args)
         step.execute()
         print("[Pipeline] Finished %s" % self.pipeline[i].__name__)