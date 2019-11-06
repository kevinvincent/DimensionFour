import argparse

from dimensionfour.pipeline.pipeline import Pipeline

from dimensionfour.stages.detect_stage import DetectStage
from dimensionfour.stages.motion_detect_stage import MotionDetectStage
from dimensionfour.stages.track_stage import TrackStage
from dimensionfour.stages.motion_track_stage import MotionTrackStage
from dimensionfour.stages.filter_motion_stage import FilterMotionStage
from dimensionfour.stages.frame_assign_stage import FrameAssignStage
from dimensionfour.stages.visualize_stage import VisualizeStage
from dimensionfour.stages.package_stage import PackageStage

def main():
   pipelineDef = [DetectStage, TrackStage, FilterMotionStage, FrameAssignStage, VisualizeStage, PackageStage]
   # pipelineDef = [MotionDetectStage, MotionTrackStage, FilterMotionStage, FrameAssignStage, VisualizeStage, PackageStage]

   parser = argparse.ArgumentParser(description='Generates a dimensionfour preprocess artifact file for input video file')
   parser.add_argument('--input', help='Path to video file.', required=True)
   parser.add_argument('--output', help='Path to output file.', required=True)
   parser.add_argument('--start', help='Enter the start stage number', required=False, type=int, choices=range(0, len(pipelineDef)))
   args = parser.parse_args()

   pipeline = Pipeline(pipelineDef, args)
   pipeline.run()

if __name__ == "__main__":
   main()