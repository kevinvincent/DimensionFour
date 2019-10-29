import argparse

from pipeline.pipeline import Pipeline

from stages.detect_stage import DetectStage
from stages.track_stage import TrackStage
from stages.filter_motion_stage import FilterMotionStage
from stages.frame_assign_stage import FrameAssignStage
from stages.visualize_stage import VisualizeStage

def main():
   pipelineDef = [DetectStage, TrackStage, FilterMotionStage, FrameAssignStage, VisualizeStage]

   parser = argparse.ArgumentParser(description='Object Detection and Tracking using YOLO in OPENCV')
   parser.add_argument('--input', help='Path to video file.', required=True)
   parser.add_argument('--output', help='Path to output file.', required=True)
   parser.add_argument('--start', help='Enter the start stage number', required=True, type=int, choices=range(0, len(pipelineDef)))
   args = parser.parse_args()

   pipeline = Pipeline(pipelineDef, args)
   pipeline.run()

if __name__ == "__main__":
   main()