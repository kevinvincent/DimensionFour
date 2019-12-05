import argparse

from dimensionfour.pipeline.pipeline import Pipeline

from dimensionfour.stages.assemble_stage import AssembleStage

def main():
   pipelineDef = [AssembleStage]

   parser = argparse.ArgumentParser(description="Generates dimensionfour video summary from input files' preprocess artifacts")
   parser.add_argument('--input', nargs='+', help='Path to input file(s) that have been preprocessed.', required=True)
   parser.add_argument('--output', help='Path to output summarized file.', required=True)
   parser.add_argument('--fps', help='Enter the output fps', required=False, type=int, default=30)
   parser.add_argument('--start', help='Enter the start stage number', required=False, type=int, choices=range(0, len(pipelineDef)))
   parser.add_argument('--filter', nargs='*', help='Start pipeline from a specific stage', required=False)
   args = parser.parse_args()

   pipeline = Pipeline(pipelineDef, args)
   pipeline.run()

if __name__ == "__main__":
   main()