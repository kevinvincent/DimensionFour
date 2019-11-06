# DimensionFour

### Installation

Make sure you have have the `pipenv` package manager installed
```bash
pip3 install pipenv
```

Install dependencies using `pipenv`
```bash
pipenv shell
pipenv install
```

### Running the code

Command line usage

1. Enter the environment using:
```bash
pipenv shell
```

2. Preprocess your video files:
```bash
python3 -m dimensionfour.preprocess --input dataset/1.mp4 --output video_1
python3 -m dimensionfour.preprocess --input dataset/2.mp4 --output video_2
...
```
The first time you run dimensionfour.preprocess, it will download a pretrained yolov3 model from dropbox.
Preprocessing will output a file called `<output>.d4artifact.zip` which contains metadata about your video

3. Assemble your `.d4artifact.zip` files into a final video:
```bash
python3 -m dimensionfour.assemble --input video_1.d4artifact.zip video_2.d4artifact.zip --output summary.avi
```
This will output a summarized video of all inputs as `summary.avi`

You can filter the summarized video to only include certain object classes using the `--filter` option:
```bash
python3 -m dimensionfour.assemble --input video_1.d4artifact.zip video_2.d4artifact.zip --output summary.avi --filter person
```


