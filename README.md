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

* Individally as:
```bash
python3 -m dimensionfour.preprocess --input dataset/1.mp4
python3 -m dimensionfour.preprocess --input dataset/2.mp4
...
```

* Or in bulk as:
```bash
python3 -m dimensionfour.preprocess --input dataset/*
```
The first time you run `dimensionfour.preprocess`, it will download a pretrained yolov3 model from dropbox.
Preprocessing will output a file called `<filename>.d4artifact.zip` into the current directory which contains metadata about your video

3. Assemble your `.d4artifact.zip` files into a final video:
```bash
python3 -m dimensionfour.assemble --input *.d4artifact.zip --output summary.avi
```
This will output a summarized video of all d4artifacts in the current directory as `summary.avi`

#### Options
You can filter the summarized video to only include certain object classes using the `--filter` option:
```bash
python3 -m dimensionfour.assemble --input *.d4artifact.zip --output summary.avi --filter person
```

You can set the fps of the output using the `--fps` option:
```bash
python3 -m dimensionfour.assemble --input *.d4artifact.zip --output summary.avi --fps 10
```


