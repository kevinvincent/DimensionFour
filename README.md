# DimensionFour

### Installation

1. Run the `getModels.sh` file from command line to download the needed model files
```bash
sudo chmod a+x getModels.sh
./getModels.sh
```

2. Install requirements from `requirements.txt`
```bash
pip3 install -r requirements.txt
```


### Running the code

Command line usage for object detection using YOLOv3

```bash
python3 object_detection_yolo.py --video=dataset/VIRAT_S_010204_11_001524_001607.mp4
```