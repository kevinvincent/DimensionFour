import cv2 as cv
import argparse
import sys
import random
import string
import numpy as np
import os.path
from lib.centroidmatcher import CentroidMatcher
from lib.multitracker import Multitracker


# Initialize the parameters
confThreshold = 0.25  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--input', help='Path to video file.')
parser.add_argument('--output', help='Path to output file.')
args = parser.parse_args()

# Load names of classes
classesFile = "yolo/coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolo/yolov3.cfg"
modelWeights = "yolo/yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Create Object Tracker
cm = CentroidMatcher()
multitracker = Multitracker()
currentTrackerObjs = []

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box.
    # NOTE: We MUST ALWAYS add +1 to calculate area when working in
    # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    # is the bottom right pixel. If we DON'T add +1, the result is wrong.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    detections = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        detections.append({
            "class": classes[classId],
            "confidence": confidences[i],
            "bounding": (left, top, width, height)
        })

    return detections

# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

if args.input:
    # Open the video file
    if not os.path.isfile(args.input):
        print("Input video file ", args.input, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.input)
else:
    raise Exception("No input file provided")

# Get the video writer initialized to save the output video
if not args.output:
    raise Exception("No output file provided")

vid_writer = cv.VideoWriter(args.output, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

frameCounter = 0
while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing %d frame(s)." % frameCounter)
        print("Output file is stored as ", args.output)
        cv.waitKey(3000)
        cap.release()
        break

    # Every 5 frames run object detector
    if frameCounter % 5 == 0:
    
        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Get efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        print('[Frame %d] Inference time: %.2f ms' % (frameCounter, t * 1000.0 / cv.getTickFrequency()))

        # Remove the bounding boxes with low confidence, nms, get detection objects
        detections = postprocess(frame, outs)
        
        # Get new detections, that is, detections for which we don't yet have an object tracker
        if not currentTrackerObjs:
            for detection in detections:
                t = cv.TrackerMedianFlow_create()
                t.init(frame, detection["bounding"])
                multitracker.add(t, ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)))
        else:
            currentTrackerObjs = multitracker.update(frame)
            currentBoundings = list(map(lambda trackerObj: trackerObj["bbox"], currentTrackerObjs))
            latestBoundings = list(map(lambda detection: detection["bounding"], detections))

            for latestBounding in latestBoundings:
                maxiou = 0
                for currentBounding in currentBoundings:
                    (a, b, c, d) = currentBounding
                    (e, f, g, h) = latestBounding
                    iou = get_iou({
                        "x1": c,
                        "y1": d,
                        "x2": a,
                        "y2": b
                    },{
                        "x1": g,
                        "y1": h,
                        "x2": e,
                        "y2": f
                    })
                    if iou > maxiou:
                        maxiou = iou

                if maxiou < .50:
                    print("Adding new")
                    t = cv.TrackerMedianFlow_create()
                    t.init(frame, latestBounding)
                    multitracker.add(t, ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)))
                

    # Every time run our object tracker
    currentTrackerObjs = multitracker.update(frame)

    for i, currentTrackerObj in enumerate(currentTrackerObjs):
        newbox = currentTrackerObj["bbox"]
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv.rectangle(frame, p1, p2, (255, 178, 50), 3)

    # Write out frame
    vid_writer.write(frame.astype(np.uint8))
    frameCounter += 1

    cv.imshow(winName, frame)

