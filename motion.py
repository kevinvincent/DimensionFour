import numpy as np
import cv2
import time
import sys
import statistics
from lib.centroidtracker import CentroidTracker

OPTIONS = {
    "bgSegm" : "MOG2", # GSOC, MOG2
}

# Helpers
def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended

# Setup capture from input file
cap = cv2.VideoCapture(sys.argv[1])

# Create backgroundSubtractor
backgroundSubtractor = None
if OPTIONS["bgSegm"] == "GSOC":
    backgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()
elif OPTIONS["bgSegm"] == "MOG2":
    backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()

# Create Object Tracker
ct = CentroidTracker()

# Create Video Writer
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2))
# fps = 25
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# vout = cv2.VideoWriter()
# success = vout.open(sys.argv[2],fourcc,fps,size,True) 

# Read in Frame
ret, avg = cap.read()

while True:

    ret, orig_frame = cap.read()
    
    if not ret:
        break

    cv2.imshow("1. Input", orig_frame)

    # Pre process frame
    frame = cv2.GaussianBlur(orig_frame, (15, 15), 0)

    # Calculate Diff
    difference = backgroundSubtractor.apply(frame)

    cv2.imshow("2. Subtractor", difference)

    # Erode and Dilate to eliminate noise
    kernel = np.ones((3,3),np.uint8)
    difference = cv2.erode(difference, kernel, iterations=3)
    difference = cv2.dilate(difference, kernel, iterations=10)

    # Find Countours
    contours, _ = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        rects.append((x,y,x+w,y+h))

    uniqueObjects = ct.update(rects)

    # Generate mask with contours
    mask = np.zeros(orig_frame.shape, np.uint8)
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255), -1)

    cv2.imshow("3. Mask", mask)

	# loop over the tracked objects
    for (objectID, centroid) in uniqueObjects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(orig_frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(orig_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
    # Generate mask with contours
    # mask = np.zeros(orig_frame.shape, np.uint8)
    # for contour in contours:
    #     [x,y,w,h] = cv2.boundingRect(contour)
    #     cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255), -1)

    cv2.imshow("4. Post Process", orig_frame)

    # Blur the mask
    # mask = cv2.blur(mask, (50,50))

    # # Apply to image
    # cv2.imshow("4. Mask", alphaBlend(np.zeros(orig_frame.shape, np.uint8), orig_frame, mask))
    # # cv2.imshow("Blend", alphaBlend(np.zeros(orig_frame.shape, np.uint8), orig_frame, mask))
    # # cv2.imshow("Avg", r)
    # final = alphaBlend(cv2.convertScaleAbs(avg), orig_frame, mask)
    # vout.write(final) 

    # cv2.imshow("result", cv2.bitwise_and(orig_frame, orig_frame, mask = img2.astype(np.uint8)))

    # if the `q` key is pressed, break from the lop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
# vout.release()
cv2.destroyAllWindows()