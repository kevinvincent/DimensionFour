import numpy as np
import cv2
import time
import sys
import statistics

cv2.setUseOptimized(True) 

cap = cv2.VideoCapture(sys.argv[1])
# backgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()
backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()


size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2))
fps = 25
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vout = cv2.VideoWriter()
success = vout.open(sys.argv[2],fourcc,fps,size,True) 

def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended

ret, avg = cap.read()

height, width, layers =  avg.shape
new_h=int(height/2)
new_w=int(width/2)
avg = cv2.resize(avg, (new_w, new_h))

avg = np.float32(avg)

while True:

    ret, orig_frame = cap.read()
    
    if not ret:
        break

    height, width, layers =  orig_frame.shape
    new_h=int(height/2)
    new_w=int(width/2)
    orig_frame = cv2.resize(orig_frame, (new_w, new_h)) 

    cv2.accumulateWeighted(orig_frame, avg, 0.001)

    cv2.imshow("1. Input", orig_frame)

    # Pre process frame
    frame = cv2.GaussianBlur(orig_frame, (15, 15), 0)

    # Calculate Diff
    difference = backgroundSubtractor.apply(frame)

    cv2.imshow("2. Subtractor", difference)

    # Post Process to eliminate noise
    # thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
    # difference = filter_small_components(difference)
    # difference = cv2.dilate(difference, None, iterations=5)

    kernel = np.ones((3,3),np.uint8)
    difference = cv2.erode(difference, kernel, iterations=3)
    difference = cv2.dilate(difference, kernel, iterations=10)

    # Find Countours
    raw_countours, _ = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours, convert to Convex Hull
    # contours = []
    # if(len(raw_countours) > 2):
    #     a = np.array([cv2.contourArea(cont) for cont in raw_countours])
    #     median = np.percentile(a, 75)
    #     for x in raw_countours:
    #         if cv2.contourArea(x) > median:
    #             contours.append(x)
    # else:
    #     contours = raw_countours

    contours = raw_countours
        
    # Generate mask with rectangles
    mask = np.zeros(orig_frame.shape, np.uint8)

    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255), -1)

    cv2.imshow("3. Post Process", mask)

    # Blur the mask
    mask = cv2.blur(mask, (50,50))

    # Apply to image
    cv2.imshow("4. Mask", alphaBlend(np.zeros(orig_frame.shape, np.uint8), orig_frame, mask))
    # cv2.imshow("Blend", alphaBlend(np.zeros(orig_frame.shape, np.uint8), orig_frame, mask))
    # cv2.imshow("Avg", r)
    final = alphaBlend(cv2.convertScaleAbs(avg), orig_frame, mask)
    vout.write(final) 

    # cv2.imshow("result", cv2.bitwise_and(orig_frame, orig_frame, mask = img2.astype(np.uint8)))

    # if the `q` key is pressed, break from the lop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
vout.release()
cv2.destroyAllWindows()