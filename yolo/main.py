import cvlib as cvlib
from cvlib.object_detection import draw_bbox
import cv2

cap = cv2.VideoCapture("test5.mp4")

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2))
fps = 25
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vout = cv2.VideoWriter()
success = vout.open('output.mp4',fourcc,fps,size,True) 

while True:
    ret, orig_frame = cap.read()

    if not ret:
        break

    orig_frame = cv2.resize(orig_frame,None,fx=0.5,fy=0.5)

    bbox, label, conf = cvlib.detect_common_objects(orig_frame, confidence=0.50, model='yolov3')
    output_image = draw_bbox(orig_frame, bbox, label, conf)

    cv2.imshow("Output", output_image)
    vout.write(output_image) 

    # if the `q` key is pressed, break from the lop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
vout.release()
cv2.destroyAllWindows()