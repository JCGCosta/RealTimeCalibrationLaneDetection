import cv2
from source import LaneDetector as ld
import numpy as np
import time
import sys

filepath = r"videos/test2.mp4"
cap = cv2.VideoCapture(filepath)
Detector = ld.LaneDetector((cap.get(3), cap.get(4)), "distances.txt")
cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)

while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    if ret:
        output_frame = Detector.frame_processor(frame)
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(output_frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        dist = Detector.get_distance()
        if dist != -1: cv2.putText(output_frame, f'DISTANCE: {dist}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else: cv2.putText(output_frame, f'DISTANCE: NO MEASURE', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Lane Detection", output_frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()