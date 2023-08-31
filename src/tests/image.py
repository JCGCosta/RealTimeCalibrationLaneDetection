import cv2
from source import LaneDetector as ld

filepath = r"images/lane.jpg"
image = cv2.imread(filepath)
Detector = ld.LaneDetector((cap.get(3), cap.get(4)), "distances.txt")
Output_image = Detector.frame_processor(image)
cv2.imwrite(filepath.split('.')[0] + "_output.jpg", Output_image)
cv2.imshow("Lane Detector", Output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()