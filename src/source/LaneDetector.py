import cv2
import numpy as np
from source import filters as af
import math as mt

class LaneDetector:
    def __init__(self, resolution, distances_path, frame_skip=0, controls=True, record=False, record_fps=30):
        self.DISTANCES_PATH = distances_path
        self.RESOLUTION = resolution
        self.FRAME_SKIP = frame_skip
        self.POLYGON = None
        self.LAST_POLYGON = None
        self.RECORD = None
        self.COUNTER = 0

        self.CHANGE_MASK = 90
        self.CHANGE_METHOD = 3
        self.CHANGE_CENTER_X = 50
        self.CHANGE_CENTER_Y = 50
        self.CHANGE_GAP = 20
        self.CHANGE_SLOPE = 50
        self.CHANGE_WIDTH = 50
        self.CHANGE_HEIGHT = 50
        self.CHANGE_DETECT_DIST = 60

        if controls: self._setup_controls()
        if record: self._setup_record(record_fps)

    def get_distance(self):
        try: return float(open(self.DISTANCES_PATH, "r").read())
        except e: return -1

    def _on_change_lowerth(self, a):
        self.CHANGE_MASK = a

    def _on_change_rm(self, a):
        self.CHANGE_METHOD = a

    def _on_change_slope(self, a):
        self.CHANGE_SLOPE = a

    def _on_change_centerx(self, a):
        self.CHANGE_CENTER_X = a

    def _on_change_centery(self, a):
        self.CHANGE_CENTER_Y = a

    def _on_change_gap(self, a):
        self.CHANGE_GAP = a

    def _on_change_width(self, a):
        self.CHANGE_WIDTH = a

    def _on_change_height(self, a):
        self.CHANGE_HEIGHT = a

    def _on_change_detdist(self, a):
        self.CHANGE_DETECT_DIST = a

    def _setup_controls(self):
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Step", "Controls", self.CHANGE_METHOD, 3, self._on_change_rm)
        cv2.createTrackbar("MaskLwrThd", "Controls", self.CHANGE_MASK, 255, self._on_change_lowerth)
        cv2.createTrackbar("ROICenterX", "Controls", self.CHANGE_CENTER_X, 100, self._on_change_centerx)
        cv2.createTrackbar("ROICenterY", "Controls", self.CHANGE_CENTER_Y, 100, self._on_change_centery)
        cv2.createTrackbar("ROIGap", "Controls", self.CHANGE_GAP, 100, self._on_change_gap)
        cv2.createTrackbar("ROIWidth", "Controls", self.CHANGE_SLOPE, 100, self._on_change_slope)
        cv2.createTrackbar("ROIHeight", "Controls", self.CHANGE_HEIGHT, 100, self._on_change_height)
        cv2.createTrackbar("ROISlope", "Controls", self.CHANGE_WIDTH, 100, self._on_change_width)
        cv2.createTrackbar("DetDist", "Controls", self.CHANGE_DETECT_DIST, 100, self._on_change_detdist)

    def _setup_record(self, record_fps):
        self.RECORD = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), record_fps, self.RESOLUTION)

    def frame_processor(self, image):
        if self.COUNTER == self.FRAME_SKIP:
            self.COUNTER = 0
            color_select = af.color_selection(image, self.CHANGE_MASK)  # Filtro de cor amarela e branca
            if self.CHANGE_METHOD == 0: return color_select
            region = af.double_region_selection(color_select, self.CHANGE_CENTER_X, self.CHANGE_CENTER_Y, self.CHANGE_SLOPE, self.CHANGE_WIDTH, self.CHANGE_HEIGHT, self.CHANGE_GAP)  # Filtro da região de interesse
            if self.CHANGE_METHOD == 1: return region
            smooth = cv2.GaussianBlur(region, (7, 7), 0)  # Atenuador gausiano
            edges = cv2.Canny(region, 180, 200)  # Filtro detector de bordas
            if self.CHANGE_METHOD == 2: return edges
            lines = af.lane_lines(image, cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300), self.CHANGE_DETECT_DIST)
            dist = self.get_distance()
            if dist == -1: # Distancia nao calculado pelo modulo
                self.POLYGON = af.draw_lane_polygon(image, lines, color=[0, 100, 0])
            elif dist > 2.5: # Limiar de distancia
                self.POLYGON = af.draw_lane_polygon(image, lines, color=[0, 100, 0])
            else:
                self.POLYGON = af.draw_lane_polygon(image, lines, color=[0, 0, 100])
        else:
            self.COUNTER = self.COUNTER + 1
        if np.any(self.POLYGON):
            self.LAST_POLYGON = self.POLYGON  # O programa conseguiu encontrar uma faixa
        if np.any(self.LAST_POLYGON):
            if self.RECORD is not None: self.RECORD.write(cv2.addWeighted(image, 1, self.LAST_POLYGON, 0.01, 0))
            return cv2.addWeighted(image, 1, self.LAST_POLYGON, 1, 0)
        return image