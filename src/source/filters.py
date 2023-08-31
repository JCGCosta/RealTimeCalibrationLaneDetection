import cv2
import numpy as np
import math as mt

def color_selection(image, lower_th):
    # Convert the input image to HSL
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # White color mask
    lower_threshold = np.uint8([lower_th])
    upper_threshold = np.uint8([255])
    white_mask = cv2.inRange(gray, lower_threshold, upper_threshold)

    # Yellow color mask
    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image

def get_rectangle_points(center_x, center_y, slope, width, height):
    # Calculate half the width and height
    half_width = width / 2
    half_height = height / 2

    # Calculate the x-component of the slope angle
    x_component = slope * half_height

    # Calculate the four points of the rectangle
    point1 = (center_x - half_width - x_component, center_y - half_height)
    point2 = (center_x + half_width - x_component, center_y + half_height)
    point3 = (center_x + half_width + x_component, center_y + half_height)
    point4 = (center_x - half_width + x_component, center_y - half_height)

    return point1, point2, point3, point4

def double_region_selection(image, center_x, center_y, slope, width, height, gap):
    lmask = np.zeros_like(image)
    rmask = np.zeros_like(image)
    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    rows, cols = image.shape[:2]

    gap_between = gap * (cols / 100)
    center_x1 = (center_x * (cols / 100)) - gap_between / 2
    center_x2 = (center_x * (cols / 100)) + gap_between / 2
    center_yb = center_y * (rows / 100)
    real_width = width * (cols / 100)
    real_height = height * (rows / 100)
    real_slope = slope * 0.01

    lvertices = np.array([get_rectangle_points(center_x1, center_yb, real_slope, -real_width, real_height)], dtype=np.int32)
    rvertices = np.array([get_rectangle_points(center_x2, center_yb, real_slope, real_width, real_height)], dtype=np.int32)

    cv2.fillPoly(lmask, lvertices, ignore_mask_color)
    cv2.fillPoly(rmask, rvertices, ignore_mask_color)

    masked_image = cv2.bitwise_or(lmask, rmask)
    masked_image = cv2.bitwise_and(image, masked_image)
    return masked_image

def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_weights.append(length)
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append(length)
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
        return left_lane, right_lane
    else:
        return None, None

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)
        return (x1, y1), (x2, y2)
    except OverflowError:
        return None

def lane_lines(image, lines, detect_dist):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * ((100 - detect_dist) * 0.01)
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_polygon(image, lines, color=None, thickness=8):
    if color is None:
        color = [255, 0, 0]
    polygon_image = np.zeros_like(image)
    if not None in lines:
        points = np.array([lines[0][0], lines[1][0], lines[1][1], lines[0][1]])
        cv2.fillPoly(polygon_image, np.int32([points]), color)
    return polygon_image