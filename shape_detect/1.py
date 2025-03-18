import numpy as np
import cv2

def mark(contour, frame_copy, originX, originY):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame_copy, (originX+x-5, originY+y-5),
                    (originX+x+w+5, originY+y+h+5), (0, 255, 0), 2)
    M = cv2.moments(contour)
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])
    cv2.circle(frame_copy, (originX+center_x, originY+center_y),
                1, (255, 0, 255), 1)  # 圆心
    cv2.putText(frame_copy, "[" + str(originX+center_x) + "," + str(originY+center_y) + "]", (originX+center_x, originY+center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)  # 坐标
    return frame_copy

def filter_contours_by_centroid(contours, min_dist=20):
    contour_centers = []
    filtered_contours = []

    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if all(np.hypot(cx - x, cy - y) > min_dist for x, y in contour_centers):
                contour_centers.append((cx, cy))
                filtered_contours.append(cnt)  # 仅保留间距足够的轮廓

    return filtered_contours

# 红圈检测
def red_circle_detect(frame, frame_copy):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=3, minDist=500,
                               param1=60, param2=135, minRadius=50, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame_copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.putText(frame_copy, f"red_circle", (i[0] - 40, i[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.putText(frame_copy, "[" + str((i[0] + i[1]) // 2) + "," + str((i[0] + i[1]) // 2) + "]",
                        (i[0], i[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)  # 坐标
    return frame_copy

# 绿圆检测
def green_circle_detect(frame, frame_copy):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 70, 50])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=3, minDist=500,
                               param1=70, param2=155, minRadius=50, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame_copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.putText(frame_copy, f"green_circle", (i[0] - 40, i[1] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.putText(frame_copy, "[" + str((i[0] + i[1])//2) + "," + str((i[0] + i[1])//2) + "]",
                        (i[0], i[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)  # 坐标
    return frame_copy

# 黄方检测
def yellow_square_detect(frame, frame_copy):
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 70, 70])  # 降低S和V的下界，提高对暗黄色的识别
    upper_yellow = np.array([50, 255, 255])  # 增大H范围，以适应不同黄色
    mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    result = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    possible_contours = filter_contours_by_centroid(valid_contours, min_dist=20)
    for contour in possible_contours:
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            M = cv2.moments(contour)
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
            cv2.putText(frame_copy, f"yellow_square", (center_x-40, center_y-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            frame_copy = mark(contour, frame_copy, 0, 0)
    return frame_copy

# 红三角检测
def red_triangle(frame, frame_copy):
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    result = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
    possible_contours = filter_contours_by_centroid(valid_contours, min_dist=20)
    for contour in possible_contours:
        approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            M = cv2.moments(contour)
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
            cv2.putText(frame_copy, f"red_triangle", (center_x-40, center_y-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            frame_copy = mark(contour, frame_copy, 0, 0)
    return frame_copy

# 蓝矩形检测
def blue_square_detect(frame, frame_copy):
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 70, 70])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
    possible_contours = filter_contours_by_centroid(valid_contours, min_dist=20)
    for contour in possible_contours:
        approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            M = cv2.moments(contour)
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
            cv2.putText(frame_copy, f"blue_square", (center_x-40, center_y-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            frame_copy = mark(contour, frame_copy, 0, 0)
    return frame_copy

capture = cv2.VideoCapture(1)
# 设置摄像头分辨率为720p/1080p
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# 检查是否正确打开
if capture.isOpened():
    open, frame = capture.read()
else:
    open = False

while open:
    ret, frame = capture.read()
    frame_copy=frame.copy()
    frame_copy=red_circle_detect(frame, frame_copy)
    frame_copy=green_circle_detect(frame, frame_copy)
    frame_copy=yellow_square_detect(frame, frame_copy)
    frame_copy=red_triangle(frame, frame_copy)
    frame_copy=blue_square_detect(frame, frame_copy)
    cv2.imshow('frame',frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        capture.release()
        cv2.destroyAllWindows()
        break