import cv2

# 逆光补偿
def backlight_compensation(frame):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


# 标记坐标
def mark(contour):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame_copy, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2) # 画外接矩形
    # 找到图形轮廓中心坐标
    M = cv2.moments(contour)
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])
    cv2.circle(frame_copy, (center_x, center_y), 1, (255, 0, 255), 1)
    cv2.putText(frame_copy, "[" + str(center_x) + "," + str(center_y) + "]", (center_x, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    

# 获取视频信息
def get_video_info(capture):
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    return width, height, fps

