import cv2
import numpy as np

#---------------------------------------------
#--------------------基本类--------------------
#---------------------------------------------

# 获取视频信息
def get_video_info(capture):
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) # 创建VideoWriter对象

    return width, height, fps, out


# 展示&保存视频
def imshow_and_save(frame, out):
    windows_name: str = 'Camera 720p'
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL) #窗口大小任意调节
    cv2.imshow(windows_name, frame)
    out.write(frame) # 写入视频


# 标记坐标
def mark(contour, frame):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2) # 画外接矩形
    # 找到图形轮廓中心坐标
    M = cv2.moments(contour)
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])
    cv2.circle(frame, (center_x, center_y), 1, (255, 0, 255), 1)
    cv2.putText(frame, "[" + str(center_x) + "," + str(center_y) + "]", (center_x, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    

# 逆光补偿
def backlight_compensation(frame):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


#-------------------------------------------------
#--------------------任务驱动类--------------------
#-------------------------------------------------

# 过滤轮廓，并执行检测
def filter_contours(contours, frame, frame_copy):
    if contours is not None:
        for contour in contours:
            area = cv2.contourArea(contour) # 轮廓面积
            x, y, w, h = cv2.boundingRect(contour) # 外接矩形
            if area > 10000 and area < 30000 and w / h > 0.8 and w / h < 1.25:
                frame_ROI = frame[y-5:y + h+5,x-5:x + w+5]
                if frame_ROI is not None and frame_ROI.shape[0] > 0 and frame_ROI.shape[1] > 0:
                    frame_copy = hsv_detect(frame_copy, frame_ROI, contour, area) # 霍夫图形检测，注意这里操作的是frame_copy
    return frame_copy


# hsv空间的霍夫检测
def hsv_detect(frame, ROI_img, ROI_contour, ROI_contour_area):
    x, y, w, h = cv2.boundingRect(ROI_contour)

    hsv_colors = {
        "blue": ([90, 50, 50], [130, 255, 255]),
        "green": ([40, 50, 50], [80, 255, 255]),
        "red": ([120, 50, 50], [180, 255, 250]),
        "yellow": ([20, 100, 100], [40, 255, 255])
    }

    hsv_img = cv2.cvtColor(ROI_img, cv2.COLOR_BGR2HSV)
    for color_name, (lower, upper) in hsv_colors.items():
        hsv_mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
        hsv_result = cv2.bitwise_and(ROI_img, ROI_img, hsv_mask)
        hsv_open = cv2.morphologyEx(hsv_result, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        hsv_gray = cv2.cvtColor(hsv_open, cv2.COLOR_BGR2GRAY)
        hsv_edges = cv2.Canny(hsv_gray, 100, 200)
        _, hsv_thresh = cv2.threshold(hsv_edges, 150, 255, cv2.THRESH_TRUNC)
        hsv_contours, _ = cv2.findContours(hsv_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        shift_x = x-5
        shift_y = y-5
        for contour in hsv_contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < ROI_contour_area + 1000 and contour_area > ROI_contour_area - 1000:
                # param1用于边缘Canny算子的高阈值。大值检测更少的边缘，减少圆数量。
                # param2用于圆心的累加器阈值。小值更多的累加器投票，检测到更多的假阳性圆。
                circles = cv2.HoughCircles(hsv_edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
                                        param1=10, param2=40, minRadius=0, maxRadius=0)

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        center = (shift_x+i[0], shift_y+i[1])
                        radius = i[2]
                        cv2.circle(frame, center, radius, (0, 0, 255), 2)
                        cv2.putText(frame, f"{color_name}_circle", (shift_x+i[0] - 40, shift_y+i[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 0), 1)
                    #mark(ROI_contour, frame)

                # 多边形
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True) # 逼近精度4%轮廓周长
                if approx is not None:
                    M = cv2.moments(contour)
                    center_x = int(M['m10'] / M['m00'])
                    center_y = int(M['m01'] / M['m00']) # 当前轮廓中心

                    if len(approx) == 3:  # 三边
                        cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)
                        cv2.putText(frame, f"{color_name}_triangle", (center_x - 40, center_y - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 0), 1)
                        #mark(ROI_contour, frame)
                    elif len(approx) == 4:  # 四边
                        cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)
                        cv2.putText(frame, f"{color_name}_square", (center_x - 40, center_y - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 0), 1)
                        #mark(ROI_contour, frame)
        return frame
 
# 霍夫直线
def line_detect(img):
    img = cv2.medianBlur(img, 5) #  椒盐滤波
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 50, 50, 5)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img , (x1, y1), (x2, y2), (0, 0, 255), 2) # 绘制直线
    return img

