import cv2
import numpy as np

# 标记坐标
def mark(contour):
    x, y, w, h = cv2.boundingRect(contour)
    # 画外接矩形
    cv2.rectangle(frame_copy, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
    # 找到图形轮廓中心坐标
    M = cv2.moments(contour)
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])
    cv2.circle(frame_copy, (center_x, center_y), 1, (255, 0, 255), 1)
    cv2.putText(frame_copy, "[" + str(center_x-680) + "," + str(center_y-360) + "]", (center_x, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

# 霍夫检测不同图形
def detect_color_shapes(img, frame,hsv_img, lower, upper, color_name, import_contour, area):
    mask = cv2.inRange(hsv_img, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    _, thresh = cv2.threshold(edges, 150, 255, cv2.THRESH_TRUNC)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    x, y, w, h = cv2.boundingRect(import_contour)
    shift_x=x-5
    shift_y=y-5
    # 对每个轮廓进行坐标调整
    for contour in contours:
        for i in range(len(contour)):
            contour[i][0][0] += shift_x  # 对 x 坐标进行加操作
            contour[i][0][1] += shift_y  # 对 y 坐标进行加操作

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < area + 2000 and contour_area > area - 2000:
            # param1：这是用于边缘检测的Canny算子的高阈值参数。较高的param1值会导致更少的边缘被检测到，会减少检测到的圆的数量。
            # param2：这是用于确定圆心的累加器阈值参数。较小的param2值会导致更多的累加器投票，因此可能会导致检测到更多的假阳性圆。较大的param2值会导致更少的累加器投票，因此可能会减少检测到的圆的数量。
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=10, param2=20, minRadius=0,
                                       maxRadius=0)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (shift_x+i[0], shift_y+i[1])
                    radius = i[2]
                    cv2.circle(frame, center, radius, (0, 0, 255), 2)
                    cv2.putText(frame, f"{color_name}_circle", (shift_x+i[0] - 40, shift_y+i[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)

                mark(import_contour) #标记


            # 逼近精度为4%闭合轮廓周长
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            if approx is not None:
                # 找到当前轮廓中心
                M = cv2.moments(contour)
                center_x = int(M['m10'] / M['m00'])
                center_y = int(M['m01'] / M['m00'])
                if len(approx) == 3:  # 当逼近函数返回为4时确认为四边形
                    cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)
                    cv2.putText(frame, f"{color_name}_triangle", (center_x - 40, center_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)
                    mark(import_contour)  # 标记
                elif len(approx) == 4:  # 当逼近函数返回为4时确认为四边形
                    cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)
                    cv2.putText(frame, f"{color_name}_square", (center_x - 40, center_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)
                    mark(import_contour) #标记

    return frame

# hsv空间的霍夫检测
def hsv_detect(img, frame, import_contour, area):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #为hsv颜色掩膜设置字典
    colors = {
        "blue": ([90, 50, 50], [130, 255, 255]),
        "green": ([40, 50, 50], [80, 255, 255]),
        "red": ([120, 50, 50], [180, 255, 250]),
        "yellow": ([20, 100, 100], [40, 255, 255])
    }

    for color_name, (lower, upper) in colors.items():
        frame=detect_color_shapes(img, frame,hsv_img, np.array(lower), np.array(upper), color_name, import_contour, area)

    return frame


# cv识别程序主体
capture = cv2.VideoCapture(0)
# 设置摄像头分辨率为720p/1080p
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# 检查是否正确打开
if capture.isOpened():
    open, frame = capture.read()
else:
    open = False

# 获取视频信息
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv2.CAP_PROP_FPS)

# 创建VideoWriter对象
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 当正确打开时
while open:
    ret, frame = capture.read()
    # 对每一帧进行处理，例如进行目标检测等

    if frame is not None:
        # 逆光补偿
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 白色边界填充
        # top_size, bottom_size, left_size, right_size = (10, 10, 10, 10)  # 边界宽度
        # frame2 = cv2.copyMakeBorder(frame, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT,
        # value=(255, 255, 255))

        # 灰度图处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 二值化处理
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        # 提取轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        frame_copy=frame.copy() #画图展示拷贝副本

        if contours is not None:
            for contour in contours:
                # 面积
                area = cv2.contourArea(contour)
                # 边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                if area > 8000 and area < 30000 and w / h > 0.8 and w / h < 1.25:
                    # 剪裁可能存在图形的ROI区域
                    frame_ROI = frame[y-5:y + h+5,x-5:x + w+5]
                    # 添加边界检查和非空检查
                    if frame_ROI is not None and frame_ROI.shape[0] > 0 and frame_ROI.shape[1] > 0:
                        frame_copy=hsv_detect(frame_ROI,frame_copy, contour, area) #进行hsv霍夫检测


        # 获取摄像头的帧率
        cv2.putText(frame, "FPS:" + str(fps), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # 显示完整图
        windows_name: str = 'Camera 1080p'
        # cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL) #窗口大小任意调节
        # cv2.imshow('Video1', thresh)
        cv2.imshow(windows_name, frame_copy)

        # 写入视频帧
        out.write(frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            out.release()
            capture.release()
            cv2.destroyAllWindows()
            break