import cv2
import numpy as np

#霍夫检测四边形
def detect_color_square_triangle(img, hsv_img, lower, upper, color_name, area):
    ret=0
    mask = cv2.inRange(hsv_img, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    #result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    ret, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_TRUNC)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < area + 1000 and contour_area > area - 1000:
            #逼近精度为10%闭合轮廓周长
            approx = cv2.approxPolyDP(contour, 0.10 * cv2.arcLength(contour, True), True)
            if approx is not None:
                cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
                #找到当前轮廓中心
                M = cv2.moments(contour)
                center_x = int(M['m10'] / M['m00'])
                center_y = int(M['m01'] / M['m00'])
                if len(approx) == 3: #当逼近函数返回为4时确认为四边形
                    cv2.putText(img, f"{color_name}_triangle", (center_x - 60, center_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)
                    ret=1
                elif len(approx) == 4: #当逼近函数返回为4时确认为四边形
                    cv2.putText(img, f"{color_name}_square", (center_x - 60, center_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)
                    ret=1

    return ret, img

# 霍夫检测不同颜色圆
def detect_color_circles(img, hsv_img, lower, upper, color_name, area):
    ret=0
    mask = cv2.inRange(hsv_img, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    #result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    _, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_TRUNC)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < area + 1000 and contour_area > area - 1000:
            # param1：这是用于边缘检测的Canny算子的高阈值参数。较高的param1值会导致更少的边缘被检测到，会减少检测到的圆的数量。
            # param2：这是用于确定圆心的累加器阈值参数。较小的param2值会导致更多的累加器投票，因此可能会导致检测到更多的假阳性圆。较大的param2值会导致更少的累加器投票，因此可能会减少检测到的圆的数量。
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=10, param2=15, minRadius=0,
                                       maxRadius=0)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    radius = i[2]
                    cv2.circle(img, center, radius, (0, 255, 0), 2)
                    cv2.putText(img, f"{color_name}_circle", (i[0] - 60, i[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)
                    ret=1

    return ret, img

# hsv空间的霍夫圆检测，外函数
def hsv_detect(img, area):
    ret=0
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #为hsv颜色掩膜设置字典
    colors = {
        "blue": ([90, 50, 50], [130, 255, 255]),
        "green": ([40, 50, 50], [80, 255, 255]),
        "red": ([120, 50, 50], [180, 255, 250]),
        "yellow": ([20, 100, 100], [40, 255, 255])
    }

    for color_name, (lower, upper) in colors.items():
        # 检测圆
        ret1,img=detect_color_circles(img, hsv_img, np.array(lower), np.array(upper), color_name, area)
        # 检测三角形和矩形
        ret2,img=detect_color_square_triangle(img, hsv_img, np.array(lower), np.array(upper), color_name, area)

    if ret2 == 1 or ret1 == 1:
        ret=1

    return ret, img


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

# 创建VideoWriter对象
# out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 当正确打开时
while open:
    ret, frame = capture.read()
    # 对每一帧进行处理，例如进行目标检测等

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

    #灰度图处理
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 如果缺少会导致CV_8UC1报错

    # 二值化处理
    _, thresh = cv2.threshold(frame2, 175, 175, cv2.THRESH_TRUNC)
    _, thresh2 = cv2.threshold(frame2, 150, 255, cv2.THRESH_BINARY)
    # 提取轮廓
    contours, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if contours is not None:
        for contour in contours:
            # 面积
            area = cv2.contourArea(contour)
            # 边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            if area > 3000 and area < 50000 and w / h > 0.8 and w / h < 1.25:
                # 剪裁可能存在图形的ROI区域
                frame_ROI=frame[x:x+w,y:y+h]
                # 添加边界检查和非空检查
                if frame_ROI is not None and frame_ROI.shape[0]>0 and frame_ROI.shape[1]>0:
                    # cv2.imshow("1", frame_ROI)
                    ret,frame[x:x+w,y:y+h]=hsv_detect(frame_ROI, area) #进行hsv霍夫检测，处理后缝合
                    if ret == 1:
                        # 画外接矩形
                        cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
                        # 画轮廓
                        result = cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)
                        # 找到图形轮廓中心坐标
                        M = cv2.moments(contour)
                        center_x = int(M['m10'] / M['m00'])
                        center_y = int(M['m01'] / M['m00'])
                        # print("图形的中心坐标为：({}, {})".format(center_x, center_y))
                        # 画圆心
                        cv2.circle(frame, (center_x, center_y), 1, (255, 0, 255), 1)
                        # put中心坐标
                        cv2.putText(frame, "[" + str(center_x) + "," + str(center_y) + "]", (center_x, center_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # 获取摄像头的帧率
    fps = capture.get(cv2.CAP_PROP_FPS)
    # print("摄像头帧率:", fps)
    cv2.putText(frame, "FPS:" + str(fps), (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # 显示完整图
    windows_name: str = 'Camera 1080p'
    # cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL) #窗口大小任意调节
    # cv2.imshow('Video1', thresh2)
    cv2.imshow(windows_name, frame)

    # 写入视频帧
    # out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # out.release()
        capture.release()
        cv2.destroyAllWindows()
        break