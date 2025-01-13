import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv_tools

class vision_pub(Node):
    def __init__(self):
        super().__init__('vision_pub')
        self.pub = self.create_publisher(String, 'vision', 5)
        self.timer = self.create_timer(0.05, self.timer_callback) # 20hz

    def timer_callback(self):
        msg = String()
        msg.data = 'vision_msg'
        self.pub.publish(msg)

# hsv空间的霍夫检测
def hsv_detect(img, frame, import_contour, area):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #hsv字典
    colors = {
        "blue": ([90, 50, 50], [130, 255, 255]),
        "green": ([40, 50, 50], [80, 255, 255]),
        "red": ([120, 50, 50], [180, 255, 250]),
        "yellow": ([20, 100, 100], [40, 255, 255])
    }

    for color_name, (lower, upper) in colors.items():
        frame = detect_color_shapes(img, frame, hsv_img, np.array(lower), np.array(upper), color_name, import_contour, area)

    return frame

# 霍夫图形分类
def detect_color_shapes(img, frame, hsv_img, lower, upper, color_name, import_contour, area):
    mask = cv2.inRange(hsv_img, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    _, thresh = cv2.threshold(edges, 150, 255, cv2.THRESH_TRUNC)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
        if contour_area < area + 1000 and contour_area > area - 1000:
            # param1用于边缘Canny算子的高阈值。大值检测更少的边缘，减少圆数量。
            # param2用于圆心的累加器阈值。小值更多的累加器投票，检测到更多的假阳性圆。
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
                                       param1=10, param2=40, minRadius=0, maxRadius=0)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (shift_x+i[0], shift_y+i[1])
                    radius = i[2]
                    cv2.circle(frame, center, radius, (0, 0, 255), 2)
                    cv2.putText(frame, f"{color_name}_circle", (shift_x+i[0] - 40, shift_y+i[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)

                #mark(import_contour)

            # 逼近精度为4%闭合轮廓周长
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            if approx is not None:
                # 找到当前轮廓中心
                M = cv2.moments(contour)
                center_x = int(M['m10'] / M['m00'])
                center_y = int(M['m01'] / M['m00'])
                if len(approx) == 3:  # 三边
                    cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)
                    cv2.putText(frame, f"{color_name}_triangle", (center_x - 40, center_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)
                    #mark(import_contour)
                elif len(approx) == 4:  # 四边
                    cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)
                    cv2.putText(frame, f"{color_name}_square", (center_x - 40, center_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)
                    #mark(import_contour)
    return frame

#霍夫直线
def line_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,50,50,5)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            #cv2.line(img , (x1, y1), (x2, y2), (0, 0, 255), 2) # 绘制直线
            #计算直线方程
            #
            #
            #
    return img





if __name__ == '__main__':
    rclpy.init(args=None)
    vision_pub = vision_pub()
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if capture.isOpened():
        open, frame = capture.read()
        if open:
            width, height, fps = get_video_info(capture)
            out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) # 创建VideoWriter对象

            while open:
                bl_frame = backlight_compensation(frame) # 逆光补偿
                # cv2.imshow('逆光补偿效果', bl_frame)

                hough_line_frame = line_detect(result) # 霍夫直线检测
                # cv2.imshow('霍夫直线效果', hough_line_frame)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 灰度图处理
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY) # 二值化处理
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # 提取轮廓

                frame_copy=frame.copy() #画图展示拷贝副本

                if contours is not None:
                    for contour in contours:
                        area = cv2.contourArea(contour) # 轮廓面积
                        x, y, w, h = cv2.boundingRect(contour) # 外接矩形
                        if area > 10000 and area < 30000 and w / h > 0.8 and w / h < 1.25:
                            frame_ROI = frame[y-5:y + h+5,x-5:x + w+5]
                            if frame_ROI is not None and frame_ROI.shape[0] > 0 and frame_ROI.shape[1] > 0:
                                frame_copy=hsv_detect(frame_ROI,frame_copy, contour, area) #进行hsv霍夫检测

                cv2.putText(frame_copy, "FPS:" + str(fps), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                windows_name: str = 'Camera 7200p'
                # cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL) #窗口大小任意调节
                cv2.imshow('Video1', thresh)
                cv2.imshow(windows_name, frame_copy)
                out.write(frame_copy) # 写入视频

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    out.release()
                    capture.release()
                    cv2.destroyAllWindows()
                    break
    else:
        open = False
    
    # 打开失败
    rclpy.spin(vision_pub)
    vision_pub.destroy_node()
    rclpy.shutdown()
