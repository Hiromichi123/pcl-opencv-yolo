import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import cv_tools

class vision_pub_node(Node):
    def __init__(self):
        super().__init__('vision_pub')
        self.pub = self.create_publisher(String, 'vision', 10)
        self.rate = rclpy.rate.Rate(20) # 20hz

    def main_loop(self):
        try:
            while rclpy.ok():
                msg = String()
                msg.data = "Hello!"
                self.publisher_.publish(msg)
                

                capture = cv2.VideoCapture(0)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                if capture.isOpened():
                    open, frame = capture.read()
                    if open:
                        width, height, fps, out = get_video_info(capture)

                        while open:
                            bl_frame = backlight_compensation(frame) # 逆光补偿
                            # cv2.imshow('逆光补偿效果', bl_frame)

                            hough_line_frame = line_detect(bl_frame) # 霍夫直线
                            # cv2.imshow('霍夫直线效果', hough_line_frame)

                            gray_frame = cv2.cvtColor(bl_frame, cv2.COLOR_BGR2GRAY) # 灰度
                            _, thresh_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY) # 二值化处理
                            # cv2.imshow('预处理最终效果', thresh_frame)

                            contours, _ = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # 提取轮廓
                            
                            frame_copy = frame.copy() # 展示拷贝
                            frame_copy = filter_contours(contours, frame, frame_copy) # 过滤轮廓，并检测

                            imshow_and_save(frame_copy, out) # 展示结果，写入视频
                            self.rate.sleep()

        except Exception as e:
            self.get_logger().error(f"Error occurred: {e}")
        finally:
            self.release_resources()

    def release_resources(self):
        if self.capture.isOpened():
            self.capture.release()
        self.out.release()
        self.get_logger().info("Resources released.")


if __name__ == '__main__':
    rclpy.init(args=None)
    node = vision_pub_node()
    node.main_loop()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
