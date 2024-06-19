#!/usr/bin/env python
#-*-coding:utf-8-*-
import cv2
import numpy as np

def detect_box(image, width, height):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_brown = np.array([10, 100, 20])  # 调整为适当的深棕色范围
    upper_brown = np.array([20, 255, 200])  # 调整为适当的深棕色范围

    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)

        if (len(approx) == 4 or len(approx) == 5) and area > 10000:
            M = cv2.moments(contour)
            delta_x = int(M['m10'] / M['m00']-width/2)
            delta_y = -int(M['m01'] / M['m00']-height/2)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)

            return (delta_x, delta_y), image_copy

    return None, None

def main():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        if not capture.isOpened():
            print("无法打开相机")
        else:
            while True:
                ret, frame = capture.read()
                if not ret:
                    print("无法读取帧")
                    break

                x1,x2=160,320
                y1,y2=120,360
                frame=frame[y1:y2,x1:x2]
                cv2.imshow('Frame', frame) # 展示原图

                frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                height, width = frame.shape[:2]
                delta, image = detect_box(frame, width, height)

                if image is not None:
                    cv2.imshow("detect", image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    capture.release()
                    cv2.destroyAllWindows()
                    break



if __name__ == '__main__':
    main()