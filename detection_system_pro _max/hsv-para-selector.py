from typing import Optional
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from numpy import ndarray as mat


class HSVSelectorTool:
    def __init__(self, windows_width: int = 720, windows_height: int = 640, windows_name: str = "HSV Selector Tool"):
        self.__windows_width: int = windows_width
        self.__windows_height: int = windows_height
        self.__windows_name: str = windows_name
        self.__img_hsv: Optional[mat] = None

    def run_bgr(self, img_bgr):
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        self.run_hsv(img_hsv)

    def run_hsv(self, img_hsv):
        self.__img_hsv = img_hsv
        cv2.namedWindow(self.__windows_name, cv2.WINDOW_NORMAL)
        debug_resize_times = 0.6
        cv2.resizeWindow(self.__windows_name, np.int_(self.__windows_width * debug_resize_times),
                         np.int_(self.__windows_height * debug_resize_times * 1.5))
        cv2.createTrackbar('H_Min', self.__windows_name, 0, 180, self.__cv_select_callback)
        cv2.createTrackbar('S_Min', self.__windows_name, 0, 255, self.__cv_select_callback)
        cv2.createTrackbar('V_Min', self.__windows_name, 0, 255, self.__cv_select_callback)
        cv2.createTrackbar('H_Max', self.__windows_name, 0, 180, self.__cv_select_callback)
        cv2.createTrackbar('S_Max', self.__windows_name, 0, 255, self.__cv_select_callback)
        cv2.createTrackbar('V_Max', self.__windows_name, 0, 255, self.__cv_select_callback)

        min_h, min_s, min_v, max_h, max_s, max_v = 0, 0, 0, 0, 0, 0

        while True:
            min_h = cv2.getTrackbarPos('H_Min', self.__windows_name)
            min_s = cv2.getTrackbarPos('S_Min', self.__windows_name)
            min_v = cv2.getTrackbarPos('V_Min', self.__windows_name)
            max_h = cv2.getTrackbarPos('H_Max', self.__windows_name)
            max_s = cv2.getTrackbarPos('S_Max', self.__windows_name)
            max_v = cv2.getTrackbarPos('V_Max', self.__windows_name)

            range_min = np.array([min_h, min_s, min_v])
            range_max = np.array([max_h, max_s, max_v])
            print(f"Range [H, S, V]: MIN - [{min_h}, {min_s}, {min_v}], MAX - [{max_h}, {max_s}, {max_v}]")

            img_hsv_debug = cv2.inRange(self.__img_hsv, range_min, range_max)
            cv2.imshow(self.__windows_name, img_hsv_debug)

            key = cv2.waitKey(10)
            if key & 0xFF == ord('q'):
                cv2.destroyWindow(self.__windows_name)
                break

        return min_h, min_s, min_v, max_h, max_s, max_v

    @staticmethod
    def __cv_select_callback(value: int):
        pass


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    print("Selected file:", file_path)
    tool: HSVSelectorTool = HSVSelectorTool()
    tool.run_bgr(cv2.imread(file_path))
