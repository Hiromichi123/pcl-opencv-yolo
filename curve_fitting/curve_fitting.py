import cv2
import numpy as np
import time

# 定义卷积核，用于图像处理
kernel = np.ones((5, 5), np.uint8)
# 定义黑色的颜色范围
lower_black = np.array([0, 0, 0])
upper_black = np.array([255, 50, 50])

# 闭运算函数
def close(n):
    # 先膨胀再腐蚀
    n = cv2.dilate(n, kernel, iterations=2)
    n = cv2.erode(n, kernel, iterations=2)
    return n

# 开运算函数
def open(n):
    # 先腐蚀再膨胀
    n = cv2.erode(n, kernel, iterations=2)
    n = cv2.dilate(n, kernel, iterations=2)
    return n

# 线性回归函数
def linear_regression(img, img1):
    height = img.shape[0]
    width = img.shape[1]

    # 创建一个和原图像一样大小的黑色图像
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 二值化图像
    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 边缘处理
    cannny_image = cv2.Canny(binary_image, 50, 150)

    # 初始化点集列表
    points = []

    # 遍历图像的每一行
    for row in range(img.shape[0]):
        white_pixels = np.where(cannny_image[row] == 255)[0]
        if white_pixels.size:
            A_x = round(np.mean(white_pixels)) - img.shape[1] / 2
            A_y = img.shape[1] / 2 - row
            black_image[row, round(np.mean(white_pixels))] = (255, 255, 255)
            points.append((A_x, A_y))
    if len(points) > 4:
        points_array = np.array(points)

        x = points_array[:, 0]
        y = points_array[:, 1]

        p = np.polyfit(x, y, 1)

        slope = p[0]
        fslope = 1 / slope
        intercept = p[1]

        print("拟合直线方程为：x = {:.2f}y - {:.2f}".format(fslope, intercept * fslope))

        x1 = -400
        y1 = int(slope * x1 + intercept)

        x2 = 400
        y2 = int(slope * x2 + intercept)

        cv2.line(img1, (center_x + x1, center_y - y1), (center_x + x2, center_y - y2), (0, 165, 255), 2)

# 计算黑色区域离中心点的距离
def nearby_distance(img, x1, y1):
    mask_black = cv2.inRange(img, lower_black, upper_black)
    image = close(mask_black)

    cropped_1 = image[center_y - 10:center_y + 10, 0:image.shape[1]]
    cropped_2 = image[center_y - y1:center_y + y1, center_x - x1:center_x + x1]

    cropped_1 = cv2.dilate(cropped_1, kernel, iterations=2)

    cropped_1[0, :] = 0
    cropped_1[cropped_1.shape[0] - 1, :] = 0

    edges_1 = cv2.Canny(cropped_1, 50, 150)

    contours_1, _ = cv2.findContours(edges_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_1:
        M = cv2.moments(contour)
        if M["m00"]:
            cX = int(M["m10"] / M["m00"])
            print(center_x - cX)
        else:
            continue
    cv2.imshow('cropped_2', cropped_2)
    cv2.imshow('image', image)
    return cropped_2


# 主函数
"""
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    image0 = frame.copy()

    center_y = image0.shape[0] // 2
    center_x = image0.shape[1] // 2
    linear_regression(nearby_distance(image0, 40, 40), image0)

    cv2.imshow('image0', image0)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
"""

frame=cv2.imread('curve.png')
image0 = frame.copy()

center_y = image0.shape[0] // 2
center_x = image0.shape[1] // 2
linear_regression(nearby_distance(image0, 40, 40), image0)

cv2.imshow('image0', image0)
key = cv2.waitKey(0)
cv2.destroyAllWindows()