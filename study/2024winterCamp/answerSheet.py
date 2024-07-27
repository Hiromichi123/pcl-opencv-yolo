import cv2
import numpy as np
A,B,C,D,E=1,2,3,4,5


#设置答案
answer=[A,B,C,D,E,A,B]
# 总得分以及得分矩阵
total_point=0
matrix=[[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]]
row=column=0 #行列

#获取角坐标点（透视变换需要）
def order_points(pts):
    rect=np.zeros((4,2),dtype="float32")

    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]

    diff=np.diff(pts,axis=1)
    rect[1]=pts[np.argmin(diff)]
    rect[3]=pts[np.argmax(diff)]

    return rect

# 透视变换
def four_point_transform(image,pts):
    rect=order_points(pts)
    (tl,tr,br,bl)=rect

    widthA=np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
    widthB=np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
    maxWidth=max(int(widthA),int(widthB))

    heightA=np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
    heightB=np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
    maxHeight=max(int(heightA),int(heightB))

    # 变换后对应坐标位置
    dst=np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth - 1, maxHeight-1],
        [0,maxHeight-1]],dtype="float32")

    #计算变换矩阵
    M=cv2.getPerspectiveTransform(rect,dst)
    wraped=cv2.warpPerspective(image,M,(maxWidth,maxHeight))

    return wraped

# 等比缩放（透视变换需要）
def resize(image,width=None,height=None,inter=cv2.INTER_AREA):
    dim=None
    (h,w)=image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r=height/float(h)
        dim=(int(w*r),height)
    else:
        r=width/float(w)
        dim=(width,int(h*r))
    resized=cv2.resize(image,dim,interpolation=inter)

    return resized

#移动函数
def move():
    global row,column
    if column==4:
        column=0
        row+=1
    else:
        column+=1

# 霍夫检测圆形区域
def detect_color_circle(img, frame,hsv_img, lower, upper, color_name, import_contour, area):
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
        if contour_area < area + 500 and contour_area > area - 500:
            # param1：这是用于边缘检测的Canny算子的高阈值参数。较高的param1值会导致更少的边缘被检测到，会减少检测到的圆的数量。
            # param2：这是用于确定圆心的累加器阈值参数。较小的param2值会导致更多的累加器投票，因此可能会导致检测到更多的假阳性圆。较大的param2值会导致更少的累加器投票，因此可能会减少检测到的圆的数量。
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=10, param2=20, minRadius=0,
                                       maxRadius=0)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    move()
                    if color_name == "dark":
                        center = (shift_x+i[0], shift_y+i[1])
                        radius = i[2]
                        if column == list[row]:
                            cv2.circle(frame, center, radius, (0, 255, 0), 5)
                            matrix[row][column]=True
                        else:
                            cv2.circle(frame, center, radius, (0, 0, 255), 5)
                            matrix[row][column]=False

    return frame

# hsv空间的霍夫检测
def hsv_detect(img, frame, import_contour, area):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #为hsv颜色掩膜设置字典
    colors = {
        "dark": ([0, 0, 30], [179, 30, 100]),
        "white": ([0, 0, 180], [179, 20, 255]),
    }

    for color_name, (lower, upper) in colors.items():
        frame=detect_color_circle(img, frame,hsv_img, np.array(lower), np.array(upper), color_name, import_contour, area)

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

    # 逆光补偿
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    ratio = frame.shape[0] / 500.0  # 计算变换比例
    orig = frame.copy()
    image = resize(orig, height=500)  # 等比缩放

    # 边缘检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 200)
    edges = cv2.Canny(gray, 75, 200)

    # 获取轮廓
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)  # 膨胀连接处理
    cnts, _ = cv2.findContours(dilation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5] #面积排序

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is not None:
        # 透视变换
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        # 二值化
        warped2 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        ref = cv2.threshold(warped2, 100, 255, cv2.THRESH_BINARY)[1]

        # 提取轮廓
        contours, _ = cv2.findContours(ref, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        warped_copy=warped.copy() #画图展示拷贝副本

        if contours is not None:
            for contour in contours:
                # 面积
                area = cv2.contourArea(contour)
                # 边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                if area > 2000 and area < 5000 and w / h > 0.8 and w / h < 1.25:
                    # 剪裁可能存在图形的ROI区域
                    warped_ROI = warped[y-5:y + h+5,x-5:x + w+5]
                    # 添加边界检查和非空检查
                    if warped_ROI is not None and warped_ROI.shape[0] > 0 and warped_ROI.shape[1] > 0:
                        warped_copy=hsv_detect(warped_ROI,warped_copy, contour, area) #进行hsv霍夫检测


        # 获取摄像头的帧率
        cv2.putText(warped_copy, "total_point:" + str(total_point), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # 显示完整图
        cv2.imshow("original", resize(orig, height=500))
        cv2.imshow("scan", resize(ref, height=500))

cv2.waitKey(0)
cv2.destroyAllWindows()
