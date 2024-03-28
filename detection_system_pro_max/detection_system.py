import cv2
import numpy as np

#cv识别程序主体
capture = cv2.VideoCapture(0)
#检查是否正确打开
if capture.isOpened():
    open,frame=capture.read()
else:
    open=False

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(cv2.CAP_PROP_FPS))

# 创建VideoWriter对象
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


#当正确打开时
while open:
    ret, frame = capture.read()
    
    # 对每一帧进行处理，例如进行目标检测等
    # 逆光补偿
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 白色边界填充
    #top_size, bottom_size, left_size, right_size = (10, 10, 10, 10)  # 边界宽度
    #frame2 = cv2.copyMakeBorder(frame, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT,
                               #value=(255, 255, 255))
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # 如果缺少会导致CV_8UC1报错

    # 二值化处理
    ret, thresh = cv2.threshold(frame2, 175, 175, cv2.THRESH_TRUNC)
    ret, thresh2 = cv2.threshold(frame2, 150, 255, cv2.THRESH_BINARY)
    #提取轮廓
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        # 面积
        area = cv2.contourArea(contour)
        # 边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        if area > 3000 and area < 50000 and w / h > 0.8 and w / h < 1.25:
            # 画外接矩形
            cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
            # 画轮廓
            result = cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)
            # 找到图形轮廓中心坐标
            M = cv2.moments(contour)
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
            #print("图形的中心坐标为：({}, {})".format(center_x, center_y))
            #画圆心
            cv2.circle(frame, (center_x,center_y), 1, (255, 0, 255), 1)
            #put中心坐标
            cv2.putText(frame, "["+str(center_x)+","+str(center_y)+"]", (center_x , center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # 获取摄像头的帧率
    fps = capture.get(cv2.CAP_PROP_FPS)
    #print("摄像头帧率:", fps)
    cv2.putText(frame, "FPS:" + str(fps), (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    #显示完整图
    #cv2.imshow('Video1', thresh2)
    cv2.imshow('Video2', frame)

    #写入视频帧
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        out.release()
        capture.release()
        cv2.destroyAllWindows()
        break