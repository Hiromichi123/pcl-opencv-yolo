import cv2
import numpy as np

#霍夫检测直线
def line_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,50,50,5)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 绘制直线
            cv2.line(img , (x1, y1), (x2, y2), (0, 0, 255), 2)
            #计算直线方程
            #
            #
            #

    return img


#主程序
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if capture.isOpened():
    open, frame = capture.read()
else:
    open = False

while open:
    #逆光补偿
    ret, frame = capture.read()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 进行霍夫直线检测
    frame2=line_detect(frame2)

    cv2.imshow('1',frame2)

    #获取相机帧率
    fps = capture.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, "FPS:" + str(fps), (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        capture.release()
        cv2.destroyAllWindows()
        break