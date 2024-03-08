import cv2
import numpy as np
import imutils
from imutils import contours
import myutils

#绘图展示函数
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img1=cv2.imread("su.png")
img2=cv2.imread("hui.png")
#灰度图
ref1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
ref2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#二值图
ref1=cv2.threshold(ref1,220,255,cv2.THRESH_BINARY_INV)[1]
ref2=cv2.threshold(ref2,220,255,cv2.THRESH_BINARY_INV)[1]
#提取轮廓
refCnts1,hierarchy1=cv2.findContours(ref1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img1,refCnts1,-1,(0,0,255),1)
refCnts2,hierarchy1=cv2.findContours(ref2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img2,refCnts2,-1,(0,0,255),1)

max=cv2.contourArea(refCnts1[0])
j=0
for i in refCnts1:
    if cv2.contourArea(i)>max:
        max=cv2.contourArea(i)
        j+=1

#计算外接矩形并且resize成合适的大小
(x1,y1,w1,h1)=cv2.boundingRect(refCnts1[j])
#print((x1,y1,w1,h1))
roi1=ref1[y1:y1+h1,x1:x1+w1]
roi1=cv2.resize(roi1,(40,40))

max=cv2.contourArea(refCnts2[0])
k=0
for i in refCnts2:
    if cv2.contourArea(i)>max:
        max=cv2.contourArea(i)
        k+=1

(x2,y2,w2,h2)=cv2.boundingRect(refCnts2[k])
#print((x2,y2,w2,h2))
roi2=ref2[y2:y2+h2,x2:x2+w2]
roi2=cv2.resize(roi2,(40,40))

# 每字符对应每一个模板
chars={}
chars[1] = roi1
chars[2] = roi2

# 读取图像
image = cv2.imread('img.png')

# 将图像转换为HSV颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义黄色的HSV颜色范围
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# 定义紫色的HSV颜色范围
lower_purple = np.array([130, 100, 100])
upper_purple = np.array([160, 255, 255])

# 创建黄色和紫色的掩膜
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

# 对掩膜进行形态学操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
#yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)

#合并掩膜
mask=yellow_mask+purple_mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#计算轮廓
img_copy=image.copy()
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_copy,cnts,-1,(0,0,255),2)
#cv_show('img',img_copy)

#筛选出实际有价值的区域
locs = [] #定义一个list
#遍历所有轮廓
for (i,c) in enumerate(cnts):
    (x,y,w,h)=cv2.boundingRect(c) #计算外接矩形
    ar=w/float(h) #计算长宽比


    #选择合适的区域，根据实际任务算出筛选条件范围(调参工程师)
    if ar>0.8 and ar<1.5:
        if(w>30 and w<100)and(h>30 and h<100):
            #符合条件的添加到list中
            locs.append((x,y,w,h))
            print((x,y,w,h))

#将符合条件的轮廓从左到右排序
locs=sorted(locs,key=lambda x:x[0])
output=[]

#遍历每个字
for(i,(gX,gY,gW,gH))in enumerate(locs):
    groupOutput=[]
    #根据坐标提取每一个组
    group=image[gY-5:gY+gH+5,gX-5:gX+gW+5]
    group=cv2.cvtColor(group,cv2.COLOR_BGR2GRAY)
    #cv_show('group',group)

    #计算每一组的轮廓(记得copy)
    cnts,hierarchy=cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=contours.sort_contours(cnts,method='left-to-right')[0] #从左到右排序

    for c in cnts:
        #找到当前数值的轮廓，resize成合适的大小
        (x,y,w,h)=cv2.boundingRect(c)
        roi=group[y:y+h,x:x+w]
        roi=cv2.resize(roi,(40,40))
        #cv_show('roi',roi)

        #模板匹配分别的得分
        scores=[]
        for(char,charROI) in chars.items():
            #模板匹配
            result=cv2.matchTemplate(roi,charROI,cv2.TM_CCOEFF)
            (_,score,_,_)=cv2.minMaxLoc(result)
            print(str(char)+" "+str(score))
            scores.append(score)

        #找到匹配度最高的数字
        groupOutput.append(str(np.argmax(scores)+1))

    #画出对应数字
    cv2.rectangle(image,(gX-5,gY-5),
                  (gX+gW+5,gY+gH+5),(0,0,255),1)
    cv2.putText(image,"".join(groupOutput),(gX+20,gY+35),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)


cv2.imshow('image',image)
cv2.waitKey(0)