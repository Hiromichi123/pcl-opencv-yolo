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

img=cv2.imread("template.png")
#cv_show("img",img)
#灰度图
ref=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv_show("ref",ref)
#二值图
ref=cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1] #threshold返回一个元组，[1]表示第一个参数
#cv_show("ref",ref)

#提取轮廓
#findContours()函数只接受二值图像,cv2.RETR_EXTERNAL只检测外轮廓,cv2.CHAIN_APPROX_SIMPLE
#返回的list中每个元素都是图像中一个轮廓
refCnts,hierarchy=cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,refCnts,-1,(0,0,255),3)
#cv_show('img',img)
refCnts=imutils.contours.sort_contours(refCnts,method="Left-to-right")[0] #排序，从左到右，从上到下
digits={}

#遍历每一个轮廓
for(i,c) in enumerate(refCnts):
    #计算外接矩形并且resize成合适的大小(合适的大小就行)
    (x,y,w,h)=cv2.boundingRect(c)
    roi=ref[y:y+h,x:x+w]
    roi=cv2.resize(roi,(57,88))

    #每一个数字对应每一个模板
    digits[i]=roi

#初始化两个卷积核
rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)) #黑帽礼帽常用方核，这里被我乱改掉了
sqKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)) #开闭运算常用圆核

#输入图像预处理
image=cv2.imread("card.png")
#cv_show('image',image)
image=imutils.resize(image,width=300)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv_show('gray',gray)

#礼帽操作，突出更明亮的区域
tophat=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
#cv_show('tophat',tophat)

#Sobel算x方向，只用一个方向比xy叠加效果更好
gradX=cv2.Sobel(tophat,cv2.CV_32F,1,0,ksize=-1) #ksize=-1相当于使用默认3*3全模板
gradX=np.absolute(gradX) #计算梯度值的绝对值，以避免负值的出现
(minVal,maxVal)=(np.min(gradX),np.max(gradX)) #找到梯度值的最小和最大值，这样就可以得到图像中边缘的强度范围
gradX=(255*((gradX-minVal)/(maxVal-minVal))) #对梯度值进行归一化处理，将其映射到 0 到 255 的范围
gradX=gradX.astype("uint8") #将归一化后的梯度值转换为 8 位无符号整数，确保梯度值在 0 到 255 的范围内，避免了浮点数计算的误差
#cv_show("gradX",gradX)

#通过闭运算(先膨胀后腐蚀)操作让数字成块
gradX=cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel)
#cv_show('gradX',gradX)

#THRESHOLD_OTSU会自动寻找合适的阈值，适合双峰型，需把阈值参数设置为0
thresh=cv2.threshold(gradX,0,255,
                     cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
#cv_show('thresh',thresh)

#再进行一次闭运算，填补空隙
thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)
#cv_show('thresh',thresh)

#计算轮廓，记得拷贝
threshCnts,hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=threshCnts
copy_img=image.copy()
cv2.drawContours(copy_img,cnts,-1,(0,0,255),1)
#cv_show('img',copy_img)

#筛选出实际有价值的区域
locs = [] #定义一个list
#遍历所有轮廓
for (i,c) in enumerate(cnts):
    (x,y,w,h)=cv2.boundingRect(c) #计算外接矩形
    ar=w/float(h) #计算长宽比


    #选择合适的区域，根据实际任务算出筛选条件范围(调参工程师)
    if ar>2.5 and ar<4.0:
        if(w>40 and w<60)and(h>10 and h<30):
            #符合条件的添加到list中
            locs.append((x,y,w,h))

#将符合条件的轮廓从左到右排序
locs=sorted(locs,key=lambda x:x[0])
output=[]
#遍历每一组中的数字
for(i,(gX,gY,gW,gH))in enumerate(locs):
    groupOutput=[]
    #根据坐标提取每一个组
    group=gray[gY-5:gY+gH+5,gX-5:gX+gW+5]
    #cv_show('group',group)

    #再处理每个数字块并拆分每个数字，标准流程
    #预处理二值化
    group=cv2.threshold(group,0,255,
                    cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    #cv_show('group',group)
    #计算每一组的轮廓(记得copy)
    digitCnts,hierarchy=cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts=contours.sort_contours(digitCnts,method='left-to-right')[0] #从左到右排序

    #计算每一组中每一个数字
    for c in digitCnts:
        #找到当前数值的轮廓，resize成合适的大小
        (x,y,w,h)=cv2.boundingRect(c)
        roi=group[y:y+h,x:x+w]
        roi=cv2.resize(roi,(57,88))
        #cv_show('roi',roi)

        #计算和0-9十个模板匹配分别的得分
        scores=[] #空分数list
        for(digit,digitROI) in digits.items():
            #模板匹配
            result=cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF) #这里用的TM_CCOEFF方法
            (_,score,_,_)=cv2.minMaxLoc(result)
            scores.append(score)

        #找到匹配度最高的数字
        groupOutput.append(str(np.argmax(scores)))

    #画出对应数字
    cv2.rectangle(image,(gX-5,gY-5),
                  (gX+gW+5,gY+gH+5),(0,0,255),1)
    cv2.putText(image,"".join(groupOutput),(gX,gY-15),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
    #得到结果
    output.extend(groupOutput)

#最后控制台打印全部结果
print("".join(output))
cv2.imshow('image',image)
cv2.waitKey(0)