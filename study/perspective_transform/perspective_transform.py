import cv2
import numpy as np
import argparse

# 设置参数
# ap=argparse.ArgumentParser() # 创建一个ArgumentParser对象ap，用于管理命令行参数的解析
# 添加一个参数-i（短参数）或--image（长参数），并指定了该参数是必需的（required=True），并提供了帮助信息，说明这个参数用于指定要扫描的图像的路径
# ap.add_argument("-i","--image",required=True,
                # help="Path to the image to be scanned")
#解析命令行参数，并将结果存储在一个字典args中。args将包含用户输入的参数--image对应的值，即用户提供的图像路径
# args=vars(ap.parse_args())

#获取角坐标点
def order_points(pts):
    #一共四个坐标点
    rect=np.zeros((4,2),dtype="float32")

    #按顺序找到对应坐标为0123分别是左上，右上，左下，右下
    #计算左上右下
    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]

    #计算右上左下
    diff=np.diff(pts,axis=1)
    rect[1]=pts[np.argmin(diff)]
    rect[3]=pts[np.argmax(diff)]

    return rect

# 透视变换
def four_point_transform(image,pts):
    #获取输入坐标点
    rect=order_points(pts)
    (tl,tr,br,bl)=rect

    #距离公式计算输入的w和h值
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
        [maxWidth - 1, maxHeight-1], #像素点之间的距离通常要多减一
        [0,maxHeight-1]],dtype="float32")

    #计算变换矩阵(一系列拓扑操作的集合)，cv帮算M矩阵
    M=cv2.getPerspectiveTransform(rect,dst)
    wraped=cv2.warpPerspective(image,M,(maxWidth,maxHeight))

    # 返回变换后结果
    return wraped

# 经典等比缩放函数
def resize(image,width=None,height=None,inter=cv2.INTER_AREA): # inter及插值方法
    dim=None
    (h,w)=image.shape[:2]
    if width is None and height is None:
        return image

    if width is None: # 根据参数情况执行不同程序
        r=height/float(h)
        dim=(int(w*r),height)
    else:
        r=width/float(w)
        dim=(width,int(h*r))
    resized=cv2.resize(image,dim,interpolation=inter) # 注意dim是一个存储当前宽高数据的元组
    return resized

#CV主程序
# image=cv2.imread(args["image"])
image=cv2.imread("text.png")

ratio=image.shape[0]/500.0 # 计算变换比例
orig=image.copy()

image=resize(orig,height=500) # 等比缩放

#Step1：边缘检测
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),200)
edges=cv2.Canny(gray,75,200)

#展示预处理结果
cv2.imshow("image",image)
cv2.imshow("egdes",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Step2：获取轮廓
kernel=np.ones((3,3),np.uint8)
dilation=cv2.dilate(edges,kernel,iterations=1) #调试发现边缘比较零散，需要膨胀连接处理
cnts,_=cv2.findContours(dilation.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5] #排序key为轮廓面积,降序，取前五个

# 遍历轮廓
screenCnt=None
for c in cnts:
    # 计算轮廓近似，C表示输入的点集
    peri=cv2.arcLength(c,True)
    # epsilon准确度参数，表示从原始轮廓到近似轮廓的最大距离，True封闭
    approx=cv2.approxPolyDP(c,0.05*peri,True)

    # 多边形逼近
    if len(approx)==4:
        screenCnt=approx
        break

# 展示结果
if screenCnt is not None:
    cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
    cv2.imshow("outline",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# step3:透视变换
# orig没有resize所以要乘ratio还原,(4,2)为多边形逼近拿到的四个原始顶点
warped=four_point_transform(orig,screenCnt.reshape(4,2)*ratio)

# 二值化处理，保留重要信息
warped=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
ref=cv2.threshold(warped,100,255,cv2.THRESH_BINARY)[1]
# cv2.imshow("scan",ref)

# 展示结果
cv2.imshow("original",resize(orig,height=500))
cv2.imshow("scan",resize(ref,height=500))
cv2.waitKey(0)
cv2.destroyAllWindows()