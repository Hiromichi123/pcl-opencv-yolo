中文变量=2**3
print(中文变量)

import math
print(math.sin(1))

s="hello"
print(len(s))
print(s[0])

T=None    #空值类型
print(type(T))

user_age=int(input("请输入年龄："))     #input只能返回字符串

if s==T:
    print("1")
elif s!=T:
    print("2")
else:
    print("3")
#and,or,not三个逻辑运算符


#列表list
shopping_list=["水果","蔬菜",123,None]      #动态可变
shopping_list.append("肉类")      #方法，直接增添
shopping_list.remove("蔬菜")      #直接删减

#字典dictionary
contact={"小明":1370000000,
         "小刚":1500000000}     #存入多个{键：值}对
contact["小红"]=159000000000    #添加/覆盖
print("小明"in contact)         #查询返回True/False
del contact["小明"]             #删除键和值
contact.keys()   #所有键
contact.values() #所有值
contact.items()  #所有键值对

#元组tuple
contact2={("张三",18):1300000000,
         ("李四",20):1500000000}      #元组（键盘，键帽），将元组作为键存入字典

for i in range(5,10,2):      #range范围左开右闭，步长
    print(i)

while s==T:
    s=1

#format方法
message_content="""
我是{name},
我叫{name},
我来自{country},
我是{country}人
""".format(name="张三",
           country="中国")

#定义函数
def summrize(a,b):
    sum=a+b
    print(f"两数之和为{sum}")      #字符串前加f也起format作用
    return sum       #返回值仅供外部使用，默认为None

summrize( 10,20)     #调用函数

import statistics           #导入了文件夹路径
from statistics import median,mean     #引入指定函数和变量
from statistics import*                #引入全部函数和变量

#pip install+第三方库

#创建对象
class CuteCat:
    def __init__(self,cat_name):        #构造函数，self不需要手动传入
        self.name=cat_name

cat1=CuteCat("Jojo")          #实例化对象

class CuteCuteCat(CuteCat):
    def __init__(self,cat_name):
        super().__init__(cat_name)     #super()调用父类方法
        self.name=cat_name

#读文件
f=open("./test.txt","r",encoding="utf-8")        #解码方式一般为utf-8
print(f.read(10))
print(f.read(10))   #从上一次读结尾结束接着往后读
print(f.readline())  #读整行

lines=f.readlines()      #readlines()读取整个文件，常和for循环一起使用
for line in lines:
    print(line)

#捕捉异常
try:
    weight=float(input("请输入体重:"))
    height=float(input("请输入身高:"))
    BMI=weight/height**2
except ValueError:
    print("")
except ZeroDivisionError:
    print("")
except:
    print(""+str(BMI))
finally:                       #无论是否错误都执行
    print("")

import unittest