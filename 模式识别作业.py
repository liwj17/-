#读取图片，并显示
%matplotlib inline
from skimage import io,data,transform  
import matplotlib.pyplot as plt #plt 用于显示图片
import matplotlib.image as mpimg #mpimg 用于读取图片
import numpy as np
import cv2
#lena = io.imread('F:/研究生课程文件/课程文件/模式识别/271206628_1_作业1_2018/test/1.bmp', as_grey=True) #读取和代码处于同一目录下的lena.png
number1=io.imread('F:/研究生课程文件/课程文件/模式识别/271206628_1_作业1_2018/train/1.bmp', as_grey=True) #读取和代码处于同一目录下的lena.png
lena=io.imread('F:/研究生课程文件/课程文件/模式识别/271206628_1_作业1_2018/test/11.bmp', as_grey=True) #读取和代码处于同一目录下的lena.png
#lena = mpimg.imread('F:/研究生课程文件/课程文件/模式识别/271206628_1_作业1_2018/test/1.bmp')
[a1,b1]=number1.shape
[a,b]=lena.shape
#二值化处理
#ret,thresh1=cv2.threshold(number1,0.3,1,cv2.THRESH_BINARY)
ret,thresh1=cv2.threshold(lena,0.50,1,cv2.THRESH_BINARY)
ret,thresh2=cv2.threshold(number1,140,1,cv2.THRESH_BINARY)
number1=thresh2
lena=thresh1
d=np.random.randn(a-a1,b-b1)
w=[]
row=[]
for i in range(0,a-a1):
    for j in range(0,b-b1):
        picture=lena[i:i+a1,j:j+b1]
        d[i,j]=np.sqrt(np.sum(np.square(number1-picture)))
        if d[i,j]<11.3:
            w.append(d[i,j])
            row.append([i,j])
count=[]
for i in range(0,5):
    for j in range(0,6):
        count.append([i,j])
count
        
a = np.array(row)
b=np.array(count)
c=np.c_[a,b]#将两个ndarry合并一起，形成N*4维数据
#将多个图像绘制在一个表格上
import matplotlib.pyplot as plt
figure,axes=plt.subplots(5,6)
a=1
b=1
#同时操作四个数字
for a11,b11,c11,d11 in c:
    i=a11
    j=b11
    picture=lena[i:i+a1,j:j+b1]
    axes[c11,d11].imshow(picture)
#在原始图像上标记出数字
for i,j in row:
    plt.imshow(lena,'gray') 
    x =[j,j,j+b1,j+b1,j] 
    y =[i,i+a1,i+a1,i,i]  
    plt.plot(x[:5],y[:5])

--------------------------------
#二值化处理
#ret,thresh1=cv2.threshold(number1,0.3,1,cv2.THRESH_BINARY)
ret,thresh1=cv2.threshold(lena,0.50,1,cv2.THRESH_BINARY)
ret,thresh2=cv2.threshold(number1,140,1,cv2.THRESH_BINARY)
plt.imshow(thresh1,'gray') 
plt.show()
-----------------------------
#利用欧氏距离获取最近的数字
d=np.random.randn(a-a1,b-b1)
w=10000
for i in range(0,a-a1):
    for j in range(0,b-b1):
        picture=lena[i:i+a1,j:j+b1]
        d[i,j]=np.sqrt(np.sum(np.square(number1-picture)))
        if d[i,j]<w:
            w=d[i,j]
            row=i
            col=j
-----------------------------------------
#获取图像位置
i=row
j=col
picture=lena[i:i+a1,j:j+b1]
---------------------------------------
d=np.random.randn(a-a1,b-b1)
w=[]
row=[]
for i in range(0,a-a1):
    for j in range(0,b-b1):
        picture=lena[i:i+a1,j:j+b1]
        d[i,j]=np.sqrt(np.sum(np.square(number1-picture)))
        if d[i,j]<10:
            w.append(d[i,j])
            row.append([i,j])
count=[]
for i in range(0,4):
    for j in range(0,6):
        count.append([i,j])
count
        
a = np.array(row)
b=np.array(count)
c=np.c_[a,b]#将两个ndarry合并一起，形成N*4维数据
------------------------------------------
#将多个图像绘制在一个表格上
import matplotlib.pyplot as plt
figure,axes=plt.subplots(4,6)
a=1
b=1
#同时操作四个数字
for a11,b11,c11,d11 in c:
    i=a11
    j=b11
    picture=lena[i:i+a1,j:j+b1]
    axes[c11,d11].imshow(picture)
plt.show()
--------------------------------
#绘制图像直方图
a2=mat.flatten()
n, bins, patches = plt.hist(arr, bins=50, normed=1, facecolor='green', alpha=0.75)
---------------------------
第三次作业
import skimage.io as io
import matplotlib.pyplot as plt #plt 用于显示图片
import numpy as np
data_dir='F:/研究生课程文件/课程文件/模式识别/61406879_3_第三次作业/第三次作业/face/train'
str=data_dir + '/*.jpg'
coll = io.ImageCollection(str)#批量读入图片
array_list=[]
for i in range(0,len(coll)):
    array_list.append(np.float32(coll[i].reshape(1,19*19)))
data = np.vstack((array_list))#将所有读入的图片进行展开后合并
# define PCA
def pca(data,k):
    data = np.mat(data)
    rows,cols = data.shape#取大小
    data_mean = np.mean(data,0)#对列求均值
    data_mean_all = np.tile(data_mean,(rows,1))
    Z = data - data_mean_all
    T1 = Z*Z.T #使用矩阵计算，所以前面mat
    D,V = np.linalg.eig(T1) #特征值与特征向量
    V1 = V[:,0:k]#取前k个特征向量
    V1 = Z.T*V1
    for i in range(0,k): #特征向量归一化
        L = np.linalg.norm(V1[:,i])
        V1[:,i] = V1[:,i]/L

    data_new = Z*V1 # 降维后的数据
    return data_new,data_mean,V1

def error_K(data,k,num):
    data_new,data_mean,V1=pca(data,k)
    picture=data_new*V1.T
    error=np.sqrt(np.sum((np.square(data[num,:]-picture[num,:])/255))#计算第m个数据的误差
    return  error
error=[]
for k in range(1,50):
    error.append(error_K(data,k,1))
plt.plot(error)
plt.show()
---------------------------------
第二问
test=test.reshape(1,19*19)
errortest=200
K=40
while (errortest>194.12854):
    data_new,data_mean,V1=pca(data,K)
    picture=data_mean+(test-data_mean)*V1*V1.T#重构图像
    errortest=np.sqrt(np.sum(np.square(test-picture)))#计算第m个数据的误差
    K+=1
    print(K)
最小结果为：18.901133
第三问：相同的误差需要175个特征
	




