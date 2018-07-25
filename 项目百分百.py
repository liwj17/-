import pandas as pd
table=pd.read_excel('F:/硕士课题/资产百分百项目/录音和案件v1/录音和案件/录音对应案件.xlsx','Sheet1')
table.head()
#--------------
#import os 
#import os.path


import os
from os.path import join, getsize
#filePath = raw_input('F:/硕士课题/资产百分百项目/录音和案件v1/录音和案件/audios ')
df_mp3 = pd.DataFrame(columns=['录音编号', '音频大小']) 
def getdirsize(dir):
    df_mp3 = pd.DataFrame(columns=['录音编号', '音频大小'])
    size = []
    name=[]
    i=0
    for root, dirs, files in os.walk(dir):
        for name in files:
			file='F:/硕士课题/资产百分百项目/录音和案件v1/录音和案件/audios/'+name+'.mp3'
			y, sr = librosa.load(file)
			S = librosa.stft(y)
            s=pd.DataFrame({"录音编号":name[::-1][4:][::-1],"音频大小":librosa.get_duration(S=S, sr=sr) },index=[i])
            df_mp3 = df_mp3.append(s)
            i=i+1
        #size += [getsize(join(root, name)) for name in files]
        #name+=[name[] for name in files]
 
    return df_mp3
 
file = getdirsize(r'F:\硕士课题\资产百分百项目\录音和案件v1\录音和案件\audios')
#
result[result['音频大小']<30]#过滤掉噪音音频
#dataframe合并
table=pd.read_excel('F:/硕士课题/资产百分百项目/录音和案件v1/录音和案件/合并音频大小.xlsx','Sheet1')
table.head()
result = pd.merge(table,file1, on='录音编号')
#进行音频长度获取
import pandas as pd
import os
from os.path import join, getsize
#filePath = raw_input('F:/硕士课题/资产百分百项目/录音和案件v1/录音和案件/audios ')
df_mp3 = pd.DataFrame(columns=['录音编号', '音频大小']) 
def getdirsize(dir):
    df_mp3 = pd.DataFrame(columns=['录音编号', '音频大小'])
    size = []
    name=[]
    i=0
    for root, dirs, files in os.walk(dir):
        for name in files:
            file='F:/硕士课题/资产百分百项目/录音和案件v1/录音和案件/audios/'+name
            #y, sr = librosa.load(file)
            #S = librosa.stft(y)
            #s=pd.DataFrame({"录音编号":name[::-1][4:][::-1],"音频大小":librosa.get_duration(S=S, sr=sr) },index=[i])
            s=pd.DataFrame({"录音编号":name[::-1][4:][::-1],"音频大小":librosa.get_duration(filename=file) },index=[i])
            df_mp3 = df_mp3.append(s)
            i=i+1
        #size += [getsize(join(root, name)) for name in files]
        #name+=[name[] for name in files]
 
    return df_mp3
 
file1 = getdirsize(r'F:\硕士课题\资产百分百项目\录音和案件v1\录音和案件\audios')
#写入excel中
writer = pd.ExcelWriter('output.xlsx')
result.to_excel(writer,'Sheet1')
#获取经纬度
import json
from urllib.request import urlopen, quote
import requests
def getlnglat(address):
    url = 'http://api.map.baidu.com/geocoder/v2/'
    output = 'json'
    ak = '3lWN2c4bDbWCUHspnNEt9dxB1ChbvxEi' # 浏览器端密钥
    address = quote(address) # 由于本文地址变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + address  + '&output=' + output + '&ak=' + ak 
    req = urlopen(uri)
    res = req.read().decode() 
    temp = json.loads(res)
    lat = temp['result']['location']['lat']
    lat = temp['result']['location']['lat']
    lng = temp['result']['location']['lng']
    return lat, lng
getlnglat('浙江省嘉兴市嘉善县浙江省嘉善县姚庄镇宝群东路159号')
#
a1=result1.describe()
a2=result2.describe()
a3=result3.describe()
a4=result4.describe()
a5=result5.describe()
a6=result6.describe()
a7=result7.describe()
a8=result8.describe()
writer = pd.ExcelWriter('合并后数据.xlsx')
a1.to_excel(writer,'债务人画像1')
a2.to_excel(writer,'债务人画像2')
a3.to_excel(writer,'债务人画像3')
a4.to_excel(writer,'债务人画像4')
a5.to_excel(writer,'债务人画像5')
a6.to_excel(writer,'债务人画像6')
a7.to_excel(writer,'债务人画像7')
a8.to_excel(writer,'债务人画像8')
frames = [table1,table2,table3,table4,table5,table6,table7,table8]
tablebig = pd.concat(frames)
import pandas as pd
table1=pd.read_excel('F:/硕士课题/资产百分百项目/债务人信息副本/债务人画像1.xlsx','Sheet2')
table2=pd.read_excel('F:/硕士课题/资产百分百项目/债务人信息副本/债务人画像2.xlsx','Sheet2')
table3=pd.read_excel('F:/硕士课题/资产百分百项目/债务人信息副本/债务人画像3.xlsx','Sheet2')
table4=pd.read_excel('F:/硕士课题/资产百分百项目/债务人信息副本/债务人画像4.xlsx','Sheet2')
table5=pd.read_excel('F:/硕士课题/资产百分百项目/债务人信息副本/债务人画像5.xlsx','Sheet2')
table6=pd.read_excel('F:/硕士课题/资产百分百项目/债务人信息副本/债务人画像6.xlsx','Sheet2')
table7=pd.read_excel('F:/硕士课题/资产百分百项目/债务人信息副本/债务人画像7.xlsx','Sheet2')
table8=pd.read_excel('F:/硕士课题/资产百分百项目/债务人信息副本/债务人画像8.xlsx','Sheet2')
table=pd.read_excel(r'F:\硕士课题\资产百分百项目\录音和案件v1\录音和案件\合并音频时长.xlsx','Sheet1')
#文字转录
from aip import AipSpeech

""" 你的 APPID AK SK """
APP_ID = '11326274'
API_KEY = 'Kp3f0AeyfvcVSq0Ol7U8YMA3'
SECRET_KEY = 'UEC0iTakMnZCcTxyZb2BhzOX6EWpHXtK'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

# 识别本地文件
client.asr(get_file_content(r'C:\Users\Think\Desktop\2.wav'), 'wav', 16000, {
    'dev_pid': 1537,
})
#数据删除
table.drop('录音编号',axis=1, inplace=True)
c=table.drop_duplicates()#去除重复项目
df =pd.DataFrame(columns=['身份证', '电话次数'])
i=0
for index in a.index:
    s=pd.DataFrame({"身份证":index,"电话次数":a['身份证'][i] },index=[i])
    df = df.append(s)
    i=i+1
df
a= pd.DataFrame(table['身份证'].value_counts())#统计数目
table.drop('还款总额',axis=1, inplace=True)
#R语言读取excel文件
dataset=read_excel('C:/Users/Think/Desktop/数据处理_资产百分百/D1_A1.xlsx')
#数据去重
table1.groupby('身份证号')['家庭地址'].apply(lambda x: x.drop_duplicates()).reset_index()
len(table1['身份证号'].unique())
table1.sort_values(by='level_1')#按列进行排序
#对合并数据进行处理
table2.groupby('身份证号')['首次逾期时间'].min()#首次逾期时间
table2.groupby('身份证号')['首次逾期时间'].count()#逾期期数
table2.groupby('身份证号')['首次录催时间'].count()#催录次数
table2.groupby('身份证号')['首次录催时间'].min()#首次催录时间
table2.groupby('身份证号')['首次还款时间'].count()#还款次数
table2.groupby('身份证号')['首次还款金额'].sum()#还款总金额
table2.groupby('身份证号')['案件金额'].sum()#案件总金额
table2.groupby('身份证号')['家庭地址'].apply(long)#获取最长家庭地址
table2.groupby('身份证号')['首次还款时间'].apply(count)#获取不同还款时间次数
table2.groupby('身份证号')['首次还款时间'].max()-table2.groupby('身份证号')['首次还款时间'].min()#还款间隔
table2.groupby('身份证号')['家庭地址'].apply(size)#获取地址长度
#定义一个返回最大字符长度的函数
def long(x):
    a=''
    for item in x:
        if len(str(item))>len(a):
            a=str(item)
    return a
#定义一个统计不同还款时间次数的函数
def count(x):
    i=0
    for item in x.value_counts():
        #a=item.index[1]
        i=i+1
    return i
#去除重复项目
a=pd.DataFrame(table2['身份证号'].drop_duplicates())
#提取区域代码
i=0
combine=pd.DataFrame(columns=['身份证号', '代码'])
for item in a['身份证号']:
    s=pd.DataFrame({"身份证号":item,"代码":item[:6]},index=[i])
    combine=combine.append(s)
    i=i+1
#将整数型转换为字符型
file1['代码']=file1['代码'].astype('str') 
#获取地址长度

def size(x):
    a=0
    for item in x:
        b=len(str(item))
        if b>a:
            a=b
    return a
#dataframe列重命名
dzl=pd.DataFrame(dzl).rename(columns={'家庭地址':'家庭地址长度'})
#
#获取姓名
xingming=pd.DataFrame(table.groupby('身份证号')['姓名'].unique())
#获取性别年龄
nianling=pd.DataFrame(table.groupby('身份证号')['年龄'].max())
xingbie=pd.DataFrame(table.groupby('身份证号')['性别'].max())
#获取家庭地址
dizhi=pd.DataFrame(table.groupby('身份证号')['家庭地址'].apply(long))#获取最长家庭地址
dizhichangdu=pd.DataFrame(table.groupby('身份证号')['家庭地址'].apply(size))#获取地址长度
anjianjine=pd.DataFrame(table.groupby('身份证号')['案件金额'].sum())#案件总金额
yuqiqishu=pd.DataFrame(table.groupby('身份证号')['逾期期数'].mean())#平均逾期期数
huankuanjiangemax=pd.DataFrame(table.groupby('身份证号')['还款间隔'].max())#最大还款间隔(最大与最小还款间隔差距不大，能否说明还款一般较为集中)
huankuanjiangemin=pd.DataFrame(table.groupby('身份证号')['还款间隔'].min())#最小还款间隔
table['首次录催时间']=table['首次催录时间'].astype('str') 
shoucilucuishijian=pd.DataFrame(table.groupby('身份证号')['首次录催时间'].min())#首次录催时间
mocilucuishijian=pd.DataFrame(table.groupby('身份证号')['首次录催时间'].max())#最大还款间隔
lucuicishu=pd.DataFrame(table.groupby('身份证号')['首次录催时间'].count())#催录次数
#获取最大ID
lucuicishu['首次录催时间'].idxmax()
#按列相同索引合并
result = xingming.join(nianling)
#离散数据计算相关比例（类似混淆矩阵）
#绘图 性别与还款与否关系
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.dpi'] = 150 #分辨率
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
death_counts = pd.crosstab(result3['Province'],result3['还款与否'])
death_counts.plot(kind='bar', stacked=True, color=['black','red'], grid=False)
death_counts[1]/(death_counts[0]+death_counts[1])	
#绘制箱体图
#result3.drop('案件总金额',axis=1, inplace=True)
data = [result3[result3['年龄']>0]['年龄'], ]  
#plt.boxplot(data) 
import seaborn as sns  
ax=sns.boxplot(x=result3[result3['年龄']>0]['还款与否'],y=result3[result3['年龄']>0]['年龄'])
#因子化
size_mapping = {    
           '男': 0,    
           '女': 1}
size_mapping = {    
           '是': 1,    
           '否': 0}
result3['本人是否PTP'] = result3['本人是否PTP'].map(size_mapping)
size_mapping = {    
           'False': 0,    
           'True': 1}
result3['是否县区'] = result3['是否县区'].map(size_mapping)    
#重置索引方式：
data_train_x.reset_index(drop=True)  