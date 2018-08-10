
# coding: utf-8

# In[1]:


#引入必要的包
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
get_ipython().run_line_magic('matplotlib', 'inline')
# load the dataset


# In[115]:


#导入数据
sentiment = read_csv(r'C:\Users\l84105658\Desktop\--master\--master\Desktop\sentiment.csv')
cx=pd.read_excel(r'C:\Users\l84105658\Desktop\CX.xlsx')


# In[117]:


writer = pd.ExcelWriter(r'C:\Users\l84105658\Desktop\数据描述统计.xlsx')
result1.to_excel(writer,'Sheet2')


# In[15]:


dataframe =sentiment 
server=dataframe[dataframe['speaker']==0]
client=dataframe[dataframe['speaker']==1]
table=pd.DataFrame(client.groupby('recordName')['predictEmotionProbability'].count()>20)
df_client= pd.DataFrame(columns=['ID','last2','last5', 'last10', 'last15','last20','frist5','frist10','full'])
i=0
for index in table.index:
    if table.loc[index].any():
        last2=client[client['recordName']==index][-2:].sum().predictEmotionProbability
        last5=client[client['recordName']==index][-5:].sum().predictEmotionProbability
        last10=client[client['recordName']==index][-10:].sum().predictEmotionProbability
        last15=client[client['recordName']==index][-15:].sum().predictEmotionProbability
        last20=client[client['recordName']==index][-20:].sum().predictEmotionProbability
        frist5=client[client['recordName']==index][:5].sum().predictEmotionProbability
        frist10=client[client['recordName']==index][:10].sum().predictEmotionProbability
        full=client[client['recordName']==index].sum().predictEmotionProbability
        s=pd.DataFrame({"ID":index,"last2":last2,"last5":last5,"last10":last10,"last15":last15,"last20":last20,"frist5":frist5,"frist10":frist10,"full":full },index=[i])
        i=i+1
        df_client = df_client.append(s)
df_server= pd.DataFrame(columns=['ID','slast2','slast5', 'slast10', 'slast15','slast20','sfrist5','sfrist10','sfull'])
i=0
for index in table.index:
    if table.loc[index].any():
        slast2=server[server['recordName']==index][-2:].sum().predictEmotionProbability
        slast5=server[server['recordName']==index][-5:].sum().predictEmotionProbability
        slast10=server[server['recordName']==index][-10:].sum().predictEmotionProbability
        slast15=server[server['recordName']==index][-15:].sum().predictEmotionProbability
        slast20=server[server['recordName']==index][-20:].sum().predictEmotionProbability
        sfrist5=server[server['recordName']==index][:5].sum().predictEmotionProbability
        sfrist10=server[server['recordName']==index][:10].sum().predictEmotionProbability
        sfull=server[server['recordName']==index].sum().predictEmotionProbability
        s=pd.DataFrame({"ID":index,"slast2":slast2,"slast5":slast5,"slast10":slast10,"slast15":slast15,"slast20":slast20,"sfrist5":sfrist5,"sfrist10":sfrist10,"sfull":sfull },index=[i])
        i=i+1
        df_server = df_server.append(s)
result = pd.merge(df_server,df_client, on='ID')


# In[114]:


result1= pd.merge(result,cx, on='ID')


# In[85]:


result2=result1[result1['isr']!='未作答']


# In[86]:


label=result2.ix[:,[-1]]
label=label.fillna(method='pad')


# In[110]:


result2


# In[76]:


label=result2.ix[:,[-1]]


# In[111]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(result2, label, test_size=0.1, random_state=0)


# In[112]:


from sklearn import tree
criterions=['gini','entropy']
for criterion in criterions:
    clf = tree.DecisionTreeClassifier(criterion=criterion)
    clf.fit(X_train, y_train)
    print(criterion,"Training score:%f"%(clf.score(X_train,y_train)))
    print(criterion,"Testing score:%f"%(clf.score(X_test,y_test)))


# In[113]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(result2, label, test_size=0.1, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
n_neighbors=4
for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.X_train, X_test, y_train, y_test
    clf = KNeighborsClassifier(n_neighbors,weights=weights)
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
hello
tablecount=dataframe.groupby('recordName')['score2'].count()
table=pd.DataFrame(tablecount>20)
df_dataframe= pd.DataFrame(columns=['ID','last2','last5','last10','last15','last20','frist5','frist10','frist15','full','number'])
i=0
for index in table.index:
    if table.loc[index].any():
        last2=dataframe[dataframe['recordName']==index][-2:].sum().score2
        last5=dataframe[dataframe['recordName']==index][-5:].sum().score2
        last10=dataframe[dataframe['recordName']==index][-10:].sum().score2
        last15=dataframe[dataframe['recordName']==index][-15:].sum().score2
        last20=dataframe[dataframe['recordName']==index][-20:].sum().score2
        frist5=dataframe[dataframe['recordName']==index][:5].sum().score2
        frist10=dataframe[dataframe['recordName']==index][:10].sum().score2
        frist15=dataframe[dataframe['recordName']==index][:15].sum().score2
        full=dataframe[dataframe['recordName']==index].sum().score2
        number=tablecount.loc[index]
        s=pd.DataFrame({"ID":index,"last2":last2,"last5":last5,"last10":last10,"last15":last15,"last20":last20,"frist5":frist5,"frist10":frist10,"frist15":frist15,"full":full ,"number":number},index=[i])
        i=i+1
        df_dataframe = df_dataframe.append(s)
#时间序列处理
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
from keras.preprocessing.sequence import pad_sequences
%matplotlib inline
# load the dataset
dataframe = read_csv(r'C:\Users\l84105658\Desktop\--master\--master\Desktop\sentiment.csv')
cv = pd.read_excel(r'C:\Users\l84105658\Desktop\CX.xlsx')#读取数据
datax=[]
table=dataframe.groupby('recordName')['score2'].count()
table=pd.DataFrame((table>50)&(table<100))
for index in table.index:
    if table.loc[index].any():
        datax.append(dataframe[dataframe['recordName']==index]['score2'].values)
        #print(index)

X = pad_sequences(datax, maxlen=100, dtype='float32')#长度不足的0来填充
cv[['ID','isr']].drop_duplicates()
# LSTM with Variable Length Input Sequences to One Character Output
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from theano.tensor.shared_randomstreams import RandomStreams
# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
num_inputs = 1000
max_len = 5
dataX = []
dataY = []
for i in range(num_inputs):
	start = numpy.random.randint(len(alphabet)-2)
	end = numpy.random.randint(start, min(start+max_len,len(alphabet)-1))
	sequence_in = alphabet[start:end+1]
	sequence_out = alphabet[end + 1]
	dataX.append([char_to_int[char] for char in sequence_in])
	dataY.append(char_to_int[sequence_out])
	print(sequence_in, '->', sequence_out)
# convert list of lists to array and pad sequences if needed
X = pad_sequences(dataX, maxlen=max_len, dtype='float32')
# reshape X to be [samples, time steps, features]
X = numpy.reshape(X, (X.shape[0], max_len, 1))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# create and fit the model
batch_size = 1
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], 1)))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=batch_size, verbose=2)
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions
for i in range(20):
	pattern_index = numpy.random.randint(len(dataX))
	pattern = dataX[pattern_index]
	x = pad_sequences([pattern], maxlen=max_len, dtype='float32')
	x = numpy.reshape(x, (1, max_len, 1))
	x = x / float(len(alphabet))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	print(seq_in, "->", result)
