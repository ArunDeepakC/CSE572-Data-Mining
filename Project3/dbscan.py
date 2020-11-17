#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from pandas import read_csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy.polynomial.polynomial as poly
import statistics
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
import pickle
train=pd.DataFrame()
warnings.filterwarnings('ignore')
label=list()
np.random.seed(0)
for i in range(5):
    df = pd.read_csv("mealData{}.csv".format(i+1), sep = '\t', header = None).head(50)

    
    train = train.append(df, ignore_index = True)


train=train[0].str.split(',', expand=True)


train.fillna(value=pd.np.nan, inplace=True)
train.fillna(0,inplace=True)
train=train.replace("NaN",0)
train=train.astype(int)

#CGM Velocity
tr=train.T
cgm_velocity=tr.pct_change()
cgm_velocity=cgm_velocity.T


cgm_velocity=np.array(cgm_velocity)

x=[i*5 for i in range(train.shape[1])]
cv=list()
for i in range(train.shape[0]):
    zero_crossings = np.where(np.diff(np.sign(cgm_velocity[i])))[0]
    cv.append(zero_crossings[0:2])
cv=pd.DataFrame(cv)
cv=cv.to_numpy()


#FFT
FFT=abs(scipy.fft.fft(train))
fft_freq=scipy.fft.fftfreq(31, d=1.0)


FFT=np.array(FFT)
fft_freq=np.array(fft_freq)
Fourier_peak=list()
Fourier_frequency=list()
for i in range(len(FFT)):
    index=np.argsort(FFT)[i][-16:]

    peak=FFT[i][index]
    Fourier_peak.append(peak)
    freq=abs(fft_freq[index])
    freq.sort()
    fr=freq[[0,1,3,5,7,9,11,13]]
    Fourier_frequency.append(fr)

Fourier_peak=np.array(Fourier_peak)
Fourier_frequency=np.array(Fourier_frequency)
Fourier_peak=np.unique(Fourier_peak,axis=1)



#rolling mean
tr=train.T
rolling_mean=tr.rolling(window=3,min_periods=1).mean()
rolling_mean=rolling_mean.T

rolling_mean=np.array(rolling_mean)





#rolling std
tr=train.T
rolling_std=tr.rolling(window=3,min_periods=1).std()
rolling_std=np.array(rolling_std)
rolling_std=rolling_std.T
rolling_std=rolling_std[:,1:]

train=np.array(train)

#Polyfit
Poly=list()
x=[i*5 for i in range(31)]
for i in range(len(train)):
    poly=np.polyfit(x,train[i],4)
    Poly.append(poly)
Poly=np.array(Poly)




#Feature Set
feature=np.append(cv,Fourier_peak,axis=1)
feature=np.append(feature,Fourier_frequency,axis=1)
feature=np.append(feature,rolling_mean,axis=1)
feature=np.append(feature,rolling_std,axis=1)
feature=np.append(feature,Poly,axis=1)
#feature=np.append(feature,cv,axis=1)
feature=np.nan_to_num(feature)
#PCA
sc = StandardScaler()
X_std = sc.fit_transform(feature)
pca = decomposition.PCA(n_components=9)
#Eigen Vectors
PC = pca.fit_transform(X_std)
#Features transformed to new dimension
New_features = pca.transform(feature)
nf=pd.DataFrame(New_features)
nf=nf.head(150)
nf_test=nf.tail(100)
with open('train_db.pkl', 'wb') as f:
    pickle.dump(nf, f)
from sklearn import preprocessing
nf = preprocessing.normalize(nf)
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=0.085, min_samples=2)
clustering.fit(nf)
y=clustering.labels_
with open('dbscan.pkl', 'wb') as f:
    pickle.dump(y, f)
carb=pd.DataFrame()
for i in range(5):
    df = pd.read_csv("mealAmountData{}.csv".format(i+1), sep = '\t', header = None)[:50]
    carb = carb.append(df, ignore_index = True)
carbo=np.array(carb)
carb_list=list()
for ele in carbo:
    if ele <= 0:
        carb_list.append(0)
    if ele>0 and ele<=20:
        carb_list.append(1)
    if ele>20 and ele<=40:
        carb_list.append(2)
    if ele>40 and ele<=60:
        carb_list.append(3)
        
    if ele>60 and ele<=80:
        carb_list.append(4)
    if ele>80 and ele<=100:
        carb_list.append(5)
carb_list_test=carb_list[150:]
carb_list=carb_list[:150]
    
test={}
label={}
for i in range(len(y)):
    if y[i] not in test:
        test[y[i]]=list()
        
    test[y[i]].append(i)
for i in range(len(carb_list)):        
    if carb_list[i] not in label:
        label[carb_list[i]]=list()
    label[carb_list[i]].append(i)
data=list()
for ele in test[0]:
    data.append(nf[ele])
data=np.array(data)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0).fit(data)
x=kmeans.labels_
for i in range(len(x)):
    if x[i]==1:
        x[i]=-3
        
    elif x[i]==2:
        x[i]=-4
    elif x[i]==3:
        x[i]=-5
final=abs(x)
test_final={}

for i in range(len(final)):
    if final[i] not in test_final:
        test_final[final[i]]=list()
        
    test_final[final[i]].append(test[0][i])
test_final[1]=test[1]
test_final[2]=test[2]
ground=list()
majority=list()
for i in range(0,6):
    for ele in test_final[i]:
        ground.append(carb_list[ele])
    
    majority.append(max(set(ground),key=ground.count))    

majority.append(7)

final_label=[6]*150
for ele in test_final:
    for i in test_final[ele]:
        final_label[i]=ele
with open('bins_db.pkl', 'wb') as f:
    pickle.dump(majority, f)
from scipy.spatial import distance
ans={}
for j in range(0,len(nf_test)):
    ans[j]=list()
    for i in range(0,len(nf)):
        ans[j].append(distance.euclidean(nf[i],nf_test.iloc[j]))

min_list=[]
for ele in ans:
    x=[ans[ele].index(i) for i in ans[ele]]
    min_list.append(sorted(zip(x,ans[ele]), key=lambda t: t[1])[0:6])
min_list
test_label=list()
for j in range(len(min_list)):
    lab=list()
    for i in min_list[j]:
        
        lab.append(majority[final_label[i[0]]])
    test_label.append(max(set(lab),key=lab.count))  
        

test_copy=list()
carb_copy=list()
for i in range(len(test_label)):
    if test_label[i]!=7:
        test_copy.append(test_label[i])
        carb_copy.append(carb_list_test[i])
        
print("DBSCAN Trained")
# from sklearn.metrics import accuracy_score
# print(accuracy_score(carb_copy,test_copy))


# In[ ]:




