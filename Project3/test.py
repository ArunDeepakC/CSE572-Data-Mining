#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
import random
random.seed(0)
train=pd.DataFrame()
warnings.filterwarnings('ignore')
label=list()

with open('bins.pkl', 'rb') as f:
    majority = pickle.load(f)
with open('train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('kmeans.pkl', 'rb') as f:
    kmeans_label = pickle.load(f)
test_file=input('Enter name of test file')
df = pd.read_csv(test_file, sep = '\t', header = None)
train=df
train=train[0].str.split(',', expand=True)
#label=label+label1
#label=np.array(label)
train.replace('',pd.np.nan,inplace=True)
train.fillna(value=pd.np.nan, inplace=True)
train.fillna(0,inplace=True)
#train=train.replace(" ",0)
#train=train.replace("NaN",0)
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
x=[i*5 for i in range(train.shape[1])]
for i in range(len(train)):
    poly=np.polyfit(x,train[i],3)
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
from sklearn import preprocessing
nf = preprocessing.normalize(nf)
from scipy.spatial import distance
nf_test=nf
nf=train_data
ans={}
for j in range(0,len(nf_test)):
    ans[j]=list()
    for i in range(0,len(nf)):
        ans[j].append(distance.euclidean(nf[i],nf_test[j]))
#print(ans[0])
min_list=[]
for ele in ans:
    x=[ans[ele].index(i) for i in ans[ele]]
    min_list.append(sorted(zip(x,ans[ele]), key=lambda t: t[1])[0:6])
min_list
test_label=list()
for j in range(len(min_list)):
    lab=list()
    for i in min_list[j]:
        
        lab.append(majority[kmeans_label[i[0]]])
    test_label.append(max(set(lab),key=lab.count))  
        
#print(test_label)
#print(carb_list_test)
from sklearn.metrics import accuracy_score
#print(accuracy_score(carb_list_test,test_label))
#len(carb_list_test)

kmeans_output=test_label
random.seed(0)
train=pd.DataFrame()
warnings.filterwarnings('ignore')
label=list()
# with open('train_db.pkl', 'rb') as f:
#     train_data=pickle.load(f)
# train_data=np.array(train_data)
with open('bins_db.pkl', 'rb') as f:
    majority = pickle.load(f)

with open('dbscan.pkl', 'rb') as f:
    y= pickle.load(f)
final_label=y
kmeans_label=y
from scipy.spatial import distance

#test_file=input('Enter name of test file')
df = pd.read_csv(test_file, sep = '\t', header = None)
train=df
train=train[0].str.split(',', expand=True)
#label=label+label1
#label=np.array(label)

train.replace('',pd.np.nan,inplace=True)
train.fillna(value=pd.np.nan, inplace=True)
train.fillna(0,inplace=True)
#train=train.replace(" ",0)
#train=train.replace("NaN",0)
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
fft_freq=scipy.fft.fftfreq(30, d=1.0)


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
x=[i*5 for i in range(train.shape[1])]
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
from sklearn import preprocessing
nf = preprocessing.normalize(nf)
from scipy.spatial import distance
nf_test=nf

nf=train_data
        

ans={}
for j in range(0,len(nf_test)):
    ans[j]=list()
    for i in range(0,len(nf)):
        ans[j].append(distance.euclidean(nf[i],nf_test[j]))
#print(ans[0])
min_list=[]
for ele in ans:
    x=[ans[ele].index(i) for i in ans[ele]]
    min_list.append(sorted(zip(x,ans[ele]), key=lambda t: t[1])[0:20])
min_list
test_label=list()
for j in range(len(min_list)):
    lab=list()
    for i in min_list[j]:
        
        lab.append(majority[final_label[i[0]]])
    test_label.append(max(set(lab),key=lab.count))  
kmeans_output=np.array(kmeans_output)       
dbscan_output=test_label
dbscan_output=np.array(dbscan_output)
dbscan_output=dbscan_output+1
kmeans_output=kmeans_output+1
output=np.column_stack((dbscan_output,kmeans_output))
o=pd.DataFrame(output)
o.to_csv('output.csv',header=None,index=None)
#print(output)
print('Output can be seen in output.csv')



# In[ ]:




