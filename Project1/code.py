#!/usr/bin/env python
# coding: utf-8

# In[68]:



import pandas as pd
from pandas import read_csv
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import welch
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy.polynomial.polynomial as poly

warnings.filterwarnings('ignore')
pat1_timeframe = pd.read_csv('Datafolder//CGMDatenumLunchPat1.csv')
pat2_timeframe = pd.read_csv('DataFolder//CGMDatenumLunchPat2.csv')
pat3_timeframe = pd.read_csv('DataFolder//CGMDatenumLunchPat3.csv')
pat4_timeframe = pd.read_csv('DataFolder//CGMDatenumLunchPat4.csv')
pat5_timeframe = pd.read_csv('DataFolder//CGMDatenumLunchPat5.csv')
pat1_series= pd.read_csv('DataFolder//CGMSeriesLunchPat1.csv')
pat2_series= pd.read_csv('DataFolder//CGMSeriesLunchPat2.csv')
pat3_series= pd.read_csv('DataFolder//CGMSeriesLunchPat3.csv')
pat4_series= pd.read_csv('DataFolder//CGMSeriesLunchPat4.csv')
pat5_series= pd.read_csv('DataFolder//CGMSeriesLunchPat5.csv')
pat_series=pd.concat([pat1_series,pat2_series,pat3_series,pat4_series,pat5_series])
pat_timeframe=pd.concat([pat1_timeframe,pat2_timeframe,pat3_timeframe,pat4_timeframe,pat5_timeframe])
pat_series.fillna(0,inplace=True)
pat_timeframe.fillna(0,inplace=True)
df=pd.DataFrame(pat_series)
#Patient 1 CGM data
x=[i*5 for i in range(len(pat_timeframe.iloc[10]))]
plt.plot(x,pat_series.iloc[10])
plt.title("CGM timeseries of Patient 1")
plt.ylabel("CGM Series")
plt.xlabel("Time frame")
plt.show()
#Welch's  Method
Welch=scipy.signal.welch(pat_series)
y=np.array(Welch[1])
wel=list()
for i in range(len(y)):
    y[i].sort()
    Reverse=y[i][::-1]
    wel.append(Reverse[0:1])

wel=np.array(wel) 

x=list()
x.append(Welch[0])
x=np.array(x)
plt.stem(x.T,Welch[1][10])
plt.title("Welch's Method")
plt.ylabel("Power spectral density (Pxx)")
plt.xlabel("Frequency")
plt.show()
#FFT
FFT=abs(scipy.fft.fft(pat_series))
fft_freq=scipy.fft.fftfreq(42, d=1.0)

plt.stem(fft_freq,FFT[10])
FFT=np.array(FFT)
fft_freq=np.array(fft_freq)
Fourier_peak=list()
Fourier_frequency=list()
for i in range(len(FFT)):
    index=np.argsort(FFT)[i][-9:]

    peak=FFT[i][index]
    Fourier_peak.append(peak)
    freq=abs(fft_freq[index])
    freq.sort()
    fr=freq[[0,1,3,5,7]]
    Fourier_frequency.append(fr)

Fourier_peak=np.array(Fourier_peak)
Fourier_frequency=np.array(Fourier_frequency)
Fourier_peak=np.unique(Fourier_peak,axis=1)


plt.title("Fast Fourier Transform")
plt.ylabel("FFT result")
plt.xlabel("frequency")
plt.show()

#rolling mean
rolling_mean=df.rolling(window=10,min_periods=1).mean()
rolling_mean=np.array(rolling_mean)
pat_timeframe=np.array(pat_timeframe)
pat_series=np.array(pat_series)
x=[i*5 for i in range(len(pat_timeframe[10]))]
plt.plot(x,rolling_mean[10])
plt.title("Rolling Mean")
plt.ylabel("Rolling Mean")
plt.xlabel("Timeframe in minutes")
plt.show()
#rolling std
rolling_std=df.rolling(window=10,min_periods=1).std()
rolling_std=np.array(rolling_std)
pat_timeframe=np.array(pat_timeframe)
pat_series=np.array(pat_series)
x=[i*5 for i in range(len(pat_timeframe[10]))]
plt.plot(x,rolling_std[10])


plt.title("Rolling standard deviation")
plt.ylabel("Rolling standard deviation")
plt.xlabel("Timeframe in minutes")
plt.show()
plt.show()

#Polyfit
Poly=list()
x=[i*5 for i in range(len(pat_timeframe[100]))]
for i in range(len(pat_series)):
    poly=np.polyfit(x,pat_series[i],4)
    Poly.append(poly)
Poly=np.array(Poly)
x=[i*5 for i in range(len(pat_timeframe[10]))]
plt.plot(x,np.polyval(Poly[10],x),label='Polynomial Fit')
plt.plot(x,pat_series[10],label='CGMSeries')
plt.legend()
plt.title("Polyfit")
plt.ylabel("CGMSeries")
plt.xlabel("Timeframe in minutes")
plt.show()


#Feature Set
feature=np.append(wel,Fourier_peak,axis=1)
feature=np.append(feature,Fourier_frequency,axis=1)
feature=np.append(feature,rolling_mean,axis=1)
feature=np.append(feature,rolling_std,axis=1)
feature=np.append(feature,Poly,axis=1)
feature=np.nan_to_num(feature)


#PCA
sc = StandardScaler()
X_std = sc.fit_transform(feature)
pca = decomposition.PCA(n_components=5)
#Eigen Vectors
PC = pca.fit_transform(X_std)
#Features transformed to new dimension
New_features = pca.transform(feature)
print("Feature size after PCA:", New_features.shape)
nf=pd.DataFrame(New_features)
#PCA performance
pcaratio= pca.explained_variance_ratio_

#Variance explained by each component
variance_pc = pca.explained_variance_

plt.title("Variance of each Principal Component")
plt.ylabel("Variance")
plt.xlabel("Principal Components")
plt.bar(list(range(1, 6)), variance_pc)
plt.show()
#PC1
plt.scatter(list(range(0,216)), New_features[0:216,0],label='PC1')
plt.title("PC-1")
plt.ylabel("Features")
plt.ylim(-20000,0)
plt.xlabel("Time")
plt.show()
#PC2
plt.scatter(list(range(0,216)), New_features[0:216, 1],label='PC2')
plt.title("PC-2")
plt.ylabel("Features")
plt.ylim(0,20000)
plt.xlabel("Time")
plt.show()
#PC3
plt.scatter(list(range(0, 216)), New_features[0:216, 2],label='PC3')
plt.title("PC-3")
plt.ylabel("Features")
plt.xlabel("Time")
plt.ylim(0,20000)
plt.show()
#PC4
plt.scatter(list(range(0, 216)), New_features[0:216, 3],label='PC4')
plt.title("PC-4")
plt.ylabel("Features")
plt.xlabel("Time")
plt.ylim(-20000,1000)
plt.show()
#PC5
plt.scatter(list(range(0, 216)), New_features[0:216, 4],label='PC5')
plt.title("PC-5")
plt.ylabel("Features")
plt.xlabel("Time")
plt.ylim(-40000,1000)
plt.show()
nf=pd.DataFrame(New_features)

w = pd.DataFrame(wel)
fe=pd.DataFrame(feature)
fp =  pd.DataFrame(Fourier_peak)
ff =  pd.DataFrame(Fourier_frequency)
rm = pd.DataFrame(rolling_mean)
rstd = pd.DataFrame(rolling_std)
poly = pd.DataFrame(Poly)
print("Features extracted by Welch's method")
print(w)
print("Features extracted by Fourier peak")
print(fp)
print("Features extracted by Fourier frequency")
print(ff)
print("Features extracted by rolling mean")
print(rm)
print("Features extracted by rolling std")
print(rstd)
print("Features extracted by Polyfit")
print(poly)
print("Consolidated feature Matrix")
print(fe)
print("New Feature Matrix after PCA")
print(nf)
print("PCA variance= ", variance_pc)
print("PCA ratio:",sum(pcaratio))
# In[ ]:




