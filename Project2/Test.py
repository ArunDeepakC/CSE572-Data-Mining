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
import pickle
model = pickle.load( open("Trained_Model.pkl", "rb"))
train=pd.DataFrame()
t=pd.DataFrame()
warnings.filterwarnings('ignore')
label=list()
np.random.seed(0)
test_file=input('Enter Test file')


train=pd.read_csv(test_file, sep = '\t',header=None)


train=train[0].str.split(',', expand=True)
train.fillna(value=pd.np.nan, inplace=True)
train.fillna(0,inplace=True)
train=train.replace("NaN",0)
train=train.astype(int)
#Feature Extraction
#CGM Velocity
tr=train.T
cgm_velocity=tr.pct_change()
cgm_velocity=cgm_velocity.T
cgm_velocity=np.array(cgm_velocity)

x=[i*5 for i in range(train.shape[0])]
cv=list()
for i in range(len(train)):
    zero_crossings = np.where(np.diff(np.sign(cgm_velocity[i])))[0]
    cv.append(zero_crossings[0:2])
cv=pd.DataFrame(cv)
cv=cv.to_numpy()

#FFT
FFT=abs(scipy.fft.fft(train))
fft_freq=scipy.fft.fftfreq(train.shape[1], d=1.0)


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
x=[i*5 for i in range(len(train[0]))]
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
feature=np.nan_to_num(feature)
#PCA
sc = StandardScaler()
X_std = sc.fit_transform(feature)
pca = decomposition.PCA(n_components=5)
#Eigen Vectors
PC = pca.fit_transform(X_std)
#Features transformed to new dimension
New_features = pca.transform(feature)
result=model.predict(New_features)
mlp= pd.DataFrame(result)
mlp.to_csv('Predicted_Labels.csv',index=False,header=None)
print('Predicted Labels can be found in Predicted_Labels.csv')
