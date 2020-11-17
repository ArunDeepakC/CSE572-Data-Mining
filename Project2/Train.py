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
    df = pd.read_csv("MealNoMealData/mealData{}.csv".format(i+1), sep = '\t', header = None)

    
    train = train.append(df, ignore_index = True)
c=train.shape[0]


label=[1 for i in range(c)]

for i in range(5):
    df = pd.read_csv("MealNoMealData/Nomeal{}.csv".format(i+1), sep = '\t', header = None)

    train = train.append(df, ignore_index = True)
x=train.shape[0]-c
label1=[0 for i in range(x)]



train=train[0].str.split(',', expand=True)
label=label+label1
label=np.array(label)
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
pca = decomposition.PCA(n_components=5)
#Eigen Vectors
PC = pca.fit_transform(X_std)
#Features transformed to new dimension
New_features = pca.transform(feature)
nf=pd.DataFrame(New_features)
#PCA performance
pcaratio= pca.explained_variance_ratio_

X=New_features
y=label
k_fold = KFold(n_splits = 10, shuffle = True)
accuracy=list()
precision=list()
recall=list()
f1=list()
classifier = MLPClassifier(alpha = 0.1, max_iter = 1000)
#classifier=RandomForestClassifier(max_depth=2,random_state=0)

for train_index, test_index in k_fold.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]

  
      
    classifier.fit(Xtrain, ytrain)
    predicted = classifier.predict(Xtest)

    accuracy.append(accuracy_score(ytest, predicted))
    f1.append(f1_score(ytest, predicted))
    precision.append(precision_score(ytest, predicted))
    recall.append(recall_score(ytest, predicted))
filename = 'Trained_Model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

print("Accuracy: ", np.mean(accuracy)*100,'%')
print("Precision:",np.mean(precision)*100,'%')
print("Recall:",np.mean(recall)*100,'%')
print("F1:",np.mean(f1)*100,'%')

