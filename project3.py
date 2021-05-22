# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:28:04 2020

@author: avkgu
"""

import pandas as pd
import csv
from datetime import datetime, timedelta
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy.stats
import pandas as pd                             # Import pandas for data manipulation/handling
import numpy as np                              # Import numpy for number processing
from pandas.plotting import scatter_matrix      # Used for plotting scatter matrix
import matplotlib                               # Need to import matplotlib
import matplotlib.pyplot as plt                 # Used to plot
from sklearn import model_selection
from scipy.integrate import simps
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.cluster import KMeans,DBSCAN

###########################################
#read csv file
##########################################
df = pd.read_csv('CGMData.csv',
                 usecols=['Date','Time','Sensor Glucose (mg/dL)'])

df1 = pd.read_csv('InsulinData.csv',
                 usecols=['Date','Time','BWZ Carb Input (grams)'])

df['Date'] = pd.to_datetime(df['Date'])

df['Date'] = df['Date'].dt.strftime("%m/%d/%y")

df['Time'] = pd.to_datetime(df['Time'])

df['Time'] = df['Time'].dt.strftime("%H:%M:%S")

###########################################
#reformatting dataframe
###########################################

df['Date_Time'] = df['Date']+' '+df['Time']
df['Date_Time']=pd.to_datetime(df['Date_Time'])
df = df.set_index('Date_Time')
df = df.sort_values(by='Date_Time')
dfx=df.interpolate(method ='linear', limit_direction ='forward')

df1 = df1.dropna()


df1['Date_Time'] = df1['Date']+' '+df1['Time']
df1['Date_Time']=pd.to_datetime(df1['Date_Time'])
df1 = df1.sort_values(by='Date_Time')

mt=df1[["Date_Time"]]

####################################################
#mealtime data
####################################################
mtx=mt
mtx=mtx.set_index('Date_Time')
mtx['dt']=mtx.index
mtx['delta'] = (mtx['dt']-mtx['dt'].shift())
texma=0
te1=pd.to_datetime(texma)
te2=pd.to_datetime(texma)
del1=te1-te2
mtx['delta'].fillna(value=del1)
mtf=mtx.loc[(mtx['delta']>='02:00:00')]
mtf['time1']=mtf['dt'] - timedelta(minutes=30)
mtf['time2']=mtf['dt'] + timedelta(hours=2)
mtf['time3']=mtf['dt'] + timedelta(hours=4)
mati=mtf['dt'].values.tolist()
mati1=pd.to_datetime(mati)
df3=df1
df3=df3.set_index('Date_Time')
df3['dt']=mtx.index
df3['delta'] = (df3['dt']-df3['dt'].shift())
df3['delta'].fillna(value=del1)
df3=df3.loc[(df3['delta']>='02:00:00')]
carb=df3['BWZ Carb Input (grams)'].values.tolist()
md=[]
mdld1=[]
for i in range(len(mtf)):
    la=dfx.loc[(dfx.index>mtf['time1'][i])&(dfx.index<mtf['time2'][i])]
    l1=la['Sensor Glucose (mg/dL)'].to_list()
    l2=l1
    l2.append(carb[i])
    l2.append(mati[i])
    md.append(l1)
    mdld1.append(l2)

meld1j=pd.DataFrame(md)
meldj=meld1j.iloc[:,:30]
meldj=meldj.dropna()


meld=meldj

####################################################
#binning
####################################################


df4=pd.DataFrame(mdld1)
df4[df4.columns[31]]=pd.to_datetime(df4[df4.columns[31]])
df5=df4.iloc[:,:32]
df5=df5.dropna()
df5=df5.set_index(df5[df5.columns[31]])

maxi=df5[df5.columns[30]].max()
mini=df5[df5.columns[30]].min()

bins=[-.1]
lables=[]

p=mini
r=0
while True:
    p=p+20
    bins.append(p)
    lables.append(r)
    r=r+1
    if(p > maxi):
        break


df5['binned'] = pd.cut(df5[df5.columns[30]], bins=bins,labels = lables)
labl1=df5['binned']
#df1['binned'] = pd.cut(df1['BWZ Carb Input (grams)'], bins=bins,labels = lables)
#labelmat=df1['binned']
nob=len(lables)


######################################################
#feature1 
######################################################


mac=meld.max(axis=1)
minim=meld.min(axis=1)
cgmdiff=mac-minim
cgmdiff1=np.array(cgmdiff)
cdiff2=np.reshape(cgmdiff1, (len(cgmdiff), 1))

sc = StandardScaler()

cdiff2 = sc.fit_transform(cdiff2)


#######################################################
#feature2
#######################################################

global FFT_Feature_Matrix
FFT_coefficents = []
for it in range(meld.shape[0]):
    FFT_coefficents.append(np.abs(np.fft.fft((meld.iloc[it,::-1])))[1:9])
FFT_Feature_Matrix = pd.DataFrame(list(map(np.ravel, FFT_coefficents)))

FFT_Feature_Matrix = sc.fit_transform(FFT_Feature_Matrix)

#for it in range(meld.shape[0]):
#    
#    FFT_coefficents.append(np.abs(np.fft.fft((meld.iloc[it,::-1]))))
#    FFT_Feature_Matrix = []
#    
#for c in range(0,len(FFT_coefficents)):
#    FFT_Feature_Matrix.append(FFT_coefficents[c][1:9]) # Take top 8
#print(len(FFT_Feature_Matrix))

#########################################################
#f3
#########################################################

Group_mean1 = []
Group_mean2 = []
Group_mean3 = []
Group_mean4 = []
Group_mean5 = []
for i in range(meld.shape[0]):
    Group_mean1.append(meld.iloc[i,0:5].mean())
    Group_mean2.append(meld.iloc[i,6:11].mean())
    Group_mean3.append(meld.iloc[i,12:17].mean())
    Group_mean4.append(meld.iloc[i,18:23].mean())
    Group_mean5.append(meld.iloc[i,23:29].mean())
    
Group_mean = sc.fit_transform(pd.DataFrame([Group_mean1,Group_mean2,Group_mean3,Group_mean4,Group_mean5]))

#f4
polycoeff = []
for i in range(meld.shape[0]):
    polycoeff.append(np.polyfit(np.linspace(0,30,30), meld.iloc[i,:], 5))

poly_coeff = sc.fit_transform(pd.DataFrame(polycoeff).transpose())
    
#Feture_Matrix = np.stack((
#                              np.array(Group_mean1),
#                              np.array(Group_mean2),
#                              np.array(Group_mean3),
#                              np.array(Group_mean4),
#                              np.array(Group_mean5),
#                              ))
#fm1=np.hstack((np.transpose(Feture_Matrix),
#               np.array(cdiff2),
#               np.array(FFT_Feature_Matrix),
#               np.array(polycoeff)))
Feture_Matrix = pd.concat([
                            pd.DataFrame(Group_mean).transpose(),
                            pd.DataFrame(cdiff2),
                            pd.DataFrame(FFT_Feature_Matrix),
                            pd.DataFrame(poly_coeff).transpose()],axis=1,ignore_index=True
                            )
Feture_Matrix = Feture_Matrix.to_numpy()
pca = PCA(n_components=9)
Feture_Matrix = pca.fit_transform(Feture_Matrix)

################################################
#kmeans
################################################

km = KMeans(n_clusters=nob,init='k-means++',n_init=10,max_iter=300,tol=1e-4,random_state=5)
y_km = km.fit_predict(Feture_Matrix)
km1=km.fit(Feture_Matrix)
sse=km1.inertia_


################################################
#dbscan
################################################
dbsc = DBSCAN(eps=1.5,min_samples=2)
dbsc1 = dbsc.fit_predict(Feture_Matrix)


################################################
#entropy and purity
################################################
def confmat(n,l1,l2):
    mat=[[0]*n for _ in range (n)]
    for i in range (n):
        for j in range(n):
            mat[i][j]=sum((l1==j)&(l2==i))
    return np.array(mat)


c1= confmat(nob,labl1,km1)
#c2=confmat(nob,labl1,y_km)
fields=['SSE']
sse1=[]
sse1.append(sse)
filename = "results.csv"
    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    # writing the fields  
    csvwriter.writerow(fields)  
        
    # writing the data rows  
    csvwriter.writerow(sse1) 