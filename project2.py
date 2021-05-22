# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 03:41:34 2020

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

def feext(meld):
    ######################################################
    #feature1 
    ######################################################
    
    
    mac=meld.max(axis=1)
    minim=meld.min(axis=1)
    cgmdiff=mac-minim
    cgmdiff1=np.array(cgmdiff)
    cdiff2=np.reshape(cgmdiff1, (len(cgmdiff), 1))
    
    
    #######################################################
    #feature2
    #######################################################
    
    global FFT_Feature_Matrix
    FFT_coefficents = []
    for it in range(meld.shape[0]):
        
        FFT_coefficents.append(np.abs(np.fft.fft((meld.iloc[it,::-1]))))
        FFT_Feature_Matrix = []
        
    for c in range(0,len(FFT_coefficents)):
        FFT_Feature_Matrix.append(FFT_coefficents[c][1:9]) # Take top 8
    print(len(FFT_Feature_Matrix))
    
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
    
    #f4
    polycoeff = []
    for i in range(meld.shape[0]):
        polycoeff.append(np.polyfit(np.linspace(0,30,30), meld.iloc[i,:], 5))
        
    Feture_Matrix = np.stack((
                                  np.array(Group_mean1),
                                  np.array(Group_mean2),
                                  np.array(Group_mean3),
                                  np.array(Group_mean4),
                                  np.array(Group_mean5),
                                  ))
    fm1=np.hstack((np.transpose(Feture_Matrix),
                   np.array(cdiff2),
                   np.array(FFT_Feature_Matrix),
                   np.array(polycoeff)))
    
    
    return (fm1)

def feext1(meld):
    ######################################################
    #feature1 
    ######################################################
    
    
    mac=meld.max(axis=1)
    minim=meld.min(axis=1)
    cgmdiff=mac-minim
    cgmdiff1=np.array(cgmdiff)
    cdiff2=np.reshape(cgmdiff1, (len(cgmdiff), 1))
    
    
    #######################################################
    #feature2
    #######################################################
    
    global FFT_Feature_Matrix
    FFT_coefficents = []
    for it in range(meld.shape[0]):
        
        FFT_coefficents.append(np.abs(np.fft.fft((meld.iloc[it,::-1]))))
        FFT_Feature_Matrix = []
        
    for c in range(0,len(FFT_coefficents)):
        FFT_Feature_Matrix.append(FFT_coefficents[c][1:9]) # Take top 8
    print(len(FFT_Feature_Matrix))
    
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
    
    #f4
    polycoeff = []
    for i in range(meld.shape[0]):
        polycoeff.append(np.polyfit(np.linspace(0,24,24), meld.iloc[i,:], 5))
        
    Feture_Matrix = np.stack((
                                  np.array(Group_mean1),
                                  np.array(Group_mean2),
                                  np.array(Group_mean3),
                                  np.array(Group_mean4),
                                  np.array(Group_mean5),
                                  ))
    fm1=np.hstack((np.transpose(Feture_Matrix),
                   np.array(cdiff2),
                   np.array(FFT_Feature_Matrix),
                   np.array(polycoeff)))
    
    
    return (fm1)

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
md=[]
for i in range(len(mtf)):
    la=dfx.loc[(dfx.index>mtf['time1'][i])&(dfx.index<mtf['time2'][i])]
    l1=la['Sensor Glucose (mg/dL)'].to_list()
    md.append(l1)

meld1j=pd.DataFrame(md)
meldj=meld1j.iloc[:,:30]
meldj=meldj.dropna()

####################################################
#no mealtime data
####################################################
nmt=mtx.loc[(mtx['delta']>='04:00:00')]
nmt['time2']=nmt['dt'] + timedelta(hours=2)
nmt['time3']=nmt['dt'] + timedelta(hours=4)
nm=[]
for i in range(len(nmt)):
    lb=dfx.loc[(dfx.index>nmt['time2'][i])&(dfx.index<nmt['time3'][i])]
    l2=lb['Sensor Glucose (mg/dL)'].to_list()
    nm.append(l2)
    
nmeld1j=pd.DataFrame(nm)
nmeldj=nmeld1j.iloc[:,:24]
nmeldj=nmeldj.dropna()

mealen=len(meldj)
nmealen=len(nmeldj)

fm1=feext(meldj)
fm2=feext1(nmeldj)

fm3=np.concatenate((fm1, fm2))
mealen=len(fm1)
nmealen=len(fm2)

#########################################################################
#### Apply PCA to get top five components 
#########################################################################
pca = PCA(n_components=9)
reduced_matrix = pca.fit_transform(fm3)
PCA_Variance = pca.explained_variance_ratio_
sum_ = np.sum(np.asarray(pca.explained_variance_ratio_))
objects = ('PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5')
y_pos = np.arange(len(objects))
PCA_Var_Percent = PCA_Variance * 100


########################################################################
#training model
########################################################################

rm=pd.DataFrame(reduced_matrix)

resarr1=[1]*mealen
resarr2=[0]*nmealen
resarr=resarr1+resarr2
rm['9']=resarr

#Final_data = pd.concat([rm,rm1],ignore_index=True, sort =False)

# Preparing to split out validation dataset
array = rm.values               # Array to hold all data
X = array[:,0:9]                     # Array to hold all input data
Y = array[:,9]                       # Array to hold all answers
validation_size = 0.2                       # Use 1/5 of the data for validation

seed = 5             # Seed to feed to model_selection

# Splitting out training set
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,test_size=validation_size, random_state=seed)
kf = KFold(n_splits=5)
kf.get_n_splits(X)



sc = StandardScaler()                       # Creating an instance of a scaler class
sc.fit(X_train)                             # fitting the scaler on X_train
X_train = sc.transform(X_train)             # Fitting the training input data
X_test = sc.transform(X_test)               # Fitting the training input data
scoring = 'accuracy'                        # Test options and evaluation metric


###################################################
# KNN
###################################################
# =============================================================================
# model = KNeighborsClassifier(n_neighbors=31,algorithm='auto')                    # creating KNN model
# model.fit(X_train,Y_train)                                                      # Fitting model on training data
# Y_predictions = model.predict(X_test)                                           # Making predictions on unseen data
# correct_or_no = np.array(Y_predictions) - np.array(Y_test)                      # Taking the difference between the arrays
# number_correct = np.sum(correct_or_no == 0)                                     # Counting the number of zeros
# total_samples = len(correct_or_no)                                              # Finding total number of zeros
# knn_accuracies= number_correct/total_samples
# =============================================================================


#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, Y, cv=5)#print each cv score (accuracy) and average them
print(cv_scores)
knn_cv.fit(X_train,Y_train)

# Save to file in the current working directory
# =============================================================================
# pkl_filename = "pickle_model.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(knn_cv, file)
# =============================================================================
pickle.dump(knn_cv, open('KNN', 'wb'))

#key_max = max(knn_accuracies.keys(), key=(lambda k: knn_accuracies[k]))                 # Finding the best value of K
#print('Accuracy of best K value (%d)\t\t%f' % (int(key_max),knn_accuracies[key_max]))   # Printing out the best value of K



