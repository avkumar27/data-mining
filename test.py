# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 01:55:20 2020

@author: avkgu
"""

import pandas as pd
import csv
from datetime import datetime, timedelta
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy.stats
import pickle

df = pd.read_csv('test.csv')


nmeld=df




######################################################
#feature1 
######################################################


mac1=nmeld.max(axis=1)
minim1=nmeld.min(axis=1)
cgmdiff1=mac1-minim1
cgmdiff11=np.array(cgmdiff1)
cdiff21=np.reshape(cgmdiff11, (len(cgmdiff11), 1))


#######################################################
#feature2
#######################################################

global FFT_Feature_Matrix1
FFT_coefficents1 = []
for it in range(nmeld.shape[0]):
    
    FFT_coefficents1.append(np.abs(np.fft.fft((nmeld.iloc[it,::-1]))))
    FFT_Feature_Matrix1= []
    
for c in range(0,len(FFT_coefficents1)):
    FFT_Feature_Matrix1.append(FFT_coefficents1[c][1:9]) # Take top 8
print(len(FFT_Feature_Matrix1))

#########################################################
#f3
#########################################################

Group_mean11 = []
Group_mean21 = []
Group_mean31 = []
Group_mean41 = []
Group_mean51 = []
for i in range(nmeld.shape[0]):
    Group_mean11.append(nmeld.iloc[i,0:5].mean())
    Group_mean21.append(nmeld.iloc[i,6:11].mean())
    Group_mean31.append(nmeld.iloc[i,12:17].mean())
    Group_mean41.append(nmeld.iloc[i,18:23].mean())
   

#f4
polycoeff1 = []
for i in range(nmeld.shape[0]):
    polycoeff1.append(np.polyfit(np.linspace(0,24,24), nmeld.iloc[i,:], 5))
    
Feture_Matrix1 = np.stack((
                              np.array(Group_mean11),
                              np.array(Group_mean21),
                              np.array(Group_mean31),
                              np.array(Group_mean41)
                              ))
fm2=np.hstack((np.transpose(Feture_Matrix1),
               np.array(cdiff21),
               np.array(FFT_Feature_Matrix1),
               np.array(polycoeff1)))

########################################################################
### Apply PCA to get top five components 
########################################################################
pca = PCA(n_components=9)
reduced_matrix1 = pca.fit_transform(fm2)


rm1=pd.DataFrame(reduced_matrix1)


X = rm1
loaded_model = pickle.load(open('KNN', 'rb'))

res = loaded_model.predict(X)
print('predicted>>',res)

filename = "VenkateshKumar_results.csv"
    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
      
        
    # writing the data rows  
    csvwriter.writerows(res) 
