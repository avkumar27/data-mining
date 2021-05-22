# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:16:41 2020

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
#from mlxtend.frequent_patterns import apriori
#from mlxtend.frequent_patterns import association_rules
from efficient_apriori import apriori,itemsets_from_transactions,generate_rules_apriori

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

#####################################################
#bolus data
#####################################################

dfi = pd.read_csv('InsulinData.csv',
                 usecols=['Date','Time','BWZ Estimate (U)'])

dfi1=dfi.dropna()
dfi1['Date_Time'] = dfi1['Date']+' '+dfi1['Time']
dfi1['Date_Time']=pd.to_datetime(dfi1['Date_Time'])
dfi1 = dfi1.sort_values(by='Date_Time')
dfi1['BWZ Estimate (U)']=dfi1['BWZ Estimate (U)'].apply(np.ceil)
dfi1['BWZ Estimate (U)']=dfi1['BWZ Estimate (U)'].round(0)


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

for i in range(len(mtf)):
    la=dfx.loc[(dfx.index>mtf['time1'][i])&(dfx.index<mtf['time2'][i])]
    l1=la['Sensor Glucose (mg/dL)'].to_list()
    md.append(l1)

meld1j=pd.DataFrame(md)
meldj=meld1j.iloc[:,:30]
meldj=meldj.dropna()
meld=meldj

matib=dfi1['BWZ Estimate (U)'].values.tolist()
mdld1=[]
for i in range(len(mtf)):
    la=dfx.loc[(dfx.index>mtf['time1'][i])&(dfx.index<mtf['time2'][i])]
    l1=la['Sensor Glucose (mg/dL)'].to_list()
    l1=la['Sensor Glucose (mg/dL)'].to_list()
    l2=l1
    l2.append(carb[i])
    l2.append(matib[i])
    mdld1.append(l2)
    
df4=pd.DataFrame(mdld1)
df4=df4.dropna()

bd=df4[df4.columns[31]]  



####################################################
#binning
####################################################

ma=meld.max()
maxval=ma.max()
min1=meld.min()
minval=min1.min()

bins=[]
lables=[]
bins.append(minval-1)

p=minval
r=0
while True:
    p=p+20
    bins.append(p)
    lables.append(r)
    r=r+1
    if(p > maxval):
        break

bimlt=meld.copy()
for i in range (0,30):
    bimlt['clno'+str(i)]=pd.cut(bimlt[bimlt.columns[i]], bins=bins,labels = lables)

collist1=meld.idxmax(axis=1)
collist2=meld.idxmin(axis=1)
testbins=bimlt.iloc[:,30:]

maxbin=[]
minbin=[]
for j in range(0,len(meld)):
    temvar=testbins.iloc[j,collist1.iloc[j]]
    maxbin.append(temvar)
    temvar1=testbins.iloc[j,collist2.iloc[j]]
    minbin.append(temvar1)

bd1=bd.values.tolist()
basket_sets = pd.DataFrame(list(zip(maxbin, minbin,bd1)), 
               columns =['Bmax', 'bmin','inulinbolus'])


#################################################################
#association rule mining
#################################################################
tuples = [list(x) for x in basket_sets.to_numpy()]

itemsets, rules = apriori(tuples, min_support= 0.001)

lab1=lables.copy()

#lab1.pop(0)


rules_rhs = filter(lambda rule: set(rule.lhs).issubset(set(lab1)) and len(rule.lhs) == 2 and len(rule.rhs) == 1 and  all(isinstance(x, float) for x in rule.rhs), rules)

rules_list = []

rules_list1 = []
frequent_itemsets = []
best_rule={}

itemsets, _ = itemsets_from_transactions(tuples, min_support=0.001)

for j,item in enumerate(rules_rhs):
        rules_list.append(['({i_1},{i_2})->({i_3})'.format(i_1 = item.lhs[1], i_2=item.lhs[0], i_3 = item.rhs[0] ),item.confidence,item.support])
        frequent_itemsets.append(['({i_1},{i_2},{i_3})'.format(i_1 = item.lhs[1], i_2=item.lhs[0], i_3 = item.rhs[0] ),item.confidence,item.support])
        print(item)  # Prints the rule and its confidence, support, lift, ...



frequent_items=pd.DataFrame(rules_list,columns=['rule','confidence','support'])

frequent_items1=pd.DataFrame(frequent_itemsets,columns=['rule','confidence','support'])

best_rule= frequent_items.nlargest(15, 'confidence')

Anomoly_rule=frequent_items[frequent_items['confidence'] <= 0.5]

frequent_items1.to_csv('Task1frequent_items.csv',header = None,mode='a', index=False)
best_rule.to_csv('Task2largest_confidence_rules.csv',header = None,mode='a', index=False)
Anomoly_rule.to_csv('Task3Anomoly_rule.csv',header = None,mode='a', index=False)
       