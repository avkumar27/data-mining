# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import csv

###########################################
#read csv file
##########################################
df = pd.read_csv('CGMData.csv',
                 usecols=['Date','Time','Sensor Glucose (mg/dL)'])

df1 = pd.read_csv('InsulinData.csv',
                 usecols=['Date','Time','Alarm'])

df['Date'] = pd.to_datetime(df['Date'])

df['Date'] = df['Date'].dt.strftime("%m/%d/%y")

df['Time'] = pd.to_datetime(df['Time'])

df['Time'] = df['Time'].dt.strftime("%H:%M:%S")

##########################################
#creating list to analyse the count
##########################################

dlis=df.groupby(['Date']).count()

lis = df.loc[(df['Time'] >= '00:00:00')&
             (df['Time'] <= '06:00:00')].groupby(['Date']).count()

lis2 = df.loc[(df['Time'] >= '06:00:00')&
              (df['Time'] <= '23:59:59')].groupby(['Date']).count()

############################################
#handling excess sample and low sampled data
############################################
lisx= dlis[dlis.Time<260].index

a=lisx
indexNames = df[ df['Date'].isin(a) ].index
df.drop(indexNames , inplace=True)

###########################################
#reformatting dataframe
###########################################

df['Date_Time'] = df['Date']+' '+df['Time']
df['Date_Time']=pd.to_datetime(df['Date_Time'])
df = df.set_index('Date_Time')
df = df.sort_values(by='Date_Time')
dfx=df.interpolate(method ='linear', limit_direction ='forward')



df1 = df1[df1['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF']
df1['Date_Time'] = df1['Date']+' '+df1['Time']
df1['Date_Time']=pd.to_datetime(df1['Date_Time'])
df1 = df1.sort_values(by='Date_Time')
df1 = df1.set_index('Date_Time')


li= df1.index[df1['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].tolist()
idx = df.index[df.index.get_loc(pd.to_datetime(li[0]), method="nearest")]

###################################################
#splitting the data frame for manual and automatic
###################################################
df_m = df.loc[df.index < idx]
df_a = df.loc[df.index >= idx]

##################################################
#getting matrix for manual
##################################################



hg1om = df_m.loc[(df_m['Time'] >= '00:00:00')&
             (df_m['Time'] <= '06:00:00')&(df_m['Sensor Glucose (mg/dL)']>180)].groupby(['Date']).count()

hg1mm = df_m.loc[(df_m['Time'] >= '06:00:00')&
             (df_m['Time'] <= '23:59:59')&(df_m['Sensor Glucose (mg/dL)']>180)].groupby(['Date']).count()

hg1dm = df_m.loc[(df_m['Sensor Glucose (mg/dL)']>180)].groupby(['Date']).count()

hg2om = df_m.loc[(df_m['Time'] >= '00:00:00')&
             (df_m['Time'] <= '06:00:00')&(df_m['Sensor Glucose (mg/dL)']>250)].groupby(['Date']).count()

hg2mm = df_m.loc[(df_m['Time'] >= '06:00:00')&
             (df_m['Time'] <= '23:59:59')&(df_m['Sensor Glucose (mg/dL)']>250)].groupby(['Date']).count()

hg2dm = df_m.loc[(df_m['Sensor Glucose (mg/dL)']>250)].groupby(['Date']).count()

rangepom = df_m.loc[(df_m['Time'] >= '00:00:00')&
             (df_m['Time'] <= '06:00:00')&(df_m['Sensor Glucose (mg/dL)']>=70)&
             (df_m['Sensor Glucose (mg/dL)']<=180)].groupby(['Date']).count()

rangepmm = df_m.loc[(df_m['Time'] >= '06:00:00')&
             (df_m['Time'] <= '23:59:59')&(df_m['Sensor Glucose (mg/dL)']>=70)&
             (df_m['Sensor Glucose (mg/dL)']<=180)].groupby(['Date']).count()

rangepdm = df_m.loc[(df_m['Sensor Glucose (mg/dL)']>=70)&
             (df_m['Sensor Glucose (mg/dL)']<=180)].groupby(['Date']).count()

rangesom = df_m.loc[(df_m['Time'] >= '00:00:00')&
             (df_m['Time'] <= '06:00:00')&(df_m['Sensor Glucose (mg/dL)']>=70)&
             (df_m['Sensor Glucose (mg/dL)']<=150)].groupby(['Date']).count()

rangesmm = df_m.loc[(df_m['Time'] >= '06:00:00')&
             (df_m['Time'] <= '23:59:59')&(df_m['Sensor Glucose (mg/dL)']>=70)&
             (df_m['Sensor Glucose (mg/dL)']<=150)].groupby(['Date']).count()

rangesdm = df_m.loc[(df_m['Sensor Glucose (mg/dL)']>=70)&
             (df_m['Sensor Glucose (mg/dL)']<=150)].groupby(['Date']).count()


hgy1om = df_m.loc[(df_m['Time'] >= '00:00:00')&
             (df_m['Time'] <= '06:00:00')&(df_m['Sensor Glucose (mg/dL)']<70)].groupby(['Date']).count()

hgy1mm = df_m.loc[(df_m['Time'] >= '06:00:00')&
             (df_m['Time'] <= '23:59:59')&(df_m['Sensor Glucose (mg/dL)']<70)].groupby(['Date']).count()

hgy1dm = df_m.loc[(df_m['Sensor Glucose (mg/dL)']<70)].groupby(['Date']).count()

hgy2om = df_m.loc[(df_m['Time'] >= '00:00:00')&
             (df_m['Time'] <= '06:00:00')&(df_m['Sensor Glucose (mg/dL)']<54)].groupby(['Date']).count()

hgy2mm = df_m.loc[(df_m['Time'] >= '06:00:00')&
             (df_m['Time'] <= '23:59:59')&(df_m['Sensor Glucose (mg/dL)']<54)].groupby(['Date']).count()

hgy2dm = df_m.loc[(df_m['Sensor Glucose (mg/dL)']<54)].groupby(['Date']).count()

hg1om['Time'] = hg1om['Time'].div(288).round(2)
hg1mm['Time'] = hg1mm['Time'].div(288).round(2)
hg1dm['Time'] = hg1dm['Time'].div(288).round(2)

if hg1om.empty:
    hyperglycemia1om=0
else:
    hyperglycemia1om = hg1om['Time'].mean()
if hg1mm.empty:
    hyperglycemia1mm=0
else:
    hyperglycemia1mm = hg1mm['Time'].mean()
if hg1dm.empty:
    hyperglycemia1dm=0
else:
    hyperglycemia1dm = hg1dm['Time'].mean()


hg2om['Time'] = hg2om['Time'].div(288).round(2)
hg2mm['Time'] = hg2mm['Time'].div(288).round(2)
hg2dm['Time'] = hg2dm['Time'].div(288).round(2)

if hg2om.empty:
    hyperglycemia2om=0
else:
    hyperglycemia2om = hg2om['Time'].mean()
if hg2mm.empty:
    hyperglycemia2mm=0
else:
    hyperglycemia2mm = hg2mm['Time'].mean()
if hg2dm.empty:
    hyperglycemia2dm=0
else:
    hyperglycemia2dm = hg2dm['Time'].mean()

rangepom['Time'] = rangepom['Time'].div(288).round(2)
rangepmm['Time'] = rangepmm['Time'].div(288).round(2)
rangepdm['Time'] = rangepdm['Time'].div(288).round(2)


if rangepom.empty:
    range1om=0
else:
    range1om = rangepom['Time'].mean()
if rangepmm.empty:
    range1mm=0
else:
    range1mm = rangepmm['Time'].mean()
if rangepdm.empty:
    range1dm=0
else:
    range1dm = rangepdm['Time'].mean()


rangesom['Time'] = rangesom['Time'].div(288).round(2)
rangesmm['Time'] = rangesmm['Time'].div(288).round(2)
rangesdm['Time'] = rangesdm['Time'].div(288).round(2)

if rangesom.empty:
    range2om=0
else:
    range2om = rangesom['Time'].mean()
if rangesmm.empty:
    range2mm=0
else:
    range2mm = rangesmm['Time'].mean()
if rangesdm.empty:
    range2dm=0
else:
    range2dm = rangesdm['Time'].mean()


hgy1om['Time'] = hgy1om['Time'].div(288).round(2)
hgy1mm['Time'] = hgy1mm['Time'].div(288).round(2)
hgy1dm['Time'] = hgy1dm['Time'].div(288).round(2)

if hgy1om.empty:
    hypoglycemia1om=0
else:
    hypoglycemia1om = hgy1om['Time'].mean()
if hgy1mm.empty:
    hypoglycemia1mm=0
else:
    hypoglycemia1mm = hgy1mm['Time'].mean()
if hgy1dm.empty:
    hypoglycemia1dm=0
else:
    hypoglycemia1dm = hgy1dm['Time'].mean()


hgy2om['Time'] = hgy2om['Time'].div(288).round(2)
hgy2mm['Time'] = hgy2mm['Time'].div(288).round(2)
hgy2dm['Time'] = hgy2dm['Time'].div(288).round(2)

if hgy2om.empty:
    hypoglycemia2om=0
else:
    hypoglycemia2om = hgy2om['Time'].mean()
if hgy2mm.empty:
    hypoglycemia2mm=0
else:
    hypoglycemia2mm = hgy2mm['Time'].mean()
if hgy2dm.empty:
    hypoglycemia2dm=0
else:
    hypoglycemia2dm = hgy2dm['Time'].mean()



##################################################
#getting matrix for automatic
##################################################



hg1oa = df_a.loc[(df_a['Time'] >= '00:00:00')&
             (df_a['Time'] <= '06:00:00')&(df_a['Sensor Glucose (mg/dL)']>180)].groupby(['Date']).count()

hg1ma = df_a.loc[(df_a['Time'] >= '06:00:00')&
             (df_a['Time'] <= '23:59:59')&(df_a['Sensor Glucose (mg/dL)']>180)].groupby(['Date']).count()

hg1da = df_a.loc[(df_a['Sensor Glucose (mg/dL)']>180)].groupby(['Date']).count()

hg2oa = df_a.loc[(df_a['Time'] >= '00:00:00')&
             (df_a['Time'] <= '06:00:00')&(df_a['Sensor Glucose (mg/dL)']>250)].groupby(['Date']).count()

hg2ma = df_a.loc[(df_a['Time'] >= '06:00:00')&
             (df_a['Time'] <= '23:59:59')&(df_a['Sensor Glucose (mg/dL)']>250)].groupby(['Date']).count()

hg2da = df_a.loc[(df_a['Sensor Glucose (mg/dL)']>250)].groupby(['Date']).count()

rangepoa = df_a.loc[(df_a['Time'] >= '00:00:00')&
             (df_a['Time'] <= '06:00:00')&(df_a['Sensor Glucose (mg/dL)']>=70)&
             (df_a['Sensor Glucose (mg/dL)']<=180)].groupby(['Date']).count()

rangepma = df_a.loc[(df_a['Time'] >= '06:00:00')&
             (df_a['Time'] <= '23:59:59')&(df_a['Sensor Glucose (mg/dL)']>=70)&
             (df_a['Sensor Glucose (mg/dL)']<=180)].groupby(['Date']).count()

rangepda = df_a.loc[(df_a['Sensor Glucose (mg/dL)']>=70)&
             (df_a['Sensor Glucose (mg/dL)']<=180)].groupby(['Date']).count()

rangesoa = df_a.loc[(df_a['Time'] >= '00:00:00')&
             (df_a['Time'] <= '06:00:00')&(df_a['Sensor Glucose (mg/dL)']>=70)&
             (df_a['Sensor Glucose (mg/dL)']<=150)].groupby(['Date']).count()

rangesma = df_a.loc[(df_a['Time'] >= '06:00:00')&
             (df_a['Time'] <= '23:59:59')&(df_a['Sensor Glucose (mg/dL)']>=70)&
             (df_a['Sensor Glucose (mg/dL)']<=150)].groupby(['Date']).count()

rangesda = df_a.loc[(df_a['Sensor Glucose (mg/dL)']>=70)&
             (df_a['Sensor Glucose (mg/dL)']<=150)].groupby(['Date']).count()


hgy1oa = df_a.loc[(df_a['Time'] >= '00:00:00')&
             (df_a['Time'] <= '06:00:00')&(df_a['Sensor Glucose (mg/dL)']<70)].groupby(['Date']).count()

hgy1ma = df_a.loc[(df_a['Time'] >= '06:00:00')&
             (df_a['Time'] <= '23:59:59')&(df_a['Sensor Glucose (mg/dL)']<70)].groupby(['Date']).count()

hgy1da = df_a.loc[(df_a['Sensor Glucose (mg/dL)']<70)].groupby(['Date']).count()

hgy2oa = df_a.loc[(df_a['Time'] >= '00:00:00')&
             (df_a['Time'] <= '06:00:00')&(df_a['Sensor Glucose (mg/dL)']<54)].groupby(['Date']).count()

hgy2ma = df_a.loc[(df_a['Time'] >= '06:00:00')&
             (df_a['Time'] <= '23:59:59')&(df_a['Sensor Glucose (mg/dL)']<54)].groupby(['Date']).count()

hgy2da = df_a.loc[(df_a['Sensor Glucose (mg/dL)']<54)].groupby(['Date']).count()

hg1oa['Time'] = hg1oa['Time'].div(288).round(2)
hg1ma['Time'] = hg1ma['Time'].div(288).round(2)
hg1da['Time'] = hg1da['Time'].div(288).round(2)


if hg1oa.empty:
    hyperglycemia1oa=0
else:
    hyperglycemia1oa = hg1oa['Time'].mean()
if hg1ma.empty:
    hyperglycemia1ma=0
else:
    hyperglycemia1ma = hg1ma['Time'].mean()
if hg1da.empty:
    hyperglycemia1da=0
else:
    hyperglycemia1da = hg1da['Time'].mean()


hg2oa['Time'] = hg2oa['Time'].div(288).round(2)
hg2ma['Time'] = hg2ma['Time'].div(288).round(2)
hg2da['Time'] = hg2da['Time'].div(288).round(2)

if hg2oa.empty:
    hyperglycemia2oa=0
else:
    hyperglycemia2oa = hg2oa['Time'].mean()
if hg2ma.empty:
    hyperglycemia2ma=0
else:
    hyperglycemia2ma = hg2ma['Time'].mean()
if hg2da.empty:
    hyperglycemia2da=0
else:
    hyperglycemia2da = hg2da['Time'].mean()


rangepoa['Time'] = rangepoa['Time'].div(288).round(2)
rangepma['Time'] = rangepma['Time'].div(288).round(2)
rangepda['Time'] = rangepda['Time'].div(288).round(2)


if rangepoa.empty:
    range1oa=0
else:
    range1oa = rangepoa['Time'].mean()
if rangepma.empty:
    range1ma=0
else:
    range1ma = rangepma['Time'].mean()
if rangepda.empty:
    range1da=0
else:
    range1da = rangepda['Time'].mean()


rangesoa['Time'] = rangesoa['Time'].div(288).round(2)
rangesma['Time'] = rangesma['Time'].div(288).round(2)
rangesda['Time'] = rangesda['Time'].div(288).round(2)


if rangesoa.empty:
    range2oa=0
else:
    range2oa = rangesoa['Time'].mean()
if rangesma.empty:
    range2ma=0
else:
    range2ma = rangesma['Time'].mean()
if rangesda.empty:
    range2da=0
else:
    range2da = rangesda['Time'].mean()


hgy1oa['Time'] = hgy1oa['Time'].div(288).round(2)
hgy1ma['Time'] = hgy1ma['Time'].div(288).round(2)
hgy1da['Time'] = hgy1da['Time'].div(288).round(2)

if hgy1oa.empty:
    hypoglycemia1oa=0
else:
    hypoglycemia1oa = hgy1oa['Time'].mean()
if hgy1ma.empty:
    hypoglycemia1ma=0
else:
    hypoglycemia1ma = hgy1ma['Time'].mean()
if hgy1da.empty:
    hypoglycemia1da=0
else:
    hypoglycemia1da = hgy1da['Time'].mean()



hgy2oa['Time'] = hgy2oa['Time'].div(288).round(2)
hgy2ma['Time'] = hgy2ma['Time'].div(288).round(2)
hgy2da['Time'] = hgy2da['Time'].div(288).round(2)


if hgy2oa.empty:
    hypoglycemia2oa=0
else:
    hypoglycemia2oa = hgy2oa['Time'].mean()
if hgy2ma.empty:
    hypoglycemia2ma=0
else:
    hypoglycemia2ma = hgy2ma['Time'].mean()
if hgy2da.empty:
    hypoglycemia2da=0
else:
    hypoglycemia2da = hgy2da['Time'].mean()


####################################################
#writing into a csv file
####################################################


fields=['Mode','Percentage time in hyperglycemia (CGM > 180 mg/dL) on',
        'percentage of time in hyperglycemia critical (CGM > 250 mg/dL) on',
        'percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL) on',
        'percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL) on',
        'percentage time in hypoglycemia level 1 (CGM < 70 mg/dL) on',
        'percentage time in hypoglycemia level 2 (CGM < 54 mg/dL) on',
        'Percentage time in hyperglycemia (CGM > 180 mg/dL) day',
        'percentage of time in hyperglycemia critical (CGM > 250 mg/dL) day',
        'percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL) day',
        'percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL) day',
        'percentage time in hypoglycemia level 1 (CGM < 70 mg/dL) day',
        'percentage time in hypoglycemia level 2 (CGM < 54 mg/dL) day',
        'Percentage time in hyperglycemia (CGM > 180 mg/dL) full',
        'percentage of time in hyperglycemia critical (CGM > 250 mg/dL) full',
        'percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL) full',
        'percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL) full',
        'percentage time in hypoglycemia level 1 (CGM < 70 mg/dL) full',
        'percentage time in hypoglycemia level 2 (CGM < 54 mg/dL) full',]

rows = [['Manual Mode',hyperglycemia1om,hyperglycemia2om,range1om,range2om,
         hypoglycemia1om,hypoglycemia2om,hyperglycemia1mm,hyperglycemia2mm,
         range1mm,range2mm,hypoglycemia1mm,hypoglycemia2mm,hyperglycemia1dm,
         hyperglycemia2dm,range1dm,range2dm,hypoglycemia1dm,hypoglycemia2dm],
        ['Auto Mode',hyperglycemia1oa,hyperglycemia2oa,range1om,range2oa,
         hypoglycemia1oa,hypoglycemia2oa,hyperglycemia1ma,hyperglycemia2ma,
         range1mm,range2ma,hypoglycemia1ma,hypoglycemia2ma,hyperglycemia1da,
         hyperglycemia2da,range1da,range2da,hypoglycemia1da,hypoglycemia2da]]

# name of csv file  
filename = "VenkateshKumar_results.csv"
    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    # writing the fields  
    csvwriter.writerow(fields)  
        
    # writing the data rows  
    csvwriter.writerows(rows) 