# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 00:41:07 2022

@author: Andriu
"""

# IMPORT LIBRARIES
from datetime import date
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import warnings


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# SET APP TITLE
Title = 'BITCOIN RETURN PREDICTION (SGD REGRESSION)'
st.title(Title)



#%% ------------------------ DATA DOWNLOADING ---------------------------------
# DOWNLOAD DATA

#@st.cache
def DownloadData(SYMBOL, START):
    RAWDATA = yf.download(SYMBOL)
    PRICEDATA = RAWDATA['Adj Close']
    VOLUMEDATA = RAWDATA['Volume']
    DATA = pd.concat([PRICEDATA,VOLUMEDATA],axis=1)
    DATES = DATA.index
    DATES.strftime('%d-%m-%Y')
    FDATA = pd.DataFrame(index=DATES)
    FDATA = FDATA.loc[START:,:]
    FDATA.index = pd.to_datetime(FDATA.index)
    DATA.index = DATES
    DATA.index = pd.to_datetime(DATA.index)
    FDATA = pd.concat([FDATA,DATA],join='inner',axis=1)
    return FDATA



#FECHA DE ANALISIS
data_load_state = st.text('Select Data Year')
STARTYEAR = st.slider('Starting Year', 15, 22, 17)
STARTDATE = '01-01-20'+str(STARTYEAR)
 
data_load_state = st.text('Loading BTC data...')          # START 

btc = DownloadData('BTC-USD', STARTDATE)
btcdates = list(btc.index)
btcdates.pop(len(btcdates)-1)
todaydate = date.today()
todaydate = todaydate.strftime("%Y-%m-%d %H:%M:%S")
todaydate = datetime.strptime(todaydate,"%Y-%m-%d %H:%M:%S")
btcdates.append(todaydate)
btc.index = btcdates

data_load_state = st.text('Done! (using @st.cache)')     # END LOADING

# RAW DATA SHOW BTC
if st.checkbox('Show BTC raw data'):
    st.subheader('BITCOIN Raw data')
    st.write(btc)

# PRICE PLOT BTC
st.subheader('Bitcoin Market Price') 
st.line_chart(data=btc['Adj Close'])

# VOLUMEN PLOT BTC
st.bar_chart(data=btc['Volume']/100000000)

data_load_state = st.text('Loading TSLA data...')          # START LOADING
tsla = DownloadData('TSLA', STARTDATE)
data_load_state = st.text('Done! (using @st.cache)')     # END LOADING

# RAW DATA SHOW TESLA
if st.checkbox('Show TSLA raw data'):
    st.subheader('TESLA Raw data')
    st.write(tsla)

# PRICE PLOT TESLA
st.subheader('TESLA Market Price') 
st.line_chart(data=tsla['Adj Close'])

# VOLUMEN PLOT TESLA
st.bar_chart(data=tsla['Volume']/100000000)


#%% ------------------------ DATA CALCULATION ---------------------------------

st.subheader('PREPROCESS DATA')

data_load_state = st.text('Processing Data...')          # START 

#Prices
btcp = pd.DataFrame(btc['Adj Close'])
tslap = pd.DataFrame(tsla['Adj Close'])
btcp.columns = ['p_btc']
tslap.columns = ['p_tsla']
prices = pd.concat([btcp,tslap],axis=1)
prices.ffill(inplace=True)
prices.dropna(axis=0, inplace=True)

st.text('...Prices Done!')

#Return Calculate
returns = np.log(prices/prices.shift(1))
returns.columns = ['r_btc','r_tsla']

st.text('...Returns Done!')

returnsl1 = returns.shift(1)
returnsl1.columns = ['rl1_btc','rl1_tsla']

returnsl2 = returns.shift(2)
returnsl2.columns = ['rl2_btc','rl2_tsla']

st.text('...Return Lags Done!')


#Moving Average 
mmap = returns.rolling(3).mean().shift(1)
mmap.columns = ['ma_btc','ma_tsla']

st.text('...Moving Average Done!')

#Volumes
btcv = pd.DataFrame(btc['Volume'])
tslav = pd.DataFrame(tsla['Volume'])
btcv.columns = ['v_btc']
tslav.columns = ['v_tsla']
volumes = pd.concat([btcv,tslav],axis=1)
volumes.ffill(inplace=True)
volumes.dropna(axis=0, inplace=True)

volumesl1 = volumes.shift(1)
volumesl1.columns = ['vl1_btc','vl1_tsla']

st.text('...Volume Lags Done!')

#CONCATDATA
YDATA = returns['r_btc']
YDATA.dropna(axis=0,inplace=True)

XDATA = pd.concat([returnsl1,
                   returnsl2,
                   mmap,
                   volumesl1], axis = 1)

XDATA.dropna(axis=0,inplace=True)

st.subheader('BITCOIN HISTORIC RETURN')
st.write(YDATA)

st.subheader('PREDICT FEATURES DATA')
st.write(XDATA)


#PREPROCESS DATA
Scaler = StandardScaler()
XSDATA = pd.DataFrame(Scaler.fit_transform(XDATA))
XSDATA.index = XDATA.index
XSDATA.columns = XDATA.columns

YSDATA = pd.DataFrame(index=XSDATA.index)
YSDATA = pd.concat([YSDATA,YDATA],axis=1,join='inner')

st.text('...Data scalation Done!')

#TRAIN TEST SPLIT
Xtr, Xts, Ytr, Yts = train_test_split(XSDATA, YSDATA, test_size=0.2, 
                                      random_state=14, shuffle=False)


st.text('...Train-Test Split Done!')

#GET TODAY DATA
TomorrowDate = date.today() + timedelta(days=1)
TomorrowDate = TomorrowDate.strftime("%Y-%m-%d %H:%M:%S")
TomorrowDate = datetime.strptime(TomorrowDate,"%Y-%m-%d %H:%M:%S")

print(TomorrowDate)

TDates = list(btc.index)
TDates = [TDates[len(TDates)-3],TDates[len(TDates)-2],
          TDates[len(TDates)-1]]

print(TDates)

#Retornos
Xpr = pd.DataFrame(index=TDates)
Xpr = pd.concat([Xpr,returns],join='inner',axis=1)
Xpr = pd.concat([Xpr,pd.DataFrame(index=[TomorrowDate])],axis=0)
print(Xpr)

#Rezago1
Xprl1 = Xpr.shift(1)
Xprl1.columns = ['rl1_btc','rl1_tsla']
print(Xprl1)

#Rezago2
Xprl2 = Xpr.shift(2)
Xprl2.columns = ['rl2_btc','rl2_tsla']
print(Xprl2)

#MediaMovil
Xpma = Xpr.rolling(3).mean().shift(1)
Xpma.columns = ['ma_btc','ma_tsla']
print(Xpma)

#Volumnes
Xpv = pd.DataFrame(index=TDates)
Xpv = pd.concat([Xpv,volumes],join='inner',axis=1)
Xpv = pd.concat([Xpv,pd.DataFrame(index=[TomorrowDate])],axis=0)
Xpv = Xpv.shift(1)
Xpv.columns = ['vl1_btc','vl1_tsla']

#Concat Data
Xp = pd.concat([Xprl1, Xprl2,Xpma, Xpv],axis=1)
Xp.dropna(axis=0,inplace=True)

#Data Scalate
XPDATA = pd.DataFrame(Scaler.transform(Xp))
XPDATA.index = Xp.index
XPDATA.columns = Xp.columns


data_load_state = st.text('...Done! Processing Data')          # END 



#%% ------------------------ SGD REGRESOR MODEL -------------------------------

st.subheader('TRAINING SGD MODEL')

data_load_state = st.text('Training Model...')     

LRate = st.number_input('Learning_Rate', value=0.01)
Iter = st.number_input('Max Iterations', value=1000)
Pty = st.selectbox('Penalty Type', ['l2','l1'])

#SGD TRAINING MODEL
SGD = SGDRegressor(loss='squared_error', penalty=Pty, alpha=0.0001,  
                                  fit_intercept=True, max_iter=Iter, tol=0.001, 
                                  shuffle=False, verbose=0, random_state=14, 
                                  learning_rate='constant', eta0=LRate)

SGD.fit(Xtr, Ytr.values.ravel())

YPtr = SGD.predict(Xtr)
YPts = SGD.predict(Xts)

data_load_state = st.text('Model Trained...')   

#SGD MODEL METRICS
MSEtr = mean_squared_error(Ytr,YPtr)
MSEts = mean_squared_error(Yts,YPts)

print('Training MSE: '+str(np.round(MSEtr,4)))
print('Testing MSE: '+str(np.round(MSEts,4)))

data_load_state = st.text('SGD Model Trained...')
st.metric('Training MSE',np.round(MSEtr,4))
st.metric('Testing MSE:',np.round(MSEts,4))

#Prediction

st.subheader('PREDICTING DATA')

data_load_state = st.text('Predicting...')  

st.text('Todays Data')  
st.write(Xp)
Yp = SGD.predict(XPDATA)
data_load_state = st.text('... Prediction Done!') 

st.text('Tomorrow Return')
st.metric('Expected', np.round(Yp[0],4))
    
st.subheader('RECOMMENDED STRATEGY')
if Yp[0]>0:
    st.text('Comprar!')
elif Yp[0]<0:
    st.text('Vender!')
else:
    st.text('Indiferente!')



