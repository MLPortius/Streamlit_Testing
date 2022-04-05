# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 00:41:07 2022

@author: Andriu
"""

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import warnings

# SET APP TITLE
Title = 'BITCOIN PRICE PLOTTING'
st.title(Title)

# DOWNLOAD DATA

@st.cache
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

data_load_state = st.text('Loading BTC data...')          # START LOADING
data = DownloadData('BTC-USD', '01-01-2018')
data_load_state = st.text('Done! (using @st.cache)')     # END LOADING

# RAW DATA SHOW BTC
st.subheader('BITCOIN Raw data')
st.write(data)

# PRICE PLOT BTC
st.subheader('Bitcoin Market Price') 
st.line_chart(data=data['Adj Close'])

# VOLUMEN PLOT BTC
st.bar_chart(data=data['Volume']/100000000)



data_load_state = st.text('Loading TSLA data...')          # START LOADING
data = DownloadData('BTC-USD', '01-01-2018')
data_load_state = st.text('Done! (using @st.cache)')     # END LOADING


# RAW DATA SHOW TESLA
st.subheader('TESLA Raw data')
st.write(data)

# PRICE PLOT TESLA
st.subheader('TESLA Market Price') 
st.line_chart(data=data['Adj Close'])

# VOLUMEN PLOT TESLA
st.bar_chart(data=data['Volume']/100000000)
