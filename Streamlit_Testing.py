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

# SET APP TITLE
Title = 'BITCOIN PRICE PLOTTING'
st.title(Title)

# DOWNLOAD DATA

#@st.cache
def DownloadData(SYMBOL, PARAM):
    RAWDATA = yf.download(SYMBOL)
    DATA = RAWDATA[PARAM]
    return DATA

data_load_state = st.text('Loading data...')   # START LOADING
data = DownloadData('BTC-USD','Adj Close')
data_load_state = st.text('Loading data...done!')     # END LOADING



