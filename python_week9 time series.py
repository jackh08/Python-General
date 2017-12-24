#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Exercise – Week IX
Data Programming With Python – Fall / 2017
Time Series Analysis & Forecasting
"""

# Time Series - Rolling Mean

import pandas as pd

data = {'score': [1,1,1,2,2,2,3,3,3]}
# convert it to DataFrame
df = pd.DataFrame(data)
print(df)
print(df.rolling(window=2).mean())
print(df.rolling(window=2).std())

# Dickey-Fuller Test with Python statsmodels Package
#from statsmodels.tsa.stattools import adfuller
#result = adfuller(TimeSeries)
#print result

# ACF - Auto Correlation Function
#from statsmodels.tsa.stattools import acf
#lag_acf = acf(timeseries, nlags=NL)

# PACF - Partial Auto Correlation Function
#from statsmodels.tsa.stattools import pacf
#lag_pacf= pacf(timeseries, nlags=NL)

# CCF - Cross Correlation Function
#from statsmodels.tsa.stattools import ccf
#lag_ccf= ccf(timeseries1, timeseries2)

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
import matplotlib.pylab as plt
import pandas as pd
#----------- Pre-Procesing of Timeseries with Pandas --------------#
data_frame = pd.read_csv('/Users/c_dredmond/Downloads/Airpassenger.csv', header=0)
data_frame ['Month'] = pd.to_datetime(data_frame ['Month'])
indexed_df = data_frame .set_index('Month')
timeseries = indexed_df['Passengers']
#--------------------- Some Timeseries Analysis -------------------#
lag_acf = acf(timeseries, nlags=20)
lag_pacf = pacf(timeseries, nlags=20)
rolmean = pd.rolling_mean(timeseries, window=12)
rolstd = pd.rolling_std(timeseries, window=12)
#------------------------- Plotting -------------------------------#
plt.subplot(221)
plt.plot(timeseries, color='black', label='original')
plt.plot(rolmean, color='blue', label='Rolling Mean')
plt.plot(rolstd, color='red', label='Rolling Deviation')
plt.legend(loc='best')
plt.title('Original Data, Rolling Mean & Standard Deviation')
plt.subplot(223)
plt.plot(lag_pacf, color='orange', label='auto correlation func')
plt.legend(loc='best')
plt.title('Partial Auto Correlation Function')
plt.subplot(224)
plt.plot(lag_acf, color='green', label='partial auto correlation func ')
plt.legend(loc='best')
plt.title('Auto Correlation Function')
plt.show()

### ARIMA - Auto-Regressive Integrated Moving Average

from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
# ============ Creating a random timeseries ========== #
counts= np.arange(1, 21) + 0.2 * (np.random.random(size=(20,)) - 0.5)
start = pd.datetime.strptime("1 Nov 16", "%d %b %y")
daterange = pd.date_range(start, periods=20)
table = {"count": counts, "date": daterange}
# ================= Pre-processing ====================#
data = pd.DataFrame(table)
data.set_index("date", inplace=True)
print(data)
# =============== Setting up ARIMA model ============= #
model = ARIMA(data[0:len(data)-1], (1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.forecast())

### Examples

from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
#------ Pre-Procesing of imported Dataset with Pandas ---------#
df_all = pd.read_csv('/Users/c_dredmond/Downloads/GlobalLandTemperaturesByCountry.csv', header=0)
# Dropping 'AvergeTemperatureUncertainty' column-This column is useless for our case !!
df_all_reduced = df_all.drop('AverageTemperatureUncertainty', axis=1)
# Filtering 'France' as country
df_france = df_all_reduced [df_all_reduced .Country == 'France']
# Dropping 'Country' column
df_france = df_france.drop('Country', axis=1)
# Converting 'Date' column to a datetime format index to access data based on dates.
df_france.index = pd.to_datetime(df_france['Date'])
# dropping 'Date' column. We use dates as index from now on, so we don't need them as an extra
df_france = df_france.drop('Date', axis=1)
# Filtering data starting from 1950-01-01
df_france = df_france.ix['1960-01-01':]
# Sorting index in an ascending way.
df_france = df_france.sort_index()
# Replacing 'NaN' values with the last valid observation
df_france.AverageTemperature.fillna(method='pad', inplace=True)
# Extract Out the Timeseries values part
timeseries = df_france.AverageTemperature
#----------------------------- ARIMA ---------------------------------------
size = int(len(timeseries) - 9)
train, test = timeseries[0:size], timeseries[size:len(timeseries)]
previous_samples = [x for x in train]
for t in range(len(test)):
    model = ARIMA(previous_samples, order=(10, 0, 1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0] #get the first element which is the forecast, we don't need the rest
    obs = test[t]
    previous_samples.append(obs)
    print('predicted=%f, expected=%f' % ((yhat), (obs)))
 