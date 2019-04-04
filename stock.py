#!/usr/bin/python
#coding:utf-8

import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get("EURONEXT/ACPH", authtoken="sK4W-_oyzEruw8RVZVPS")

df = df[['Open', 'High', 'Low', 'Last', 'Volume']]

df['HL_PCT'] = (df['High']-df['Last']) / df['Last'] * 100.0
df['PCT_change'] = (df['Last']-df['Open']) / df['Open'] * 100.0

df = df[['Last', 'HL_PCT', 'PCT_change', 'Volume']]
forecast_col = 'Last'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))

df['Label'] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(['Label'], 1))
x = preprocessing.scale(x)
x = x[:-forecast_out]
x_later = x[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['Label'])
y = np.array(df['Label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = LinearRegression()
# clf = svm.SVR()               #support vendor machine 支持向量机，一种分类算法
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

forecast_set = clf.predict(x_later)
# print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


print(df['Last'])

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# print(df['Last'])
# df['Forecast'].plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()


df['Last'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

