import numpy as np
import pandas as pd
import psutil as psutil
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from accuracy import Accuracy
import statsmodels.api as sm
import math
from core.data_processor import DataLoader
import tracemalloc
import linecache
import os

from utils import Timer

tracemalloc.start()



# Global variables
FILE_LOCATION = r"C:\Users\Sergio\Desktop\ARIMA\Stock-Price-Prediction-Using-ARIMA-master/Stocks\AMZN/Minutely.csv"
TRAINING_PERCENT = 0.85

#Read file
df = pd.read_csv(FILE_LOCATION, index_col=['Date'])
df.index=pd.to_datetime(df.index, format='%d-%m-%Y %H:%M')

'''
plt.xlabel('Date')
plt.ylabel('Close')
plt.plot(df)
plt.show()
'''

# Divide dataset for training and testing
training = math.floor(int(len(df) * TRAINING_PERCENT))
df2 = df[:training]
df3 = df[training+1:]

rolling_mean = df.rolling(window = 12).mean()
rolling_std = df.rolling(window = 12).std()
'''
plt.plot(df, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()
'''

print("Original data p-values")
result = adfuller(df2['Close'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

df_log = np.log(df2)
df_log2 = np.log(df3)

'''
plt.plot(df)
plt.show()
'''

def get_stationarity(timeseries, nax):
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # Dickey–Fuller test:
    result = adfuller(timeseries['Close'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

# Option 1: Subtract mean
rolling_mean = df_log.rolling(window=12).mean()
df_log_minus_mean = df_log - rolling_mean
df_log_minus_mean.dropna(inplace=True)
#get_stationarity(df_log_minus_mean,0)

#Option 2: Apply exponential decay
rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
df_log_exp_decay = df_log - rolling_mean_exp_decay
df_log_exp_decay.dropna(inplace=True)
#get_stationarity(df_log_exp_decay,1)

#Option 3: Subtract point by the one that preceded it
df_log_shift = df_log - df_log.shift()
df_log_shift.dropna(inplace=True)
get_stationarity(df_log_shift,2)

'''
plt.title("Normalized data with shift")
plt.plot(df_log_shift)
plt.show()
'''

#Option 4: Normalize
data = DataLoader(FILE_LOCATION, TRAINING_PERCENT, ["Close"])
x, y=data.get_train_data(50,True)


'''
# Check acf and pacf values
fig2 = plt.figure(figsize=(12,8))
ax1 = fig2.add_subplot(211)
fig2 = sm.graphics.tsa.plot_acf(df_log_shift, lags=30, ax=ax1)
ax2 = fig2.add_subplot(212)
fig2 = sm.graphics.tsa.plot_pacf(df_log_shift, lags=30, ax=ax2)
plt.show()

'''

#Sort stuff
df_log = df_log.sort_values(by="Date")
df_log_shift = df_log_shift.sort_values(by="Date")
df2 = df2.sort_values(by="Date")


#Execute ARIMA (use order=(3, 1, 3)) for SP500 Daily)
print("Modeling started")
timer = Timer()
timer.start()
decomposition = seasonal_decompose(df_log,model='additive',period=1)
model = ARIMA(df_log, order=(2, 1, 2))
results = model.fit(disp=0)
timer.stop()


'''
plt.plot(results.fittedvalues, color='red')
plt.plot(df_log_shift)
plt.show()

'''

#Predict
predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

print("Prediction started")
timer = Timer()
timer.start()
predictions_ARIMA_log = pd.Series(df_log['Close'].iloc[0], index=df_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
timer.stop()

results.plot_predict(1,df2.size+df3.size)

# Get Predicted data from the plot
ax = plt.gca()
line = ax.lines[0]
predicted = line.get_ydata()[training+1:]

accuracy = Accuracy()
exp_predicted = np.exp(predicted)


print()
print("Predicted data:")
final_expected, final_predicted = accuracy.print_all(df3['Close'].to_numpy(), exp_predicted, False)
print()

plt.show()

plt.plot(df3['Close'].to_numpy(), color='blue', label='Original')
plt.plot(exp_predicted, color='red', label='Predicted')
plt.show()


snapshot = tracemalloc.take_snapshot()
display_top(snapshot)


