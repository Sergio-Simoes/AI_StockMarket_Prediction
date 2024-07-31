
# coding: utf-8

# In[1]:


import pandas
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import math
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA


# In[5]:


variables = pandas.read_csv(r"C:\Users\Sergio\Desktop\ARIMA\Stock-Price-Prediction-Using-ARIMA-master/aapl.csv")
Close =variables ["Close"]
Close


# In[3]:


lnClose=np.log(Close)
lnClose
plt.plot(lnClose)
plt.show()


# In[4]:


# In[12]:


price_matrix=lnClose.as_matrix()
model = ARIMA(price_matrix, order=(0,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[13]:


predictions=model_fit.predict(122, 127, typ='levels')
predictions
predictionsadjusted=np.exp(predictions)
predictionsadjusted
print(predictionsadjusted)

