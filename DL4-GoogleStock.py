#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


# In[33]:


dataset_train = pd.read_csv('Google_Stock_Price_Train1.csv')


# In[34]:


dataset_train.head()


# In[35]:


#keras only takes numpy array
training_set = dataset_train.iloc[:, 1: 2].values


# In[36]:


training_set.shape


# In[37]:


plt.figure(figsize=(18, 8))
plt.plot(dataset_train['Open'])
plt.title("Google Stock Open Prices")
plt.xlabel("Time (oldest -> latest)")
plt.ylabel("Stock Open Price")
plt.show()


# In[38]:


import os
if os.path.exists('config.py'):
 print(1)
else:
 print(0)


# In[39]:



sc = MinMaxScaler(feature_range = (0, 1))
#fit: get min/max of train data
training_set_scaled = sc.fit_transform(training_set)


# In[40]:


## 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
 X_train.append(training_set_scaled[i-60: i, 0])
 y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# In[41]:


X_train.shape


# In[42]:


y_train.shape


# In[43]:


X_train = np.reshape(X_train, newshape =
 (X_train.shape[0], X_train.shape[1], 1))


# In[44]:


X_train.shape


# In[67]:


regressor = Sequential()
#add 1st lstm layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(rate = 0.2))
##add 2nd lstm layer: 50 neurons
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))
##add 3rd lstm layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))
##add 4th lstm layer
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(rate = 0.2))
##add output layer
regressor.add(Dense(units = 1))


# In[68]:


regressor.summary()


# In[69]:


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[70]:


regressor.fit(x = X_train, y = y_train, batch_size = 32, epochs = 100)


# In[71]:


dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')


# In[72]:


dataset_test.head()


# In[73]:


#keras only takes numpy array
real_stock_price = dataset_test.iloc[:, 1: 2].values
real_stock_price.shape


# In[74]:


#vertical concat use 0, horizontal uses 1
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']),
 axis = 0)
##use .values to make numpy array
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values


# In[75]:


#reshape data to only have 1 col
inputs = inputs.reshape(-1, 1)
#scale input
inputs = sc.transform(inputs)
len(inputs)


# In[76]:



X_test = []
for i in range(60, len(inputs)):
 X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
#add dimension of indicator
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[77]:


predicted_stock_price = regressor.predict(X_test)


# In[78]:


#inverse the scaled value
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[80]:


##visualize the prediction and real price
plt.plot(real_stock_price, label = 'Real price')
plt.plot(predicted_stock_price, label = 'Predicted price')
plt.title('Google price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




