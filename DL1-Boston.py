#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# In[5]:


boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df.head()


# In[91]:


#dataframe from dataset
df['PRICE'] = boston.target


# In[1]:


#features and target
X = df.drop('PRICE', axis=1)
y = df['PRICE']


# In[93]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[94]:


# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[95]:


#deep neural network model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))


# In[96]:


#compile the model
model.compile(loss='mean_squared_error', optimizer='adam')


# In[97]:


#contains info about training proces
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)
#epochs = 100 means that model will go through the training data 100 times during training process
#batch size = 32 means number of samples through each step before updating weights
#model.fit - performs training process


# In[98]:


y_pred = model.predict(X_test)


# In[99]:


y_pred[0]


# In[100]:


mse = mean_squared_error(y_test, y_pred) #average square difference
r2 = r2_score(y_test, y_pred) 
print(f"Mean Squared Error: {mse}")
print(f"R-Squared Score: {r2}") # 1 is perfect fit


# In[101]:


plt.plot(history.history['loss']) #training loss - decrease
plt.plot(history.history['val_loss']) #validation loss - decrease or stable
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




