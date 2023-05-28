#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[4]:


# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
data = pd.read_csv(url, header=None)


# In[5]:


# Split features and labels
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values


# In[6]:


# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# In[7]:


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(16,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])


# In[9]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[10]:


model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)


# In[16]:


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')


# In[ ]:




