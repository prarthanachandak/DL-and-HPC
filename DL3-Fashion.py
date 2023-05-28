#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical


# In[3]:


fashion_train_df = pd.read_csv('fashion-mnist_train.csv')
fashion_test_df = pd.read_csv('fashion-mnist_test.csv')


# In[4]:


fashion_train_df.shape


# In[5]:


fashion_train_df.columns


# In[6]:


#unique values from label
print(set(fashion_train_df['label']))


# In[7]:


#drop labels and find max value across all rows
print([fashion_train_df.drop(labels='label',axis=1).min(axis=1).min(),fashion_train_df.drop(labels='label',axis=1).max(axis=1).max()])


# In[8]:


fashion_train_df.head()


# In[9]:


fashion_test_df.head()


# #Visualization

# In[10]:


training = np.asarray(fashion_train_df, dtype='float32') #convert to np array
height = 10
width =10
#for grid
fig,axes = plt.subplots(nrows=width,ncols=height,figsize=(17,17))
axes = axes.ravel() #flattening aces into 1-d array
n_train = len(training)

for i in range(0, height*width):
  index = np.random.randint(0,n_train)
  axes[i].imshow(training[index,1:].reshape(28,28)) #display image using imshow, training[]->all cols(pixel values except first (label))
  axes[i].set_title(int(training[index,0]), fontsize=8)
  axes[i].axis('off')

plt.subplots_adjust(hspace=0.5, wspace=0.5)


# #Preprocess Data 

# In[13]:


training = np.asarray(fashion_train_df, dtype='float32')
X_train = training[:,1:].reshape([-1,28,28,1]) #exclude 1st column, -1 unknown samples, 28 height width, 1color channels greyscale
X_train = X_train/255 #normalize pixel values
y_train = training[:,0] #extract labels

testing = np.asarray(fashion_test_df, dtype='float32')
X_test = testing[:,1:].reshape([-1,28,28,1])
X_test = X_test/255
y_test = testing[:,0]


# In[14]:


X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=50)


# In[15]:


print(X_train.shape,X_val.shape,X_test.shape)
print(y_train.shape,y_val.shape,y_test.shape)


# In[16]:


cnn_model = Sequential() #stack layers
#convolutional 2d layer
#filters = weights applied to the input image - to detect features or patters in input
#kernelsize = filter dimension
#reLU - max(0,x) - x is input val
cnn_model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(28,28,1), activation='relu'))
# The max pooling layer divides the input into rectangular regions (usually squares) without overlapping. The most common pooling region size is (2, 2), but other sizes can be used as well.
cnn_model.add(MaxPooling2D(pool_size = (2,2)))
#Dropout is a regularization technique that randomly sets a fraction of input units to 0 during training, which helps prevent overfitting.
cnn_model.add(Dropout(rate=0.3))
#The flatten layer is responsible for flattening the multi-dimensional output from the previous layer into a 1D vector, which can be fed into a fully connected layer.
cnn_model.add(Flatten())
#fully connected layer
cnn_model.add(Dense(units=32,activation='relu'))
#sigmoid - takes real values input and squashes it between 0 and 1
cnn_model.add(Dense(units=10, activation='sigmoid'))


# In[17]:


cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')
cnn_model.summary()
#It computes adaptive learning rates for each parameter, allowing the model to dynamically adjust the learning rate during training.
#multi-class classification problems, where the target labels are integers. It calculates the cross-entropy loss between the predicted class probabilities and the true labels, taking into account the sparsity of the labels 


# In[18]:



cnn_model.fit(x=X_train, y=y_train, batch_size=512, epochs=2, verbose=1, validation_data=(X_val,y_val))


# In[19]:


eval_result = cnn_model.evaluate(X_test,y_test)
print("Accuracy: ",eval_result)
#calculates loss function


# In[21]:


predict_x=cnn_model.predict(X_test)
classes_x=np.argmax(predict_x,axis=1) #maximum alues along axis 1
classes_x


# In[23]:


height =10
width = 10

fig,axes = plt.subplots(nrows = width, ncols=height,figsize=(20,20))
axes = axes.ravel()
for i in range(0,height*width):
  index = np.random.randint(len(classes_x))
  axes[i].imshow(X_test[index].reshape((28,28)))
  axes[i].set_title("True Class : {1} \nPrediction : {1}".format(y_test[index], classes_x[index]))
  axes[i].axis('off')
plt.subplots_adjust(hspace=0.9, wspace=0.5)
  


# In[ ]:




