#!/usr/bin/env python
# coding: utf-8

# In[141]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[142]:


(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()


# In[173]:


len(x_train)


# In[174]:


x_train[0].shape


# In[145]:


x_train[0]


# In[175]:


plt.matshow(x_train[7])


# In[176]:


y_train[7]


# In[177]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# In[178]:


x_train[0]


# In[182]:


x_train_flattend = x_train.reshape(len(x_train),28*28) / 255.0
x_test_flattend = x_test.reshape(len(x_test),28*28) / 255.0


# In[181]:


x_train_flattend.shape
x_test_flattend.shape


# In[183]:


x_test_flattend[0]


# In[184]:


x_train[0]


# In[185]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,),activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train_flattend,y_train, epochs=5)


# In[186]:


model.evaluate(x_test_flattend,y_test)


# In[187]:


plt.matshow(x_test[1])


# In[193]:


y_predicted = model.predict(x_test_flattend)
y_predicted[1]


# In[199]:


predicted_class = np.argmax(y_predicted[1])
print(predicted_class)


# In[201]:


y_test[:5]


# In[202]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[204]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[206]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[207]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,),activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train_flattend,y_train, epochs=5)


# In[209]:


model.evaluate(x_test_flattend,y_test)


# In[211]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:





# In[213]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train,y_train, epochs=5)


# In[215]:


model.evaluate(x_test,y_test)


# In[216]:


y_predicted = model.predict(x_test)
y_predicted[1]


# In[219]:


predicted_class=np.argmax(y_predicted[0])
print(predicted_class)


# In[220]:


plt.matshow(x_test[0])


# In[221]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

