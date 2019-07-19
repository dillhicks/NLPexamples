#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
import pandas as pd 
import numpy as np
import nltk, re, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from IPython.display import display, HTML
tf.logging.set_verbosity(tf.logging.ERROR)


# In[2]:


max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 175
epochs = 3


# In[ ]:





# In[3]:


df1 = pd.read_csv("imdb_labelled.txt", sep="\t", header=None)
df2 = pd.read_csv("yelp_labelled.txt", sep="\t", header=None)
df3 = pd.read_csv("amazon_cells_labelled.txt", sep="\t", header=None)
df = pd.concat([df1,df2,df3])
data = pd.concat([df1,df2,df3])
nltk.download('stopwords')
stop = stopwords.words('english')
print("Data loading is complete.")


# In[4]:


data[0] = data[0].map(lambda x: re.sub(r'\W+', ' ', x))
data[0] = data[0].map(lambda x: re.sub(r'   ', ' ', x))
data[0] = data[0].map(lambda x: re.sub(r'  ', ' ', x))
data[0] = data[0].str.lower().str.split()  
data[0] = data[0].apply(lambda x: [item for item in x if item not in stop])
print("Data cleaning is complete.")


# In[5]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data[0])
print("Fitting is complete.")
p_data = data
p_data[0] = tokenizer.texts_to_sequences(p_data[0])
print("Tokenizing is complete.")


# In[6]:


p_data_train, p_data_test = train_test_split(p_data, test_size=0.2)

p_train_list = p_data_train[0].tolist()
p_test_list = p_data_test[0].tolist()

y_train = p_data_train[1].tolist()
y_test = p_data_test[1].tolist()

x_train = keras.preprocessing.sequence.pad_sequences(p_train_list, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(p_test_list, maxlen=maxlen)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print("Created and padded test and training sets")


# In[7]:


print('Setting up model saving...')
checkpoint_path = "models/tensorflow_models/cnn_model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=2)


# In[8]:


print('Building model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[9]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
classifier = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))


# In[10]:


history_dict = classifier.history
history_dict.keys()
results = model.evaluate(x_test, y_test)
print(results)
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[11]:


plt.clf()   
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




