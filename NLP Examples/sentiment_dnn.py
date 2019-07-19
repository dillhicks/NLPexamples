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
from IPython.display import display, HTML
tf.logging.set_verbosity(tf.logging.ERROR)


# Loading data. Uses 3 datasets from IMDB, Yelp, and Amazon in order to get a diverser dataset. Also downloading datset of stopwords in order to remove words of non-interest

# In[2]:


df1 = pd.read_csv("imdb_labelled.txt", sep="\t", header=None)
df2 = pd.read_csv("yelp_labelled.txt", sep="\t", header=None)
df3 = pd.read_csv("amazon_cells_labelled.txt", sep="\t", header=None)
data = pd.concat([df1,df2,df3])
nltk.download('stopwords')
stop = stopwords.words('english')


# Cleaning up the data to remove unwanted data, removing special characters, extra spaces, and stop words that exist in the ntlk set

# In[3]:


data[0] = data[0].map(lambda x: re.sub(r'\W+', ' ', x))
data[0] = data[0].map(lambda x: re.sub(r'   ', ' ', x))
data[0] = data[0].map(lambda x: re.sub(r'  ', ' ', x))
data[0] = data[0].str.lower().str.split()  
data[0] = data[0].apply(lambda x: [item for item in x if item not in stop])


# In[4]:


vocab_size = 2000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(data[0])
print("Fitting is complete.")
p_data = data
p_data[0] = tokenizer.texts_to_sequences(data[0])
print("Tokenizing is complete.")


# A bit of analytics on the words:

# In[ ]:





# In[5]:


p_data_train, p_data_test = train_test_split(p_data, test_size=0.2)
p_train_list = p_data_train[0].tolist()
p_test_list = p_data_test[0].tolist()
train_labels = p_data_train[1].tolist()
test_labels = p_data_test[1].tolist()
train_data = keras.preprocessing.sequence.pad_sequences(p_train_list, maxlen=14, padding='post')
test_data = keras.preprocessing.sequence.pad_sequences(p_test_list, maxlen=14, padding='post')
print("Created and padded test and training sets")


# In[6]:


print('Setting up model saving...')
checkpoint_path = "models/tensorflow_models/dnn_model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=1)


# In[7]:


## print("Vocabulary size =",vocab_size)
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size+1, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[8]:


model.summary()
classifier = model.fit(train_data,
                    train_labels,
                    epochs=10,
                    batch_size=52,
                    validation_data=(test_data, test_labels),
                    verbose=0)


# In[9]:


history_dict = classifier.history
history_dict.keys()
results = model.evaluate(test_data, test_labels)
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


# In[10]:


plt.clf()   # clear figure
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

