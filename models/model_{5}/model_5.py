#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Convert this notebook to a Python script and save as a new file
get_ipython().system('jupyter nbconvert --to script model_5.ipynb')


# In[15]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import string


# In[93]:


charlist = string.ascii_lowercase + ' \n' + string.digits
print(charlist)


# In[78]:


dataset = []
labels = []

# TODO: move into function
# Loop through categories
for category, value in [('neg', 0), ('pos', 1)]:
#     Get list of files in dataset and truncate
    filenames = glob.glob(f'../train/{category}/*.txt')[:100]
    for i, f in enumerate(filenames):
        with open(f) as textfile:
            p = string.punctuation
#             Get text data and remove punctuation
            content = textfile.read().translate(str.maketrans(p, ' '*len(p)))
#             Convert to lowercase
            content = content.lower()
#             Replace spans of whitespace with single spaces
            content = ' '.join(content.split())
    
#             Add x and y to corresponding data lists
            dataset.append(content)
            labels.append(value)
            if i < 5:
                print(i, content[:200]+'...', '\n')

def encode_char(c):
    try:
        return charlist.index(c)
    except:
        return len(charlist)
        
encoded = []
max_len = max([len(d) for d in dataset])
print(max_len)
for i, d in enumerate(dataset):
    s = ' ' * (max_len - len(d))
    d = d + s
#     try:
    encoded.append(tf.one_hot([encode_char(c) for c in d], len(charlist)+1))
#     except Exception as e:
#         print('Encoding error:')
#         print(d, e, '\n')
    dataset[i] = d


# In[79]:


A = np.array(encoded)
encoded = A
print(A.shape, A.size)

labels = np.array(labels)
print(labels.shape)


# In[87]:


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(20, input_shape=(5850, 39)),
    tf.keras.layers.Dense(40, activation=tf.keras.activations.tanh),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])
model.summary()
