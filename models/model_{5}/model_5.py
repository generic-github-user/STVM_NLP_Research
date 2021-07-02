#!/usr/bin/env python
# coding: utf-8

# In[94]:


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


# In[108]:


def load_data(start=0, stop=100, log=False):
    dataset = []
    labels = []
    
    # Loop through categories
    for category, value in [('neg', 0), ('pos', 1)]:
    #     Get list of files in dataset and truncate
        filenames = glob.glob(f'../train/{category}/*.txt')[start:stop]
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
                if i < 5 and log:
                    print(i, content[:200]+'...', '\n')
    
    return dataset, labels


# In[109]:


def encode_char(c):
    try:
        return charlist.index(c)
    except:
        return len(charlist)


# In[110]:


def prep_data(*args, **kwargs):
    X, Y = load_data(*args, **kwargs)
    encoded = []
    max_len = max([len(d) for d in X])
    print(max_len)
    for i, d in enumerate(X):
        s = ' ' * (max_len - len(d))
        d = d + s
    #     try:
        encoded.append(tf.one_hot([encode_char(c) for c in d], len(charlist)+1))
    #     except Exception as e:
    #         print('Encoding error:')
    #         print(d, e, '\n')
        X[i] = d

    X = np.array(encoded)
#     encoded = A
    print(X.shape, X.size)

    Y = np.array(Y)
    print(Y.shape)
    
    return X, Y


# In[114]:


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(20, input_shape=(5850, 39)),
    tf.keras.layers.Dense(40, activation=tf.keras.activations.tanh),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])
model.summary()


# In[115]:


model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
    loss=tf.keras.losses.binary_crossentropy,
)


# In[ ]:


X, Y = prep_data(0, 100)
model.fit(X, Y, epochs=5, batch_size=16)


# In[91]:


model(encoded[:1])

