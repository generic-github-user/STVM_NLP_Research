#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('jupyter nbconvert --to script model_5.ipynb')


# In[15]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import string


# In[44]:


charlist = string.ascii_lowercase + ' \n' + string.digits
print(charlist)


# In[78]:


dataset = []
labels = []

for category, value in [('neg', 0), ('pos', 1)]:
    filenames = glob.glob(f'../train/{category}/*.txt')[:100]
    for i, f in enumerate(filenames):
        with open(f) as textfile:
            p = string.punctuation
            content = textfile.read().translate(str.maketrans(p, ' '*len(p)))
            content = content.lower()
            content = ' '.join(content.split())
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
