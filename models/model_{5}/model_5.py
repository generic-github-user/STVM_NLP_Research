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
