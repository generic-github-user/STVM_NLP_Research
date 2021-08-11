from keras.models import load_model
from keras import backend as K
from keras.layers import BatchNormalization, InputLayer, RepeatVector
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Flatten
from keras.models import Sequential

import math
import random
import re
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import matplotlib

# Use the AGG (Anti-Grain Geometry) backend since we are not displaying figures directly
matplotlib.use('agg')

# Load saved weights
# (if the h5 is not present in this directory, move it there or run the training script to create it)
model = load_model('model_a_1.h5')

NUM_WORDS = 0
DISCRIMINATOR_CUTOFF = 5
L2_ETA = 0.019

# Load all data in a DataFrame.
# shuffles the data to ensure good mix of postive and negative reviews


def sorted_files_list(directory):
    train_files_pos = []
    train_files_neg = []
    for filename in os.listdir(directory + "/pos/"):
        if filename != '.DS_Store':
              train_files_pos.append(filename)

    ll1 = list()
    ll2 = list()
    for f in train_files_pos:
        ss = f.split("_")
        ll1.append(int(ss[0]))
        ll2.append(ss[1])

    z = list(zip(ll1, ll2))
    z = sorted(z, key=lambda t: t[0])

    ll1, ll2 = zip(*z)

    train_files_pos = [str(ll1[i]) + "_" + ll2[i] for i in range(len(ll1))]
    print(train_files_pos[0], train_files_pos[1],
          train_files_pos[12498], train_files_pos[12499])

    for filename in os.listdir(directory + "/neg/"):
          if filename != '.DS_Store':
              train_files_neg.append(filename)

    ll1 = list()
    ll2 = list()
    for f in train_files_neg:
        ss = f.split("_")
        ll1.append(int(ss[0]))
        ll2.append(ss[1])

    z = list(zip(ll1, ll2))
    z = sorted(z, key=lambda t: t[0])

    ll1, ll2 = zip(*z)

    train_files_neg = [str(ll1[i]) + "_" + ll2[i] for i in range(len(ll1))]
    print(train_files_neg[0], train_files_neg[1],
          train_files_neg[12498], train_files_neg[12499])

    return train_files_pos, train_files_neg


# This function is reused from the model_1.py script; see that file for more information
def load_dataset_from_feat(directory, feat_file_name, use_for_predictions=False):
  data = {}
  data['reviews'] = []

  print(os.path.join(directory, feat_file_name))

  if not use_for_predictions:
    data['sentiments'] = []

  with open(os.path.join(directory, feat_file_name), 'r') as f:
    imdb_encoded_content = f.readlines()
    #if not use_for_predictions:
    # shuffle the reviews before using, only if training/testing but not if computing predictions for validation
    #random.shuffle(imdb_encoded_content)
    print("************************************")
    #print(imdb_encoded_content)
    print("************************************")
    review_encoding = []
    for review in imdb_encoded_content:
      review_encoding = review.split()
      if not use_for_predictions:
        if int(review_encoding[0]) > DISCRIMINATOR_CUTOFF:
          data['sentiments'].append(1)
        else:
          data['sentiments'].append(0)
      review_encoding.pop(0)
      data['reviews'].append(review_encoding)
  return pd.DataFrame.from_dict(data)


# See note above the load_dataset_from_feat function definition
def load_datasets_from_file():
    #  dataset = tf.keras.utils.get_file(
  #      fname='aclImdb.tar.gz',
  #      origin='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
  #      extract=True)
  # Assumes this script runs from the top directory containing the test and
  # train directory.
  global NUM_WORDS
  f = open("../imdb.vocab", "r")
  imdb_vocab = f.readlines()
  NUM_WORDS = len(imdb_vocab)
  print('Vocabulary size is: %d words' % (NUM_WORDS))
  '''
  with tf.io.gfile.GFile(os.path.join('..', 'imdb.vocab'), 'r') as f:
    imdb_vocab = f.readlines()
    NUM_WORDS = len(imdb_vocab)
    print('Vocabulary size is: %d words'%(NUM_WORDS))
    '''

  train_data = load_dataset_from_feat(
      os.path.join('..', 'train'), 'labeledBow.feat')
  #test_data = load_dataset_from_feat(
  #    os.path.join('..', 'test'), 'labeledBow.feat')
  return train_data  # , test_data


# See note above the load_dataset_from_feat function definition
def weighted_multi_hot_sequences(sequences):
    print("NUM_WORDS", NUM_WORDS)
    results = np.zeros((len(sequences['reviews']), NUM_WORDS))
    with open(os.path.join('..', 'imdbEr.txt'), 'r') as f:
        imdb_word_polarity = f.readlines()

    max = 0.0
    min = 0.0
    for review_index, review in enumerate(sequences['reviews']):
      for word in review:
        word_index, word_count = word.split(':')
        cumulative_polarity = int(word_count) * \
            float(imdb_word_polarity[int(word_index)])
        results[review_index, int(word_index)] = cumulative_polarity
        #accumulate statistics for the dataset
        if cumulative_polarity > max:
          max = cumulative_polarity
        elif cumulative_polarity < min:
          min = cumulative_polarity
    print('Dataset encoding stats: MIN = %f, MAX = %f\n' % (min, max))
    return results


print('Loading the large data set from disk...\n')
#train_data_full, test_data_full = load_datasets_from_file()
train_data_full = load_datasets_from_file()

train_files_list = []
train_files_pos, train_files_neg = sorted_files_list("../train")
train_files_list = train_files_pos + train_files_neg

print(train_data_full.shape)
print(train_data_full.head())
print(train_files_list[0])
print(train_files_list[1])


train_data = train_data_full[:][0:25000]

print(train_data.head())
print(train_data['reviews'].iloc[0])

train_data_mhe = weighted_multi_hot_sequences(train_data)
print(train_data_mhe.shape)

print(train_data_mhe[0])
print(len(train_data_mhe[0]))

TRAINING_SAMPLE = 5000
df_train = pd.read_csv('imdb_train_5k.csv')
SAMPLE_SIZE = len(df_train)
print(df_train.head())

assert SAMPLE_SIZE == TRAINING_SAMPLE, 'training sample not complete....'

train_y = np.zeros([TRAINING_SAMPLE, 1], dtype=np.int)
train_x = np.zeros([TRAINING_SAMPLE, 89527], dtype=np.float64)
list_of_train_files = []
for index in df_train.index:
    file_name = str(df_train['file'][index])
    label = int(df_train['label'][index])

    index_in_files_list = train_files_list.index(file_name)
    train_x[index] = train_data_mhe[index_in_files_list]
    train_y[index] = label
    list_of_train_files.append(file_name)

print(train_x[0])
print(train_y[0])

model.summary()

# Evaluate the model on the multi-hot encodings
prediction_results = None
prediction_results = model.predict(train_x, verbose=1)

list_of_proba = []
list_of_file_name = []
print("predict shape", prediction_results.shape)

# Store a counter of the number of correct predictions over the (test) datset
correct_pred = 0
for i in range(SAMPLE_SIZE):
    proba = prediction_results[i][0]
#     Check if model prediction is correct and update counter accordingly
    if (proba < float(0.5) and train_y[i] == 0) or (proba >= float(0.5) and train_y[i] == 1):
        correct_pred = correct_pred + 1
    list_of_proba.append(str(prediction_results[i][0]))
    list_of_file_name.append(list_of_train_files[i])

print("Accuracy: ", float(correct_pred)/float(SAMPLE_SIZE))

# write to file

d_prob = {'prob': list_of_proba}
d_files = {'file': list_of_file_name}

dd = {'prob': list_of_proba, 'file': list_of_file_name}

# Save the model's predictions to a CSV file
df = pd.DataFrame(dd, index=None)
df.to_csv('model_a_5ktrain.csv', index=False)
