import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, InputLayer, RepeatVector
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Flatten
from tensorflow.keras.models import Sequential
import math
import random
import re
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib
matplotlib.use('agg')


NUM_WORDS = 0
DISCRIMINATOR_CUTOFF = 5
L2_ETA = 0.019
empty_indices = np.empty((0, 2), dtype=np.int64)

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


def load_datasets_from_file():
    #  dataset = tf.keras.utils.get_file(
  #      fname='aclImdb.tar.gz',
  #      origin='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
  #      extract=True)
  # Assumes this script runs from the top directory containing the test and
  # train directory.
  global NUM_WORDS
  f = open("../imdb.vocab", "r", encoding="UTF-8")
  imdb_vocab = f.readlines()
  NUM_WORDS = len(imdb_vocab)
  print('Vocabulary size is: %d words' % (NUM_WORDS))
  

  train_data = load_dataset_from_feat(
      os.path.join('..', 'train'), 'labeledBow.feat')
  #test_data = load_dataset_from_feat(
  #    os.path.join('..', 'test'), 'labeledBow.feat')
  return train_data  # , test_data


def weighted_multi_hot_sequences(sequences):
    print("NUM_WORDS", NUM_WORDS)
#     results = np.zeros((len(sequences['reviews']), NUM_WORDS))
    
    with open(os.path.join('..', 'imdbEr.txt'), 'r') as f:
        imdb_word_polarity = f.readlines()


    max = 0.0
    min = 0.0
    indices = []
    values = []
    for review_index, review in enumerate(sequences['reviews']):
      for word in review:
        word_index, word_count = word.split(':')
        cumulative_polarity = int(word_count) * \
            float(imdb_word_polarity[int(word_index)])
#         results[review_index, int(word_index)] = cumulative_polarity
#         maybe assemble the lists of indices + values then generate the corresponding EagerTensor objects in a single TF operation?
#         results.indices = tf.concat([results.indices, np.expand_dims([review_index, int(word_index)], 0)], 0)
#         results.values = tf.concat([results.values, cumulative_polarity], 0)
        indices.append([review_index, int(word_index)])
        values.append(cumulative_polarity)

        #accumulate statistics for the dataset
        if cumulative_polarity > max:
          max = cumulative_polarity
        elif cumulative_polarity < min:
          min = cumulative_polarity
    print('Dataset encoding stats: MIN = %f, MAX = %f\n' % (min, max))
    
    results = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=(len(sequences['reviews']), NUM_WORDS)
    )
    
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

# print(train_data_mhe[0])
# print(len(train_data_mhe[0]))

TRAINING_SAMPLE = 20000
VALIDATION_SAMPLE = 5000
df_train = pd.read_csv('../../data/imdb_train_split_20000.csv')
df_validation = pd.read_csv('../../data/imdb_train_split_5000.csv')
SAMPLE_SIZE_TRD = len(df_train)
SAMPLE_SIZE_VLD = len(df_validation)
print(df_train.head())

assert SAMPLE_SIZE_TRD == TRAINING_SAMPLE, 'training sample not complete....'
assert SAMPLE_SIZE_VLD == VALIDATION_SAMPLE, 'validation sample not complete....'

df_train = df_train.sample(frac=1)

# train_y = np.zeros([TRAINING_SAMPLE, 1], dtype=np.int)
# train_x = np.zeros([TRAINING_SAMPLE, 89527], dtype=np.float64)
# validation_y = np.zeros([VALIDATION_SAMPLE, 1], dtype=np.int)
# validation_x = np.zeros([VALIDATION_SAMPLE, 89527], dtype=np.float64)

# Generate empty SparseTensors
# train_y = tf.SparseTensor(indices=empty_indices, values=[], dense_shape=[TRAINING_SAMPLE, 1])
# train_x = tf.SparseTensor(indices=empty_indices, values=[], dense_shape=[TRAINING_SAMPLE, 89527])
# validation_y = tf.SparseTensor(indices=empty_indices, values=[], dense_shape=[VALIDATION_SAMPLE, 1])
# validation_x = tf.SparseTensor(indices=empty_indices, values=[], dense_shape=[VALIDATION_SAMPLE, 89527])
# todo: add option

xt_indices, xt_values = [], []
yt_indices, yt_values = [], []
for index in df_train.index:
    file_name = str(df_train['file'][index])
    label = int(df_train['label'][index])

    index_in_files_list = train_files_list.index(file_name)
#     train_x[index] = train_data_mhe[index_in_files_list]
#     train_y[index] = label
    xt_indices.append(index)
    xt_values.append(train_data_mhe[index_in_files_list])
    
    yt_indices.append(index)
    yt_values.append(label)

train_y = tf.SparseTensor(indices=yt_indices, values=yt_values, dense_shape=[TRAINING_SAMPLE, 1])
train_x = tf.SparseTensor(indices=xt_indices, values=xt_values, dense_shape=[TRAINING_SAMPLE, 89527])
# print(train_x[0])
# print(train_y[0])

xv_indices, xv_values = [], []
yv_indices, yv_values = [], []
for index in df_validation.index:
    file_name = str(df_validation['file'][index])
    label = int(df_validation['label'][index])

    index_in_files_list = train_files_list.index(file_name)
#     validation_x[index] = train_data_mhe[index_in_files_list]
#     validation_y[index] = label
    xv_indices.append(index)
    xv_values.append(train_data_mhe[index_in_files_list])
    
    yv_indices.append(index)
    yv_values.append(label)

validation_y = tf.SparseTensor(indices=yv_indices, values=yv_values, dense_shape=[VALIDATION_SAMPLE, 1])
validation_x = tf.SparseTensor(indices=xv_indices, values=xv_values, dense_shape=[VALIDATION_SAMPLE, 89527])


model = Sequential()

model.add(Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(
    L2_ETA), input_shape=(NUM_WORDS,)))
model.add(Dropout(0.6))
model.add(Dense(128, activation="relu",
                kernel_regularizer=keras.regularizers.l2(L2_ETA)))
model.add(Dropout(0.4))
model.add(Dense(64, activation="relu",
                kernel_regularizer=keras.regularizers.l2(L2_ETA)))

model.add(BatchNormalization())
model.add(RepeatVector(96))

model.add(LSTM(64, recurrent_dropout=0.4, return_sequences=True)) 
model.add(LSTM(64, recurrent_dropout=0.7, return_sequences=True))
model.add(LSTM(64, recurrent_dropout=0.4, return_sequences=True))

model.add(Flatten())


model.add(Dense(1, activation="sigmoid"))

#optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy', 'binary_crossentropy'])


# checkpoint
filepath = "model_1.h5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(train_x, train_y, validation_data=(validation_x, validation_y),
          epochs=150, batch_size=512,
          shuffle=False, verbose=1, callbacks=callbacks_list)
model.save(filepath)
