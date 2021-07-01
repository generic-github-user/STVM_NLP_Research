import numpy as np

# from tensorflow import keras
# import tensorflow as tf

from tensorflow.keras.utils import plot_model
# from tensorflow.keras.layers import Merge
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization, InputLayer, RepeatVector, Permute
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Flatten, Bidirectional
from tensorflow.keras.models import Sequential, Model

import tensorflow.keras.backend as K
# import keras
import os
import pandas as pd
import random
import os
import pandas as pd
from collections import defaultdict
import zipfile
import requests
import re
from tensorflow.keras.models import load_model


model = load_model('model_2.h5')

TRAINING_SAMPLE = 5000

train_csv_file = 'imdb_train_5k.csv'
train_y = np.zeros([TRAINING_SAMPLE, 1], dtype=np.int)
list_of_train_reviews = list()

glove_file = "glove.6B.zip"
EMBEDDING_SIZE = 100


def download(url_):
    if not os.path.exists(glove_file):
        print("downloading glove embedding .....")
        r = requests.get(url_, glove_file)
    glove_filename = "glove.6B.{}d.txt".format(EMBEDDING_SIZE)
    if not os.path.exists(glove_filename) and EMBEDDING_SIZE in [50, 100, 200, 300]:
        print("extract glove embeddings ...")
        with zipfile.ZipFile(glove_file, 'r') as z:
            z.extractall()


def load_glove():
    with open("glove.6B.100d.txt", 'r') as glove_vectors:
        word_to_int = defaultdict(int)
        int_to_vec = defaultdict(lambda: np.zeros([EMBEDDING_SIZE]))

        index = 1
        for line in glove_vectors:
            fields = line.split()
            word = str(fields[0])
            vec = np.asarray(fields[1:], np.float32)
            word_to_int[word] = index
            int_to_vec[index] = vec
            index += 1
    return word_to_int, int_to_vec


download("http://nlp.stanford.edu/data/glove.6B.zip")
word_to_int, int_to_vec = load_glove()


list_of_filename = list()
list_of_y = list()


def create_training_sample(filename):
    df_train = pd.read_csv(train_csv_file)
    SAMPLE_SIZE = len(df_train)
    assert SAMPLE_SIZE == TRAINING_SAMPLE, 'training sample not complete....'

    for index in df_train.index:
        review = str(df_train['review'][index])
        label = int(df_train['label'][index])
        list_of_filename.append(df_train['file'][index])
        review = review.lower()
        '''
        review = review.replace("<br />", " ")
        review = re.sub("[,]", " ,", review)
        review = re.sub("[.]", " .", review)
        review = re.sub("[;]", " ;", review)
        review = re.sub("[!]", " !", review)
        review = re.sub("[/]", " / ", review)
        '''
        review = review.replace("<br />", " ")
        review = re.sub(r"[^a-z ]", " ", review)
        review = re.sub(r" +", " ", review)

        review = review.split(" ")
        #for i in range(len(review)):
        #    if review[i] == 'can't''

        list_of_train_reviews.append(review)
        list_of_y.append(label)

        #train_y[index] = label


def encode_reviews(revs):
    train_data = []
    for review in revs:
        int_review = [word_to_int[word] for word in review]
        train_data.append(int_review)
    return train_data


create_training_sample(filename=train_csv_file)
print(list_of_train_reviews[0])

for index in range(len(list_of_y)):
    train_y[index] = list_of_y[index]

print(train_y[0], train_y[6], train_y[8], train_y[10])
print(list_of_filename[0], list_of_filename[6],
      list_of_filename[8], list_of_filename[10])
train_reviews = encode_reviews(list_of_train_reviews)
print(train_reviews[0])

MAX_REVIEW_LEN = 1200


def zero_pad_reviews(revs):
    _data_padded = []
    for review in revs:
        padded = [0] * MAX_REVIEW_LEN
        stop_index = min(len(review), MAX_REVIEW_LEN)
        padded[:stop_index] = review[:stop_index]
        _data_padded.append(padded)
    return _data_padded


train_reviews = zero_pad_reviews(train_reviews)


def review_ints_to_vecs(train_reviews):
    train_data = []
    for review in train_reviews:
        vec_review = [int_to_vec[word] for word in review]
        train_data.append(vec_review)
    return train_data


train_reviews = np.array(review_ints_to_vecs(train_reviews))
print(train_reviews.shape)
print(train_reviews[0])

model.summary()

prediction_results = None
prediction_results = model.predict(train_reviews, verbose=1)

list_of_proba = []
list_of_file_names = []
print("predict shape", prediction_results.shape)
correct_pred = 0
for i in range(TRAINING_SAMPLE):
    proba = prediction_results[i][0]
    if (proba < float(0.5) and train_y[i] == 0) or (proba >= float(0.5) and train_y[i] == 1):
        correct_pred = correct_pred + 1
    list_of_proba.append(str(prediction_results[i][0]))
    list_of_file_names.append(list_of_filename[i])

print("Accuracy: ", float(correct_pred)/float(TRAINING_SAMPLE))

d_prob = {'prob': list_of_proba}
d_files = {'file': list_of_file_names}

dd = {'prob': list_of_proba, 'file': list_of_file_names}

df = pd.DataFrame(dd, index=None)
df.to_csv('model_b_2_5ktrain.csv', index=False)
