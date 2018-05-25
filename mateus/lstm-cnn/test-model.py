import numpy as np
import pandas as pd
import random

import re

import sys
import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.optimizers import Adamax
from keras.models import Model
from keras.models import model_from_json

import pickle

MAX_SEQUENCE_LENGTH = 250

filename_tokenizer = 'tokenizer.pickle'
filename_model = 'model_lstm-cnn.json'
filename_weights = 'model_lstm-cnn.h5'

# loading
with open(filename_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

with open(filename_model, 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

model.load_weights(filename_weights)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adamax(clipnorm=4.), metrics=['acc'])

test_data = pd.read_json('../datasets/test_Books_big.json', lines=True, chunksize=100000)

test_data = pd.read_json('../datasets/test_Books.json', lines=True, chunksize=100000)
for chunk in test_data:
    texts = chunk['reviewText'][chunk['overall']==i].tolist()
    labels = chunk['overall'][chunk['overall']==i].tolist()
    sequences = tokenizer.texts_to_sequences(texts)
    del texts
    x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    del sequences
    y_test = to_categorical(np.asarray(labels)-1)
    del labels
    loss, acc = model.evaluate(x_test, y_test, verbose=1)
    print('loss = ' + str(loss))
    print('acc = ' + str(acc))
    del x_test
    del y_test
    for i in range(1,6):
        texts = chunk['reviewText'][chunk['overall']==i].tolist()
        labels = chunk['overall'][chunk['overall']==i].tolist()
        sequences = tokenizer.texts_to_sequences(texts)
        del texts
        x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        del sequences
        y_test = to_categorical(np.asarray(labels)-1)
        del labels
        loss, acc = model.evaluate(x_test, y_test, verbose=1)
        print('loss_'+str(i)+' = ' + str(loss))
        print('acc _'+str(i)+' = ' + str(acc))
        del x_test
        del y_test
