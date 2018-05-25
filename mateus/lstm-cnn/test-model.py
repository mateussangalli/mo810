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
from keras.optimizers import Adam
from keras.models import Model
from keras.models import model_from_json

import pickle

MAX_SEQUENCE_LENGTH = 256

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


test_data = pd.read_json('../datasets/test_Books.json', lines=True)
test_data = test_data.sample(frac = 0.1)

texts = [text for text in test_data['reviewText']]
labels = [label for label in test_data['overall']]
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y_test = np.asarray(labels)-1
del sequences
del labels
del texts

pred_test = model.predict(x_test, verbose=1)
pred_test = pred_test.reshape(1,pred_test.size)
print(pred_test[:10])
print(y_test[:10])
print(pred_test[:-10])
print(y_test[:-10])


diff = pred_test - y_test
diff = np.absolute(diff)
print(np.sum(diff)/diff.size)

num_rights = 1*(diff < 0.5) 
acc = np.sum(num_rights)/num_rights.size
print(acc)

num_rights = 1*(diff < 1) 
acc = np.sum(num_rights)/num_rights.size
print(acc)
