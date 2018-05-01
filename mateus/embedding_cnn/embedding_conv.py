import numpy as np
import pandas as pd

import re

import sys
import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.optimizers import Adam
from keras.models import Model

MAX_SEQUENCE_LENGTH = 800
MAX_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.3

tb_callback = TensorBoard('logdir_keras_TB')

train_data = pd.read_json('datasets/train_Video_Games.json', lines=True)

texts = [text for text in train_data['reviewText']]
labels = [label for label in train_data['overall']]


# create a new Tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
del texts

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
del sequences

labels = (np.asarray(labels)-3)/2
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * train_data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

del labels
del data

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

embeddings_index = {}
glove_filename = 'glove.6B/glove.6B.100d.txt'
with open(glove_filename) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

del embeddings_index

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_cov1= Conv1D(256, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(3)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(3)(l_cov2)
l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(20)(l_cov3)  # global max pooling
l_flat = Flatten()(l_pool3)
l_dense = Dense(128, activation='relu')(l_flat)
l_dropout = Dropout(0.5)(l_dense)
preds = Dense(1, activation='linear')(l_dropout)

model = Model(sequence_input, preds)
model.compile(loss='mean_squared_error', optimizer=Adam())


print("model fitting - simplified convolutional neural network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
        nb_epoch=15, batch_size=128, callbacks=[tb_callback])


# serialize model to JSON
model_json = model.to_json()
with open('model_keras.json', 'w') as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('model_keras.h5')
print('Saved model to disk')
