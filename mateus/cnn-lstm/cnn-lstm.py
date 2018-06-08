import numpy as np
import pandas as pd

import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

from keras.layers import Dense, Input, Flatten, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import LSTM
from keras.layers import GaussianNoise, Dropout
from keras.regularizers import l2
from keras.optimizers import Adamax
from keras.models import Model

import pickle

MAX_SEQUENCE_LENGTH = 300
MAX_WORDS = 50000
EMBEDDING_DIM = 50

tb_folder = 'lstm_keras_tb'
emb_meta = 'emb_labels.tsv'
filename_tokenizer = 'tokenizer.pickle'
filename_model = 'model_cnn-lstm.h5'
filename_corpus = '../datasets/corpus.txt'
glove_filename = '../glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM

# this is the number of times we will train throught the whole training set
# the epoch value will be more than that because the dataset is divided

# just have to compute the tokenizer once
try:
    # loading
    with open(filename_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)
except:
    # create a new Tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, filters=';#&*:()')
    with open(filename_corpus) as corpus:
        tokenizer.fit_on_texts(corpus)
    # saving Tokenizer
    with open(filename_tokenizer, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

embedding_matrix = np.random.random((MAX_WORDS, EMBEDDING_DIM))
# use the pre-trained glove vectors
embeddings_index = {}
mean = np.array([0]*EMBEDDING_DIM, dtype='float32')
with open(glove_filename) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        mean += coefs
        embeddings_index[word] = coefs
mean = mean/len(embeddings_index)
inv_index = {v: k for k, v in word_index.items()}
for i in range(1,MAX_WORDS):
    word = inv_index[i]
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        # words not found in embedding index will be close to the mean.
        embedding_matrix[i] = mean + np.random.normal(scale=0.1, size=(1,EMBEDDING_DIM))
del embeddings_index

embedding_layer = Embedding(MAX_WORDS,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True,
                            name='embeddings')
del embedding_matrix

inv_index = {v: k for k, v in word_index.items()}
with open(tb_folder+'/'+emb_meta,'w') as emb:
    emb.write('unk\n')
    for i in range(1,MAX_WORDS):
        emb.write(inv_index[i]+'\n')

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

l_cov1 = Conv1D(128, 5, activation='relu', padding='valid')(embedded_sequences)
l_dropout1 = Dropout(0.1)(l_cov1)
l_pool1 = MaxPooling1D(strides=2)(l_dropout1)

l_cov2 = Conv1D(128, 5, activation='relu', padding='valid')(l_pool1)
l_dropout2 = Dropout(0.1)(l_cov2)
l_pool2 = MaxPooling1D(strides=2)(l_dropout2)

l_lstm = LSTM(128, return_sequences=False, recurrent_activation='hard_sigmoid', dropout=0.1)(l_pool2)

preds = Dense(5, activation='softmax')(l_lstm)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',optimizer=Adamax(clipnorm=3.5),metrics=['acc'])

model.summary()

# save model
model.save(filename_model)
print('Saved model to disk')
