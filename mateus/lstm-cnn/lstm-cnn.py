import numpy as np
import pandas as pd

import re

import sys
import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

from keras.layers import Dense, Input, Flatten, Embedding, Conv1D, MaxPooling1D
from keras.layers import Bidirectional, LSTM
from keras.layers import GaussianNoise, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model

import pickle

MAX_SEQUENCE_LENGTH = 150
MAX_WORDS = 30000
EMBEDDING_DIM = 100

tb_folder = 'lstm_keras_tb'
emb_meta = 'emb_labels.tsv'
filename_tokenizer = 'tokenizer.pickle'
filename_model = 'model_lstm-cnn.json'
filename_weights = 'model_lstm-cnn.h5'

tb_callback = TensorBoard(
        tb_folder,
        embeddings_freq=1,
        embeddings_layer_names=['embeddings'],
        embeddings_metadata={'embeddings':emb_meta})

train_data = pd.read_json('../datasets/train_Books.json', lines=True)

texts = [text for text in train_data['reviewText']]
labels = [label for label in train_data['overall']]
del train_data

# create a new Tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS, filters=';#&*:()')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
del texts
sizes = np.array([len(sequence) for sequence in sequences])
# saving Tokenizer
with open(filename_tokenizer, 'wb') as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
del sequences
y_train = np.asarray(labels)-1
del labels
print('Shape of data tensor:', x_train.shape)
print('Shape of label tensor:', y_train.shape)
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = to_categorical(y_train[indices])

# process validation set
val_data = pd.read_json('../datasets/val_Books.json', lines=True)
texts = [text for text in val_data['reviewText']]
labels = [label for label in val_data['overall']]
del val_data
sequences = tokenizer.texts_to_sequences(texts)
del texts
x_val = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y_val = to_categorical(np.asarray(labels)-1)
del sequences
del labels

embedding_matrix = np.random.random((MAX_WORDS, EMBEDDING_DIM))

embeddings_index = {}
glove_filename = '../vectors.txt'
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
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
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
l_noise = GaussianNoise(0.01)(embedded_sequences)
l_dropout = Dropout(0.05)(l_noise)

l_lstm = Bidirectional(LSTM(64, return_sequences=True, recurrent_activation='hard_sigmoid', kernel_regularizer=l2(0.0015)))(l_dropout)

l_cov1 = Conv1D(150, 5, activation='relu', padding='same', kernel_regularizer=l2(0.0015))(l_lstm)
l_pool1 = MaxPooling1D(strides=2)(l_cov1)  # global max pooling

l_cov2 = Conv1D(200, 5, activation='relu', padding='same', kernel_regularizer=l2(0.0015))(l_pool1)
l_pool2 = MaxPooling1D(strides=2)(l_cov2)  # global max pooling

l_cov3 = Conv1D(250, 5, activation='relu', padding='same', kernel_regularizer=l2(0.0015))(l_pool2)
l_pool3 = MaxPooling1D(strides=3)(l_cov3)  # global max pooling

l_cov4 = Conv1D(300, 3, activation='relu', padding='same', kernel_regularizer=l2(0.0015))(l_pool3)
l_pool4 = MaxPooling1D(strides=4)(l_cov4)  # global max pooling

l_flat = Flatten()(l_pool4)
preds = Dense(5, activation='softmax', kernel_regularizer=l2(0.0015))(l_flat)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',optimizer=Adam(clipnorm=4.),metrics=['acc'])

model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=3,
        batch_size=32, callbacks=[tb_callback])


# serialize model to JSON
model_json = model.to_json()
with open(filename_model, 'w') as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(filename_weights)
print('Saved model to disk')

del x_train
del y_train
del x_val
del y_val

test_data = pd.read_json('../datasets/test_Books.json', lines=True)
test_data = test_data.sample(frac = 0.2)

texts = [text for text in test_data['reviewText']]
labels = [label for label in test_data['overall']]
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y_test = to_categorical(np.asarray(labels)-1)
del sequences
del labels
del texts

loss, acc = model.evaluate(x_test, y_test, verbose=1)
print(loss)
print(acc)
