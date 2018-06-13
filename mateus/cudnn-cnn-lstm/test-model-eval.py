import numpy as np
import pandas as pd
import json
import random

import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.models import load_model

from keras.models import Model

import pickle

MAX_SEQUENCE_LENGTH = 250
MAX_WORDS = 30000
EMBEDDING_DIM = 50

tb_folder = 'lstm_keras_tb'
emb_meta = 'emb_labels.tsv'
filename_tokenizer = 'tokenizer.pickle'
filename_model = 'model_lstm-cnn.h5'

# this is the number of times we will train throught the whole training set
# the epoch value will be more than that because the dataset is divided
num_epochs = 29

folder = '../datasets'


# load tokenizer
with open(filename_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

# load model
model = load_model(filename_model)

model.summary()

# evaluation
val_data = pd.read_json('../datasets/val_Books.json', lines=True, chunksize=200000)
for chunk in val_data:
    texts = chunk['reviewText'].tolist()
    labels = chunk['overall'].tolist()
    sequences = tokenizer.texts_to_sequences(texts)
    x_val = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
    y_val = to_categorical(np.asarray(labels)-1)
    loss, acc = model.evaluate(x_val, y_val, verbose=1)
    print('loss = ' + str(loss))
    print('acc = ' + str(acc))
    del x_val
    del y_val
    #evaluate the accuracy for each individual class individually
    for i in range(1,6):
        x_val = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)[chunk['overall']==i]
        y_val = to_categorical(np.asarray(labels)-1)[chunk['overall']==i]
        loss, acc = model.evaluate(x_val, y_val, verbose=1)
        print('loss_'+str(i)+' = ' + str(loss))
        print('acc _'+str(i)+' = ' + str(acc))
        del x_val
        del y_val
