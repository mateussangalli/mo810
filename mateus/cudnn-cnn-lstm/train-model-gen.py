import numpy as np
import pandas as pd
import json
import random
from time import time

from collections import deque

import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.models import load_model

from keras.models import Model

import pickle

MAX_SEQUENCE_LENGTH = 400
MAX_WORDS = 40000
BATCH_SIZE = 32

tb_folder = 'lstm_keras_tb'
emb_meta = 'emb_labels.tsv'
filename_tokenizer = 'tokenizer.pickle'
filename_model = 'model_cnn-lstm.h5'
filename_train = '../datasets/train_Books.json'
filename_val = '../datasets/val_Books.json'

numlines = 0
with open(filename_train) as f:
    for line in f:
        numlines+=1

def input_gen(filename, tokenizer, numlines, chunk_size, batch_size):
    not_read = set(range(numlines))
    while len(not_read):
        x = list()
        y = list()
        try:
            samples = set(random.sample(not_read, chunk_size))
            not_read = not_read.difference(samples)
        except:
            samples = not_read
            not_read = set()
        with open(filename) as f:
            for n, line in enumerate(f):
                d = json.loads(line)
                if n in samples:
                    x.append(d['summary']+' * '+d['reviewText'])
                    y.append(d['overall'])
        size = len(samples)
        x = tokenizer.texts_to_sequences(x)
        x = pad_sequences(x,maxlen=MAX_SEQUENCE_LENGTH,truncating='post')
        y = to_categorical(np.asarray(y)-1)
        indices = np.arange(size)
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        for i in range(size//batch_size):
            x_tmp = x[batch_size*i:batch_size*(i+1)]
            y_tmp = y[batch_size*i:batch_size*(i+1)]
            yield x_tmp,y_tmp
        if not size % batch_size == 0:
            k = size % batch_size
            x_tmp = x[-k:]
            y_tmp = y[-k:]
            yield x_tmp,y_tmp

tb_callback = TensorBoard(
        tb_folder,
        embeddings_freq=1,
        embeddings_layer_names=['embeddings'],
        embeddings_metadata={'embeddings':emb_meta})

# load tokenizer
with open(filename_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

# load model
model = load_model(filename_model)

model.summary()

w = {0:8, 1:8, 2:4, 3:1, 4:1}

val_data = pd.read_json('../datasets/val_Books.json', lines=True).sample(100000)
texts = (val_data['summary'] + ' *  ' + val_data['reviewText']).tolist()
labels = val_data['overall'].tolist()
sequences = tokenizer.texts_to_sequences(texts)
del texts
x_val = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
del sequences
del val_data
y_val = to_categorical(np.asarray(labels)-1)
del labels
loss_val = list()
acc_val = list()

losses = deque([0.0]*20)
accs = deque([0.0]*20)
loss_sum = 0.0
acc_sum = 0.0
for epoch in range(0,3):
    i = 0
    t = time()
    for x,y in input_gen(filename_train,tokenizer,numlines,650000,BATCH_SIZE):
        loss, acc = model.train_on_batch(x,y,class_weight=w)
        loss_sum += loss-losses.popleft()
        acc_sum += acc-accs.popleft()
        losses.append(loss)
        accs.append(acc)
        i += BATCH_SIZE
        print('epoch %s,step %s/%s, loss = %.3f, acc = %.3f' % (epoch,i,numlines,loss_sum/20.0,acc_sum/20.0), end = '\r')
    # save checkpoint
    model.save('ckpt_%s.h5' % epoch)
    print('Saved checkpoint %s to disk' % epoch)
    print('time = %.3f' % (time()-t))
    loss, acc = model.evaluate(x_val, y_val, verbose=1)
    print('val loss = ' + str(loss))
    print('val acc = ' + str(acc))
    loss_val.append(loss)
    acc_val.append(acc)
    print('')

# save model
model.save(filename_model)
print('Saved model to disk')
