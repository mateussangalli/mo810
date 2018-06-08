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

train_filenames = list()
for i in range(1,6):
    train_filenames.append('train_Books'+str(i)+'stars.json')
with open('../datasets/train_numlines') as f:
    train_numlines = json.loads(f.readline())

def input_gen(folder, filenames, numlines, tokenizer, chunk_size, num_epochs):
    not_read = dict()
    epoch = -1
    for filename in filenames:
        not_read[filename] = set(range(numlines[filename]))
    while epoch<num_epochs:
        epoch += 1
        x = list()
        y = list()
        for filename in filenames:
            if len(not_read[filename]) <= chunk_size:
                not_read[filename] = set(range(numlines[filename]))
            samples = set(random.sample(not_read[filename], chunk_size))
            with open(folder+'/'+filename) as f:
                for n, line in enumerate(f):
                    d = json.loads(line)
                    if n in samples:
                        x.append(d['reviewText'])
                        y.append(d['overall'])
            not_read[filename] = not_read[filename].difference(samples)
        x = tokenizer.texts_to_sequences(x)
        x = pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
        y = to_categorical(np.asarray(y)-1)
        yield epoch, x ,y

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

for epoch, x_train, y_train in input_gen(folder,train_filenames,train_numlines,tokenizer,100000,num_epochs):
    model.fit(x_train, y_train,initial_epoch=epoch,
            epochs=epoch+1, batch_size=32,
            callbacks=[tb_callback], shuffle=True)
    if (epoch+1) % 5 == 0:
        # save model
        model.save('ckpt_%s.h5' % (epoch+1))
        print('Saved model to disk')

# save model
model.save(filename_model)
print('Saved model to disk')

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
