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

MAX_SEQUENCE_LENGTH = 300
MAX_WORDS = 50000
EMBEDDING_DIM = 100

tb_folder = 'lstm_keras_tb'
emb_meta = 'emb_labels.tsv'
filename_tokenizer = 'tokenizer.pickle'
filename_model = 'model_cnn-lstm.h5'

# this is the number of times we will train throught the whole training set
# the epoch value will be more than that because the dataset is divided
num_epochs = 24

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

val_data = pd.read_json('../datasets/val_Books.json', lines=True).sample(50000)
texts = val_data['reviewText'].tolist()
labels = val_data['overall'].tolist()
sequences = tokenizer.texts_to_sequences(texts)
x_val = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
y_val = to_categorical(np.asarray(labels)-1)

del val_data
del texts
del labels
del sequences

loss_train = np.array([0]*(num_epochs+1), dtype=np.float32)
acc_train = np.array([0]*(num_epochs+1), dtype=np.float32)

loss_val = list()
acc_val = list()

for epoch, x_train, y_train in input_gen(folder,train_filenames,train_numlines,tokenizer,100000,num_epochs):
    model.fit(x_train, y_train,initial_epoch=epoch,
            epochs=epoch+1, batch_size=32,
            callbacks=[tb_callback], shuffle=True)
    if (epoch+1) % 5 == 0:
        loss, acc = model.evaluate(x_train, y_train, verbose=1)
        print('train loss = %.3f' % loss_train[epoch])
        print('train acc = %.3f' % acc_train[epoch])
        loss_train.append(loss)
        acc_train.append(acc)
        loss, acc = model.evaluate(x_val, y_val, verbose=1)
        print('val loss = %.3f' % loss)
        print('val acc = %.3f' % acc)
        loss_val.append(loss)
        acc_val.append(acc)
        # save checkpoint
        model.save('ckpt_%s.h5' % (epoch+1))
        print('Saved checkpoint to disk')

loss_train = np.array(loss_train)
acc_train = np.array(acc_train)
loss_val = np.array(loss_val)
acc_val = np.array(acc_val)

# save model
model.save(filename_model)
print('Saved model to disk')

d = dict()
d['loss_train'] = loss_train
d['loss_val'] = loss_val
d['acc_train'] = acc_train
d['acc_val'] = acc_val
# saving losses and accuracys
with open('losses.pickle', 'wb') as f:
    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
