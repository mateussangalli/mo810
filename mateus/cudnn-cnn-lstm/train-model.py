import numpy as np
import pandas as pd

import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.models import load_model

from keras.layers import Dense, Input, Flatten, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Bidirectional, LSTM
from keras.layers import GaussianNoise
from keras.regularizers import l2
from keras.optimizers import Adamax
from keras.models import Model

import pickle

MAX_SEQUENCE_LENGTH = 300
MAX_WORDS = 50000

tb_folder = 'lstm_keras_tb'
emb_meta = 'emb_labels.tsv'
filename_tokenizer = 'tokenizer.pickle'
filename_model = 'model_cnn-lstm.h5'

# this is the number of times we will train throught the whole training set
# the epoch value will be more than that because the dataset is divided
num_epochs = 5


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

w = {0:5, 1:5, 2:3, 3:1, 4:1}

loss_train = list()
acc_train = list()
loss_val = list()
acc_val = list()

val_data = pd.read_json('../datasets/val_Books.json', lines=True).sample(100000)
texts = val_data['reviewText'].tolist()
labels = val_data['overall'].tolist()
del val_data
sequences = tokenizer.texts_to_sequences(texts)
del texts
x_val = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
del sequences
y_val = to_categorical(np.asarray(labels)-1)
del labels

# epoch will not be the real number of the epoch, because the dataset is divided
epoch = 0
true_epoch = 0
for i in range(num_epochs):
    train_data = pd.read_json('../datasets/train_Books.json', lines=True, chunksize=500000)
    for chunk in train_data:
        texts = chunk['reviewText'].tolist()
        labels = chunk['overall'].tolist()
        sequences = tokenizer.texts_to_sequences(texts)
        del texts
        x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
        del sequences
        y_train = to_categorical(np.asarray(labels)-1)
        del labels
        model.fit(x_train, y_train,
                initial_epoch=epoch, class_weight=w,
                epochs=epoch+1, batch_size=64,
                callbacks=[tb_callback], shuffle=True)
        epoch += 1
        if epoch%5 == 0:
            loss, acc = model.evaluate(x_train, y_train, verbose=1)
            print('train loss = ' + str(loss))
            print('train acc = ' + str(acc))
            loss_train.append(loss)
            acc_train.append(acc)
        del x_train
        del y_train
        # evaluate on the validation set each 5 chunks
        if epoch%5 == 0:
            loss, acc = model.evaluate(x_val, y_val, verbose=1)
            print('val loss = ' + str(loss))
            print('val acc = ' + str(acc))
            loss_val.append(loss)
            acc_val.append(acc)
    true_epoch += 1
    # save model
    model.save('ckpt_%s.h5' % true_epoch)
    print('Saved checkpoint %s to disk' % true_epoch)

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
