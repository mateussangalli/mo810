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

MAX_SEQUENCE_LENGTH = 200
MAX_WORDS = 30000
EMBEDDING_DIM = 100

tb_folder = 'lstm_keras_tb'
emb_meta = 'emb_labels.tsv'
filename_tokenizer = 'tokenizer.pickle'
filename_model = 'model_lstm-cnn.h5'

# this is the number of times we will train throught the whole training set
# the epoch value will be more than that because the dataset is divided
num_epochs = 1


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

train_data = pd.read_json('../datasets/train_Books.json', lines=True, chunksize=500000)
numc = np.array((0,0,0,0,0))
for chunk in train_data:
    numc += np.asarray(chunk['overall'].value_counts(sort=False))
w = dict()
for i in range(5):
    w[i] = float(numc[4])/float(numc[i])
print(w)

# epoch will not be the real number of the epoch, because the dataset is divided
epoch = 0
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
                epochs=epoch+1, batch_size=32,
                callbacks=[tb_callback], shuffle=True)
        epoch += 1
        del x_train
        del y_train
        # evaluate on the validation set each 5 chunks
        if epoch%5 == 0:
            val_data = pd.read_json('../datasets/val_Books.json', lines=True, chunksize=200000)
            for vchunk in val_data:
                texts = vchunk['reviewText'].tolist()
                labels = vchunk['overall'].tolist()
                sequences = tokenizer.texts_to_sequences(texts)
                del texts
                x_val = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
                del sequences
                y_val = to_categorical(np.asarray(labels)-1)
                del labels
                loss, acc = model.evaluate(x_val, y_val, verbose=1)
                print('loss = ' + str(loss))
                print('acc = ' + str(acc))
                del x_val
                del y_val



# save model
model.save(filename_model)
print('Saved model to disk')

# evaluation
val_data = pd.read_json('../datasets/val_Books.json', lines=True, chunksize=200000)
for chunk in val_data:
    texts = chunk['reviewText'].tolist()
    labels = chunk['overall'].tolist()
    sequences = tokenizer.texts_to_sequences(texts)
    x_val = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y_val = to_categorical(np.asarray(labels)-1)
    loss, acc = model.evaluate(x_val, y_val, verbose=1)
    print('loss = ' + str(loss))
    print('acc = ' + str(acc))
    del x_val
    del y_val
    #evaluate the accuracy for each individual class individually
    for i in range(1,6):
        x_val = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)[chunk['overall']==i]
        del sequences
        y_val = to_categorical(np.asarray(labels)-1)[chunk['overall']==i]
        del labels
        loss, acc = model.evaluate(x_val, y_val, verbose=1)
        print('loss_'+str(i)+' = ' + str(loss))
        print('acc _'+str(i)+' = ' + str(acc))
        del x_val
        del y_val
