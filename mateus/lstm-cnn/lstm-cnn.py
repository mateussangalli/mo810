import numpy as np
import pandas as pd

import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

from keras.layers import Dense, Input, Flatten, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Bidirectional, LSTM
from keras.layers import GaussianNoise, Dropout
from keras.regularizers import l2
from keras.optimizers import Adamax
from keras.models import Model

import pickle

MAX_SEQUENCE_LENGTH = 200
MAX_WORDS = 30000
EMBEDDING_DIM = 50

tb_folder = 'lstm_keras_tb'
emb_meta = 'emb_labels.tsv'
filename_tokenizer = 'tokenizer.pickle'
filename_model = 'model_lstm-cnn.json'
filename_weights = 'model_lstm-cnn.h5'
filename_corpus = '../datasets/corpus.txt'

# this is the number of times we will train throught the whole training set
# the epoch value will be more than that because the dataset is divided
num_epochs = 2

tb_callback = TensorBoard(
        tb_folder,
        embeddings_freq=1,
        embeddings_layer_names=['embeddings'],
        embeddings_metadata={'embeddings':emb_meta})

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
glove_filename = '../glove.6B/glove.6B.50d.txt'
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
l_noise = GaussianNoise(0.05)(embedded_sequences)

l_lstm = Bidirectional(LSTM(30, return_sequences=True, recurrent_activation='hard_sigmoid', kernel_regularizer=l2(0.0015), dropout=0.3))(l_noise)

l_cov1 = Conv1D(150, 5, activation='relu', padding='same', kernel_regularizer=l2(0.0015))(l_lstm)
l_pool1 = MaxPooling1D(strides=4)(l_cov1)  # global max pooling

l_cov2 = Conv1D(200, 5, activation='relu', padding='same', kernel_regularizer=l2(0.0015))(l_pool1)
l_pool2 = MaxPooling1D(strides=8)(l_cov2)  # global max pooling

l_cov3 = Conv1D(200, 3, activation='relu', padding='same', kernel_regularizer=l2(0.0015))(l_pool2)
l_pool3 = GlobalMaxPooling1D()(l_cov3)  # global max pooling

l_dense = Dense(150, activation='relu', kernel_regularizer=l2(0.0015))(l_pool3)
l_dropout = Dropout(0.5)(l_dense)

preds = Dense(5, activation='softmax')(l_dropout)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',optimizer=Adamax(clipnorm=4.),metrics=['acc'])

model.summary()

# epoch will not be the real number of the epoch, because the dataset is divided
epoch = 0
for i in range(num_epochs):
    train_data = pd.read_json('../datasets/train_Books.json', lines=True, chunksize=500000)
    for chunk in train_data:
        texts = chunk['reviewText'].tolist()
        labels = chunk['overall'].tolist()
        sequences = tokenizer.texts_to_sequences(texts)
        del texts
        x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        del sequences
        y_train = to_categorical(np.asarray(labels)-1)
        del labels
        model.fit(x_train, y_train,
                initial_epoch=epoch, class_weight='auto',
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



# serialize model to JSON
model_json = model.to_json()
with open(filename_model, 'w') as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(filename_weights)
print('Saved model to disk')

# evaluation
val_data = pd.read_json('../datasets/val_Books.json', lines=True, chunksize=200000)
for chunk in val_data:
    texts = chunk['reviewText'][chunk['overall']==i].tolist()
    labels = chunk['overall'][chunk['overall']==i].tolist()
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
    #evaluate the accuracy for each individual class individually
    for i in range(1,6):
        texts = chunk['reviewText'][chunk['overall']==i].tolist()
        labels = chunk['overall'][chunk['overall']==i].tolist()
        sequences = tokenizer.texts_to_sequences(texts)
        del texts
        x_val = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        del sequences
        y_val = to_categorical(np.asarray(labels)-1)
        del labels
        loss, acc = model.evaluate(x_val, y_val, verbose=1)
        print('loss_'+str(i)+' = ' + str(loss))
        print('acc _'+str(i)+' = ' + str(acc))
        del x_val
        del y_val
