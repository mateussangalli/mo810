import numpy as np
import pandas as pd
import json
import random
import seaborn as sb
import matplotlib.pyplot as plt

import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Model

from sklearn.metrics import confusion_matrix

import pickle

MAX_SEQUENCE_LENGTH = 300
MAX_SUMMARY_LENGTH = 25
MAX_WORDS = 50000
EMBEDDING_DIM = 50

tb_folder = 'lstm_keras_tb'
emb_meta = 'emb_labels.tsv'
filename_tokenizer = 'tokenizer.pickle'
filename_model = 'model_cnn-lstm.h5'

folder = '../datasets'


# load tokenizer
with open(filename_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

# load model
model = load_model(filename_model)

model.summary()

N = 500000
pred = list()
Labels = list()
# testing
test_data = pd.read_json('../datasets/test_Books.json', lines=True, chunksize=500000)
for chunk in test_data:
    texts = chunk['reviewText'].tolist()
    labels = chunk['overall'].tolist()
    Labels = Labels + labels
    sequences = tokenizer.texts_to_sequences(texts)
    x1_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
    x2_test = pad_sequences(sequences, maxlen=MAX_SUMMARY_LENGTH, truncating='post')
    y_test = to_categorical(np.asarray(labels)-1)
    del sequences
    del labels
    pred_tmp = list(model.predict([x1_test,x2_test], verbose=1).argmax(axis=1))
    pred = pred+pred_tmp
    del x1_test
    del x2_test
    del y_test


Labels = np.array(Labels)-1
pred = np.array(pred)
C = confusion_matrix(Labels, pred)
sb.heatmap(C)
plt.savefig('conf_matrix.png')
sb.heatmap(C,annot=True)
plt.savefig('conf_matrix_annot.png')
plt.show()
print(C)
acc = np.sum(1*(Labels==pred))/Labels.size
print('acc = %.3f' % acc)

Labels_bin = 1*(Labels>3)
pred_bin = 1*(pred>3)
C_bin = confusion_matrix(Labels_bin, pred_bin)
sb.heatmap(C_bin)
plt.savefig('conf_matrix_bin.png')
sb.heatmap(C_bin,annot=True)
plt.savefig('conf_matrix_bin_annot.png')
plt.show()
print(C_bin)
acc_bin = np.sum(1*(Labels_bin==pred_bin))/Labels.size
print('acc_bin = %.3f' % acc_bin)
