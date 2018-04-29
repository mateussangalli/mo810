import math
import numpy as np
import gzip
import re
import json
from nltk import ngrams
import tensorflow as tf
import os.path
import random
from nltk.stem.snowball import SnowballStemmer

#Books
#N = 8898041
#Video Games
N = 231780
#Musical Instruments
#N = 10261
tf.logging.set_verbosity(tf.logging.INFO)

def clean_string(string):
    """changes the string to lowercase and removes certain characters from it"""
    string2 = re.sub('[,:\(\)]',' ',string).lower()
    string2 = re.sub('[!?]','.',string2)
    string2 = re.sub('[~*&#]','',string2)
    string2 = re.sub('[0-9]*;','',string2)
    return string2

def tuple_to_str(tuple):
    string = str(tuple)
    return re.sub('[\(\) \']','',string)

def split_dataset(data_filename, train_filename, test_filename, ratio=0.7):
    """reads the file data_filename and outputs the files train_filename and test_filename, with the training data and the test_data, respectively"""
    K = math.floor((1-ratio)*N)
    samples = random.sample(range(0,N),K)
    samples = sorted(samples)
    f_train = open(train_filename,'w')
    f_test = open(test_filename,'w')
    k = 0
    with gzip.open(data_filename) as f:
        for n,line in enumerate(f):
            d = json.loads(line)
            d2 = {'overall' : d['overall'], 'reviewText' : clean_string(d['reviewText'])}
            if k<K and samples[k]==n:
                #write test dataframe
                json.dump(d2,f_test)
                f_test.write('\n')
                k += 1
            else:
                #write to training dataframe
                json.dump(d2,f_train)
                f_train.write('\n')
    f_train.close()
    f_test.close()
    
def maybe_split_dataset(data_filename, train_filename, test_filename, ratio=0.7):
    """split the dataset if it is not already done"""
    if not (os.path.isfile(train_filename) and os.path.isfile(test_filename)):
        split_dataset(
            data_filename,
            train_filename,
            test_filename,
            ratio)


def create_vocab(filename, frequency_range=(0.0001,0.02), ngram_range=(1,2)):
    """given the json file filename, returns a vocabulary of ngrams with n in ngram_range, and frequency in frequency_range"""
    stemmer = SnowballStemmer("english")
    k = 0
    vocab = dict()
    ngram_min, ngram_max = ngram_range
    freq_min, freq_max = frequency_range
    with open(filename,'r') as f:
        for m in range(ngram_min, ngram_max):
            total = 0
            counts = dict()
            f.seek(0)
            for n,line in enumerate(f):
                revTexts = json.loads(line)['reviewText'].split('.')
                for revText in revTexts:
                    revText_words = [stemmer.stem(word) for word in revText.split()]
                    revText_grams = ngrams(revText_words,m)
                    for gram in revText_grams:
                        total += 1
                        if gram not in counts:
                            counts[gram] = 1
                        else:
                            counts[gram] += 1
            for key in counts:
                value = counts[key]
                if value > freq_min * total and value < freq_max * total:
                    vocab[tuple_to_str(key)] = k
                    k += 1
    return vocab

def load_vocab(data_filename, vocab_filename, frequency_range=(0.0005,0.05), ngram_range=(1,2)):
    """try to load the vocabulary, creates one if it doesn't find one"""
    try:
        with open(vocab_filename,'r') as f:
            vocab = json.loads(f.readline())
        return vocab
    except(FileNotFoundError):
        vocab = create_vocab(data_filename, frequency_range, ngram_range)
        print('saving vocabulary...')
        with open(vocab_filename,'w') as f:
            json.dump(vocab, f)
        return vocab

def count_grams(string, vocab, ngram_range=(1,2)):
    """counts the ocurrences of the grams in d of a string and outputs a dictionary that maps i to the number of times the gram mapped to i in d appears in the string"""
    stemmer = SnowballStemmer('english')
    gramcount = {}
    ngram_min, ngram_max = ngram_range
    strings = string.split('.')
    for substring in strings:
        string_words = [stemmer.stem(word) for word in substring.split()]
        for m in range(ngram_min, ngram_max):
            string_grams = ngrams(string_words, m)
            for gram in string_grams:
                gram2 = tuple_to_str(gram)
                if gram2 in vocab and vocab[gram2] in gramcount:
                    gramcount[vocab[gram2]] += 1
                elif gram2 in vocab:
                    gramcount[vocab[gram2]] = 1
    return gramcount

def create_preproc_dataset(data_filename, preproc_filename, vocab, ngram_range=(1,2)):
    """preprocess datasets to be in the bag of words format"""
    print('preprocessing dataset ' + data_filename + '...')
    i = 0
    with open(preproc_filename, 'w') as fp:
        with open(data_filename, 'r') as fd:
            for line in fd:
                d = json.loads(line)
                d['reviewText'] = count_grams(d['reviewText'], vocab, ngram_range)
                json.dump(d,fp)
                fp.write('\n')

def maybe_create_preproc_dataset(data_filename, preproc_filename, vocab, ngram_range=(1,2)):
    """if it does not find a preprocessed dataset, creates one"""
    if not os.path.isfile(preproc_filename):
        create_preproc_dataset(data_filename, preproc_filename, vocab, ngram_range=(1,2))

def load_data(data_filename, vocab_filename, train_filename, test_filename, train_filename_p, test_filename_p, frequency_range=(0.0001,0.02), ngram_range=(1,2), ratio=0.7):
    """creates the preprocessed files and the vocabulary for the dataset"""
    print('splitting dataset...')
    maybe_split_dataset(
            data_filename,
            train_filename,
            test_filename,
            ratio)
    print('loading/creating vocabulary...')
    vocab = load_vocab(
            train_filename,
            vocab_filename,
            frequency_range,
            ngram_range)
    print(dict(list(vocab.items())[0:30]))
    print(dict(list(vocab.items())[-30:]))
    print('trying to load preprocessed training set...')
    maybe_create_preproc_dataset(
            train_filename,
            train_filename_p,
            vocab,
            ngram_range)
    print('trying to load preprocessed test set...')
    maybe_create_preproc_dataset(
            test_filename,
            test_filename_p,
            vocab,
            ngram_range)
    return vocab

def generate_big_batch(filename, batch_size, M, L, repeat_bad_ones = False, repeated_ratio = 0.3):
    k = 0
    samples = random.sample(range(0,L),math.ceil(batch_size*(1-repeated_ratio*(1*repeat_bad_ones))))
    samples = sorted(samples)
    revs = [None]*batch_size
    rating = [None]*batch_size
    bad_ones = []
    with open(filename) as f:
        for n,line in enumerate(f):
            if k < batch_size and samples[k]==n:
                d = json.loads(line)
                rating[k] = d['overall']
                temp = [0]*M
                for key,value in d['reviewText'].items():
                    temp[int(key)] = value
                revs[k] = temp
                k += 1
                if repeat_bad_ones and d['overall'] < 3:
                    bad_ones.append(d)
                if k == batch_size:
                    break
    if repeat_bad_ones:
        i = 0
        while k < batch_size:
            d = bad_ones[i]
            rating[k] = d['overall']
            temp = [0]*M
            for key,value in d['reviewText'].items():
                temp[int(key)] = value
            revs[k] = temp
            k += 1
            i += 1
            if i == len(bad_ones):
                i = 0
    x = np.array(revs, dtype=np.float32)
    y = np.array(rating, dtype=np.int32)-1
    return x,y

def model_fn(features, labels, mode):
    regularizer = tf.contrib.layers.l1_regularizer(scale=0.5)
    # Input Layer
    input_layer = features['reviewBoW']
    dropout0 = tf.layers.dropout(inputs=input_layer, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense1 = tf.layers.dense(inputs=dropout0, units=1000, activation=tf.nn.softplus, kernel_regularizer=regularizer)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(inputs=dropout1, units=750, activation=tf.nn.softplus, kernel_regularizer=regularizer)
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense3 = tf.layers.dense(inputs=dropout2, units=500, activation=tf.nn.softplus, kernel_regularizer=regularizer)
    dropout3 = tf.layers.dropout(inputs=dense3, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense4 = tf.layers.dense(inputs=dropout3, units=500, activation=tf.nn.softplus, kernel_regularizer=regularizer)
    dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense5 = tf.layers.dense(inputs=dropout4, units=500, activation=tf.nn.softplus, kernel_regularizer=regularizer)
    dropout5 = tf.layers.dropout(inputs=dense5, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense6 = tf.layers.dense(inputs=dropout5, units=500, activation=tf.nn.softplus, kernel_regularizer=regularizer)
    dropout6 = tf.layers.dropout(inputs=dense6, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense7 = tf.layers.dense(inputs=dropout6, units=300, activation=tf.nn.softplus, kernel_regularizer=regularizer)
    dropout7 = tf.layers.dropout(inputs=dense7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense8 = tf.layers.dense(inputs=dropout7, units=300, activation=tf.nn.softplus, kernel_regularizer=regularizer)
    dropout8 = tf.layers.dropout(inputs=dense8, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense9 = tf.layers.dense(inputs=dropout8, units=300, activation=tf.nn.softplus, kernel_regularizer=regularizer)
    dropout9 = tf.layers.dropout(inputs=dense9, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense10 = tf.layers.dense(inputs=dropout9, units=300, activation=tf.nn.softplus, kernel_regularizer=regularizer)
    dropout10 = tf.layers.dropout(inputs=dense10, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense11 = tf.layers.dense(inputs=dropout10, units=150, activation=tf.nn.softplus, kernel_regularizer=regularizer)
    dropout11 = tf.layers.dropout(inputs=dense11, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits_rating = tf.layers.dense(inputs=dropout11, units=5)
    logits_polarity = tf.layers.dense(inputs=logits_rating, units=3)

    predictions_rating = tf.argmax(input=logits_rating, axis=1)
    predictions_polarity = tf.argmax(input=logits_polarity, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_rating)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss_rating = tf.losses.sparse_softmax_cross_entropy(labels=labels['rating'], logits=logits_rating)
    loss_rating_named = tf.identity(loss_rating, name='loss_rating')

    loss_polarity = tf.losses.sparse_softmax_cross_entropy(labels=labels['polarity'], logits=logits_polarity)
    loss_polarity_named = tf.identity(loss_polarity, name='loss_polarity')

    loss = loss_rating + loss_polarity
    loss_named = tf.identity(loss, name='loss')
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        'accuracy_rating': tf.metrics.accuracy(labels=labels['rating'], predictions=predictions_rating),
        'accuracy_polarity': tf.metrics.accuracy(labels=labels['polarity'], predictions=predictions_polarity)
        }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    folder = 'datasets/'
    dataset = 'Video_Games'
    freq_range = (0.00003,0.01)
    ngram_range = (1,2)
    ratio=0.65
    #creates all the necessary files and returns the vocabulary
    vocab = load_data(
            folder+'reviews_'+dataset+'_5.json.gz',
            folder+dataset+'_vocab.json',
            folder+dataset+'_train.json',
            folder+dataset+'_test.json',
            folder+dataset+'_train_preproc.json',
            folder+dataset+'_test_preproc.json',
            frequency_range=freq_range,
            ngram_range=ngram_range,
            ratio=ratio)
    print(dict(list(vocab.items())[0:30]))
    print(dict(list(vocab.items())[-30:]))
    M = len(vocab)
    L_test = math.floor(N*(1-ratio))
    L_train = N-L_test
    # Init estimator
    estimator = tf.estimator.Estimator(
            model_fn = model_fn,
            model_dir = 'dnn_classifier_9layers_2losses'
            )
    # Set up logging for predictions
    tensors_to_log = {'loss': 'loss', 'loss_rating': 'loss_rating', 'loss_polarity': 'loss_polarity'}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log,
            every_n_iter=200)
    print('starting training...')
    train_x,train_y = generate_big_batch(
            filename = folder+dataset+'_train_preproc.json',
            batch_size=L_train+10000,
            M = M,
            L = L_train,
            repeat_bad_ones = True
            )
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'reviewBoW':train_x},
            y={'rating': train_y, 'polarity': 1*(train_y>1) + 1*(train_y>2)},
            batch_size=1000,
            num_epochs=None,
            shuffle=True,
            )
    del train_x
    del train_y
    estimator.train(
            input_fn=train_input_fn,
            steps=10000,
            hooks=[logging_hook])
    print('finished training!')
    # Prediction on the train set.
    train_results = estimator.evaluate(input_fn=train_input_fn, steps=10)
    print('train results:',train_results)
    del train_input_fn
    test_x,test_y = generate_big_batch(
            filename = folder+dataset+'_test_preproc.json',
            batch_size=L_test,
            M = M,
            L = L_test,
            )
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'reviewBoW':test_x},
            y={'rating': test_y, 'polarity': 1*(test_y>1) + 1*(test_y>2)},
            batch_size=1000,
            num_epochs = 1,
            shuffle=True,
            )
    # Prediction on the test set.
    test_results = estimator.evaluate(input_fn=test_input_fn, steps=10)
    print('test results:',test_results)
    bad_ratings_indexes = [i for i in range(0,len(test_y)) if test_y[i]<2]
    print(bad_ratings_indexes[:5])
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'reviewBoW':test_x[bad_ratings_indexes[:5],:]},
        y={'rating': test_y[bad_ratings_indexes[:5]], 'polarity': 1*(test_y[bad_ratings_indexes[:5]]>1) + 1*(test_y[bad_ratings_indexes[:5]]>2)},
            batch_size=1000,
            num_epochs = 1,
            shuffle=True,
            )
    print('labels:')
    print(test_y[bad_ratings_indexes[:5]])
    print('predictions = ')
    for pred in estimator.predict(test_input_fn):
        print(pred)
        


if __name__ == "__main__":
    tf.app.run()
