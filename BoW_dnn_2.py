import math
import numpy as np
import gzip
import os.path
import re
import json
import time
import random
import tensorflow as tf

#Books
#N = 8898041
#Video Games
N = 231780
#Musical Instruments
#N = 10261

def clean_string(string):
    """changes the string to lowercase and removes certain characters from it"""
    string2 = re.sub('[.,!?:\(\)]',' ',string).lower()
    string2 = re.sub('[~*&#]','',string2)
    string2 = re.sub('[0-9]*;','',string2)
    return string2

def split_dataset(data_filename, train_filename, test_filename, ratio=0.7):
    """reads the file data_filename and outputs the files train_filename and test_filename, with the training data and the test_data, respectively"""
    print('spliting dataset...')
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

def create_vocab(filename, threshold):
    """reads the dataset lines in filename and creates a vocabulary for the words with at least threshold ocurrences, outputs the vocabulary vocab and the reverse vocabulary reverse_vocab"""
    print('creating vocabulary...')
    k = 0
    counts = {}
    vocab = {}
    inverse_vocab = {}
    with open(filename,'r') as f:
        for n,line in enumerate(f):
            revText_words = json.loads(line)['reviewText'].split()
            for word in revText_words:
                if word not in counts:
                    counts[word] = 1
                elif counts[word]<threshold:
                    counts[word] += 1
                elif word not in vocab:
                    vocab[word] = k
                    inverse_vocab[k] = word
                    k += 1
    return (vocab,inverse_vocab)

def load_vocab(data_filename, vocab_filename, threshold):
    """try to load the vocabulary, creates one if it doesn't find one"""
    try:
        print('trying to load vocabulary...')
        with open(vocab_filename,'r') as f:
            vocab = json.loads(f.readline())
        inverse_vocab = {v: k for k, v in vocab.items()}
        return (vocab,inverse_vocab)
    except(FileNotFoundError):
        (vocab,inverse_vocab) = create_vocab(data_filename, threshold)
        print('saving vocabulary...')
        with open(vocab_filename,'w') as f:
            json.dump(vocab, f)
        return (vocab,inverse_vocab)

def count_words(string, vocab):
    """counts the ocurrences of the words in d of a string and outputs a dictionary that maps i to the number of times the word mapped to i in d appears in the string"""
    wordcount = {}
    string_words = string.split()
    for word in string_words:
        if word in vocab and word in wordcount:
            wordcount[vocab[word]] += 1
        elif word in vocab and word not in wordcount:
            wordcount[vocab[word]] = 1
    return wordcount

def create_preproc_dataset(data_filename, preproc_filename, vocab):
    """preprocess datasets to be in the bag of words format"""
    print('preprocessing dataset ' + data_filename + '...')
    i = 0
    with open(preproc_filename, 'w') as fp:
        with open(data_filename, 'r') as fd:
            for line in fd:
                d = json.loads(line)
                d['reviewText'] = count_words(d['reviewText'], vocab)
                json.dump(d,fp)
                fp.write('\n')

def maybe_create_preproc_dataset(data_filename, preproc_filename, vocab):
    """if it does not find a preprocessed dataset, creates one"""
    if not os.path.isfile(preproc_filename):
        create_preproc_dataset(data_filename, preproc_filename, vocab)

def load_data(data_filename, vocab_filename, train_filename, test_filename, train_filename_p, test_filename_p, threshold=500, ratio=0.7):
    """creates the preprocessed files and the vocabulary for the dataset"""
    maybe_split_dataset(
            data_filename,
            train_filename,
            test_filename,
            ratio)
    (vocab, inverse_vocab) = load_vocab(
            train_filename,
            vocab_filename,
            threshold)
    maybe_create_preproc_dataset(
            train_filename,
            train_filename_p,
            vocab)
    maybe_create_preproc_dataset(
            test_filename,
            test_filename_p,
            vocab)
    return (vocab, inverse_vocab)

def proto_input_fn(filename, batch_size, M, L):
    """function intended to use as a input_fn for estimator when the  parameters are fixed"""
    k = 0
    samples = random.sample(range(0,L),batch_size)
    samples = sorted(samples)
    revs = [None]*batch_size
    rating = [None]*batch_size
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
                if k == batch_size:
                    break
    features = {'reviewBoW':tf.convert_to_tensor_or_sparse_tensor(
        np.array(revs, dtype=np.float32),
        np.float32)}
    labels = tf.convert_to_tensor(
            np.array(rating, dtype=np.int32)-1 > 3,
            np.int32)
    return features, labels

def json_input_fn(filename, M, L, batch_size=1024):
    """generates an input_fn"""
    return lambda:proto_input_fn(filename, batch_size, M, L)

def model_fn(features, labels, mode):
    # Input Layer
    input_layer = features['reviewBoW']
    dense1 = tf.layers.dense(inputs=input_layer, units=2048, activation=tf.nn.softplus)
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.softplus)
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense3 = tf.layers.dense(inputs=dropout2, units=1024, activation=tf.nn.softplus)
    dropout3 = tf.layers.dropout(
        inputs=dense3, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense4 = tf.layers.dense(inputs=dropout3, units=32, activation=tf.nn.sigmoid)
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout3, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    lossTensor = tf.identity(loss, name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.003)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
   



def main(unused_argv):
    dataset = 'Video_Games'
    threshold=300
    #creates all the necessary files and returns the vocabulary
    (vocab, inverse_vocab) = load_data(
            'reviews_'+dataset+'_5.json.gz',
            dataset+'_vocab.json',
            dataset+'_train.json',
            dataset+'_test.json',
            dataset+'_train_preproc.json',
            dataset+'_test_preproc.json',
            threshold=threshold,
            ratio=0.6)
    print(dict(list(vocab.items())[0:30]))
    print(dict(list(vocab.items())[10000:10030]))
    print(dict(list(vocab.items())[-30:]))
    M = len(vocab)
    L = N-math.ceil(N*0.6)
    estimator = tf.estimator.Estimator(
            model_fn = model_fn,
            model_dir = 'custom_estimator_sigmoid'
            )
    train_input_fn = json_input_fn(
            filename = dataset+'_train_preproc.json',
            batch_size=256,
            M = M,
            L = L,
            )
    test_input_fn = json_input_fn(
            filename = dataset+'_test_preproc.json',
            batch_size=256,
            M = M,
            L = L,
            )
    # Set up logging for predictions
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log,
            every_n_iter=10)

    #I do this because I want to be able to know how much times it take per training iteration
    time1 = time.time()
    estimator.train(
            input_fn=train_input_fn,
            steps=10)
    time2 = time.time()
    print(time2-time1)
    print('starting training...')
    for i in range(0,300):
        estimator.train(
                input_fn=train_input_fn,
                steps=10,
                hooks=[logging_hook])
        # Prediction on the train set.
        train_results = estimator.evaluate(input_fn=train_input_fn, steps=10)
        print('train results:',train_results)
        # Prediction on the test set.
        test_results = estimator.evaluate(input_fn=test_input_fn, steps=10)
        print('test results:',test_results)
    print('finished training!')



if __name__ == "__main__":
    tf.app.run()
