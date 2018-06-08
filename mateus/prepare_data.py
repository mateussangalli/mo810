import json
import gzip
import re
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='folder with the data')
parser.add_argument('--data', help='filename of the data file')
parser.add_argument('--size', help='number of reviews in the data')
parser.add_argument('--train', help='filename of the train file')
parser.add_argument('--val', help='filename of the validation file')
parser.add_argument('--test', help='filename of the test file')
parser.add_argument('--rtrainval', help='ratio of the training+validation set size to the total')
parser.add_argument('--rtrain', help='ratio of the training set size to the training+validation')

args = parser.parse_args()
folder = args.folder
filename_data = folder+'/'+args.data
filename_train = folder+'/'+args.train
filename_val = folder+'/'+args.val
filename_test = folder+'/'+args.test
ratio1 = float(args.rtrainval)
ratio2 = float(args.rtrain)
N = int(args.size)

def clean_string(string):
    string = re.sub('(?<=;)(\d+)','',string)
    string = re.sub('[\'&#]','',string)
    string = re.sub('([\(\)\.,:!?\"/;])',' \\1 ',string)
    string = re.sub(' +',' ',string)
    return string.strip().lower()

def split_dataset(filename_in, filename_train, filename_test, filename_val, ratio1, ratio2):
    with open(filename_train,'w') as f_train:
        with open(filename_test,'w') as f_test:
            with open(filename_val,'w') as f_val:
                with gzip.open(filename_in,'r') as f_in:
                    num_tv = int(N*ratio1)
                    num_t = int(num_tv*ratio2)
                    samples1 = random.sample(range(0,N),num_tv)
                    samples2 = [samples1[i] for i in random.sample(range(num_tv),num_t)]
                    samples1 = set(samples1)
                    samples2 = set(samples2)
                    for n, line in enumerate(f_in):
                        d = json.loads(line)
                        d = {'reviewText': clean_string(d['summary'] + ' * ' + d['reviewText']), 'overall': d['overall']}
                        if n in samples2:
                            f_train.write(json.dumps(d)+'\n')
                        elif n in samples1:
                            f_val.write(json.dumps(d)+'\n')
                        else:
                            f_test.write(json.dumps(d)+'\n')

split_dataset(filename_data, filename_train, filename_test, filename_val, ratio1, ratio2)
