import json
import gzip
import re
import random

# Video Games reviews size
N = 231780

def clean_string(string):
    string = re.sub('[\(\)\.,:!?]',' ',string)
    string = re.sub('[\'\"]','',string)
    string = re.sub('[~*&#]','',string)
    return string.strip().lower()

def create_train_set(filename_in, filename_out, ratio):
    with open(filename_out,'w') as f_out:
        with gzip.open(filename_in,'r') as f_in:
            num_samps = int(N*ratio)
            samples = set(random.sample(range(0,N),num_samps))
            for n, line in enumerate(f_in):
                if n in samples:
                    d = json.loads(line)
                    d = {'reviewText': clean_string(d['reviewText']), 'overall': d['overall']}
                    f_out.write(json.dumps(d)+'\n')

folder = 'datasets/'
filename_data = folder+'reviews_Video_Games_5.json.gz'
filename_train = folder+'train_Video_Games.json'
create_train_set(filename_data, filename_train, 0.7)
