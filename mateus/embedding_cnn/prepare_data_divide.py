import json
import gzip
import re
import random

def clean_string(string):
    string = re.sub('[\(\)\.,:!?]',' ',string)
    string = re.sub('[\'\"]','',string)
    string = re.sub('[~*&#]','',string)
    return string.strip().lower()

def divide_dataset(filename_in, filenames_out, filename_num):
    f_out = list()
    num_lines = dict()
    for filename in filenames_out:
        f_out.append(open(filename,'w'))
        num_lines[filename] = 0
    with gzip.open(filename_in,'r') as f_in:
        for line in f_in:
            d = json.loads(line)
            d = {'reviewText': d['reviewText'], 'overall': d['overall']}
            d['reviewText'] = clean_string(d['reviewText'])
            d['overall'] = int(d['overall'])
            f_out[d['overall']-1].write(json.dumps(d)+'\n')
            num_lines[filenames_out[d['overall']-1]] += 1
    for f in f_out:
        f.close()
    with open(filename_num,'w') as f:
        f.write(json.dumps(num_lines))
    return num_lines

def create_train_set(filenames_in, filename_out, num_lines, samples_each):
    with open(filename_out,'w') as f_out:
        for filename in filenames_in:
            with open(filename,'r') as f_in:
                samples = set(
                        random.sample(range(0,num_lines[filename]),samples_each))
                for n, line in enumerate(f_in):
                    if n in samples:
                        f_out.write(json.dumps(json.loads(line))+'\n')

folder = 'datasets/'
filename_data = folder+'reviews_Books_5.json.gz'
filename_numlines = folder+'num_lines.json'
filename_train = folder+'train_Books.json'
filenames_stars = list()
for i in range(1,6):
    filenames_stars.append(folder+'Books_'+str(i)+'stars.json')
try:
    with open(filenames_numline,'r') as f:
        num_lines = json.loads(f.readlines())
    create_train_set(filenames_stars, filename_train, num_lines, 70000)
except:
    print('something happened... :(')
    num_lines = divide_dataset(filename_data, filenames_stars, filename_numlines)
    create_train_set(filenames_stars, filename_train, num_lines, 70000)
