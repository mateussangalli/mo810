import json
import gzip
import re
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='folder with the data')
parser.add_argument('--data', help='filename of the data file')
parser.add_argument('--newdata', help='name of the new datasets')
parser.add_argument('--numlines', help='filename of the file containing the number of lines of each new dataset')

args = parser.parse_args()
folder = args.folder
filename_data = args.data
filename_numlines = args.numlines
data_name  = args.newdata

def divide_dataset(folder, filename_in, filenames_out, filename_num):
    f_out = list()
    num_lines = dict()
    for filename in filenames_out:
        f_out.append(open(folder+'/'+filename,'w'))
        num_lines[filename] = 0
    with open(folder+'/'+filename_in,'r') as f_in:
        for line in f_in:
            d = json.loads(line)
            d = {'reviewText': d['reviewText'], 'overall': d['overall']}
            d['overall'] = int(d['overall'])
            f_out[d['overall']-1].write(json.dumps(d)+'\n')
            num_lines[filenames_out[d['overall']-1]] += 1
    for f in f_out:
        f.close()
    with open(folder+'/'+filename_num,'w') as f:
        f.write(json.dumps(num_lines))

filenames_stars = list()
for i in range(1,6):
    filenames_stars.append(data_name+str(i)+'stars.json')

divide_dataset(folder, filename_data, filenames_stars, filename_numlines)
