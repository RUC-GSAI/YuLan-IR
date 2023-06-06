import os
import json
import re
from tqdm import tqdm
from collections import Counter
import re

json_directory = '../json_data/'

def load_json_files(directory):
    files = os.listdir(directory)
    files = [i for i in files if i.endswith('.json')]
    data = []
    for file in files:
        with open(os.path.join(directory, file), encoding='utf-8') as f:
            data.append([file, json.load(f)])
    return data

def collect_ngrams(data, n):
    ngrams = []
    print('collecting n-grams of n={}...'.format(n))
    for d in tqdm(data):
        text = d[1]['contents']
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i+n])
    return Counter(ngrams)

data = load_json_files(json_directory)

for length in [100, 50]:
    ngrams = collect_ngrams(data, length)
    print('reduplicating...')
    ngrams_duplicated = [(k, v) for k, v in ngrams.items() if v > 10]
    for i in tqdm(ngrams_duplicated):
        flag = False
        for k in range(len(data)):
            if i[0] in data[k][1]['contents']:
                if flag == False:
                    flag = True # do not remove the n-gram of the first document it appears
                else:
                    data[k][1]['contents'] = data[k][1]['contents'].replace(i[0], "")
                    
# save the processed data
for d in data:
    file_name = d[0]
    with open(os.path.join(json_directory, file_name), 'w', encoding='utf-8') as f:
            f.write(json.dumps(d[1], ensure_ascii=False))