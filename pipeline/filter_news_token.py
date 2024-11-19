import os
import json
import nltk
from tqdm import tqdm

def wr_dict(filename,dic):
    if not os.path.isfile(filename):
        data = []
        data.append(dic)
        with open(filename, 'w') as f:
            json.dump(data, f)
    else:      
        with open(filename, 'r') as f:
            data = json.load(f)
            data.append(dic)
        with open(filename, 'w') as f:
            json.dump(data, f)
            
def rm_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        
def count_tokens(text):
    tokens = nltk.word_tokenize(text)
    return len(tokens)

with open('datasource/source/news_data.json', 'r') as file:
    data = json.load(file)
    
save_file = 'datasource/news_filter_token.json'
count = 0
for d in tqdm(data):
    body = d['body']
    if count_tokens(body)>1024:
        wr_dict(save_file,d)
        count = count + 1

print(f'final count = {count}')