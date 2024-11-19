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

with open('datasource/news_filter_token.json', 'r') as file:
    data = json.load(file)
    
save_path = 'datasource/news_filter_dup.json'
count = 0
print(f"Before: {len(data)}")

doc_list = []
for d in tqdm(data):
    if d['body'] not in doc_list:
        doc_list.append(d['body'])
        wr_dict(save_path,d)
        
print(f"After: {len(doc_list)}")