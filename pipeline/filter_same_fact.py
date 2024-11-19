import os
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from colorama import Fore,Back,Style
from colorama import init,Fore
init(autoreset=True)

# Initialize the model only once to avoid repeating this expensive operation
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Function to write a dictionary to a JSON file
def wr_dict(filename, dic):
    data = []
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    data.append(dic)
    with open(filename, 'w') as f:
        json.dump(data, f)
            
# Function to remove a file
def rm_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


with open('datasource/news_filter_fact.json', 'r') as file:
    data = json.load(file)

print(data[0]['fact_list'])
save_path = 'datasource/news_filter_fact_out_same.json'
rm_file(save_path)
fact_count = 0
save_count = 0


# Encoding process
print('start to encode...')
doc_embeddings = {}
for idx, d in enumerate(tqdm(data)):
    fact_list = d['fact_list']
    for jdx, fact in enumerate(fact_list):
        embedding = model.encode(fact, convert_to_tensor=True)
        key = f'{idx}*{jdx}'
        doc_embeddings[key] = embedding
        

torch.save(doc_embeddings, 'tensor_list.pt')

doc_embeddings = torch.load('tensor_list.pt')

# Filtering process
print('start to filter...')
for idx, d in enumerate(tqdm(data)):
    fact_list = d['fact_list']
    fact_save_list = []
    print(Fore.YELLOW +f'Before: {len(fact_list)}')
    for jdx, fact in enumerate(tqdm(fact_list)):
        flag = True
        key = f'{idx}*{jdx}'
        query_embedding = doc_embeddings[key]
        for index in (i for i in range(len(data)) if i != idx):
            if flag == False:
                continue

            passage_embeddings = [
                doc_embeddings[f'{index}*{kdx}'] 
                for kdx in range(len(data[index]['fact_list'])) 
                if f'{index}*{kdx}' in doc_embeddings.keys()
            ]
            if len(passage_embeddings) < 1:
                continue
            passage_embeddings_tensor = torch.stack(passage_embeddings)
            scores = util.dot_score(query_embedding, passage_embeddings_tensor).squeeze().tolist()
            # If scores is a single float, make it a list
            if isinstance(scores, float):
                scores = [scores]
            max_value = max(scores)
            if max_value >= 0.9:
                print(f'\033[93mFailed For Checking: {fact}\033[0m')  # Using ANSI escape codes for color
                flag = False
        if flag:
            fact_save_list.append(fact)
    d['fact_list'] = fact_save_list
    if fact_save_list:
        wr_dict(save_path, d)
        print(Fore.YELLOW +f'After: {len(fact_save_list)}')
    print('-' * 20)

# Summary of the operation
fact_count = sum(len(d['fact_list']) for d in data)
save_count = sum(len(d['fact_list']) for d in json.load(open(save_path)))
print(f"Total facts: {fact_count}, Saved facts: {save_count}")