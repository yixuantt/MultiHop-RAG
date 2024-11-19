# Generate Keywords
import torch
import os
from tqdm import tqdm
import json
import nltk
import numpy as np

from span_marker import SpanMarkerModel

model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd")
model.cuda()
# pipe = pipeline("token-classification", model="jasminejwebb/KeywordIdentifier", aggregation_strategy="simple",device= 'cuda')

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


with open('datasource/news_filter_fact_out_same.json', 'r') as file:
    data = json.load(file)
    

save_path = 'datasource/news_filter_entity.json'
rm_file(save_path)
save_list = []

fact_count = sum(len(d['fact_list']) for d in data)
print(f'Before: {fact_count}')

# Run inference
for item in tqdm(data):
    fact_list = item['fact_list']
    item['fact_key'] = []
    for sample in fact_list:
        facts = {}
        sample = sample.strip()
        if not sample.endswith('.'):
            continue    
        facts['fact'] = sample
        tokens = nltk.word_tokenize(sample)
        length = len(tokens)
        if length<18:
            continue
        entity_list = []
        entities = model.predict(sample)
        for e in entities:
            entity_list.append(e['span'])
        if len(entity_list)==0:
            continue
        # stripped_text = []
        # for group in generated_text:
        #     if group['entity_group'] == 'LABEL_1':
        #         stripped_text.append(group['word'].strip())
        # generated_text = generated_text[0]['generated_text']
        # generated_text = generated_text.split(',')
        stripped_text = [text.strip() for text in entity_list]

        print(stripped_text)

        facts['keywords'] = stripped_text

        # RAKE
        # rake_nltk_var.extract_keywords_from_text(sample)
        # keywords = rake_nltk_var.get_ranked_phrases()
        # print(facts['keywords'])
        item['fact_key'].append(facts)
    # print(item['fact_key'])
    save_list.append(item)
    
with open(save_path, "w") as json_file:
    json.dump(save_list, json_file)

fact_count = sum(len(d['fact_key']) for d in data)
print(f'After: {fact_count}')

