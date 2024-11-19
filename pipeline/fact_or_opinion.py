import os
import re
import json
import nltk
from tqdm import tqdm
from transformers import pipeline
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
## Check If Fact or Opinion
#lighteternal/fact-or-opinion-xlmr-el

fact_opinion_classifier = pipeline("text-classification", model="lighteternal/fact-or-opinion-xlmr-el")

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

with open('datasource/news_filter_dup.json', 'r') as file:
    data = json.load(file)

save_path = 'datasource/news_filter_fact.json'
print(len(data))
print(data[0].keys())
rm_file(save_path)

for d in tqdm(data):
    fact_list = []
    body = d['body']
    paragraphs = re.split(r'\n{2,}', body)
    for text in paragraphs:
        sentences = sent_tokenize(text)
        for sentence in sentences:
            try:
                sentence_result = fact_opinion_classifier(sentence)[0]
                # If Fact
                if sentence_result["label"] == "LABEL_1":
                    fact_list.append(sentence)
            except:
                print(sentence)
    d['fact_list'] = fact_list
    wr_dict(save_path,d)