
import json
import os
from tqdm import tqdm
import pandas as pd
import re
import random
from colorama import Fore,Back,Style
from colorama import init,Fore
from util import rm_file,wr_dict,claim_prompt,convert_string_to_list,query_openai_3
init(autoreset=True)

categories = ['business','entertainment','health','science','sports','technology']


with open('datasource/news_group_keyphrase.json', 'r') as file:
    data = json.load(file)
    
with open('datasource/news_filter_dup.json', 'r') as file:
    doc = json.load(file)

def get_bodies_by_url(target_url):
    # Consider using a dictionary for faster lookups if 'doc' is large
    return next((d.get('body', None) for d in doc if d.get('url') == target_url), None)

save_file = 'datasource/news_claim_full_init.json'
rm_file(save_file)  # Remove the file before starting the process

print('start pipeline...')

selected_sublist = []
hop_list = [4,5,6]

def sample_list(i, hops, max_retries=3):
    label = categories[int(i % len(categories))]

    filtered_data = [item for item in data if item[-1]['category'] == label]
    if len(filtered_data) < hops:
        return []
    attempts = 0  
    while attempts < max_retries:
        random_list = random.sample(filtered_data, 1)

        if random_list not in selected_sublist and len(random_list[0])>=hops:
            random_list = random_list[0]
            return random_list 
        attempts += 1  

    return []

count = 0
max_count = 6000
pbar = tqdm(total=max_count)
save_item = []
while count != max_count:
    # Confirm Hop
    i = count
    hops = hop_list[int(i%3)]
    label = categories[int(i%6)]
    print(Style.DIM + f'Label: {label}')
    print(Style.DIM + f'Hops: {hops}')
    random_list = sample_list(i,hops)
    if len(random_list) == 0:
        continue
    # Generate Claims
    target_news_list = []
    if not len(random_list) == hops:
        print(Style.DIM + f'len(random_list:{len(random_list)}')
        target_news_list.append(random_list[-1])
        target_news_list.extend(random.sample(random_list[:-1],hops-1))
    else:
        target_news_list = random_list

    selected_sublist.append(target_news_list)
    for items in target_news_list:
        wr_dict(save_file, items) 
    # for items in target_news_list:
    #     body = get_bodies_by_url(items['url'])
    #     if body is None:
    #         continue
    #     message = f"{claim_prompt}{body}\n\n##Evidence:{items['fact']}\n\n##Claims:"
    #     response = query_openai(message)
    #     # print(Fore.CYAN + f'Evidence: {items["fact"]}')
    #     # print(Fore.CYAN + f'Keywords: {items["keywords"]}')
    #     # print(Fore.BLUE + response + '\n\n')
    #     result_list = convert_string_to_list(response)
    #     if result_list:
    #         items['claim'] = result_list[0]['claims']
    #         items['claim_topic'] = result_list[0]['topic']
    #         items['claim_target'] = result_list[0]['target']
    #         # save_item.append(items)
    #         wr_dict(save_file, items)

    pbar.update(1)
    count = count + 1
    

print(f'finish generating claims')

with open('datasource/news_claim_full_init.json', 'r') as file:
    data = json.load(file)

print(len(data))
print(data[0].keys())
# print(have[0].keys())
def unique_dicts(list_of_dicts):
    count = 0
    unique_list = []
    for entry in list_of_dicts:
        if entry not in unique_list:
            unique_list.append(entry)
        else:
            count = count + 1
    print(len(unique_list))
    return unique_list

def unique_by_fact_combination(lst):
    seen_facts = set()
    unique_lst = []

    for sublst in lst:
        facts = tuple(sorted(item['fact'] for item in sublst if isinstance(item, dict) and 'fact' in item))
        if facts not in seen_facts:
            seen_facts.add(facts)
            unique_lst.append(sublst)

    return unique_lst


unique_lists = unique_dicts(data)
print(len(unique_lists))

with open('datasource/news_claim_full_init_unique.json', 'w') as json_file:
    json.dump(unique_lists, json_file)

with open('datasource/news_filter_dup_full.json', 'r') as file:
    doc = json.load(file)

with open('datasource/news_claim_full_init_unique.json', 'r') as file:
    data = json.load(file)
    
def get_bodies_by_url(target_url):
    # Consider using a dictionary for faster lookups if 'doc' is large
    return next((d.get('body', None) for d in doc if d.get('url') == target_url), None)

def get_item_by_fact(fact,find):
    # Consider using a dictionary for faster lookups if 'have' is large
    return next((d for d in find if d.get('fact') == fact), None)

save_file = 'datasource/news_claim/news_claim_for_time.json'
print('start pipeline...')
rm_file(save_file)
count = 0
for idx, item in enumerate(tqdm(data)):
    # if idx<already_have_len:
    #     continue
    fact = item['fact']
    # d = get_item_by_fact(fact,have)
    # d_2 = get_item_by_fact(fact,have_2)
    # if 'claim' in item.keys():
    #     wr_dict(save_file, item)
    # elif d != None:
    #     wr_dict(save_file, d)
    # elif d_2 != None:
    #     wr_dict(save_file, d_2)
    # else:
    count = count + 1
    body = get_bodies_by_url(item['url'])
    if body == None:
        print('body None')
    message = f"{claim_prompt}{body}\n\n##Evidence:{item['fact']}\n\n##Claims:"
    response = query_openai_3(message)
    if response == 0:
        wr_dict(save_file, item)
    if isinstance(response, str):
        print(Fore.CYAN + f'Evidence: {item["fact"]}')
        print(Fore.CYAN + f'Keywords: {item["keywords"]}')
        print(Fore.BLUE + response + '\n\n')
        result_list = convert_string_to_list(response)
        if result_list:
            item['claim'] = result_list[0]['claims']
            item['claim_topic'] = result_list[0]['topic']
            item['claim_target'] = result_list[0]['target']
            wr_dict(save_file, item)
print(count)
print(f'finish generating claims')