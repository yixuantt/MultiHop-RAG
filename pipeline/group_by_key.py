import json
from tqdm import tqdm
import nltk

# It's good practice to tokenize text once if it's going to be used multiple times.
nltk.download('punkt')

def count_tokens(text):
    tokens = nltk.word_tokenize(text)
    return len(tokens)

def is_almost_subset(list1, list2):
    """
    Check if one of the lists is almost a subset of the other.
    'Almost' means that the list can have up to 2 elements different.
    """
    set1, set2 = set(list1), set(list2)
    if min(len(set1), len(set2)) == 1:
        return is_any_subset(list1, list2)
    
    return len(set1 - set2) <= 1 or len(set2 - set1) <= 1

def is_any_subset(list1, list2):
    """
    Check if one of the lists is a subset of the other.
    """
    if len(list1) ==0 or len(list2) ==0:
        return True
    return set(list1).issubset(list2) or set(list2).issubset(list1)

def process_fact_keys(data):
    """
    Processes the fact keys in the data.
    """
    for d in data:
        for fact_key in d['fact_key']:
            fact_key['if_use'] = False

def deduplicate_by_fact(dict_list):
    seen_facts = set()
    unique_items = []
    for item in dict_list:
        fact = item['fact']
        if fact not in seen_facts:
            seen_facts.add(fact)
            unique_items.append(item)
    return unique_items

def group_facts(data):
    """
    Groups facts by keywords and saves them if they meet certain conditions.
    """
    save_list = []
    for idx, d in enumerate(tqdm(data)):
        for fact_key in d['fact_key']:
            if fact_key['if_use']:
                continue

            fact_key['if_use'] = True
            # group = [extract_fact_data(d, fact_key)]
            group = []
            for i, other_d in enumerate(data):
                if i != idx and other_d['url']!= d['url']:
                    for existing in group:
                        if existing['url'] == other_d['url']:
                            continue
                    for other_fact_key in other_d['fact_key']:
                        if (not other_fact_key['if_use'] and
                            is_any_subset(other_fact_key['keywords'], fact_key['keywords'])):
                            # Check if the new fact key is a subset/superset of every other item in the group
                            if all(is_any_subset(other_fact_key['keywords'], existing['keywords']) for existing in group):
                                other_fact_key['if_use'] = True
                                group.append(extract_fact_data(other_d, other_fact_key))
            group.append(extract_fact_data(d, fact_key))
            group = deduplicate_by_fact(group)
            if len(group) > 1:
                # print(f'Group by: {fact_key["fact"]} \n Save: {len(group)}\n\n')
                # for g in group:
                #     print(g['fact'])
                # print('\n\n')
                save_list.append(group)
    return save_list

def extract_fact_data(d, fact_key):
    """
    Extracts and returns fact data from a dictionary.
    """
    return {
        'title': d['title'],
        'author': d['author'],
        'url': d['url'],
        'source': d['source'],
        'category': d['category'],
        'published_at': d['published_at'],
        'fact': fact_key['fact'],
        'keywords': fact_key['keywords']
    }

def save_to_json(filename, data):
    """
    Saves data to a JSON file with the given filename.
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)

# Main execution
def main():
    with open('datasource/news_filter_keyphrase.json', 'r') as file:
        data = json.load(file)

    print('Initializing...')
    process_fact_keys(data)

    print('Starting group...')
    grouped_facts = group_facts(data)
    print(f'Facts grouped: {len(grouped_facts)}')

    save_to_json('datasource/news_group_keyphrase.json', grouped_facts)

if __name__ == '__main__':
    main()