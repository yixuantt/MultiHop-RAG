from newspaper import Article
from newspaper import Config
import json
from tqdm import tqdm
import os
import requests

with open('datasource/api/news_api.json', 'r') as file:
    data = json.load(file)

print(len(data))
save_path = 'datasource/source/news_data.json'
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
# rm_file(save_path)

with open(save_path, 'r') as file:
    have = json.load(file)

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'

config = Config()
config.headers = {'Cookie': "cookie1=xxx;cookie2=zzzz"}
config.browser_user_agent = USER_AGENT
config.request_timeout = 10

RETRY_ATTEMPTS = 1
count = 0
def parse_article(url):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            article = Article(url, config=config)
            article.download()
            article.parse()
            return article.text
        except:
            return None
            # print(f"Error retrieving article from URL '{url}'")
    return None


for idx, d in enumerate(tqdm(data)):
    if idx<len(have):
        continue
    url = d['url']
    maintext = parse_article(url.strip())
    if maintext == None:
        continue
    d['body'] = maintext
    wr_dict(save_path,d)
    count = count + 1
print(count+len(have))
