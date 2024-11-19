# https://mediastack.com/documentation
# collect news api date
import http.client, urllib.parse
from datetime import datetime, timedelta
import json
import os
from tqdm import tqdm
categories = ['business','entertainment','health','science','sports','technology']
save_path = 'datasource/api/news_api.json'
start_date = '2023-09-26'
end_date = '2023-12-26'

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

def adjust_date(date_str, days_to_add=2, cutoff_date_str=end_date):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    cutoff_date = datetime.strptime(cutoff_date_str, "%Y-%m-%d")
    new_date_obj = date_obj + timedelta(days=days_to_add)
    # if new_date_obj > cutoff_date:
    #     new_date_obj = cutoff_date
    return new_date_obj.strftime("%Y-%m-%d")

conn = http.client.HTTPConnection('api.mediastack.com')

def getApi(date,type,offset):
    params = urllib.parse.urlencode({
        'access_key': '',
        'date':date,
        'categories': type,
        'sort': 'published_desc',
        'languages': 'en,-ar,-de,-es,-fr,-it,-nl,-no,-pt,-ru,-zh',
        'limit': 100,
        'offset': offset,
        })
    conn.request('GET', '/v1/news?{}'.format(params))

    res = conn.getresponse()
    data = res.read()
    decoded_data = data.decode('utf-8')
    try:
        json_data = json.loads(decoded_data)
        # if error 
        if 'error' in json_data.keys():
            print(f"API ERROR: {json_data}")
            return False
        data_list = json_data['data']

        print(f"Date:{date} | Type: {type}")
        for d in tqdm(data_list):
            wr_dict(save_path,d)
        return True

    except json.JSONDecodeError as e:
        print(f"JSON ERROR: {e}")
        return False
    
date = start_date
result = True
rm_file(save_path)
for i in range(1000):
    if not result:
        break
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    cutoff_date = datetime.strptime(end_date, "%Y-%m-%d")
    if date_obj > cutoff_date:
        break
    else:
        for t in categories:
            for offset in range(2):
                result = getApi(date,t,offset)
                if not result:
                    break
    date = adjust_date(date)