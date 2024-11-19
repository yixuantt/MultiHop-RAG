import json
import os
from colorama import init, Fore
# from openai import OpenAI
import time
import openai
# Initialize colorama only once
init(autoreset=True)

openai.api_key = ''
openai.base_url = ''

def query_openai(message):

    try:
        response = openai.chat.completions.create(
            temperature=0.1,
            model="gpt-4-1106-preview",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant who can complete all instructions. Please do not mention the detailed information of the given excerpts. "},
                {"role": "user", "content": message}
            ]
        )
        print(response)
        return response.choices[0].message.content
    except Exception as e:
        print(Fore.RED + "OpenAI error: An error occurred: " + str(e))
        time.sleep(20)  # It's usually better to handle retries with a back-off strategy
        return query_openai(message)  # Recursive call on error
    
def query_openai_3(message):

    try:
        response = openai.chat.completions.create(
            temperature=0.1,
            model="gpt-3.5-turbo-1106",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ]
        )
        print(response)
        return response.choices[0].message.content
    except Exception as e:
        print(Fore.RED + "OpenAI error: An error occurred: " + str(e))
        time.sleep(20)  # It's usually better to handle retries with a back-off strategy
        return query_openai(message)  # Recursive call on error
    
    
# Function to write a dictionary to a JSON file
def wr_dict(filename,dic):
    try:
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
    except Exception as e:
        print(Fore.RED + "Save Error:", str(e))
        return
            
# Function to remove a file
def rm_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print(Fore.RED + f"Error: {file_path} : {e.strerror}")

def save_to_json(filename, data):
    """
    Saves data to a JSON file with the given filename.
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)

def convert_string_to_list(string):
    try:
        lines = string.strip().split('\n\n')
        result = []
        for line in lines:
            parts = line.split('\n')
            topic = parts[-1].split('##Claim Topic: ')[1] if '##Claim Topic:' in parts[-1] else ''
            target = parts[-2].split('##Claim Target: ')[1] if '##Claim Target:' in parts[-2] else ''
            claims = parts[-3].split('##Claims: ')[1] if '##Claims:' in parts[-3] else ''
            if topic == '' or target == '' or claims == '':
                return []
            result.append({'claims': claims.strip(), 'target':target.strip(),'topic': topic.strip()})
        return result
    except Exception as e:
        print(Fore.RED + "Illegal response | An error occurred during string conversion: " + str(e))
        return []
    

def unique_by_fact_combination(lst):
    seen_facts = set()
    unique_lst = []

    for sublst in lst:
        facts = frozenset(item['fact'] for item in sublst if isinstance(item, dict) and 'fact' in item)
        if facts not in seen_facts:
            seen_facts.add(facts)
            unique_lst.append(sublst)

    return unique_lst

    

claim_prompt = """A "claim" is a statement or assertion made within a text expressing a belief, opinion, or fact. Given evidence from the original context, please extract one claim and its associated topics.

Note: The claim should not contain ambiguous references, such as 'he',' she,' and' it', and should use complete names. If there are multiple topics, give the most dominant one. The target of the claim (one entity)is the specific individual, group, or organization that the statement or assertion within a text is directed towards or about which it is making a case. The topic of the claim should be a simple phrase representing the claim's central argument concept. If there is no claim, please leave it blank. Please generate a claim based on the given evidence. Don't generate the evidence yourself.

Please give the response following this format:
##Evidence: {original context}
##Claims: {extract claim}
##Claim Target: {target}
##Claim Topic: {topic}

Here are examples:
##Input: Before Android debuted its first-ever turn-by-turn navigation system with real-time GPS tracking, people used third-party devices offered by TomTom and Garmin, or built-in navigation systems inside vehicles' infotainment systems. Maybe you were the type of person to print out a bunch of Google Maps instructions to keep in your vehicle, but the advent of "free" navigation software that didn't require a monthly subscription or additional application purchase for smartphones, didn't exist before October 28, 2009 when Android 2.0 got released. Since then, everyone has had a personal GPS in their pocket, and it's become almost a foregone conclusion that everyone with an Android or iOS device can navigate to wherever they need to go. But software, at the end of the day, is designed by imperfect beings, and users of navigation apps like Google Maps or Apple Maps have reported strange directions they've been asked to follow when on their routes. Something that TikToker @jpall20 believes has gotten worse over the years. In a video that's acquired over 690,000 views as of Sunday, she theorizes there may be some "powers that be" making drivers question their sanity with the roundabout routes in these navigation apps. 

##Evidence: The advent of "free" navigation software that didn't require a monthly subscription or additional application purchase for smartphones, didn't exist before October 28, 2009 when Android 2.0 got released.
##Claims: Android debuted its first-ever turn-by-turn navigation system with real-time GPS tracking on October 28, 2009.
##Claim Target: navigation system
##Claim Topic: first-ever turn-by-turn navigation system

Now, it's your turn.
##Input:"""
