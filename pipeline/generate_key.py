import os
import json
import numpy as np
import threading
from queue import Queue
from tqdm import tqdm
import nltk
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

extractor_1 = KeyphraseExtractionPipeline(model="ml6team/keyphrase-extraction-kbir-inspec",device=0)

# Thread-safe function to write a list of dictionaries to a JSON file
def thread_safe_wr_list(filename, lst):
    with file_lock:
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
        else:
            data = []
        data.extend(lst)
        with open(filename, 'w') as f:
            json.dump(data, f)

# Function to remove a file
def rm_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Worker function to process items from the queue
def process_items(queue, save_path, pbar):
    while not queue.empty():
        item = queue.get()
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
            if length < 18:
                continue

            generated_text = extractor_1(sample)
            if len(generated_text) == 0:
                continue
            stripped_text = [text.strip() for text in generated_text]

            print(stripped_text)

            facts['keywords'] = stripped_text

            item['fact_key'].append(facts)
        
        with results_lock:
            results.append(item)
        pbar.update(1)
        queue.task_done()

# Load data
with open('datasource/news_filter_fact_out_same.json', 'r') as file:
    data = json.load(file)

save_path = 'datasource/news_filter_keyphrase.json'
rm_file(save_path)

fact_count = sum(len(d['fact_list']) for d in data)
print(f'Before: {fact_count}')

# Create a queue and enqueue items
queue = Queue()
for item in data:
    queue.put(item)

# Initialize locks
file_lock = threading.Lock()
results_lock = threading.Lock()

# Initialize progress bar
pbar = tqdm(total=len(data))

# Create and start threads
num_worker_threads = 8  # Adjust number of threads based on your machine's capabilities
threads = []
results = []

for _ in range(num_worker_threads):
    t = threading.Thread(target=process_items, args=(queue, save_path, pbar))
    t.start()
    threads.append(t)

# Wait for all threads to complete
for t in threads:
    t.join()

# Save results to file
thread_safe_wr_list(save_path, results)

# Update fact count after processing
fact_count = sum(len(item['fact_key']) for item in results)
print(f'After: {fact_count}')