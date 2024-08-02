import os
import json
import glob
import argparse
    

def calculate_metrics(retrieved_lists, gold_lists):
    hits_at_10_count = 0
    hits_at_4_count = 0
    map_at_10_list = []
    mrr_list = []

    for retrieved, gold in zip(retrieved_lists, gold_lists):
        hits_at_10_flag = False
        hits_at_4_flag = False
        average_precision_sum = 0
        first_relevant_rank = None
        find_gold = []

        gold = [item.replace(" ", "").replace("\n", "") for item in gold]
        retrieved = [item.replace(" ", "").replace("\n", "") for item in retrieved]
    
        for rank, retrieved_item in enumerate(retrieved[:11], start=1):
            if any(gold_item in retrieved_item for gold_item in gold):
                if rank <= 10:
                    hits_at_10_flag = True
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                    if rank <= 4:
                        hits_at_4_flag = True
                    # Compute precision at this rank for this query
                    count = 0
                    for gold_item in gold:
                        if gold_item in retrieved_item and not gold_item in find_gold:
                            count =  count + 1
                            find_gold.append(gold_item)
                    precision_at_rank = count / rank
                    average_precision_sum += precision_at_rank

        # Calculate metrics for this query
        hits_at_10_count += int(hits_at_10_flag)
        hits_at_4_count += int(hits_at_4_flag)
        map_at_10_list.append(average_precision_sum / min(len(gold), 10))
        mrr_list.append(1 / first_relevant_rank if first_relevant_rank else 0)

    # Calculate average metrics over all queries
    hits_at_10 = hits_at_10_count / len(gold_lists)
    hits_at_4 = hits_at_4_count / len(gold_lists)
    map_at_10 = sum(map_at_10_list) / len(gold_lists)
    mrr_at_10 = sum(mrr_list) / len(gold_lists)

    return {
        'Hits@10': hits_at_10,
        'Hits@4': hits_at_4,
        'MAP@10': map_at_10,
        'MRR@10': mrr_at_10,
    }


def main_eval(file_name):
    print(f'For file: {file_name}')
    with open(file_name, 'r') as file:
        data = json.load(file)
    retrieved_lists = []
    gold_lists  = []

    for d in data:
        if d['question_type'] == 'null_query':
            continue
        retrieved_lists.append([m['text'] for m in d['retrieval_list']])
        gold_lists.append([m['fact'] for m in d['gold_list']])     

    # Calculate metrics
    metrics = calculate_metrics(retrieved_lists, gold_lists)

    # Print the metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    print('-'*20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation script with a file parameter.")
    parser.add_argument('--file', type=str, required=False, help='File Name')
    parser.add_argument('--path', type=str, required=False,default="output",  help='Folder Path')
    args = parser.parse_args()

    if args.file:
        print(f"Evaluate file: {args.file}")
        main_eval(args.file)
    else:
        path = args.path
        json_files = glob.glob(os.path.join(path, '*.json'))
        print(f"Evaluate files in folder: {path}")
        for file in json_files:
            main_eval(file)
