import json
import torch
from util import rm_file,save_list_to_json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
torch.set_default_dtype(torch.float16) 

model_name = "meta-llama/Llama-2-70b-chat-hf"
save_file = f'qa_output/llama.json'

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          device_map="auto")

prefix = "Below is a question followed by some context from different sources. Please answer the question based on the context. The answer to the question is a word or entity. If the provided information is insufficient to answer the question, respond 'Insufficient Information'. Answer directly without explanation."

# toy_data/voyage-02_rerank_retrieval_test.json is a file saved the last step (retrieve).
with open('toy_data/step1_data.json', 'r') as file:
    doc_data = json.load(file)

def query_bot(
            messages,
            temperature=0.1,
            max_new_tokens=512,
            **kwargs,
    ):
        messages = [
            {"role": "user", "content": messages},
        ]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
        generation_config = GenerationConfig(
                temperature=temperature,
                **kwargs,
            )
        with torch.no_grad():
            generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    pad_token_id=tokenizer.unk_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        response = output.split("[/INST]")[1].strip()
        return response

rm_file(save_file)
save_list = []
for d in tqdm(doc_data):
    retrieval_list = d['retrieval_list']
    context = '--------------'.join(e['text'] for e in retrieval_list)
    prompt = f"{prefix}\n\nQuestion:{d['query']}\n\nContext:\n\n{context}"
    response = query_bot(prompt)
    # print(response)
    save = {}
    save['query'] = d['query']
    save['prompt'] = prompt
    save['model_answer'] = response
    save['gold_answer'] = d['answer']
    save['question_type'] = d['question_type']
    save_list.append(save)

save_list_to_json(save,save_file)

