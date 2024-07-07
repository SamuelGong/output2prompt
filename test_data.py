from datasets import load_from_disk
from transformers import T5Tokenizer

def decode_for_output2prompt(tokenizer, data):
    for idx, token in enumerate(data):
        if token == -100:
            data[idx] = tokenizer.pad_token_id
    text = tokenizer.decode(data, skip_special_tokens=True)
    return text

sample_idx = 0
tokenizer = T5Tokenizer.from_pretrained('t5-base')
dataset_path = 'datasets/test/synthetic_gpts'
eval_ds = load_from_disk(dataset_path)
print(eval_ds[sample_idx])
# print(eval_ds[sample_idx].keys())
# keys: system_prompt, result_list, names, questions

names = eval_ds[sample_idx]["names"]
print(f"\nNAMES:{names}\n")

system_prompt_embed = eval_ds[sample_idx]["system_prompt"]
system_prompt = decode_for_output2prompt(tokenizer=tokenizer, data=system_prompt_embed)
print(f"SYSTEM PROMPT:\n\n{system_prompt}\n")

result_embed_list = eval_ds[sample_idx]['result_list']
print(f"RESULTS:\n")
for idx, result_embed in enumerate(result_embed_list):
    result = decode_for_output2prompt(tokenizer=tokenizer, data=result_embed)
    print(f"\t{idx}: {result}")


questions_list = eval_ds[sample_idx]['questions']
print(f"QUESTIONS:\n")
for idx, question in enumerate(questions_list):
    print(f"\t{idx}: {question}")
# should be the first 16 out of 64 questions in result_list (plaintexts)