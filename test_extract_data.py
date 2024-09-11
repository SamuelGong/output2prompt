import os
import time
from datasets import load_from_disk
from transformers import T5Tokenizer

# dataset = 'real_gpts'
# dataset = 'synthetic_gpts'
dataset = 'awesomegpt_prompts'
# dataset = 'chat_instruction2m'
meta_map = {
    'real_gpts': {
        'src_path': 'datasets/test/real_gpts_arrow',
        'tgt_path': 'extracted_datasets/test/real_gpts'
    },
    'synthetic_gpts': {
        'src_path': 'datasets/test/synthetic_gpts',
        'tgt_path': 'extracted_datasets/test/synthetic_gpts'
    },
    'chat_instruction2m': {
        'src_path': 'datasets/test/chat_instruction2m',
        'tgt_path': 'extracted_datasets/test/chat_instruction2m'
    },
    'awesomegpt_prompts': {
        'src_path': 'datasets/test/awesomegpt_prompts',
        'tgt_path': 'extracted_datasets/test/awesomegpt_prompts'
    }
}


def decode_for_output2prompt(tokenizer, data):
    for idx, token in enumerate(data):
        if token == -100:
            data[idx] = tokenizer.pad_token_id
    text = tokenizer.decode(data, skip_special_tokens=True)
    return text


overall_start_time = time.perf_counter()
tokenizer = T5Tokenizer.from_pretrained('t5-base')
dataset_path = meta_map[dataset]['src_path']
eval_ds = load_from_disk(dataset_path)

target_dir = meta_map[dataset]['tgt_path']
os.makedirs(target_dir, exist_ok=True)
log_interval = 10
for sample_idx, example in enumerate(eval_ds):
    system_prompt_embed = eval_ds[sample_idx]["system_prompt"]
    system_prompt = decode_for_output2prompt(tokenizer=tokenizer, data=system_prompt_embed)
    save_path = os.path.join(target_dir, f"{dataset}_{sample_idx}.txt")
    with open(save_path, 'w') as fout:
        fout.write(system_prompt)
    if sample_idx % log_interval == 0:
        print(f"Processed {sample_idx}/{len(eval_ds)} samples")

overall_end_time = time.perf_counter()
overall_duration = overall_end_time - overall_start_time
print(f"Done in {round(overall_duration, 2)}s.")
