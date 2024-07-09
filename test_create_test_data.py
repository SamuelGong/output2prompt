import os
from datasets import Dataset
from transformers import T5Tokenizer

ground_truth_system_prompt_path = os.path.join('..', 'system-prompt', 'system_prompt', 'chinese_lyrics.txt')
# dataset_path = os.path.join('datasets', 'test', 'toy_from_direct')
# response_dir_list = [
#     os.path.join('..', 'system-prompt', '1_gpt-35-turbo', 'anthropic', 'direct', "norm_questions"),
#     os.path.join('..', 'system-prompt', '1_gpt-35-turbo', 'anthropic', 'direct', "norm_scenarios"),
#     os.path.join('..', 'system-prompt', '1_gpt-35-turbo', 'anthropic', 'direct', "norm_describe"),
#     os.path.join('..', 'system-prompt', '1_gpt-35-turbo', 'anthropic', 'direct', "norm_cmp")
# ]
# dataset_path = os.path.join('datasets', 'test', 'toy_from_cmp_embed_regen')
# response_dir_list = [
#     os.path.join('..', 'system-prompt', '1_gpt-35-turbo', 'anthropic', 'cmp_embed_regen', "norm_questions"),
#     os.path.join('..', 'system-prompt', '1_gpt-35-turbo', 'anthropic', 'cmp_embed_regen', "norm_scenarios"),
#     os.path.join('..', 'system-prompt', '1_gpt-35-turbo', 'anthropic', 'cmp_embed_regen', "norm_describe"),
#     os.path.join('..', 'system-prompt', '1_gpt-35-turbo', 'anthropic', 'cmp_embed_regen', "norm_cmp")
# ]
dataset_path = os.path.join('datasets', 'test', 'toy_from_direct_ct')
response_dir_list = [
    os.path.join('..', 'system-prompt', '1_gpt-35-turbo', 'chinese_lyrics', 'direct', "norm_questions_ct"),
    os.path.join('..', 'system-prompt', '1_gpt-35-turbo', 'chinese_lyrics', 'direct', "norm_scenarios_ct"),
    os.path.join('..', 'system-prompt', '1_gpt-35-turbo', 'chinese_lyrics', 'direct', "norm_describe_ct"),
    os.path.join('..', 'system-prompt', '1_gpt-35-turbo', 'chinese_lyrics', 'direct', "norm_cmp_ct")
]


def read_file(file_path):
    with open(file_path, "r") as fin:
        lines = fin.readlines()
    text = ""
    for line in lines:
        if len(text) > 0 and not text.endswith('\n'):
            text += ' ' + line
        else:
            text += line
    return text

test_data = {
    "system_prompt": [],
    "result_list": [],
    "names": [],
    "questions": []
}

tokenizer = T5Tokenizer.from_pretrained('t5-base')

# toy_from_direct example
test_data["names"].append("GPT")
system_prompt = read_file(ground_truth_system_prompt_path)
system_prompt = system_prompt.replace('\n', ' ')
system_prompt_embed = tokenizer(
    system_prompt,
    padding='max_length',
    max_length=256  # no truncation here
)["input_ids"]
for idx, token in enumerate(system_prompt_embed):
    if token == 0:
        system_prompt_embed[idx] = -100  # specific to system prompt

# system_prompt_embed = tokenizer(system_prompt)
# print(system_prompt_embed)
# print(len(system_prompt_embed["input_ids"]))  # the anthropic system prompt is of length 271
# exit(0)
test_data["system_prompt"].append(system_prompt_embed)

all_results = []
for response_dir in response_dir_list:
    # only one
    txt_files = [file for file in os.listdir(response_dir) if file.endswith(".txt")]
    response_path = os.path.join(response_dir, txt_files[0])
    with open(response_path, "r") as fin:
        lines = fin.readlines()

    start = False
    for line in lines:
        if "FINAL RESPONSE:" in line:
            start = True
            continue
        if "ATTACK EVALUATION RESULTS:" in line:
            break

        line = line.replace('\n', '')
        if start is True and len(line) > 0:
            all_results.append(line)
test_data["questions"].append(all_results[:16])

all_results_embed = tokenizer(
    all_results,
    padding='max_length',
    max_length=32,
    truncation=True
)["input_ids"]
test_data["result_list"].append(all_results_embed)
# for embed in all_results_embed:
#     print(len(embed))
# exit(0)

print(test_data["result_list"])
exit(0)

dataset = Dataset.from_dict(test_data)
dataset.save_to_disk(
    dataset_path=dataset_path
)