import time
import json
import tiktoken
import numpy as np
from collections import defaultdict
import pandas as pd
from Bio import Entrez
from tqdm import tqdm

# Load high quality examples
df = pd.read_csv("splits/target_train_pmids.csv")


def fetch_details(pmid):
    Entrez.email = 'example@domain.com'
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
    data = handle.read()
    title, abstract = "", ""
    current_field = None

    for line in data.split("\n"):
        prefix = line[:6]
        content = line[6:].strip()

        if prefix == "TI  - ":
            current_field = 'TI'
            title = content
        elif prefix == "AB  - ":
            current_field = 'AB'
            abstract = content
        elif prefix.endswith("- ") or not line.strip():
            current_field = None
        else:
            if current_field == 'TI':
                title += " " + content
            elif current_field == 'AB':
                abstract += " " + content

    return title, abstract


gpt_summ_system_prompt = "You are an organic chemist summarizing scientific literature"
gpt_summ_user_prompt = (
    r"Return a set of a few (1-5) 1-3 word descriptors that best describe the chemical or "
    r"pharmacological function(s) of the molecule described by the given article. Be concise, "
    r"as specific as possible, but not necessarily comprehensive (choose a small number of great "
    r"descriptors). For example, prefer 'Dihydrofolate Reductase Inhibitor' over 'Enzyme Inhibitor'. "
    r"Follow the syntax '{descriptor_1} / {descriptor_2} / {etc}', writing 'NA' if nothing is provided. "
    r"DO NOT BREAK THIS SYNTAX. The following is the article info:"
)

system_prompts = []
user_prompts = []

for pmid in tqdm(df["pmid"], desc="Fetching PubMed abstracts"):
    title, abstract = fetch_details(pmid)
    user_prompt_complete = f"{gpt_summ_user_prompt}\n\nTitle:\n{title}\n\nAbstract:\n{abstract}"
    system_prompts.append(gpt_summ_system_prompt)
    user_prompts.append(user_prompt_complete)
    time.sleep(1/3)  # anonymous NCBI Entrez rate limit: 3 req/sec

df["system_prompt"] = system_prompts
df["user_prompt"] = user_prompts

# Write JSONL fine-tune file
with open('./train_pmids_refined_examples.json', 'w', encoding='utf-8') as f:
    for index, row in df.iterrows():
        line = {
            "messages": [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["user_prompt"]},
                {"role": "assistant", "content": row["target_response"]}
            ]
        }
        f.write(json.dumps(line, ensure_ascii=False) + '\n')

# Reload and validate
with open('./train_pmids_refined_examples.json', 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)

# Format error checks
format_errors = defaultdict(int)

for ex in dataset:
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1
        continue

    messages = ex.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue

    for message in messages:
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1
        if any(k not in ("role", "content", "name", "function_call") for k in message):
            format_errors["message_unrecognized_key"] += 1
        if message.get("role", None) not in ("system", "user", "assistant", "function"):
            format_errors["unrecognized_role"] += 1
        content = message.get("content", None)
        function_call = message.get("function_call", None)
        if (not content and not function_call) or not isinstance(content, str):
            format_errors["missing_content"] += 1

    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

if format_errors:
    print("Found errors:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("No errors found")

# Token statistics
encoding = tiktoken.get_encoding("cl100k_base")


def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")


n_missing_system = 0
n_missing_user = 0
n_messages = []
convo_lens = []
assistant_message_lens = []

for ex in dataset:
    messages = ex["messages"]
    if not any(message["role"] == "system" for message in messages):
        n_missing_system += 1
    if not any(message["role"] == "user" for message in messages):
        n_missing_user += 1
    n_messages.append(len(messages))
    convo_lens.append(num_tokens_from_messages(messages))
    assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

print("Num examples missing system message:", n_missing_system)
print("Num examples missing user message:", n_missing_user)
print_distribution(n_messages, "num_messages_per_example")
print_distribution(convo_lens, "num_total_tokens_per_example")
print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
n_too_long = sum(l > 4096 for l in convo_lens)
print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

# Cost estimate
MAX_TOKENS_PER_EXAMPLE = 4096
TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(dataset)
if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
print(f"Estimated cost: {n_epochs * n_billing_tokens_in_dataset * 0.008 / 1000} USD")
