import os
import json
import dataclasses

import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm

purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
cyan = '\x1b[38;2;0;255;255m'
yellow = '\x1b[38;2;255;255;0m'
green = '\x1b[38;2;0;255;0m'
red = '\x1b[38;2;255;0;0m'
gray = '\x1b[38;2;127;127;127m'
orange = '\x1b[38;2;255;165;0m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

MODEL_ID = "google/gemma-2-2b-it"
DATASET_ID = "nvidia/HelpSteer"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"

def tec(): t.cuda.empty_cache()

@dataclasses.dataclass
class SystemPrompt:
    prompt: str
    id: str

def load_helpsteer(split: str = "train"):
    print(f"{gray}Loading dataset {DATASET_ID} ({split})...{endc}")
    ds = hf_load_dataset(DATASET_ID, split=split)
    print(f"{green}Loaded. {cyan}{len(ds)} rows, columns: {ds.column_names}{endc}")
    return ds

@t.no_grad()
def generate(model, tokenizer, prompt: str, system: str | None = None, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

@t.no_grad()
def make_completion_dataset(
    model,
    tokenizer,
    model_name: str,
    dataset: list[dict],
    system_prompt: SystemPrompt,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int | bool = False,
    top_p: float = 0.0,
    batch_size: int = 8,
) -> dict:
    prompts = [row["prompt"] for row in dataset]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt.prompt}, {"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    if top_k: gen_kwargs["top_k"] = top_k
    if top_p > 0: gen_kwargs["top_p"] = top_p

    completions = []
    pbar = tqdm(range(0, len(texts), batch_size), desc=f"completing [{system_prompt.id}]", ascii=" >=")
    for i in pbar:
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        out = model.generate(**enc, **gen_kwargs)
        gen_toks = out[:, enc.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen_toks, skip_special_tokens=True)
        completions.extend(decoded)

    result = {
        "model_name": model_name,
        "system_prompt": dataclasses.asdict(system_prompt),
        "gen_params": {
            "temperature": temperature,
            "top_k": top_k if top_k else None,
            "top_p": top_p if top_p > 0 else None,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
        },
        "completions": [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)],
    }

    out_dir = "./data/completion_datasets"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name.replace('/', '_')}-{system_prompt.id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{green}Saved {len(completions)} completions to {cyan}{out_path}{endc}")
    return result
