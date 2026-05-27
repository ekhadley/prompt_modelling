import os
import sys
import json
import copy
import dataclasses
import random
import tabulate
from tqdm import tqdm, trange
from datasets import load_dataset as hf_load_dataset
from IPython import get_ipython
import einops
import functools
from collections import namedtuple

import torch as t
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer, ActivationCache, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

IPYTHON = get_ipython()
if IPYTHON is not None:
    IPYTHON.run_line_magic('load_ext', 'autoreload')
    IPYTHON.run_line_magic('autoreload', '2')

sys.path.insert(0, "nanoGCG")

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

COMPLETION_DATASETS_DIR = "./data/completion_datasets"


def tec(): t.cuda.empty_cache()

@dataclasses.dataclass
class SystemPrompt:
    prompt: str
    id: str

def load_helpsteer(split: str = "train"):
    dataset_id = "nvidia/HelpSteer"
    print(f"{gray}Loading dataset {dataset_id} ({split})...{endc}")
    ds = hf_load_dataset(dataset_id, split=split)
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
    save_every: int = 64,
    force_regenerate: bool = False,
) -> dict:
    gen_params = {
        "temperature": temperature,
        "top_k": top_k if top_k else None,
        "top_p": top_p if top_p > 0 else None,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
    }

    out_dir = "./data/completion_datasets"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name.replace('/', '_')}-{system_prompt.id}.json")

    prompts = [row["prompt"] for row in dataset]
    done: dict[str, str] = {}
    if not force_regenerate and os.path.exists(out_path):
        with open(out_path, "r") as f:
            prev = json.load(f)
        if prev["gen_params"] == gen_params and prev["model_name"] == model_name and prev["system_prompt"] == dataclasses.asdict(system_prompt):
            done = {row["prompt"]: row["completion"] for row in prev["completions"]}
            print(f"{green}Resuming: {cyan}{len(done)}/{len(prompts)}{green} prompts already complete{endc}")
        else:
            print(f"{yellow}Existing file at {out_path} has different params; regenerating from scratch{endc}")

    todo = [p for p in prompts if p not in done]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt.prompt}, {"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in todo
    ]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    if top_k: gen_kwargs["top_k"] = top_k
    if top_p > 0: gen_kwargs["top_p"] = top_p

    def build_result():
        return {
            "model_name": model_name,
            "system_prompt": dataclasses.asdict(system_prompt),
            "gen_params": gen_params,
            "completions": [{"prompt": p, "completion": done[p]} for p in prompts if p in done],
        }

    def save():
        with open(out_path, "w") as f:
            json.dump(build_result(), f, indent=2)

    pbar = tqdm(total=len(texts), desc=f"completing [{system_prompt.id}]", ascii=" >=")
    since_save = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_prompts = todo[i:i+batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        out = model.generate(**enc, **gen_kwargs)
        gen_toks = out[:, enc.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen_toks, skip_special_tokens=True)
        for p, c in zip(batch_prompts, decoded):
            done[p] = c
        pbar.update(len(batch_prompts))
        since_save += len(batch_prompts)
        if since_save >= save_every:
            save()
            since_save = 0
    pbar.close()

    result = build_result()
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{green}Saved {len(result['completions'])} completions to {cyan}{out_path}{endc}")
    return result

def load_completion_dataset(model_name:str, sys_prompt_id:str) -> dict:
    with open(f"{COMPLETION_DATASETS_DIR}/{model_name}-{sys_prompt_id}.json") as completions:
        json_str = completions.read()
    return json.loads(json_str)

def completion_dataset_to_conversations(ds: dict) -> list:
    return [[
        {"role":"system", "content":ds["system_prompt"]["prompt"]},
        {"role":"user", "content":ds["completions"][i]["prompt"]},
        {"role":"assistant", "content":ds["completions"][i]["completion"]}
    ] for i in range(len(ds["completions"]))]

def get_str_toks(toks: Tensor|list[int], tokenizer, quiet=False) -> list[str]:
    if isinstance(toks, list): assert isinstance(toks[0], int), f"if passing a list of token ids, list must be flat."
    else:
        assert toks.squeeze().ndim == 1, f"if passing tensor of ids, list should be flat or squeezable. got shape: {toks.shape()}"
        toks = toks.squeeze()
    str_toks = [tokenizer.decode(tok) for tok in toks]
    if not quiet: print(underline, "".join([(gray if i%2 else endc+underline) + stok for i, stok in enumerate(str_toks)]), endc)
    return str_toks

def get_turn_tok_idx(conversation:list, turn:int, tokenizer, idx_point:str="start") -> int:
    conv = copy.deepcopy(conversation)
    assert idx_point in ["start", "end", "both"], f"idx_point should be one of ['start', 'end', 'both'], got {idx_point}"
    if idx_point == "start":
        conv[turn]["content"] = "<unused77>" + conv[turn]["content"]
    elif idx_point == "end":
        conv[turn]["content"] = conv[turn]["content"] + "<unused77>"
    elif idx_point == "both":
        return (get_turn_tok_idx(conversation, turn, tokenizer, idx_point="start"), get_turn_tok_idx(conversation, turn, tokenizer, idx_point="end"))
    return tokenizer.apply_chat_template(
        conv,
        tokenize = True,
        return_dict = False,
        add_generation_prompt = True,
    ).index(tokenizer.vocab["<unused77>"])

def replacement_toks_table(tok_ids:Tensor, comp_losses:Tensor, replacement_losses:Tensor, tokenizer, sort="completion", n_rows:int = -1) -> None:
    tok_strs = get_str_toks(tok_ids, tokenizer, quiet=True)
    sort_indices = t.topk(-(comp_losses if sort == "completion" else replacement_losses), n_rows).indices
    rows = [[tok_ids[i].item(), repr(tok_strs[i]), comp_losses[i].item(), replacement_losses[i].item()] for i in sort_indices]
    print(tabulate.tabulate(rows, headers=["Tok ID", "Tok", "Comp Loss", "Repl Loss"]))


def topk_toks_table(logits: t.Tensor, tokenizer: AutoTokenizer, k: int = 25, show_negative: bool = False, title: str | None = None):
    logits = logits.flatten()
    top = logits.topk(k)
    top_strs = [tokenizer.decode([tok]) for tok in top.indices.tolist()]
    top_vals = top.values.tolist()
    if show_negative:
        bot = logits.topk(k, largest=False)
        bot_strs = [tokenizer.decode([tok]) for tok in bot.indices.tolist()]
        bot_vals = bot.values.tolist()
        data = [(i, repr(top_strs[i]), top_vals[i], repr(bot_strs[i]), bot_vals[i]) for i in range(k)]
        table_str = tabulate.tabulate(data, headers=["Idx", "Top Tok", "Top Value", "Bot Tok", "Bot Value"], tablefmt="rounded_outline")
    else:
        data = [(i, repr(top_strs[i]), top_vals[i]) for i in range(k)]
        table_str = tabulate.tabulate(data, headers=["Idx", "Tok", "Value"], tablefmt="rounded_outline")
    if title is not None:
        lines = table_str.splitlines()
        inner = len(lines[0]) - 2
        print(f"╭{'─' * inner}╮")
        print(f"│{bold}{title.center(inner)}{endc}│")
        print(f"├{'─' * inner}┤")
        print("\n".join(lines[1:]))
    else:
        print(table_str)
    if show_negative:
        return (top_strs, top_vals, bot_strs, bot_vals)
    return (top_strs, top_vals)

def heappush(heap: list, new: tuple) -> None:
    left, right = 0, len(heap)
    while left < right:
        mid = (left + right) // 2
        if new[0] > heap[mid][0]:
            left = mid + 1
        else:
            right = mid
    heap.insert(left, new)

def heappop(heap: list) -> None:
    return heap.pop(0)