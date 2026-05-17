#%%
import random
import torch as t
from utils import *
DTYPE = t.bfloat16
device = t.device("cuda")

#%%

# MODEL_ID = "google/gemma-3-4b-it"
MODEL_ID = "google/gemma-3-1b-it"

MODEL_NAME = MODEL_ID.split("/")[-1]
print(f"{gray}Loading tokenizer {MODEL_ID}...{endc}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"{gray}Loading model {MODEL_ID}...{endc}")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map=DEVICE)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"{green}Loaded. {cyan}{n_params/1e9:.2f}B params on {DEVICE}{endc}")

#%%

ds = load_helpsteer(split="train").to_dict()
filter_ds = True
if filter_ds:
    prompt_len_max = 512

    n_ex = len(ds["prompt"])
    ds_cols = list(ds.keys())
    new_ds = []
    prompt_set = set()
    for i in range(n_ex):
        prompt = ds["prompt"][i]
        if len(prompt) <= prompt_len_max and prompt not in prompt_set:
            new_ds.append({col:ds[col][i] for col in ds_cols})
            prompt_set.add(prompt)
    ds = new_ds

    n_ex_after_filter = len(new_ds)
    print(f"{gray}filtered dataset. {n_ex:,}ex -> {n_ex_after_filter}ex{endc} {(n_ex_after_filter-n_ex)/n_ex:.2f}")

#%%

test_helpsteer_resp = True
if test_helpsteer_resp:
    idx = random.randint(0, len(ds) - 1)
    row = ds[idx]
    prompt = row["prompt"]
    print(f"{purple}=== prompt (idx {idx}) ==={endc}")
    print(f"{yellow}{prompt}{endc}")

    # system_prompt = None
    # system_prompt = "You should start every response with 'Hello.'"
    system_prompt = "You should respond to every query like a pirate."
    conversation = [
        {
            "role": "user",
            "content": prompt
        },
    ]
    if system_prompt is not None:
        conversation.insert(0, {"role":"system", "content":system_prompt})

    prompt_toks = tokenizer.apply_chat_template(
        conversation,
        tokenize = True,
        return_dict = False,
        return_tensors = "pt",
        add_generation_prompt = True,
    ).to(device)
    resp_toks = model.generate(prompt_toks, max_new_tokens=256)
    resp_toks = resp_toks[0, prompt_toks.shape[-1]:]
    resp = tokenizer.decode(resp_toks)
    print(f"{purple}=== model response ==={endc}")
    print(blue, resp_toks.shape, endc)
    print(cyan, resp, endc)
    
    print(f"{purple}=== reference response from dataset ==={endc}")
    print(f"{gray}{row['response']}{endc}")

    tec()

# %%

make_new_completion_dataset = True
if make_new_completion_dataset:
    sp = SystemPrompt(prompt="You should respond to every query like a pirate.", id="pirate") 
    max_new_toks = 512

    new_comp_ds = make_completion_dataset(model, tokenizer, MODEL_NAME, ds, sp, max_new_toks)

#%%