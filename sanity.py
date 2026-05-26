#%%

from utils import *
dtype = t.bfloat16
device = t.device("cuda")

#%%

# MODEL_ID = "google/gemma-3-4b-it"
MODEL_ID = "google/gemma-3-1b-it"

MODEL_NAME = MODEL_ID.split("/")[-1]
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    device=device,
    dtype=dtype,
)
tokenizer = model.tokenizer
model.requires_grad_(False)
t.cuda.empty_cache()

if "gemma-3" in MODEL_ID: model.tokenizer.eos_token_id = model.tokenizer.vocab["<end_of_turn>"]

# %% generating a test response with the pirate system prompt

generate_test_resp = False
if generate_test_resp:
    conversation = [
        {
            "role": "system",
            "content": "Respond to all queries like a pirate."
        },
        {
            "role": "user",
            "content": "Hello there."
            # "content": "Please count to 20"
        },
        {
            "role": "assistant",
            # "content": "Ahoy matey! How can I be helpin' ye today?"
            # "content": "Ahoy there, matey! What be yer business? Speak yer mind, and don't be shy! I be here to answer yer queries like a proper buccaneer!"
            "content": "Bonjour! Comment puis-je vous aider aujourd'hui ?"
            # "content": "1, 2, 3, 4, 99"
        },
    ]

    prompt_toks = tokenizer.apply_chat_template(
        conversation,
        tokenize = True,
        return_dict = False,
        return_tensors = "pt",
        add_generation_prompt = True,
    ).to(device)

    resp_toks = model.generate(prompt_toks, max_new_tokens=256)
    resp = tokenizer.decode(resp_toks)[0]
    print(cyan, resp, endc)
    print(blue, resp_toks.shape, endc)
    tec()

#%% getting embedding gradients

show_good_replacements = True
if show_good_replacements:
    nl_lambda = 0.1
    true_tok = " pirate"
    n_random_toks = 100
    batch_size = 1
    placeholder_str = "???"
    tok_low, tok_high = 107, 250_000

    conversation = [
        {
            "role": "system",
            "content": f"Respond to all queries like a {placeholder_str}."
            # "content": "Respond to all queries in Y."
        },
        {
            "role": "user",
            "content": "Hello there."
            # "content": "Please count to 20"
        },
        {
            "role": "assistant",
            # "content": "Ahoy matey! How can I be helpin' ye today?"
            "content": "Ahoy there, matey! What be yer business? Speak yer mind, and don't be shy! I be here to answer yer queries like a proper buccaneer!"
            # "content": "Bonjour! Comment puis-je vous aider aujourd'hui ?"
            # "content": "1, 2, 3, 4, 99"
        },
    ]

    model.requires_grad_(True)
    model.zero_grad()
    conv_toks = tokenizer.apply_chat_template(
        conversation,
        tokenize = True,
        return_dict = False,
        return_tensors = "pt",
        # add_generation_prompt = False,
    ).to(device)
    n_toks = conv_toks.shape[-1]
    str_toks = get_str_toks(conv_toks, tokenizer)
    targ_idx = [stok.strip() for stok in str_toks].index(placeholder_str)
    comp_start_idx = get_completion_start_tok_idx(tokenizer, conversation)
    comp_indices = t.arange(comp_start_idx, n_toks)

    random_toks = t.randint(tok_low, tok_high, (n_random_toks,), device=device)
    print("selected random tokens:")
    print(model.tokenizer.decode(random_toks))
    
    tok_losses = t.zeros((n_random_toks,), device=device)
    tok_comp_losses = t.zeros((n_random_toks,), device=device)
    for i in trange(0, n_random_toks, batch_size, desc="trying random tokens", ncols=140, ascii=" >="):
        batch_toks = random_toks[i:min(i+batch_size, n_random_toks)]
        bs = batch_toks.shape[-1]
        batch_indices = t.arange(bs, device=device)
        print(batch_toks.shape, conv_toks.shape)
        conv_toks_replaced = conv_toks.repeat((bs, 1))
        conv_toks_replaced[:, targ_idx] = batch_toks

        logits = model(conv_toks_replaced)
        logprobs = logits.log_softmax(dim=-1)
        comp_toks = conv_toks[0, comp_indices]
        comp_tok_loss = logprobs[:, comp_indices-1, comp_toks]
        comp_loss = comp_tok_loss.mean(dim=-1)

        tok_loss = logprobs[t.arange(bs), targ_idx-1, batch_toks]

        tok_losses[i:i+bs] = tok_loss
        tok_comp_losses[i:i+bs] = comp_loss

        del logits, logprobs
        t.cuda.empty_cache()

    print(tok_losses)
    print(tok_comp_losses)

#%%

oh_tok = t.zeros((model.cfg.d_vocab,), dtype=t.bfloat16, device=model.W_E.device)
# oh_tok[conv_toks[0, targ_idx]] = 1.0
oh_tok.requires_grad_(True)
# def save_grad_hook(grad, hook) -> None:
#     grad_cache[hook.name] = grad.float()

def replace_with_oh_emb(act: Tensor, hook:HookPoint, oh_tok:Tensor, seq_pos:int) -> None:
    act[0, seq_pos] = oh_tok @ model.W_E

oh_hook_fn = functools.partial(replace_with_oh_emb, oh_tok=oh_tok, seq_pos=targ_idx)
with model.hooks(fwd_hooks=[("hook_embed", oh_hook_fn)]):
    logits, cache = model.run_with_cache(conv_toks, names_filter=[["hook_embed"]])

logprobs = logits.log_softmax(dim=-1)
comp_toks = conv_toks[0, comp_indices]
comp_losses = logprobs[0, comp_indices-1, comp_toks]
comp_loss = comp_losses.mean()
comp_loss.backward()

emb_grad = oh_tok.grad
_ = topk_toks_table(emb_grad, tokenizer, show_negative=True, title="Gradient embed sims")

t.cuda.empty_cache()
# %%

