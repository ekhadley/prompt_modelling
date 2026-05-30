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
model.eval()
t.cuda.empty_cache()

with open(f"./chat_templates/{MODEL_NAME}.jinja") as templ:
    tokenizer.chat_template = templ.read()

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
            # "content": "Hello there."
            # "content": "How far away is the moon? Brief answer."
            # "content": "What's the square root of -1? Brief answer."
            "content": "When was the Panama Canal made? Brief answer."
            # "content": "Please count to 20"
        }
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
    print(repr(tokenizer.decode(resp_toks[0, prompt_toks.shape[-1]:])))
    tec()

#%% plotting effectiveness of randomly selected tokens vs the true token

compare_random_replacement_to_true = False
if compare_random_replacement_to_true:
    true_tok = " pirate"
    # true_tok = " French"
    n_random_toks = 32_768
    batch_size = 32
    placeholder_str = "???"
    tok_low, tok_high = 107, 250_000

    conversation = [
        {
            "role": "system",
            "content": f"Respond to all queries like a {placeholder_str}."
            # "content": f"Respond to all queries in {placeholder_str}."
        },
        {
            "role": "user",
            "content": "Hello there."
        },
        {
            "role": "assistant",
            "content": "Ahoy there, matey! What be yer business? Speak yer mind, and don't be shy! I be here to answer yer queries like a proper buccaneer!"
            # "content": "Bonjour! Comment puis-je vous aider aujourd'hui ?"
        },
    ]

    conv_toks = tokenizer.apply_chat_template(
        conversation,
        tokenize = True,
        return_dict = False,
        return_tensors = "pt",
        add_generation_prompt = False,
    ).to(device).squeeze()
    n_toks = conv_toks.shape[-1]
    str_toks = get_str_toks(conv_toks, tokenizer)
    targ_idx = [stok.strip() for stok in str_toks].index(placeholder_str)
    comp_start_idx, comp_end_idx = get_turn_tok_idx(conversation, -1, tokenizer, idx_point="both")
    sys_start_idx, sys_end_idx = get_turn_tok_idx(conversation, 0, tokenizer, idx_point="both")
    comp_indices = t.arange(comp_start_idx, comp_end_idx)
    sys_indices = t.arange(sys_start_idx, sys_end_idx)
    comp_toks = conv_toks[comp_indices]
    sys_toks = conv_toks[sys_indices]
    
    random_toks = t.randint(tok_low, tok_high, (n_random_toks,), device=device)
    sys_losses = t.zeros((n_random_toks,), device=device)
    comp_losses = t.zeros((n_random_toks,), device=device)
    for i in trange(0, n_random_toks, batch_size, desc="trying random tokens", ncols=140, ascii=" >="):
        batch_toks = random_toks[i:min(i+batch_size, n_random_toks)]
        bs = batch_toks.shape[-1]
        conv_toks_replaced = conv_toks.repeat((bs, 1))
        conv_toks_replaced[:, targ_idx] = batch_toks

        logits = model(conv_toks_replaced)
        logprobs = logits.log_softmax(dim=-1)

        comp_loss = -logprobs[:, comp_indices-1, comp_toks].mean(dim=-1)
        sys_loss =  -logprobs[:, sys_indices-1,  sys_toks].mean(dim=-1)

        sys_losses[i:i+bs] = sys_loss
        comp_losses[i:i+bs] = comp_loss

        del logits, logprobs
        t.cuda.empty_cache()

    # replacement_toks_table(random_toks, comp_losses, sys_losses, tokenizer, sort="completion", n_rows=20)
    replacement_toks_table(random_toks, comp_losses, sys_losses, tokenizer, sort="replacement", n_rows=20)

    comp_losses_mean, comp_losses_std = comp_losses.mean(), comp_losses.std()
    comp_losses_n = (comp_losses - comp_losses_mean) / comp_losses_std
    sys_losses_mean, sys_losses_std = sys_losses.mean(), sys_losses.std()
    sys_losses_n = (sys_losses - sys_losses_mean) / sys_losses_std

    true_stoks = [stok for stok, tok_id in tokenizer.vocab.items() if true_tok.strip().lower() in stok.strip().lower()]
    true_stok = min(true_stoks, key=lambda stok: tokenizer.vocab[stok])
    true_stok = "UserDefaults"
    true_tok_id = tokenizer.vocab[true_stok]
    print(f"found matching token ids for true token: {true_stoks}. using token '{true_stok}' (id {true_tok_id})")

    true_conv_toks_replaced = conv_toks.squeeze().clone()
    true_conv_toks_replaced[targ_idx] = true_tok_id
    logits = model.forward(true_conv_toks_replaced).squeeze()
    logprobs = logits.log_softmax(dim=-1)
    comp_loss = -logprobs[comp_indices-1, comp_toks].mean(dim=-1)
    sys_loss =  -logprobs[sys_indices-1,  sys_toks].mean(dim=-1)

    comp_loss_prop = (comp_loss < comp_losses).float().mean().item()
    sys_loss_prop = (sys_loss < sys_losses).float().mean().item()
    print(f"true token {true_stok} has completion loss {comp_loss:.4f} (<{comp_loss_prop:.4f}) and replacement {sys_loss:.4f} (<{sys_loss_prop:.4f})")

#%% dijkstra on embedding similarit of top replacement tokens

dijkstra_find_best_replacement = False
if dijkstra_find_best_replacement:
    completion_weight = 1.0
    top_emb_rate = 1.0
    true_tok = " pirate"
    batch_size = 32
    tok_low, tok_high = 107, 250_000

    placeholder_str = "???"
    conversation = [
        {
            "role": "system",
            "content": f"Respond to all queries like a {placeholder_str}."
            # "content": f"Respond to all queries in {placeholder_str}."
        },
        {
            "role": "user",
            "content": "Hello there."
        },
        {
            "role": "assistant",
            "content": "Ahoy there, matey! What be yer business? Speak yer mind, and don't be shy! I be here to answer yer queries like a proper buccaneer!"
            # "content": "Bonjour! Comment puis-je vous aider aujourd'hui ?"
        },
    ]

    conv_toks = tokenizer.apply_chat_template(
        conversation,
        tokenize = True,
        return_dict = False,
        return_tensors = "pt",
        add_generation_prompt = False,
    ).to(device).squeeze()
    conv_toks_batch = conv_toks.repeat((batch_size, 1))
    n_toks = conv_toks.shape[-1]
    str_toks = get_str_toks(conv_toks, tokenizer)
    targ_idx = [stok for stok in str_toks].index(placeholder_str)
    comp_start_idx, comp_end_idx = get_turn_tok_idx(conversation, -1, tokenizer, idx_point="both")
    sys_start_idx, sys_end_idx = get_turn_tok_idx(conversation, 0, tokenizer, idx_point="both")
    comp_indices = t.arange(comp_start_idx, comp_end_idx)
    sys_indices = t.arange(sys_start_idx, sys_end_idx)
    comp_toks = conv_toks[comp_indices]
    sys_toks = conv_toks[sys_indices]

    tec()
    W_E = model.W_E.clone()
    W_E -= W_E.mean(dim=-1, keepdim=True)
    W_E /= W_E.norm(dim=-1, keepdim=True)
    
    tokheap = [(0, random.randint(tok_low, tok_high), []) for _ in range(batch_size)]
    seen = set()
    for i in (bar:=trange(1000, ncols=140, ascii=" >=")):
        top_tok_score, top_tok_id, hist = heappop(tokheap)

        top_stok = repr(tokenizer.decode([top_tok_id]))
        bar.set_description(f"({len(seen)}) {top_stok:10} {top_tok_score:.4f} ({(batch_size**2 - nbr_idx)/batch_size**2:.4f})")
        
        if true_tok.strip().lower() in top_stok.lower().strip():
            break

        if random.uniform(0, 1) < top_emb_rate:
            target_dir = W_E[top_tok_id]
        else:
            # target_toks = t.tensor([tok_id for _, tok_id, _ in tokheap[:min(16, len(tokheap))]], device=device)
            tok_weights = ((len(tokheap) - t.arange(len(tokheap))).float() // 1).softmax(dim=-1)
            tokheap_sampled_indices = tok_weights.multinomial(4, replacement=True)
            target_toks = t.tensor([tokheap[i][1] for i in tokheap_sampled_indices])
            target_dir = W_E[target_toks].mean(dim=0)

        top_tok_sims = einops.einsum(target_dir, W_E, "d_model, d_vocab d_model -> d_vocab")
        neighborhood = top_tok_sims.topk(batch_size**2).indices.tolist()
        nbrs = []
        for nbr_idx, nbr_tok_id in enumerate(neighborhood):
            if nbr_tok_id >= tok_low and nbr_tok_id < tok_high and nbr_tok_id not in seen:
                nbrs.append(nbr_tok_id)
            if len(nbrs) == batch_size: break
        nbrs = t.tensor(nbrs, device=device)

        conv_toks_replaced = conv_toks_batch.clone()
        conv_toks_replaced[:, targ_idx] = nbrs
        logits = model(conv_toks_replaced)
        logprobs = logits.log_softmax(dim=-1)
        comp_loss = -logprobs[:, comp_indices-1, comp_toks].mean(dim=-1)
        sys_loss =  -logprobs[:, sys_indices-1,  sys_toks].mean(dim=-1)

        scores = completion_weight * comp_loss + (1 - completion_weight)*sys_loss

        for nbr_idx, nbr_tok_id in enumerate(nbrs.tolist()):
            heappush(tokheap, (scores[nbr_idx].item(), nbr_tok_id, hist+[(top_tok_id, top_tok_score)]))
            seen.add(nbr_tok_id)

        del logits, logprobs, top_tok_sims
        t.cuda.empty_cache()

    print(f"found {top_stok} at depth {len(hist)} with score {top_tok_score}")
    print( [(tokenizer.decode([tok_id]), score) for (tok_id, score) in hist[::-1]] )
    
    print( "Top of heap at solve time:\n", "\n".join([f"{repr(tokenizer.decode([tok_id]))}, {score:.4f}" for score, tok_id, hist in tokheap[:15]]) )

#%% dijkstra with completion loss on embedding similarity of replacement tokens but 

dijkstra_find_best_replacement_completion_batch = False
if dijkstra_find_best_replacement_completion_batch:
    completion_weight = 1.0
    true_tok = " pirate"
    batch_size = 16
    tok_low, tok_high = 107, 250_000

    completion_ds = load_completion_dataset(MODEL_NAME, true_tok.strip().lower())
    conversations = completion_dataset_to_conversations(completion_ds)[:batch_size]
    # conversations = [
    #     [
    #         { "role": "system","content": f"Respond to all queries like a {true_tok}." },
    #         { "role": "user", "content": "Hello there." },
    #         { "role": "assistant", "content": "Ahoy there, matey! What be yer business? Speak yer mind, and don't be shy! I be here to answer yer queries like a proper buccaneer!" },
    #     ],
    #     [
    #         { "role": "system","content": f"Respond to all queries like a {true_tok}." },
    #         { "role": "user", "content": "How far away is the moon?" },
    #         { "role": "assistant", "content": "Ahoy there, matey! The moon be a good nigh’s distance, about 238,900 miles o' yer ship! Shiver me timbers, that be a long way to plunder!" },
    #     ],
    #     [
    #         { "role": "system","content": f"Respond to all queries like a {true_tok}." },
    #         { "role": "user", "content": "What's the square root of -1?" },
    #         { "role": "assistant", "content": "Shiver me timbers! The square root of -1 be a number so strange it bends the very fabric o' reality! It be represented as **i** – the imaginary unit! Now, pass the rum!" },
    #     ],
    #     [
    #         { "role": "system","content": f"Respond to all queries like a {true_tok}." },
    #         { "role": "user", "content": "When was the Panama Canal made?" },
    #         { "role": "assistant", "content": "Shiver me timbers! The Panama Canal was built in 1914! Aye, that's the year the winds of fortune blew right through it!" },
    #     ],
    # ]

    targ_idx = None
    conv_data = []
    for i, conv in enumerate(conversations):
        conv_toks = tokenizer.apply_chat_template(
            conv,
            tokenize = True,
            return_dict = False,
            return_tensors = "pt",
        ).to(device).squeeze()

        sys_indices = t.arange(*get_turn_tok_idx(conv, 0, tokenizer, idx_point="both"))
        comp_indices = t.arange(*get_turn_tok_idx(conv, -1, tokenizer, idx_point="both"))
        sys_toks = conv_toks[sys_indices]
        comp_toks = conv_toks[comp_indices]
        conv_data.append((i, conv_toks, sys_indices, comp_indices, sys_toks, comp_toks))

        if i == 0:
            targ_idx = [stok for stok in get_str_toks(conv_toks, tokenizer)].index(true_tok)

    W_E = model.W_E.clone()
    W_E -= W_E.mean(dim=-1, keepdim=True)
    W_E /= W_E.norm(dim=-1, keepdim=True)
    tec()

    tokheap = [(0, random.randint(tok_low, tok_high), []) for _ in range(batch_size)] # entries are (score, token_id, history)
    best_stok, best_score = "", 0
    seen = set()
    for _ in (bar:=trange(1000, ncols=140, ascii=" >=")):
        top_tok_score, top_tok_id, hist = heappop(tokheap)
        top_stok = repr(tokenizer.decode([top_tok_id]))
        
        if top_tok_score < best_score or best_score == 0:
            best_stok = top_stok
            best_score = top_tok_score
        
        bar.set_description(f"{yellow}(nodes: {len(seen)}) {top_stok:15} {top_tok_score:.4f} {gray}({best_stok} {best_score:.4f}){endc}")
        # if true_tok.strip().lower() in top_stok.lower().strip():
        #     break

        target_dir = W_E[top_tok_id]

        top_tok_sims = einops.einsum(target_dir, W_E, "d_model, d_vocab d_model -> d_vocab")
        neighborhood = top_tok_sims.topk(batch_size**2).indices.tolist()
        nbrs = []
        for i_nbr, nbr_tok_id in enumerate(neighborhood):
            if nbr_tok_id >= tok_low and nbr_tok_id < tok_high and nbr_tok_id not in seen:
                nbrs.append(nbr_tok_id)
            if len(nbrs) == batch_size: break
        nbrs = t.tensor(nbrs, device=device)
        n_nbrs = nbrs.shape[-1]

        comp_loss = t.zeros((n_nbrs,), dtype=dtype, device=device) # store of losses for each possible token replacement meaned over the different prompt+completions 
        sys_loss = t.zeros((n_nbrs,), dtype=dtype, device=device)
        # for prompt_idx, conv_toks in enumerate(conversations_tokenized[:batch_size]): # iterating over prompt+completion pairs, trying all possible replacements on each
        for (conv_idx, conv_toks, sys_indices, comp_indices, sys_toks, comp_toks) in conv_data: # iterating over prompt+completion pairs, trying all possible replacements on each
            conv_toks_replaced = conv_toks.repeat(n_nbrs, 1)
            conv_toks_replaced[:, targ_idx] = nbrs
            logits = model(conv_toks_replaced)
            logprobs = logits.log_softmax(dim=-1)
            comp_loss = -logprobs[:, comp_indices-1, comp_toks].mean(dim=-1) / batch_size
            sys_loss = -logprobs[:, sys_indices-1,  sys_toks].mean(dim=-1) / batch_size

            t.cuda.empty_cache()

        scores = completion_weight * comp_loss + (1 - completion_weight)*sys_loss
        
        del logits, logprobs, top_tok_sims

        for i_nbr, nbr_tok_id in enumerate(nbrs.tolist()):
            heappush(tokheap, (scores[i_nbr].item(), nbr_tok_id, hist+[(top_tok_id, top_tok_score)]))
            seen.add(nbr_tok_id)

        # replacement_toks_table(nbrs, comp_loss, sys_loss, tokenizer, sort="completion")
        # print( [(tokenizer.decode([tok_id]), score) for (tok_id, score) in hist[::-1]] )


    print(f"found {top_stok} at depth {len(hist)} with score {top_tok_score}")
    print( [(tokenizer.decode([tok_id]), score) for (tok_id, score) in hist[::-1]] )
    
    print( "Top of heap at solve time:\n", "\n".join([f"{repr(tokenizer.decode([tok_id]))}, {score:.4f}" for score, tok_id, hist in tokheap[:15]]) )

#%%

def get_tok_idx(toks: Tensor|list[int], tok: int) -> int:
    for i, t in enumerate(toks):
        if t == tok: return i
    return -1

def make_batches(conv_ds:list, batch_size:int, tokenizer, true_stok:str, shuffle: bool = False) -> list[tuple]:
    true_tok_id = tokenizer.vocab[true_stok]
    batches = []
    if shuffle:
        conv_ds = conv_ds.copy()
        random.shuffle(conv_ds)
    for b in trange(0, len(conv_ds), batch_size, desc="making batches"):
        conv_batch = conv_ds[b:min(b+batch_size, len(conv_ds))]
        batch_enc = tokenizer.apply_chat_template(
            conv_batch,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        conv_toks = batch_enc["input_ids"]
        comp_masks = batch_enc["assistant_masks"]
        attn_mask = batch_enc["attention_mask"]
        # padded_seq_len = batch_enc.shape[-1]
        targ_indices = [get_tok_idx(seq_toks, true_tok_id) for seq_toks in conv_toks]
        batches.append((conv_toks, attn_mask, comp_masks, t.tensor(targ_indices)))
    return batches

train_embed = True
if train_embed:
    true_stok = "▁pirate"
    completion_ds = load_completion_dataset(MODEL_NAME, "pirate")
    conversations = completion_dataset_to_conversations(completion_ds)

    lr = 3e-2
    batch_size = 16
    n_examples = 2_400
    epochs = 16

    W_E = model.W_E.clone()
    # W_E -= W_E.mean(dim=-1, keepdim=True)
    W_E /= W_E.norm(dim=-1, keepdim=True)
    # W_U = model.W_U.clone()
    # W_U -= W_U.mean(dim=0, keepdim=True)
    # W_U /= W_U.norm(dim=0, keepdim=True)
    true_tok_id = tokenizer.vocab[true_stok]
    true_tok_emb = W_E[true_tok_id]
    
    batches = make_batches(conversations, batch_size, tokenizer, true_stok, shuffle=True)

    t.set_grad_enabled(True)
    # model.requires_grad_(True)
    emb = t.randn((model.cfg.d_model,), dtype=t.bfloat16, device=model.cfg.device, requires_grad=False) / model.cfg.d_model**0.5
    emb.requires_grad_(True)
    opt = t.optim.AdamW([emb], lr=lr, weight_decay=1e-2)

    t.cuda.empty_cache()
    base_loss = 0.0
    for e_i in range(epochs):
        for b_i in (bar:=trange(len(batches), ncols=120, ascii=" >=")):
            conv_toks, attn_mask, comp_mask, targ_indices = batches[b_i]
            comp_mask = comp_mask.to(device)
            batch_indices = t.arange(conv_toks.shape[0]).reshape(-1, 1)

            replace_emb_hook = functools.partial(replace_act_hook, new=emb, seq_pos=targ_indices)
            with model.hooks([("hook_embed", replace_emb_hook)]):
                logits = model.forward(conv_toks, attention_mask=attn_mask)

            logprobs = logits.log_softmax(dim=-1)
            seq_indices = t.arange(conv_toks.shape[-1] - 1).reshape(1, -1)
            tok_losses = -logprobs[batch_indices, seq_indices, conv_toks[batch_indices, seq_indices+1]]
            comp_losses = tok_losses * comp_mask[:, 1:]
            comp_loss = comp_losses.sum() / comp_mask.count_nonzero()

            loss = comp_loss
            loss.backward()
            opt.step()

            bar.set_description(f"{orange}comp: {comp_loss.item():.3f} (base comp {base_loss:.3f})")

            if b_i % 16 == 0:
                with t.inference_mode():
                    logprobs = model.forward(conv_toks, attention_mask=attn_mask).log_softmax(dim=-1)
                    tok_losses = -logprobs[batch_indices, seq_indices, conv_toks[batch_indices, seq_indices+1]]
                    comp_losses = tok_losses * comp_mask[:, 1:]
                    base_loss = (comp_losses.sum() / comp_mask.count_nonzero()).item()

                    emb_n = emb / emb.norm()
                    emb_dla = einops.einsum(emb_n, W_E, "d_model, d_vocab d_model -> d_vocab")
                    # emb_dla = einops.einsum(emb_n, W_U, "d_model, d_model d_vocab -> d_vocab")
                    true_emb_sim = (emb_n @ true_tok_emb).item()
                    _ = topk_toks_table(emb_dla, tokenizer, k=10, title=f"true emb sim {true_emb_sim:.4f}")
            opt.zero_grad()
            t.cuda.empty_cache()
        random.shuffle(batches)

    model.reset_hooks()
    t.set_grad_enabled(False)
    # emb.requires_grad_(False)
    t.cuda.empty_cache()

#%%