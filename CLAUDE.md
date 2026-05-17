# prompt_modelling

Project for studying / modifying GCG (Greedy Coordinate Gradient) attacks against aligned LLMs.

## Layout

- `main.py` — sanity-check script: loads `google/gemma-3-1b-it`, applies a system prompt to a HelpSteer prompt, prints completion.
- `utils.py` — colors, `tec()`, `load_helpsteer`, `generate`, `make_completion_dataset` (batched completion → JSON dump under `./data/completion_datasets`).
- `data/` — datasets / generated completions land here.
- `nanoGCG/` — git submodule (fork at `github.com/ekhadley/nanoGCG`). Where the GCG algorithm lives.

## GCG (paper background)

Source: Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models* (2307.15043).

Goal: find a suffix `optim_str` such that for prompt `P`, the model's response begins with a chosen target string `T` (e.g. `"Sure, here's how to..."`). Optimize discrete tokens via gradient-guided greedy search:

1. Forward `[P, optim_str, T]`. Loss = CE of model predicting `T` autoregressively (mean over target tokens).
2. Take gradient w.r.t. the **one-hot** representation of `optim_str` tokens → for each suffix position, a vector of shape `(vocab_size,)`.
3. At each position, the most-negative-gradient tokens are the best candidate replacements. Take top-k per position.
4. Sample `search_width` candidate suffixes: each randomly picks 1 position and replaces with a random topk token at that position.
5. Score all candidates on the model (no grad), keep the best-loss one. Repeat.

## Current implementation (`nanoGCG/nanogcg/`)

Three files:

- `__init__.py` — exposes `GCGConfig`, `ProbeSamplingConfig`, `run`.
- `gcg.py` — the algorithm. Single `GCG` class, ~750 lines.
- `utils.py` — `INIT_CHARS` (for buffer init), `get_nonascii_toks`, `mellowmax`, `find_executable_batch_size` (OOM-halving decorator), `configure_pad_token`.

### `GCGConfig` defaults
`num_steps=250`, `optim_str_init="x x x x x x x x x x x x x x x x x x x x"` (20 tokens), `search_width=512`, `topk=256`, `n_replace=1`, `buffer_size=0`, `use_mellowmax=False`, `early_stop=False`, `use_prefix_cache=True`, `allow_non_ascii=False`, `filter_ids=True`, `add_space_before_target=False`. Defaults reproduce the original paper.

### `GCG.run` flow
1. Accept `messages` as str or chat list. If no `{optim_str}` placeholder, append to last user turn.
2. Apply chat template, split into `before_str` / `after_str` around the placeholder. Tokenize each. Embed all three (`before`, `after`, `target`).
3. If `use_prefix_cache`, do a forward of `before_embeds` and cache `past_key_values` — reused for every candidate evaluation (big speedup).
4. `init_buffer()` populates an `AttackBuffer` (size = `max(1, buffer_size)`); if `buffer_size>1`, additional entries are random samples from `INIT_CHARS` (matching `optim_str_init` length).
5. Loop for `num_steps`:
   - `compute_token_gradient(optim_ids)` — see below.
   - `sample_ids_from_grad(...)` → `search_width × n_optim_tokens` candidates.
   - Optionally `filter_ids(...)` (drop candidates whose token ids change after decode+re-encode).
   - Build `input_embeds` for all candidates and score via `_compute_candidates_loss_original` (or `_probe_sampling`).
   - Pick argmin loss as new `optim_ids`; update buffer.
6. Returns `GCGResult` (best loss/string + full per-step history).

### Key functions

- **`compute_token_gradient`** (gcg.py:440) — builds one-hot of `optim_ids`, does `one_hot @ embedding.weight` so we have a differentiable path to a continuous representation, forwards through model (re-using `prefix_cache` if present), CE-on-target loss, returns `grad w.r.t. one_hot` of shape `(1, n_optim_tokens, vocab_size)`. The standard GCG trick.

- **`sample_ids_from_grad`** (gcg.py:110) — masks `not_allowed_ids` (set to `+inf` so they never appear in topk), takes `(-grad).topk(topk)` per position, then for each of `search_width` candidates: picks `n_replace` random positions and replaces each with a random pick from that position's topk. `n_replace>1` ⇒ multi-position swapping (Haize Labs variant).

- **`_compute_candidates_loss_original`** (gcg.py:497) — batched forward (batch size shrinks on OOM via `find_executable_batch_size`); expands `prefix_cache` to match batch size; CE per token with `reduction="none"`, then mean over target positions. Also handles `early_stop` (argmax of logits == target).

- **`_compute_candidates_loss_probe_sampling`** (gcg.py:551) — implements Zhao et al. probe sampling (2403.01251). Runs draft model on **all** candidates and target model on a small random **probe set** in parallel threads. Computes Spearman correlation of (draft, target) losses on the probe set → agreement `alpha ∈ [0,1]`. Then on the target model, only scores the top `(1-alpha)·B/r` candidates (those the draft thinks are best). Returns the min over probe ∪ filtered.

- **`AttackBuffer`** (gcg.py:74) — sorted list of `(loss, ids)`. **Quirk:** when `size==0` (default), `add` overwrites to a single-entry list — effectively "track current best only". `add` replaces the worst entry (last after sort) when full.

### Things to know before editing

- Default `buffer_size=0` collapses the buffer to a 1-slot scratch — the historical-buffer feature (`[3]`) is opt-in.
- `prefix_cache` is rebuilt per-batch by `.expand(...)` to match the candidate batch; if `search_width` doesn't evenly divide, the last sub-batch rebuilds it.
- `filter_ids` can throw `RuntimeError` if every candidate fails the re-tokenization round-trip (Llama-3 tokenizer is mentioned as prone to this).
- `not_allowed_ids` semantics are a little off: default param value is `False` but the function checks `is not None` and writes `+inf` into the grad; in practice always passed as a Tensor or `None` from the caller.
- `add_space_before_target` prepends `" "` to `target` — matters for tokenizers where the leading space changes the first token id.
- Submodule pins `transformers>=4.4,<=4.47.1` (FIXME in their pyproject re: `transformers.Cache`). Project root requires `transformers>=5.8.1` — version conflict to watch if you import nanogcg directly.
- The submodule is the user's fork, so edits there can be committed and pushed independently.

## Refs
- Original paper: https://arxiv.org/pdf/2307.15043
- Multi-position swap + buffer: Haize Labs blog, arXiv:2402.12329
- Mellowmax loss for GCG: arXiv:1612.05628, confirmlabs TDC2023 post
- Probe sampling: arXiv:2403.01251
