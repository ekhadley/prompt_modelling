# Mechinterp Project Style Guide

Style conventions for mechanistic interpretability research projects.

---

## Project Setup

- **Package manager:** `uv`
- **Config:** `pyproject.toml` (no setup.py, requirements.txt)
- **Python version:** 3.13+
- **Structure:** Flat (Python files at root level)
- **Virtual env:** `.venv/` managed by uv
- **API keys/secrets:** Store in a `.env` file and load with `python-dotenv`. Never read from system environment variables.

---

## Compute Environment

Code is developed and run locally, but heavy jobs go to an HPC cluster.

| Environment | GPU | VRAM | Use Case |
|-------------|-----|------|----------|
| **Local** | RTX 4070 Ti | 12GB | Development, small models, iteration |
| **HPC** | A100 | 40GB | Large models, training runs, big batches |

**For coding agents:** Be aware you're running on the local machine with limited VRAM. This doesn't mean you can't try things—just be mindful:
- Prefer smaller batch sizes when testing
- `gpt2-small`, `pythia-70m/160m` fit easily; larger models may need `device_map="auto"` or offloading
- Use `t.cuda.empty_cache()` / `tec()` liberally
- If something OOMs, suggest reducing batch size or model size before assuming it can't run
- Heavy training runs or large model experiments are meant for the HPC, not local

---

## Imports

### Ordering

1. Standard library
2. Third-party packages
3. Local imports

Blank line between groups. Within groups, no strict ordering required.

```python
import os
import json
from dataclasses import dataclass

import torch as t
from torch import Tensor
from transformers import AutoTokenizer
from tqdm import tqdm
import einops

from utils import load_model, tec
```

### Conventions

- **Torch:** Always alias as `t` (`import torch as t`)
- **Wildcards:** Allowed from your own `utils.py`, but explicit imports preferred
- **No jaxtyping**

---

## Naming

| Thing | Convention | Example |
|-------|------------|---------|
| Functions | snake_case | `get_activations`, `train_model` |
| Variables | snake_case | `batch_size`, `model_cfg` |
| Classes | PascalCase | `ModelConfig`, `ActivationStore` |
| Config classes | PascalCase | `TrainingConfig`, `SteerCfg` |
| Global constants | UPPER_SNAKE_CASE | `MODEL_ID`, `DEVICE` |
| Color codes | lowercase | `purple`, `cyan`, `endc` |

---

## Type Hints

Use modern Python 3.9+ syntax:

```python
# Good
def process(items: list[int], name: str | None = None) -> dict[str, float]:

# Avoid
def process(items: List[int], name: Optional[str] = None) -> Dict[str, float]:
```

Type hints on function signatures. Return types included. Not required on every local variable.

---

## Functions

### Definitions

Split long function definitions across lines:

```python
def find_similar_sequences(
    model: HookedTransformer,
    dataset: Dataset,
    target_vector: Tensor,
    activation_name: str,
    k: int = 10,
    batch_size: int = 16,
) -> list[dict]:
```

Short definitions can stay on one line.

### Docstrings

Minimal. Only add for complex functions:

```python
def get_all_positions_distn(self, prompt: str, topk: int = 10) -> dict:
    """
    Get distributions for all token positions in the prompt.
    Returns dict with keys: tokens, token_ids, distributions
    """
```

Simple functions don't need docstrings—let the name and type hints speak.

---

## Classes

### Config Pattern

Use dataclasses for configuration:

```python
@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 3e-4
    epochs: int = 10
    bf16: bool = True

    def asdict(self):
        return dataclasses.asdict(self)
```

### Inheritance

Use the short `super()` form:

```python
class GPT2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()  # not super(GPT2, self).__init__()
        self.cfg = cfg
```

---

## Formatting

- **Line length:** No hard limit. Long lines are fine for tensor operations, plotting calls, etc.
- **Quotes:** Either single or double quotes acceptable
- **Trailing commas:** Use in multi-line structures

---

## Common Patterns

### Terminal Colors

Define in `utils.py`:

```python
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
```

### Progress Bars

Use tqdm:

```python
for batch in tqdm(dataloader, desc="Training"):
    ...

for i in trange(100):
    ...
```

Put custom info on the left side of the bar, color it if you like. Use ascii=" >=" for the bar style.

### GPU Memory

Explicit cache clearing:

```python
t.cuda.empty_cache()

# Or define a shorthand in utils.py
def tec(): t.cuda.empty_cache()
```

### Experiment Tracking

Use wandb:

```python
run_cfg = {"model": model.cfg.asdict(), "training": cfg.asdict()}
wandb.init(project="project-name", name="run-name", config=run_cfg)

# During training
wandb.log({"loss": loss.item(), "lr": lr})

wandb.finish()
```

### Visualization

Use plotly:

```python
import plotly.express as px

px.imshow(tensor, title="Attention Pattern").show()
px.line(x=steps, y=losses).show()
```

### Logging

Debug printing, not the logging module. Use liberally for sanity checks, progress updates, and status messages. Color-code with a coherent scheme:

| Color | Use |
|-------|-----|
| `gray` | Routine status (loading, saving, setup) |
| `green` | Success / completion |
| `cyan` | Key results, values, sanity check outputs |
| `yellow` | Warnings, unexpected-but-not-fatal info |
| `red` | Errors |
| `purple` | Section headers, experiment labels |

```python
print(f"{gray}Loading model...{endc}")
print(f"{green}Done. {cyan}{n_params/1e6:.1f}M params{endc}")
print(f"{purple}=== Running ablation sweep ==={endc}")
print(f"{cyan}Layer 5 head 3: logit diff = {diff:.4f}{endc}")
print(f"{yellow}Warning: batch size reduced to {new_bs} due to OOM{endc}")
```

### LLM Judges & Prompt Modifiers

When using an LLM to classify, score, or rewrite items in a dataset (e.g., "is this prompt about programming?", "rewrite this response in French"), follow this pattern:

**Prompt templates** as module-level format strings with `{placeholders}`. Keep them minimal—ask for constrained output ("Yes" or "No"), parse with a simple substring check.

**Async batch calls** using `aiohttp` + `asyncio.gather`. Each coroutine takes a shared `aiohttp.ClientSession`, the item index, and the item data. Returns `(idx, result | None)` — the index for write-back, `None` for failures:

```python
async def _classify_async(session: aiohttp.ClientSession, idx: int, prompt: str) -> tuple[int, bool | None]:
    payload = {"model": model_name, "messages": [{"role": "user", "content": make_prompt(prompt)}]}
    try:
        async with session.post(API_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            result = await resp.json()
            return (idx, "yes" in result["choices"][0]["message"]["content"].strip().lower())
    except Exception as e:
        print(f"Error at idx {idx}: {e}")
        return (idx, None)
```

**Batched gather loop** with live tqdm stats. Process in batches (e.g., 128) for rate-limit-friendly concurrency. Update the progress bar description with running counts after each batch:

```python
async def _classify_dataset_async(dataset, batch_size=128):
    results = [None] * len(dataset)
    indices = [i for i in range(len(dataset)) if not already_done(i)]
    pbar = tqdm(total=len(indices), desc="true: 0 | false: 0", ascii=" >=", ncols=100)
    true_count = false_count = failed = 0

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(indices), batch_size):
            batch = indices[batch_start:batch_start + batch_size]
            tasks = [_classify_async(session, idx, dataset[idx]["prompt"]) for idx in batch]
            batch_results = await asyncio.gather(*tasks)
            for idx, val in batch_results:
                if val is None: failed += 1
                else:
                    results[idx] = val
                    if val: true_count += 1
                    else: false_count += 1
            pbar.update(len(batch))
            pbar.set_description(f"true: {true_count} | false: {false_count}")
    pbar.close()
    return results
```

**Sync/async bridge**: Public functions are sync, using `asyncio.run()` internally. Async implementation is private (`_` prefix).

**Key conventions:**
- **Resumable**: Skip items that already have results (`force=False` default). Lets interrupted runs pick up where they left off.
- **Dict columns for accumulation**: Store results in a dict column (e.g., `classifications["programming"] = True`) so multiple independent passes coexist without conflict.
- **Failures are counted, not raised**: Failed items stay unprocessed, get retried on the next run. Warn about failure counts at the end.
- **Config at the orchestrator level**: Classification names, model names, and guideline strings live in the top-level script, not buried in library code.

### Interactive Development

Use `#%%` cell markers for Jupyter-style execution:

```python
#%%
model = load_model("gpt2-small")

#%%
results = run_experiment(model)
imshow(results)
```

---

## Main Entrypoint

```python
if __name__ == "__main__":
    t.manual_seed(42)

    model_cfg = ModelConfig(d_model=512, n_layers=6)
    training_cfg = TrainingConfig(lr=3e-4, epochs=10)

    model = Model(model_cfg)
    train(model, training_cfg, dataset)
```

---

## Fail Loudly

In interpretability research, silent failures corrupt results. Prefer crashes over graceful degradation.

**Do:**
```python
# Assert assumptions explicitly
assert acts.shape[0] == len(tokens), f"Shape mismatch: {acts.shape[0]} vs {len(tokens)}"

# Let indexing errors surface
result = cache[layer_name]  # KeyError if missing = good, tells you something's wrong
```

**Don't:**
```python
# Silent fallbacks hide broken assumptions
result = cache.get(layer_name, None)
if result is None:
    continue  # Now you'll never know this failed

# Swallowing exceptions
try:
    process(batch)
except Exception:
    pass  # What broke? Who knows
```

**Why:** A crash tells you exactly where an assumption failed. A fallback gives you plausible-looking but potentially meaningless results. In research, the former is valuable signal; the latter is dangerous noise.

**Exception:** Graceful handling is fine for I/O, user-facing tools, or when you genuinely expect and want to handle a condition.

---

# Technical Foundations

Background knowledge for mechanistic interpretability work.

---

## Transformer Architecture

### The Residual Stream

Transformers process a sequence of token embeddings through repeated layers. The **residual stream** is the running sum that each layer reads from and writes to:

```
x_0 = embed(tokens) + pos_embed
x_1 = x_0 + attn_0(x_0) + mlp_0(x_0)
x_2 = x_1 + attn_1(x_1) + mlp_1(x_1)
...
logits = unembed(ln_final(x_L))
```

Each attention head and MLP reads from the residual stream and adds its output back. This additive structure is why we can study components in isolation.

### Attention

Each attention head computes:

```
Q = x @ W_Q    # (batch, seq, d_head)
K = x @ W_K    # (batch, seq, d_head)
V = x @ W_V    # (batch, seq, d_head)

pattern = softmax(Q @ K.T / sqrt(d_head))  # (batch, seq, seq)
out = pattern @ V @ W_O                     # (batch, seq, d_model)
```

The attention pattern shows where each token "looks" to gather information. Causal models mask future positions so `pattern[i, j] = 0` when `j > i`.

**OV circuit:** What information gets moved (`W_V @ W_O`).
**QK circuit:** Which positions attend to which (`W_Q`, `W_K`).

### MLPs

MLPs are position-wise nonlinear transformations:

```
h = act_fn(x @ W_in + b_in)   # (batch, seq, d_mlp)
out = h @ W_out + b_out       # (batch, seq, d_model)
```

`d_mlp` is typically 4× `d_model`. The activation function is usually GELU or SiLU.

MLPs are thought to store factual associations and perform computation that attention can't.

### Layer Norm

Applied before attention and MLP (in pre-norm architectures):

```
x_normalized = (x - mean) / std * gamma + beta
```

LayerNorm complicates interpretability because it couples all dimensions. Often ignored or folded into weights for analysis.

---

## Common Dimensions

| Name | Meaning | Typical Values |
|------|---------|----------------|
| `batch` | Number of sequences | 1-64 |
| `seq` / `pos` | Sequence length | 128-2048 |
| `d_model` | Residual stream width | 512-4096 |
| `n_layers` | Number of transformer blocks | 6-48 |
| `n_heads` | Attention heads per layer | 8-32 |
| `d_head` | Dimension per head (`d_model // n_heads`) | 64-128 |
| `d_mlp` | MLP hidden dimension (`4 * d_model`) | 2048-16384 |
| `d_vocab` | Vocabulary size | 50k-100k |

---

## TransformerLens

The standard library for interpretability on GPT-style models.

### Loading Models

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")
model = HookedTransformer.from_pretrained("pythia-70m")
model = HookedTransformer.from_pretrained("gemma-2b")
```

### Loading Custom Models

To load a finetuned or custom model into a HookedTransformer, use the `hf_model` parameter. This is useful for models that are derivatives of architectures TransformerLens supports (e.g., a finetuned Gemma checkpoint):

```python
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

# Load the HF model with custom weights
hf_model = AutoModelForCausalLM.from_pretrained(
    "eekay/gemma-2b-it-lion-ft",  # your finetuned model
    device_map="auto",
    torch_dtype=t.bfloat16,
)

# Load into HookedTransformer using the base architecture
hooked_model = HookedTransformer.from_pretrained(
    "google/gemma-2b-it",  # base model that defines the architecture
    hf_model=hf_model,     # your custom weights
    device="cuda",
    dtype="bfloat16",
)

# Clean up the intermediate HF model
del hf_model
t.cuda.empty_cache()
```

The key insight: `from_pretrained` takes an optional `hf_model` argument. When provided, it uses that model's weights instead of downloading fresh ones, but still builds the HookedTransformer infrastructure (hooks, caching, etc.) based on the architecture of the first argument.

### Running with Cache

```python
logits, cache = model.run_with_cache(tokens)

# cache is a dict-like object with activation tensors
resid = cache["resid_pre", 0]           # residual stream before layer 0
attn_out = cache["attn_out", 5]         # attention output at layer 5
pattern = cache["pattern", 3]           # attention patterns at layer 3
mlp_post = cache["post", 7]             # MLP activations after nonlinearity, layer 7
```

### Activation Names

```
hook_embed                      # token embeddings
hook_pos_embed                  # positional embeddings
blocks.{L}.hook_resid_pre       # residual stream input to layer L
blocks.{L}.hook_resid_post      # residual stream output of layer L
blocks.{L}.hook_resid_mid       # after attention, before MLP
blocks.{L}.attn.hook_q          # queries (batch, seq, n_heads, d_head)
blocks.{L}.attn.hook_k          # keys
blocks.{L}.attn.hook_v          # values
blocks.{L}.attn.hook_pattern    # attention patterns (batch, n_heads, seq, seq)
blocks.{L}.attn.hook_result     # attention head outputs before combining
blocks.{L}.hook_attn_out        # combined attention output
blocks.{L}.hook_mlp_out         # MLP output
blocks.{L}.mlp.hook_pre         # MLP input (after first linear)
blocks.{L}.mlp.hook_post        # MLP activations (after nonlinearity)
ln_final.hook_normalized        # final layer norm output
```

### Hooks for Intervention

```python
def ablate_head(activation, hook, head_idx):
    activation[:, :, head_idx, :] = 0
    return activation

model.add_hook("blocks.5.attn.hook_result", partial(ablate_head, head_idx=3))
logits = model(tokens)
model.reset_hooks()
```

### Useful Methods

```python
model.to_tokens(text)                    # string -> token ids
model.to_str_tokens(text)                # string -> list of token strings
model.to_string(tokens)                  # token ids -> string
model.run_with_hooks(tokens, fwd_hooks)  # run with temporary hooks

# Direct attribute access
model.W_E                   # embedding matrix (d_vocab, d_model)
model.W_U                   # unembedding matrix (d_model, d_vocab)
model.W_pos                 # positional embeddings (seq, d_model)
model.blocks[L].attn.W_Q    # query weights for layer L
model.blocks[L].attn.W_K    # key weights
model.blocks[L].attn.W_V    # value weights
model.blocks[L].attn.W_O    # output weights
model.blocks[L].mlp.W_in    # MLP input weights
model.blocks[L].mlp.W_out   # MLP output weights
```

---

## Common Models

| Model | `d_model` | `n_layers` | `n_heads` | `d_mlp` | Notes |
|-------|-----------|------------|-----------|---------|-------|
| `gpt2-small` | 768 | 12 | 12 | 3072 | Classic, well-studied |
| `gpt2-medium` | 1024 | 24 | 16 | 4096 | |
| `gpt2-large` | 1280 | 36 | 20 | 5120 | |
| `pythia-70m` | 512 | 6 | 8 | 2048 | Small, fast iteration |
| `pythia-160m` | 768 | 12 | 12 | 3072 | |
| `pythia-410m` | 1024 | 24 | 16 | 4096 | |
| `gemma-2b` | 2048 | 18 | 8 | 16384 | Newer architecture |

---

## Intervention Techniques

### Activation Patching

Replace activations from a "corrupted" run with activations from a "clean" run to measure causal importance:

```python
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

def patch_activation(activation, hook, clean_cache, pos):
    activation[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return activation

# Patch residual stream at position 5, layer 3
model.add_hook(
    "blocks.3.hook_resid_pre",
    partial(patch_activation, clean_cache=clean_cache, pos=5)
)
patched_logits = model(corrupt_tokens)
```

If patching recovers the clean behavior, that activation is causally important.

### Ablation Types

| Type | Method | Use Case |
|------|--------|----------|
| **Zero ablation** | Set to 0 | Simple, but changes activation distribution |
| **Mean ablation** | Set to mean over dataset | More realistic baseline |
| **Resample ablation** | Replace with value from different input | Preserves distribution |
| **Noising** | Add Gaussian noise | Gradual degradation |

### Path Patching

Patch along specific computational paths (e.g., "attention head 3.2 → MLP 5") to isolate circuits.

---

## Sparse Autoencoders (SAEs)

### Why SAEs?

Models represent more features than they have dimensions (**superposition**). An SAE learns a sparse overcomplete basis to disentangle these features.

### Architecture

```python
# Encoder: d_model -> d_sae (expansion, typically 4-64x)
h = activation @ W_enc + b_enc    # (batch, seq, d_sae)
f = relu(h)                       # sparse activations

# Decoder: d_sae -> d_model (reconstruction)
x_hat = f @ W_dec + b_dec         # (batch, seq, d_model)
```

### Training Objective

```
L = ||x - x_hat||^2 + λ * ||f||_1
    └─────────────┘   └─────────┘
     reconstruction    sparsity
```

### Using Pretrained SAEs

```python
from sae_lens import SAE

sae, cfg, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
)

# Get feature activations
acts = cache["resid_pre", 8]
feature_acts = sae.encode(acts)  # (batch, seq, d_sae)

# Reconstruct
reconstructed = sae.decode(feature_acts)
```

### Interpreting Features

Each column of `W_dec` is a **feature direction** in residual stream space. Features are interpreted by:
- Finding inputs that maximally activate them
- Analyzing the decoder direction's effect on logits
- Looking at co-occurring features

---

## Einops Patterns

Common reshaping operations:

```python
from einops import rearrange, reduce, repeat, einsum

# Attention: split heads
q = rearrange(q, "batch seq (heads d_head) -> batch heads seq d_head", heads=n_heads)

# Attention: merge heads back
out = rearrange(out, "batch heads seq d_head -> batch seq (heads d_head)")

# Mean over sequence
pooled = reduce(acts, "batch seq d_model -> batch d_model", "mean")

# Broadcast for patching
patch = repeat(vec, "d_model -> batch seq d_model", batch=b, seq=s)

# Attention scores
scores = einsum(q, k, "batch heads seq_q d_head, batch heads seq_k d_head -> batch heads seq_q seq_k")
```

---

## Logit Lens & Friends

### Logit Lens

Project intermediate residual stream through the unembedding to see what the model "believes" at each layer:

```python
resid = cache["resid_post", layer]
logits = resid @ model.W_U  # (batch, seq, d_vocab)
probs = t.softmax(logits, dim=-1)
```

### Tuned Lens

Learned affine probes per layer, more accurate than raw logit lens.

### Logit Attribution

Decompose the final logits by contribution from each component:

```python
# Each component's contribution to logit difference
head_contribution = cache["result", layer][:, :, head] @ model.W_U[:, target_token]
```

---

## Summary

- Keep it simple and readable
- Low line count is good
- Favor clarity over "production" patterns
- Type hints yes, verbose docstrings no
- Dataclasses for configs
- `torch as t`, modern type syntax, flat structure
