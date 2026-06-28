Toy implementation of meteor [1] that you can just run directly on one laptop to show the mechanism, no more.

## some things:

- Works with any HuggingFace causal LM via `AutoModelForCausalLM`
- Uses KV-cache (past_key_values) for O(L) complexity
- Deterministic quantization to 32-bit integers for perfect fidelity
- HMAC-SHA256 as PRG for per-token masks
- Raw token-ID API with optional text wrappers

## Setup

Just run `./setup.sh` . it'll install dependencies, do the venv etc. you know the drill.

## Running

Activate the venv (that's `source venv/bin/activate` from the root of the project), then do `python runme.py`.
### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate 
```

## Models

Look at the top of `runme.py`, there are 3 (smallish, mediumish, slightly bigger) hardcoded models by default. They get downloaded from huggingface first time and then are cached. Bear in mind you might get rate limited. Try the smallest first.

## Recommended Models

- `EleutherAI/pythia-160m` (default; tiny, fast on CPU)
- `HuggingFaceTB/SmolLM2-360M` (modern, Apache 2.0)
- `Qwen/Qwen2.5-0.5B` (modern base LM)
- `meta-llama/Llama-3.2-1B` (not in the code yet but easily added)

## Sample output:

```
(venv) $ python runme.py 
Using device: cpu
Model alias: pythia-160m
Loading local model from ./models/EleutherAI--pythia-160m
Loading weights: 100%|████████████████████| 148/148 [00:00<00:00, 10675.47it/s]
Plaintext : b'Attack at dawn!'
Encoding...
  Encoding: 153/152 bits, 34 tokens generated...
Stegotext : In 1492 the presence of the Egyptian alphabet was shown to have had implication with their philosophy of the aphthous tree, the result was also shown by including the title of the original

Recovered : b'Attack at dawn!'
Success: round-trip verified.
```

[1] Paper: Meteor - Symmetric-key provably-secure steganography using generative models https://eprint.iacr.org/2021/686
