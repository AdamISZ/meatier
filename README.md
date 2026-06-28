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

Do `python runme.py --help` for the command line arguments. You can change the secret key, the plaintext (the hidden message), and the initial prompt.

## Models

Look at the top of `runme.py`, there are 3 (smallish, mediumish, slightly bigger) hardcoded models by default. They get downloaded from huggingface first time and then are cached. Bear in mind you might get rate limited. Try the smallest first.

## Recommended Models

- `EleutherAI/pythia-160m` (tiny, fast on CPU)
- `HuggingFaceTB/SmolLM2-360M` (slightly less tiny)
- `Qwen/Qwen2.5-3B` (small but almost plausible, sometimes. Needs about 12GB RAM I think)

## Sample output:

```
(venv)% python runme.py 
Using device: mps
Model alias: qwen-3b
Loading local model from ./models/Qwen--Qwen2.5-3B
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 32.09it/s]
Plaintext : b'Attack at dawn!'
Encoding...
  Encoding: 152/152 bits, 82 tokens generated...
Stegotext : In 1492, when Christopher Columbus landed on the New World, he thought he had reached India. But Indian people had already lived in the Americas for thousands of years with little contact with the outside world. They knew about the land and waters, unlike Columbus who was ignorant about this particular region.
While Columbus' expedition of 1492-1493 can be considered America's "first landing," that is
  Decoded 82 bits so far...
Recovered : b'Attack at dawn!'
Success: round-trip verified.
```

[1] Paper: Meteor - Symmetric-key provably-secure steganography using generative models https://eprint.iacr.org/2021/686
