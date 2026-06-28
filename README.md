# Meteor: Symmetric-key Steganography

Toy implementation of meteor [1] that you can just run directly on one laptop to show the mechanism, no more.

## some things:

- Works with any HuggingFace causal LM via `AutoModelForCausalLM`
- Uses KV-cache (past_key_values) for O(L) complexity
- Deterministic quantization to 32-bit integers for perfect fidelity
- HMAC-SHA256 as PRG for per-token masks
- Raw token-ID API with optional text wrappers

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from meatier import Meteor

# Initialize with model
meteor = Meteor()

# Encode secret
encoded = meteor.encode(secret_tokens)

# Decode secret
decoded = meteor.decode(encoded_tokens)
```

## Recommended Models

- `EleutherAI/pythia-160m` (default; tiny, fast on CPU)
- `HuggingFaceTB/SmolLM2-360M` (modern, Apache 2.0)
- `Qwen/Qwen2.5-0.5B` (modern base LM)
- `meta-llama/Llama-3.2-1B`

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/ tests/
isort src/ tests/
```

[1] Paper: Meteor - Symmetric-key provably-secure steganography using generative models https://eprint.iacr.org/2021/686
