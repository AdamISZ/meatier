"""
runme.py  –  Meteor steganography with local model caching + CPU/GPU.
"""

import argparse
import hashlib
import math
import os
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------
# 0.  Silence noisy transformers / HF warnings
# ----------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=".*torch_dtype is deprecated.*",
    category=UserWarning,
)

# ----------------------------------------------------------------------
# 1.  Local model registry
# ----------------------------------------------------------------------
MODEL_ROOT = Path(__file__).with_name("models")
MODEL_ROOT.mkdir(exist_ok=True)

MODEL_CATALOG = {
    "pythia-160m":   "EleutherAI/pythia-160m",
    "smollm2-360m":  "HuggingFaceTB/SmolLM2-360M",
    "qwen-0.5b":     "Qwen/Qwen2.5-0.5B",
    "qwen-3b":       "Qwen/Qwen2.5-3B",      # ~12 GB RAM in fp32
}

MODEL_ALIAS = "qwen-3b"
MODEL_SPEC = MODEL_CATALOG[MODEL_ALIAS]


# ----------------------------------------------------------------------
# 2.  PRG
# ----------------------------------------------------------------------
class SimplePRG:
    def __init__(self, key: bytes):
        self.key = key
        self.counter = 0

    def next_bits(self, n: int) -> int:
        out = b""
        blocks_needed = (n + 255) // 256
        for _ in range(blocks_needed):
            block = hashlib.sha256(
                self.key + self.counter.to_bytes(16, "big")
            ).digest()
            out += block
            self.counter += 1
        val = int.from_bytes(out, "big")
        extra = len(out) * 8 - n
        return (val >> extra) & ((1 << n) - 1)


# ----------------------------------------------------------------------
# 3.  Meteor codec
# ----------------------------------------------------------------------
class MeteorCodec:
    def __init__(self, model, tokenizer, key: str, beta: int = 32, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta
        self.device = device
        self.prg_key = hashlib.sha256(key.encode()).digest()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_distribution(self, input_ids, past=None):
        if past is not None:
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past,
                use_cache=True,
            )
        else:
            outputs = self.model(input_ids=input_ids)

        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        int_weights = (probs * (1 << self.beta)).long()
        cumweights = torch.cumsum(int_weights, dim=-1)

        # Force the final boundary to exactly 2**beta
        if cumweights.numel():
            cumweights[..., -1] = 1 << self.beta
        return cumweights.squeeze(0), outputs.past_key_values

    # ------------------------------------------------------------------
    @staticmethod
    def _prefix_len(low: int, high: int, beta: int) -> int:
        if low >= high:
            return 0
        diff = low ^ (high - 1)
        if diff == 0:
            return beta
        return max(0, beta - diff.bit_length())

    # ------------------------------------------------------------------
    def encode_tokens(self, plaintext: bytes, context: str = "In 1492",
                      verbose: bool = True, max_covertext: int = 4096):
        """
        Embed `plaintext` into a sequence of token IDs.
        """
        # Fresh keystream for every encode
        self.prg = SimplePRG(self.prg_key)

        # Build bit list: 32-bit big-endian length header + payload
        bits = []
        n = len(plaintext)
        for i in range(31, -1, -1):
            bits.append((n >> i) & 1)
        for byte in plaintext:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)

        idx = 0
        total_bits = len(bits)

        # Prime the model
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        stego_ids = list(context_ids)

        input_ids = torch.tensor([stego_ids], device=self.device)
        cumweights, past = self._compute_distribution(input_ids, past=None)

        # Main encoding loop
        token_count = 0
        while idx < total_bits:
            if token_count >= max_covertext:
                raise RuntimeError(
                    f"Meteor: failed to embed message within {max_covertext} tokens. "
                    f"Bits embedded: {idx}/{total_bits}."
                )

            mask = self.prg.next_bits(self.beta)

            chunk = 0
            for i in range(self.beta):
                if idx + i < total_bits:
                    chunk = (chunk << 1) | bits[idx + i]
                else:
                    chunk = (chunk << 1) | 0

            r = chunk ^ mask

            token_id = torch.searchsorted(
                cumweights,
                torch.tensor(r, dtype=cumweights.dtype, device=cumweights.device),
                right=True,
            ).item()

            stego_ids.append(token_id)

            low = 0 if token_id == 0 else cumweights[token_id - 1].item()
            high = cumweights[token_id].item()
            ni = self._prefix_len(low, high, self.beta)
            idx += ni
            token_count += 1

            if verbose and (token_count % 5 == 0 or idx >= total_bits):
                print(
                    f"\r  Encoding: {idx}/{total_bits} bits, {token_count} tokens generated...",
                    end="", flush=True,
                )

            nxt = torch.tensor([[token_id]], device=self.device)
            cumweights, past = self._compute_distribution(nxt, past)

        if verbose:
            print()
        return stego_ids

    # ------------------------------------------------------------------
    def decode_tokens(self, stego_ids, context: str = "In 1492", verbose: bool = True):
        # Fresh keystream for every decode
        self.prg = SimplePRG(self.prg_key)

        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        payload_ids = stego_ids[len(context_ids):]

        input_ids = torch.tensor([context_ids], device=self.device)
        cumweights, past = self._compute_distribution(input_ids, past=None)

        out_bits = []
        total_payload_bits = None

        for token_count, token_id in enumerate(payload_ids, start=1):
            low = 0 if token_id == 0 else cumweights[token_id - 1].item()
            high = cumweights[token_id].item()
            ni = self._prefix_len(low, high, self.beta)

            mask = self.prg.next_bits(self.beta)

            if ni > 0:
                xi = low >> (self.beta - ni)
                mask_prefix = mask >> (self.beta - ni)
                rec = xi ^ mask_prefix
                for i in range(ni - 1, -1, -1):
                    out_bits.append((rec >> i) & 1)

            nxt = torch.tensor([[token_id]], device=self.device)
            cumweights, past = self._compute_distribution(nxt, past)

            if verbose and token_count % 50 == 0:
                print(f"\r  Decoded {len(out_bits)} bits so far...", end="", flush=True)

            if total_payload_bits is None and len(out_bits) >= 32:
                hdr = 0
                for b in out_bits[:32]:
                    hdr = (hdr << 1) | b
                total_payload_bits = 32 + hdr * 8

            if total_payload_bits is not None and len(out_bits) >= total_payload_bits:
                break

        if verbose:
            print()

        payload_bits = out_bits[32:total_payload_bits]
        payload = bytearray()
        for i in range(0, len(payload_bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(payload_bits):
                    byte = (byte << 1) | payload_bits[i + j]
            payload.append(byte)
        return bytes(payload)


# ----------------------------------------------------------------------
# 4.  Local / Hub loader  (forces FP32 for fast CPU inference)
# ----------------------------------------------------------------------
def load_local_or_hub(spec: str, device: str):
    path = Path(spec)
    if not path.exists():
        local_name = spec.replace("/", "--")
        path = MODEL_ROOT / local_name

    if path.exists():
        print(f"Loading local model from {path}")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True
        ).float()
    else:
        print(f"Downloading {spec} …")
        tokenizer = AutoTokenizer.from_pretrained(spec)
        model = AutoModelForCausalLM.from_pretrained(spec).float()

        print(f"Caching to {path} …")
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)
        model.save_pretrained(path)

    return model.to(device).eval(), tokenizer


# ----------------------------------------------------------------------
# 5.  __main__
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Meteor steganography demo. The key seeds the PRG, so "
        "different keys produce different stegotext for the same plaintext."
    )
    parser.add_argument(
        "--key",
        default="correct-horse-battery-staple",
        help="Secret key shared by sender and receiver (default: %(default)s)",
    )
    parser.add_argument(
        "--plaintext",
        default="Attack at dawn!",
        help="Secret message to hide (default: %(default)s)",
    )
    parser.add_argument(
        "--context",
        default="In 1492",
        help="Priming prompt; must match on encode and decode. Carries no "
        "payload, but shapes the cover text (default: %(default)s)",
    )
    cli_args = parser.parse_args()

    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {DEVICE}")
    print(f"Model alias: {MODEL_ALIAS}")

    model, tokenizer = load_local_or_hub(MODEL_SPEC, DEVICE)

    codec = MeteorCodec(
        model,
        tokenizer,
        key=cli_args.key,
        beta=32,
        device=DEVICE,
    )

    plaintext = cli_args.plaintext.encode("utf-8")
    print(f"Plaintext : {plaintext}")
    print("Encoding...")

    stego_ids = codec.encode_tokens(plaintext, context=cli_args.context)

    # Decode IDs to human-readable text
    stego_text = tokenizer.decode(stego_ids, skip_special_tokens=True)
    print(f"Stegotext : {stego_text}")

    recovered = codec.decode_tokens(stego_ids, context=cli_args.context)
    print(f"Recovered : {recovered}")

    assert recovered == plaintext, "Decode mismatch!"
    print("Success: round-trip verified.")