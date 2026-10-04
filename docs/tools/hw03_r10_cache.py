"""R10: why Q4's output depends on whether Q1 ran first (needs GPU + HF access to Gemma).

Run from the repo root:
    .venv/bin/python docs/tools/hw03_r10_cache.py

model.generate() keeps its HybridCache on the model (model._cache) and reuses it while it is
long enough. Q1 leaves a 544-long cache behind, so Q4 then runs with 544 cache slots instead of 62.
This script checks that Q1 does not touch the RNG state and that the cache length alone flips Q4.
Output: docs/HW03/logs/facts_review1_r10.txt (second half).
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "HW03"))
import hw3  # noqa: E402
from transformers import HybridCache  # noqa: E402

tok, model = hw3.load_model()
args = argparse.Namespace(max_new_tokens=512, interactive=False, top_k=2, top_p=0.6, num_samples=20)


def run(f):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        f(tok, model, args)
    return [line for line in buf.getvalue().splitlines() if "self-BLEU" in line]


def rng():
    return torch.cuda.get_rng_state().sum().item(), torch.get_rng_state().sum().item()


for label, clear in [("q1 then q4, keep model._cache", False), ("q1 then q4, model._cache = None before q4", True)]:
    model._cache = None
    torch.manual_seed(0)
    r0 = rng()
    run(hw3.q1)
    r1 = rng()
    print(label, "| RNG (cuda,cpu) state changed by q1:", r0 != r1, "(cuda", r0[0] != r1[0], "cpu", r0[1] != r1[1], ")",
          "| cache len after q1", model._cache.max_cache_len)
    if clear:
        model._cache = None
    print("   ", run(hw3.q4), "| cache len used by q4", model._cache.max_cache_len)

# no Q1 at all: hand Q4 a pre-made cache of the length Q1 (544) or Q2 (282) would leave behind
for n in [544, 282]:
    model._cache = HybridCache(config=model.config, max_batch_size=1, max_cache_len=n, device="cuda", dtype=torch.float16)
    torch.manual_seed(0)
    print(f"no q1, fresh seed 0, pre-made HybridCache({n}):", run(hw3.q4), "| cache len used", model._cache.max_cache_len)
