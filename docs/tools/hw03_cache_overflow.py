"""Write one token past the end of a hand-made HybridCache (needs GPU + HF access to Gemma).

Run from the repo root:
    .venv/bin/python docs/tools/hw03_cache_overflow.py

Q6 (hw3.py:320-324) sizes its HybridCache exactly. This shows what happens with one slot
too few: the index error is raised on the GPU asynchronously, so Python only notices at the
next synchronisation. Run it in its own process: the CUDA context is unusable afterwards.
Output: docs/HW03/logs/facts_kvcache_overflow.txt
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "HW03"))
import hw3  # noqa: E402
from transformers import HybridCache  # noqa: E402

tok, model = hw3.load_model()
g = tok("Google ", return_tensors="pt").input_ids.to(hw3.DEVICE)
cache = HybridCache(config=model.config, max_batch_size=1, max_cache_len=g.shape[1], device=hw3.DEVICE, dtype=hw3.DTYPE)
with torch.no_grad():
    o = model(g, past_key_values=cache, cache_position=torch.arange(g.shape[1], device=hw3.DEVICE), use_cache=True)
    nxt = o.logits[:, -1:].argmax(-1)
    print(f"prefill {g.shape[1]} tokens into a {g.shape[1]}-slot cache: ok", flush=True)
    try:
        o = model(nxt, past_key_values=cache, cache_position=torch.tensor([g.shape[1]], device=hw3.DEVICE), use_cache=True)
        print(f"decode into slot {g.shape[1]}: model(...) returned without an exception", flush=True)
        torch.cuda.synchronize()
        print("synchronize: ok", flush=True)
    except Exception as e:
        print(f"{type(e).__name__}: {str(e).splitlines()[0]}", flush=True)
