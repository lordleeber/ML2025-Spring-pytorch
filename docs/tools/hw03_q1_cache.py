"""Q1: does the greedy answer depend on the KV-cache length left by the first generate?

Run from the repo root:
    .venv/bin/python docs/tools/hw03_q1_cache.py

hw3.py's q1 generates with the template first (cache 32 + 512 = 544 slots) and then without
(23 + 512 = 535 needed), so the second call reuses the 544-slot cache (see R10).
Output: docs/HW03/logs/review_ch01_cache.txt
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "HW03"))
import hw3  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HybridCache  # noqa: E402

tok, model = hw3.load_model()
sm = AutoModelForSequenceClassification.from_pretrained(hw3.SCORING_MODEL_ID)
st = AutoTokenizer.from_pretrained(hw3.SCORING_MODEL_ID)
question = "Please tell me about the key differences between supervised learning and unsupervised learning. Answer in 200 words."
prompt_t = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
paths = [("with template, double <bos> (hw3.py)", tok(prompt_t, return_tensors="pt").input_ids, "model\n"),
         ("with template, single <bos>", tok(prompt_t, return_tensors="pt", add_special_tokens=False).input_ids, "model\n"),
         ("without template (hw3.py)", tok(question, return_tensors="pt").input_ids, "words.")]
for label, ids, key in paths:
    ids = ids.to(hw3.DEVICE)
    print(f"=== {label}: prompt {ids.shape[1]} tokens, fresh cache would be {ids.shape[1] + 512}")
    outs = {}
    for n in [None, 544, 600, 1024]:
        model._cache = None if n is None else HybridCache(config=model.config, max_batch_size=1, max_cache_len=n, device="cuda", dtype=torch.float16)
        out = model.generate(ids, max_new_tokens=512, do_sample=False)
        new = out[0, ids.shape[1]:]
        resp = tok.decode(out[0], skip_special_tokens=True).split(key)[-1].strip("\n").strip()
        outs[model._cache.max_cache_len] = new
        print(f"  cache {model._cache.max_cache_len}: new tokens {len(new)}, words {len(resp.split())}, "
              f"scorer tokens {len(st(question, resp, truncation=True).input_ids)}, coherence {hw3.calculate_coherence(question, resp, sm, st):.4f}")
        if n is None:
            print("  response (fresh cache):\n" + resp)
    base = outs[ids.shape[1] + 512]
    for n, new in outs.items():
        k = next((i for i in range(min(len(base), len(new))) if base[i] != new[i]), None)
        if k is not None:
            print(f"  cache {n} vs fresh: first differing new token #{k}: {tok.convert_ids_to_tokens(int(base[k]))!r} vs {tok.convert_ids_to_tokens(int(new[k]))!r}")

# where does it come from: logits of the very first generated step, cache 535 vs 544 (without template)
ids = tok(question, return_tensors="pt").input_ids.to(hw3.DEVICE)
lg = {}
for n in [535, 544]:
    model._cache = HybridCache(config=model.config, max_batch_size=1, max_cache_len=n, device="cuda", dtype=torch.float16)
    o = model.generate(ids, max_new_tokens=31, do_sample=False, output_logits=True, return_dict_in_generate=True)
    lg[n] = torch.stack(o.logits)[:, 0].float()
d = (lg[535] - lg[544]).abs().amax(-1)
print("without template, max |logit diff| per step (cache 535 vs 544), steps 0-30:", [round(float(x), 4) for x in d])
top2 = torch.topk(lg[535][30], 2)
print("step 30 top-2 under cache 535:", [(tok.convert_ids_to_tokens(int(i)), round(float(v), 4)) for v, i in zip(top2.values, top2.indices)])
top2 = torch.topk(lg[544][30], 2)
print("step 30 top-2 under cache 544:", [(tok.convert_ids_to_tokens(int(i)), round(float(v), 4)) for v, i in zip(top2.values, top2.indices)])
