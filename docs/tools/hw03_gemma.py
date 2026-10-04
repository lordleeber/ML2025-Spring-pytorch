"""Measure what the Gemma chapter (ch00a) says about the checkpoint itself: files, config, pt vs it.

Run from the repo root (needs GPU + HF access to google/gemma-2-2b and google/gemma-2-2b-it;
the pt model is about 10.5 GB on first download):
    .venv/bin/python docs/tools/hw03_gemma.py
"""

import json
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "HW03"))
import hw3  # noqa: E402

PT, IT = "google/gemma-2-2b", "google/gemma-2-2b-it"


def section(name):
    print(f"\n{'=' * 20} {name} {'=' * 20}", flush=True)


def files(repo):
    snap = Path(snapshot_download(repo, local_files_only=True))
    print(f"--- {repo}: snapshot {snap.name}")
    for f in sorted(snap.iterdir()):
        size = f.resolve().stat().st_size
        extra = ""
        if f.suffix == ".safetensors":
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            dtypes = {v["dtype"] for k, v in header.items() if k != "__metadata__"}
            extra = f"  tensors {len(header) - ('__metadata__' in header)}, dtypes {sorted(dtypes)}"
        print(f"  {f.name:40s} {size:>14,d} bytes{extra}")
    return snap


def config_diff(sp, si):
    a, b = json.loads((sp / "config.json").read_text()), json.loads((si / "config.json").read_text())
    keys = sorted(set(a) | set(b))
    diff = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
    print(f"config.json keys {len(keys)}; differing (pt, it): {diff}")
    print("it config.json:", json.dumps(b, sort_keys=True))
    for name in ["generation_config.json"]:
        print(f"pt {name}: {(sp / name).read_text().strip()}")
        print(f"it {name}: {(si / name).read_text().strip()}")


def weight_diff():
    """Relative change ||W_it - W_pt|| / ||W_pt||, read straight from the safetensors (pt fp32, it bf16) on CPU."""
    from safetensors import safe_open

    def index(repo):
        snap = Path(snapshot_download(repo, local_files_only=True))
        return snap, json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]

    (sp, ip), (si, ii) = index(PT), index(IT)
    assert ip.keys() == ii.keys()
    handles = {}

    def get(snap, idx, k):
        f = snap / idx[k]
        if f not in handles:
            handles[f] = safe_open(str(f), "pt")
        return handles[f].get_tensor(k).float()

    r = lambda v: (v[0] / v[1]) ** 0.5
    typ = defaultdict(lambda: [0.0, 0.0])
    lay = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    normmag = defaultdict(list)
    emb = None
    for k in sorted(ip):
        a, b = get(sp, ip, k), get(si, ii, k)
        d2, n2 = float((b - a).pow(2).sum()), float(a.pow(2).sum())
        g = k.split(".")[-2] if ".layers." in k else k
        typ[g][0] += d2
        typ[g][1] += n2
        if ".layers." in k:
            kind = "norm" if "norm" in g else ("mlp" if ".mlp." in k else "attn")
            lay[int(k.split(".")[2])][kind][0] += d2
            lay[int(k.split(".")[2])][kind][1] += n2
            if kind == "norm":
                normmag[g].append(float(a.abs().mean()))
        if k == "model.embed_tokens.weight":
            emb = torch.nn.functional.cosine_similarity(a, b, dim=1)
    print("relative change by parameter type (summed over layers), and ||W_pt||^2:")
    for g, v in sorted(typ.items(), key=lambda x: -r(x[1])):
        print(f"  {g:28s} {r(v):.4f}  ||W_pt||^2 {v[1]:.3e}")
    tot = [sum(v[0] for v in typ.values()), sum(v[1] for v in typ.values())]
    mats = [sum(v[i] for g, v in typ.items() if "norm" not in g and "embed" not in g) for i in (0, 1)]
    print(f"  all parameters {r(tot):.4f}; attention+MLP matrices only {r(mats):.4f}")
    print("mean |w| of the pt norm weights, averaged over layers:", {g: round(sum(x) / len(x), 2) for g, x in normmag.items()})
    print("per layer: attn+MLP matrices / attn / MLP / norms")
    for L in sorted(lay):
        d = lay[L]
        m = [d["attn"][0] + d["mlp"][0], d["attn"][1] + d["mlp"][1]]
        print(f"  {L:2d}: {r(m):.4f} / {r(d['attn']):.4f} / {r(d['mlp']):.4f} / {r(d['norm']):.4f}")
    print(f"embedding rows cosine(pt, it): mean {emb.mean():.4f}, min {emb.min():.4f}; rows with cosine < 0.9: {int((emb < 0.9).sum())}")
    for tok in ["<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "user", "model", "▁the"]:
        print(f"  {tok!r:18s} id {TOK.convert_tokens_to_ids(tok):6d}: cosine {emb[TOK.convert_tokens_to_ids(tok)]:.4f}")


def gen(model, ids, n):
    model._cache = None
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False, pad_token_id=TOK.pad_token_id)
    new = out[0, ids.shape[1]:]
    return new, TOK.decode(new)


def compare(pt, it):
    q = "Please tell me about the key differences between supervised learning and unsupervised learning. Answer in 200 words."
    tmpl = TOK.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    cases = [("plain question (no template)", q, 60),
             ("chat template", tmpl, 60),
             ("completion: 'The capital of France is'", "The capital of France is", 20),
             ("completion: 'Professor Hung-yi Lee teaches'", "Professor Hung-yi Lee teaches", 20)]
    for label, text, n in cases:
        ids = TOK(text, return_tensors="pt", add_special_tokens=not text.startswith("<bos>")).input_ids.to("cuda")
        print(f"--- {label}: {ids.shape[1]} prompt tokens, greedy {n} new tokens")
        for name, m in [("pt", pt), ("it", it)]:
            with torch.no_grad():
                p = torch.softmax(m(ids).logits[0, -1].float(), -1)
            tp, ti = torch.topk(p, 5)
            new, txt = gen(m, ids, n)
            print(f"  [{name}] next-token top-5: " + ", ".join(f"{TOK.decode([i])!r} {v:.4f}" for i, v in zip(ti.tolist(), tp.tolist())))
            print(f"  [{name}] {len(new)} tokens, last {TOK.convert_ids_to_tokens(int(new[-1]))!r}, contains <end_of_turn>: {107 in new.tolist()}")
            print(f"  [{name}] text: {txt!r}")


def main():
    global TOK
    section("files")
    sp, si = files(PT), files(IT)
    section("config")
    config_diff(sp, si)
    section("tokenizer")
    TOK = AutoTokenizer.from_pretrained(IT)
    tpt = AutoTokenizer.from_pretrained(PT)
    print(f"it tokenizer len {len(TOK)}, pt tokenizer len {len(tpt)}; same vocab: {TOK.get_vocab() == tpt.get_vocab()}")
    print(f"chat_template present: it {TOK.chat_template is not None}, pt {tpt.chat_template is not None}; identical: {TOK.chat_template == tpt.chat_template}")
    print(f"max token id {max(TOK.get_vocab().values())}; special tokens map {TOK.special_tokens_map}")
    print(f"GenerationConfig it: eos {GenerationConfig.from_pretrained(IT).eos_token_id}, pt: eos {GenerationConfig.from_pretrained(PT).eos_token_id}")
    try:
        tpt.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=False, add_generation_prompt=True)
        print("pt apply_chat_template: no error")
    except Exception as e:
        print(f"pt apply_chat_template: {type(e).__name__}: {e}")
    section("weights pt vs it (safetensors, CPU)")
    weight_diff()
    section("load")
    t0 = time.time()
    it = AutoModelForCausalLM.from_pretrained(IT, torch_dtype=hw3.DTYPE, device_map="cuda", attn_implementation="eager").eval()
    t1 = time.time()
    pt = AutoModelForCausalLM.from_pretrained(PT, torch_dtype=hw3.DTYPE, device_map="cuda", attn_implementation="eager").eval()
    print(f"it loaded {t1 - t0:.1f}s, pt loaded {time.time() - t1:.1f}s; both fp16 on GPU, memory_allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    section("behaviour pt vs it")
    compare(pt, it)


if __name__ == "__main__":
    main()
