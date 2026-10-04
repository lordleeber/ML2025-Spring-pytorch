"""Measure every number the HW03 textbook cites (needs GPU + HF access to Gemma).

Run from the repo root:
    .venv/bin/python docs/tools/hw03_facts.py [env model tok q1 q2 q4 q5 q6 q7 shapes perq q4steps ptit rescale26 review_ch01 kvcache review_ch02 pre_ch03 review_ch03 pre_ch04 review_ch04 pre_ch06 pre_ch07 review_ch07]

With no arguments every section runs. Output is plain text meant to be pasted
(after review) into docs/HW03/FACTS.md. Experiment figures go to docs/HW03/img/.
Functions from HW03/hw3.py are reused where possible so the numbers come from the
same code the book quotes.
"""

import importlib.metadata as md
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "HW03"))
import hw3  # noqa: E402

IMG = ROOT / "docs" / "HW03" / "img"
DEVICE = hw3.DEVICE


def section(name):
    print(f"\n{'=' * 20} {name} {'=' * 20}", flush=True)


def save_exp_fig(name):
    import matplotlib.pyplot as plt

    IMG.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMG / name, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[saved] docs/HW03/img/{name}")


# ---------------------------------------------------------------------------
def env():
    section("env")
    print("python", sys.version.split()[0])
    for pkg in ["torch", "transformers", "tokenizers", "accelerate", "huggingface_hub", "sae-lens",
                "transformer-lens", "nltk", "scikit-learn", "numpy", "matplotlib", "seaborn"]:
        print(f"{pkg} {md.version(pkg)}")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU {p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.1f} GiB, CUDA {torch.version.cuda}")
    from huggingface_hub import scan_cache_dir

    for repo in scan_cache_dir().repos:
        print(f"HF cache {repo.repo_type} {repo.repo_id}: {repo.size_on_disk / 1e9:.3f} GB")


# ---------------------------------------------------------------------------
def model_facts(tokenizer, model):
    section("model")
    cfg = model.config
    for k in ["num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
              "head_dim", "vocab_size", "max_position_embeddings", "sliding_window", "attn_logit_softcapping",
              "final_logit_softcapping", "query_pre_attn_scalar", "rope_theta", "rms_norm_eps", "hidden_activation",
              "torch_dtype", "tie_word_embeddings", "_attn_implementation"]:
        print(f"config.{k} = {getattr(cfg, k, None)}")
    print("generation_config:", model.generation_config.to_diff_dict())
    print("model dtype:", model.dtype, "device:", model.device)

    total = sum(p.numel() for p in model.parameters())
    emb = model.model.embed_tokens.weight.numel()
    print(f"params total {total:,}; embed_tokens {emb:,} ({emb / total:.1%}); non-embedding {total - emb:,}")
    print("lm_head tied to embed_tokens:", model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())
    layer = model.model.layers[0]
    for name, p in layer.named_parameters():
        print(f"  layers.0.{name} {tuple(p.shape)} {p.numel():,}")
    per_layer = sum(p.numel() for p in layer.parameters())
    print(f"per layer {per_layer:,}; x26 = {per_layer * 26:,}; final norm {model.model.norm.weight.numel():,}")
    print("sliding layers:", [i for i, l in enumerate(model.model.layers) if l.is_sliding])
    print(f"weights in fp16: {total * 2 / 2**30:.2f} GiB; cuda allocated after load {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    print(model.model.layers[0])

    # hidden_states indexing: compare with forward hooks on each decoder layer
    outs = {}
    hooks = [l.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k, o[0])) for k, l in enumerate(model.model.layers)]
    ids = tokenizer("Time travel will become a reality.", return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    for h in hooks:
        h.remove()
    print(f"len(hidden_states) = {len(hs)}")
    emb_scaled = model.model.embed_tokens(ids) * torch.tensor(cfg.hidden_size**0.5, dtype=model.dtype)
    print("hidden_states[0] == embed_tokens * sqrt(2304):", torch.equal(hs[0], emb_scaled))
    for i in [1, 20, 21, 25]:
        print(f"hidden_states[{i}] == output of layers[{i - 1}]:", torch.equal(hs[i], outs[i - 1]))
    print("hidden_states[26] == output of layers[25]:", torch.equal(hs[26], outs[25]))
    print("hidden_states[26] == norm(output of layers[25]):", torch.equal(hs[26], model.model.norm(outs[25])))
    print("per-position L2 norm of hidden_states (mean over tokens excl. <bos> / <bos>):")
    for i in [0, 1, 5, 10, 15, 20, 21, 24, 25, 26]:
        n = hs[i][0].float().norm(dim=-1)
        print(f"  [{i:2d}] {n[1:].mean():9.1f} / {n[0]:9.1f}")


def attn_default():
    section("attn_implementation default")
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(hw3.MODEL_ID, device_map=DEVICE, torch_dtype=hw3.DTYPE)
    print("without attn_implementation:", m.config._attn_implementation)
    del m
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
def tok_facts(tokenizer):
    section("tokenizer")
    print(type(tokenizer).__name__, "len", len(tokenizer), "padding_side", tokenizer.padding_side)
    print("special:", tokenizer.special_tokens_map)
    for t in ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>", "\n", "▁"]:
        print(f"  {t!r} -> {tokenizer.convert_tokens_to_ids(t)}")
    print("chat_template:", repr(tokenizer.chat_template))

    q = "Please tell me about the key differences between supervised learning and unsupervised learning. Answer in 200 words."
    p = tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    print("Q1 prompt_with_template repr:", repr(p))
    ids = tokenizer(p).input_ids
    print(f"tokenizer(prompt).input_ids: len {len(ids)}, first 6 {ids[:6]} -> {tokenizer.convert_ids_to_tokens(ids[:6])}, last 5 {tokenizer.convert_ids_to_tokens(ids[-5:])}")
    ids2 = tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True)
    print(f"apply_chat_template(tokenize=True): len {len(ids2)}, first 3 {tokenizer.convert_ids_to_tokens(ids2[:3])}")
    ids3 = tokenizer(q).input_ids
    print(f"plain question: len {len(ids3)}, first 3 {tokenizer.convert_ids_to_tokens(ids3[:3])}")

    for s in [hw3.Q2_TURNS[0], "Google ", "Google", "Hung-yi", "Hungyi", " Hung-yi Lee", "機器學習", "李宏毅", "2025",
              "12345", "unsupervised", "Time travel will become a reality as technology continues to advance."]:
        i = tokenizer.encode(s, add_special_tokens=False)
        print(f"  {s!r}: {list(zip(tokenizer.convert_ids_to_tokens(i), i))}")
    print("decode([235285, 2182]) =", repr(tokenizer.decode([235285, 2182])))
    print("Q3 slide check: '▁you' ->", tokenizer.convert_tokens_to_ids("▁you"), "'you' ->", tokenizer.convert_tokens_to_ids("you"),
          "'?' ->", tokenizer.convert_tokens_to_ids("?"), "id 23533 ->", tokenizer.convert_ids_to_tokens(23533))


# ---------------------------------------------------------------------------
def q1_facts(tokenizer, model):
    section("q1")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    sm = AutoModelForSequenceClassification.from_pretrained(hw3.SCORING_MODEL_ID)
    st = AutoTokenizer.from_pretrained(hw3.SCORING_MODEL_ID)
    print("scoring model params:", f"{sum(p.numel() for p in sm.parameters()):,}", "num_labels", sm.config.num_labels,
          "max_len", st.model_max_length, "device", sm.device)
    q = "Please tell me about the key differences between supervised learning and unsupervised learning. Answer in 200 words."
    p = tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

    def run(label, ids, split):
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=512, do_sample=False)
        dt = time.time() - t0
        new = out[0, ids.shape[1]:]
        full = tokenizer.decode(out[0], skip_special_tokens=True)
        resp = split(full)
        score = hw3.calculate_coherence(q, resp, sm, st)
        n_score_tokens = len(st(q, resp, truncation=True).input_ids)
        print(f"--- {label}: new tokens {len(new)}, last token {tokenizer.convert_ids_to_tokens(new[-1].item())!r}, "
              f"{dt:.1f}s ({len(new) / dt:.1f} tok/s), words {len(resp.split())}, scorer tokens {n_score_tokens}, "
              f"score {score:.4f}")
        print(f"decoded full (repr, first 300): {full[:300]!r}")
        print("response:\n" + resp)
        return resp

    run("with template (double <bos>, as hw3.py)", tokenizer(p, return_tensors="pt").input_ids.to(DEVICE),
        lambda s: s.split("model\n")[-1].strip("\n").strip())
    run("with template, single <bos> (add_special_tokens=False)",
        tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE),
        lambda s: s.split("model\n")[-1].strip("\n").strip())
    run("without template", tokenizer(q, return_tensors="pt").input_ids.to(DEVICE),
        lambda s: s.split(q.split(" ")[-1])[-1].strip("\n").strip())
    print("split key for no-template:", repr(q.split(" ")[-1]))


# ---------------------------------------------------------------------------
def q2_facts(tokenizer, model):
    section("q2")
    hist = []
    for r, u in enumerate(hw3.Q2_TURNS, 1):
        hist.append({"role": "user", "content": u})
        p = tokenizer.apply_chat_template(hist, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(p, return_tensors="pt").to(DEVICE)
        print(f"--- round {r}: prompt tokens {inputs.input_ids.shape[1]}, first 3 {tokenizer.convert_ids_to_tokens(inputs.input_ids[0, :3].tolist())}")
        print("prompt repr:", repr(p))
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        p32 = torch.softmax(logits.float(), -1)
        p16 = torch.softmax(logits, -1)
        tp, ti = torch.topk(p32, 10)
        print(f"logits dtype {logits.dtype}; top-10 sum {tp.sum():.4f}")
        for prob, idx in zip(tp[0].tolist(), ti[0].tolist()):
            print(f"  {idx:6d} {tokenizer.decode([idx])!r:>14} fp32 {prob:.4f}  fp16-softmax {p16[0, idx].item():.4f}  logit {logits[0, idx].item():.3f}")
        out = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        new = out[0][inputs.input_ids.shape[1]:]
        resp = tokenizer.decode(new, skip_special_tokens=True)
        print(f"generated ids {new.tolist()} -> {tokenizer.convert_ids_to_tokens(new.tolist())}; response {resp!r}")
        hist.append({"role": "assistant", "content": resp})


# ---------------------------------------------------------------------------
def q4_facts(tokenizer, model):
    section("q4")
    from nltk.translate.bleu_score import sentence_bleu

    prompt = "Generate a paraphrase of the sentence 'Professor Hung-yi Lee is one of the best teachers in the domain of machine learning'. Just response with one sentence."
    enc = tokenizer(prompt, return_tensors="pt")
    n_prompt = enc.input_ids.shape[1]
    print("prompt tokens", n_prompt, "max_length", n_prompt + 30)
    gp = model.generation_config
    print(f"generation_config top_k={gp.top_k} top_p={gp.top_p} temperature={gp.temperature} eos={gp.eos_token_id}; tokenizer.eos_token_id={tokenizer.eos_token_id}")
    base = dict(do_sample=True, max_length=30 + n_prompt, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id, attention_mask=enc.attention_mask.to(DEVICE), use_cache=True,
                return_dict_in_generate=True, output_scores=False)
    ids = enc.input_ids.to(DEVICE)
    from transformers import GenerationConfig

    gc = model._prepare_generation_config(None, top_p=0.6, do_sample=True)[0]
    print(f"GenerationConfig() class defaults: top_k={GenerationConfig().top_k} top_p={GenerationConfig().top_p}; "
          f"effective config for generate(top_p=0.6): top_k={gc.top_k} top_p={gc.top_p} temperature={gc.temperature}")

    def sample(n, **kw):
        sents, lens = [], []
        for _ in range(n):
            o = model.generate(input_ids=ids, **kw, **base)
            g = o.sequences[0, n_prompt:]
            lens.append(len(g))
            d = tokenizer.decode(g, skip_special_tokens=True)
            sents.append(d.replace(" ,", ",").replace(" 's", "'s").replace(" .", ".").strip())
        return sents, lens

    configs = [("top_k=2", dict(top_k=2)), ("top_k=200", dict(top_k=200)), ("top_p=0.6", dict(top_p=0.6)),
               ("top_p=0.999", dict(top_p=0.999)), ("top_p=0.6,top_k=0", dict(top_p=0.6, top_k=0)),
               ("top_p=0.999,top_k=0", dict(top_p=0.999, top_k=0))]
    for seed in range(5):
        row = []
        for name, kw in configs:
            torch.manual_seed(seed)
            s, lens = sample(20, **kw)
            row.append(f"{name}: {hw3.compute_self_bleu(s):.4f} ({len(set(s))} uniq, hit max_length {sum(l == 30 for l in lens)}/20)")
            if seed == 0:
                print(f"--- seed 0 {name} sentences:")
                for i, x in enumerate(s):
                    print(f"  {i}. {x}")
        print(f"seed {seed}: " + " | ".join(row))

    for name, kw in [("top_k=1", dict(top_k=1)), ("top_p=0", dict(top_p=0.0))]:
        outs = set()
        for seed in range(3):
            torch.manual_seed(seed)
            outs.update(sample(3, **kw)[0])
        print(f"{name}: {len(outs)} distinct over 9 samples: {sorted(outs)}")
    with torch.no_grad():
        g = model.generate(ids, max_new_tokens=30, do_sample=False)
    print("greedy:", repr(tokenizer.decode(g[0, n_prompt:], skip_special_tokens=True)))

    torch.manual_seed(0)
    s, lens = sample(20, top_k=2)
    print("raw decode example (seed 0, top_k=2, #0):", repr(tokenizer.decode(model.generate(input_ids=ids, top_k=2, **base).sequences[0, n_prompt:])))
    print("hit max_length (30 new tokens) count, seed0 top_k=2:", sum(l == 30 for l in lens))
    try:
        hw3.compute_self_bleu(["only one sentence"])
    except Exception as e:
        print("compute_self_bleu with 1 sentence ->", type(e).__name__, e)
    print("BLEU of identical sentences:", sentence_bleu(["a b c d e".split()], "a b c d e".split()))


# ---------------------------------------------------------------------------
def q5_facts(tokenizer, model):
    section("q5")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    sentences = ["I ate a fresh apple.", "Apple released the new iPhone.", "I peeled an orange and ate it.",
                 "The Orange network has great coverage.", "Microsoft announced a new update.", "Banana is my favorite fruit."]
    labels = ["Apple (fruit)", "Apple (company)", "Orange (fruit)", "Orange (telecom)", "Microsoft (company)", "Banana (fruit)"]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    print("input_ids shape", tuple(inputs.input_ids.shape))
    for s, row, m in zip(sentences, inputs.input_ids, inputs.attention_mask):
        print(f"  {s!r}: real tokens {int(m.sum())}, pads {int((m == 0).sum())}, tokens {tokenizer.convert_ids_to_tokens(row.tolist())}")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-1].float()
    print("hidden_states[-1] shape", tuple(h.shape), "dtype (before .float())", out.hidden_states[-1].dtype)
    m = inputs.attention_mask.unsqueeze(-1).float()
    emb_all = h.mean(1).cpu().numpy()
    emb_masked = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
    emb_nobos = ((h * m)[:, :, :].sum(1) - (h * m)[torch.arange(6), (m[:, :, 0] == 0).sum(1)]) / (m.sum(1) - 1)
    emb_nobos = emb_nobos.cpu().numpy()
    print("per-position norms of row 0 (pads...<bos>...):", [round(x, 1) for x in h[0].norm(dim=-1).tolist()])

    def cos(e):
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        return e @ e.T

    for name, e in [("mean over all positions incl. pad (hw3.py)", emb_all), ("masked mean (no pad)", emb_masked),
                    ("masked mean, no pad, no <bos>", emb_nobos)]:
        print(f"--- cosine similarity, {name}:")
        c = cos(e)
        for lab, r in zip(labels, c):
            print(f"  {lab:>20} " + " ".join(f"{x:6.3f}" for x in r))
        e2 = TSNE(n_components=2, perplexity=2, random_state=42).fit_transform(e)
        print("  t-SNE 2D:", [(lab, round(float(a), 1), round(float(b), 1)) for lab, (a, b) in zip(labels, e2)])
        if name.startswith("masked mean (no pad)"):
            plt.figure(figsize=(8, 6))
            colors = ["red", "blue", "orange", "purple", "green", "brown"]
            for i, lab in enumerate(labels):
                plt.scatter(e2[i, 0], e2[i, 1], color=colors[i], s=100)
                plt.text(e2[i, 0] + 0.1, e2[i, 1] + 0.1, lab, fontsize=12, color=colors[i])
            plt.xlabel("t-SNE Dim 1")
            plt.ylabel("t-SNE Dim 2")
            plt.title("t-SNE (masked mean pooling, padding excluded)")
            save_exp_fig("exp_q5_tsne_masked.png")
    for seed in [0, 1, 42, 123]:
        e2 = TSNE(n_components=2, perplexity=2, random_state=seed).fit_transform(emb_all)
        d = np.linalg.norm(e2[:, None] - e2[None], axis=-1)
        nn = [labels[int(np.argsort(r)[1])] for r in d]
        print(f"t-SNE random_state={seed} (hw3 embeddings) nearest neighbour:", dict(zip(labels, nn)))


# ---------------------------------------------------------------------------
def q6_facts(tokenizer, model):
    section("q6")
    from transformers import HybridCache

    for layer_idx, head_idx in [(10, 7), (0, 0), (25, 0)]:
        prompt = "Google "
        enc = tokenizer(prompt, return_tensors="pt")
        nxt = enc.input_ids.to(DEVICE)
        mask = enc.attention_mask.to(DEVICE)
        pos = torch.arange(mask.shape[1], device=DEVICE)
        total = 20 + nxt.size(1) - 1
        cache = HybridCache(config=model.config, max_batch_size=1, max_cache_len=total, device=DEVICE, dtype=hw3.DTYPE)
        row_tokens = tokenizer.convert_ids_to_tokens(nxt[0].tolist())
        gen, rows, shapes = [], [], []
        for step in range(20):
            with torch.no_grad():
                o = model(nxt, attention_mask=mask, cache_position=pos, use_cache=True, past_key_values=cache, output_attentions=True)
            a = o.attentions[layer_idx][0][head_idx].float().cpu().numpy()
            shapes.append(tuple(o.attentions[layer_idx].shape))
            rows.append(a)
            t = o.logits[:, -1, :].argmax(-1)
            gen.append(t.item())
            if step < 19:
                row_tokens.append(tokenizer.convert_ids_to_tokens(t.item()))
            mask = torch.cat([mask, torch.ones(1, 1, device=DEVICE)], dim=-1)
            nxt = t.unsqueeze(0)
            cache = o.past_key_values
            pos = pos[-1:] + 1
        A = np.concatenate(rows, 0)
        print(f"--- layer {layer_idx} head {head_idx} (sliding={model.model.layers[layer_idx].is_sliding})")
        if layer_idx == 10:
            print("prompt ids", enc.input_ids[0].tolist(), tokenizer.convert_ids_to_tokens(enc.input_ids[0].tolist()))
            print("max_cache_len", total, "; attentions per step shape: step0", shapes[0], "step1", shapes[1], "; len(outputs.attentions)", len(o.attentions))
            print("generated ids", gen)
            print("generated tokens", tokenizer.convert_ids_to_tokens(gen))
            full = prompt + tokenizer.decode(gen, skip_special_tokens=True)
            labels = tokenizer.tokenize(full)
            print("full_text", repr(full))
            print(f"tokenize(full_text) {len(labels)}: {labels}")
            print(f"actual query/key tokens {len(row_tokens)}: {row_tokens}")
            print("row | label used by hw3.py | actual token")
            for i in range(A.shape[0]):
                print(f"  {i:2d} | {labels[i] if i < len(labels) else '-'!r} | {row_tokens[i]!r}")
            print("last generated token (never fed back, no row):", repr(tokenizer.convert_ids_to_tokens(gen[-1])))
            print("upper triangle (future) max:", float(np.triu(A, 1).max()))
            print("row sums min/max:", float(A.sum(1).min()), float(A.sum(1).max()))
        col0 = A[:, 0]
        print("attention to column 0 (<bos>) per row:", [round(float(x), 3) for x in col0])
        print(f"mean attention to <bos> over rows 1..: {col0[1:].mean():.3f}; diag mean {np.diag(A).mean():.3f}")
        print("argmax key per row:", [int(x) for x in A.argmax(1)])
    # all heads of layer 10: share of attention on <bos>
    enc = tokenizer("Google is a multinational technology company", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        att = model(**enc, output_attentions=True).attentions
    bos = torch.stack([a[0, :, 1:, 0].float().mean(-1) for a in att])  # [layer, head]
    print("mean attention to <bos> (rows 1..) per layer, avg over heads:", [round(x, 2) for x in bos.mean(1).tolist()])
    print("layer 10 per head:", [round(x, 2) for x in bos[10].tolist()])

    # corrected-label figure (same matrix as hw3 default, labels = actual tokens)
    import matplotlib.pyplot as plt
    import seaborn as sns

    prompt = "Google "
    enc = tokenizer(prompt, return_tensors="pt")
    nxt, mask = enc.input_ids.to(DEVICE), enc.attention_mask.to(DEVICE)
    pos = torch.arange(mask.shape[1], device=DEVICE)
    cache = HybridCache(config=model.config, max_batch_size=1, max_cache_len=22, device=DEVICE, dtype=hw3.DTYPE)
    toks = tokenizer.convert_ids_to_tokens(nxt[0].tolist())
    rows = []
    for step in range(20):
        with torch.no_grad():
            o = model(nxt, attention_mask=mask, cache_position=pos, use_cache=True, past_key_values=cache, output_attentions=True)
        rows.append(o.attentions[10][0][7].float().cpu().numpy())
        t = o.logits[:, -1, :].argmax(-1)
        if step < 19:
            toks.append(tokenizer.convert_ids_to_tokens(t.item()))
        mask = torch.cat([mask, torch.ones(1, 1, device=DEVICE)], dim=-1)
        nxt, cache, pos = t.unsqueeze(0), o.past_key_values, pos[-1:] + 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.concatenate(rows), xticklabels=toks, yticklabels=toks, cmap="viridis")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.title("Layer 10 Head 7, labels = tokens actually fed to the model")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    save_exp_fig("exp_q6_attention_true_labels.png")


# ---------------------------------------------------------------------------
def q7_facts(tokenizer, model):
    section("q7")
    from sae_lens import SAE

    t0 = time.time()
    sae, cfg_dict, sparsity = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id="layer_20/width_16k/canonical")
    print(f"SAE load {time.time() - t0:.1f}s; type {type(sae).__name__}; sparsity {sparsity}")
    print("cfg_dict:", cfg_dict)
    for n, p in sae.named_parameters():
        print(f"  {n} {tuple(p.shape)} {p.dtype} {p.numel():,}")
    print("SAE params", f"{sum(p.numel() for p in sae.parameters()):,}")
    sae = sae.to(DEVICE)
    th = sae.threshold.detach().float()
    print(f"threshold: min {th.min():.3f} median {th.median():.3f} max {th.max():.3f}; feature 10004 threshold {th[10004]:.4f}")
    print("cfg.hook_layer", sae.cfg.hook_layer, "cfg.hook_name", sae.cfg.hook_name, "cfg.model_name", sae.cfg.model_name)
    print("encode signature uses dtype:", sae.dtype if hasattr(sae, "dtype") else None)

    def acts(prompt, idx):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
            fa = sae.encode(hs[idx]).squeeze()
        return tokenizer.convert_ids_to_tokens(ids[0].tolist()), fa, hs

    prompts = {"a": "Time travel offers me the opportunity to correct past errors, but it comes with its own set of risks.",
               "b": "I accept that my decisions shape my future, and though mistakes are inevitable, they define who I become.",
               "c": "Time travel will become a reality as technology continues to advance."}
    for key, pr in prompts.items():
        for idx in [20, 21]:
            toks, fa, hs = acts(pr, idx)
            v = fa[:, 10004].float().cpu().numpy()
            l0 = (fa[1:] > 0).sum(-1).float()
            note = "(hw3.py: hidden_states[hook_layer])" if idx == 20 else "(output of block 20 = blocks.20.hook_resid_post)"
            print(f"--- prompt {key} hidden_states[{idx}] {note}: encode dtype {fa.dtype}, shape {tuple(fa.shape)}")
            print(f"  feature 10004 max {v.max():.4f} at {toks[int(v.argmax())]!r}; nonzero tokens {int((v > 0).sum())}/{len(v)}; "
                  f"max excl <bos> {v[1:].max():.4f}; L0 per token (excl <bos>) mean {l0.mean():.1f}")
            print("  " + ", ".join(f"{t}:{a:.2f}" for t, a in zip(toks, v)))
            if idx == 21:
                top = torch.topk(fa[1:].max(0).values.float(), 5)
                print("  top-5 features over tokens (excl <bos>):", [(int(i), round(float(x), 1)) for x, i in zip(top.values, top.indices)])
            rec = sae.decode(sae.encode(hs[idx].float()))
            x = hs[idx].float()
            fvu = ((rec - x)[0, 1:] ** 2).sum() / ((x[0, 1:] - x[0, 1:].mean(0)) ** 2).sum()
            print(f"  reconstruction FVU (excl <bos>) {fvu:.3f}")
    toks, _, hs = acts(prompts["c"], 20)
    print("--- prompt c, feature 10004 for every token x every hidden_states index (rows = tokens):")
    print("  idx " + " ".join(f"{i:>5d}" for i in range(len(hs))))
    with torch.no_grad():
        table = torch.stack([sae.encode(h).squeeze()[:, 10004].float() for h in hs], 1).cpu().numpy()
    for t, r in zip(toks, table):
        print(f"  {t!r:>14} " + " ".join(f"{x:5.1f}" for x in r))
    print("hidden_states L2 norm per index for token 'Time' (1):", [round(float(h[0, 1].float().norm()), 0) for h in hs])


# ---------------------------------------------------------------------------
# Review round 1 (outline TODOs): shapes, perq, q4steps, ptit, rescale26
def shapes_facts(tokenizer, model):
    section("shapes")
    ids = tokenizer("Time travel will become a reality as technology continues to advance.", return_tensors="pt").input_ids.to(DEVICE)
    print("input_ids", tuple(ids.shape), "(prompt c of Q7, with <bos>)")
    seen = []

    def hook(name):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            seen.append((name, tuple(inp[0].shape) if inp else None, tuple(o.shape), o.dtype))
        return f

    hs = []
    for li in [0, 1]:
        layer = model.model.layers[li]
        mods = [("embed_tokens", model.model.embed_tokens)] if li == 0 else []
        mods += [(f"layers[{li}].input_layernorm", layer.input_layernorm), (f"layers[{li}].self_attn.q_proj", layer.self_attn.q_proj),
                 (f"layers[{li}].self_attn.k_proj", layer.self_attn.k_proj), (f"layers[{li}].self_attn.v_proj", layer.self_attn.v_proj),
                 (f"layers[{li}].self_attn.o_proj", layer.self_attn.o_proj), (f"layers[{li}].self_attn", layer.self_attn),
                 (f"layers[{li}].post_attention_layernorm", layer.post_attention_layernorm),
                 (f"layers[{li}].pre_feedforward_layernorm", layer.pre_feedforward_layernorm),
                 (f"layers[{li}].mlp.gate_proj", layer.mlp.gate_proj), (f"layers[{li}].mlp.up_proj", layer.mlp.up_proj),
                 (f"layers[{li}].mlp.down_proj", layer.mlp.down_proj), (f"layers[{li}].post_feedforward_layernorm", layer.post_feedforward_layernorm),
                 (f"layers[{li}]", layer)]
        hs += [m.register_forward_hook(hook(n)) for n, m in mods]
    hs.append(model.model.norm.register_forward_hook(hook("norm")))
    hs.append(model.lm_head.register_forward_hook(hook("lm_head")))
    with torch.no_grad():
        out = model(ids, output_attentions=True, output_hidden_states=True)
    for h in hs:
        h.remove()
    for name, i, o, dt in seen:
        print(f"  {name:38s} in {i} -> out {o} {dt}")
    print("attentions: len", len(out.attentions), "each", tuple(out.attentions[0].shape), out.attentions[0].dtype)
    print("hidden_states: len", len(out.hidden_states), "each", tuple(out.hidden_states[0].shape))
    print("logits", tuple(out.logits.shape), out.logits.dtype, f"max |logit| {out.logits.abs().max():.2f} (final soft-cap 30)")
    print("past_key_values", type(out.past_key_values).__name__)
    cfg = model.config
    print(f"q_proj 2304->{cfg.num_attention_heads}x{cfg.head_dim}={cfg.num_attention_heads * cfg.head_dim}, "
          f"k/v_proj 2304->{cfg.num_key_value_heads}x{cfg.head_dim}={cfg.num_key_value_heads * cfg.head_dim}, "
          f"query_pre_attn_scalar {cfg.query_pre_attn_scalar}, sliding_window {cfg.sliding_window}")


def perq_facts(tokenizer, model):
    import argparse

    section("perq")
    args = argparse.Namespace(max_new_tokens=512, interactive=False, sentence="I love taking a Machine Learning course by Professor Hung-yi Lee, What about you?",
                              top_k=2, top_p=0.6, num_samples=20, layer_idx=10, head_idx=7, sae_layer_idx=24, token_idx=[1])
    gib = 2**30
    print(f"after load_model: allocated {torch.cuda.memory_allocated() / gib:.2f} GiB, reserved {torch.cuda.memory_reserved() / gib:.2f} GiB")
    torch.manual_seed(0)
    rows = []
    for q, fn in hw3.QUESTIONS.items():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        fn(tokenizer, model, args)
        torch.cuda.synchronize()
        c = getattr(model, "_cache", None)  # generate() keeps its HybridCache on the model for reuse
        cache = sum(t.numel() * t.element_size() for t in c.key_cache + c.value_cache) / gib if c is not None else 0.0
        rows.append((q, time.time() - t0, torch.cuda.max_memory_allocated() / gib, torch.cuda.memory_allocated() / gib,
                     cache, getattr(c, "max_cache_len", None)))
    print("\n--- per question (one process, model loaded once, --seed 0, default flags; Q1/Q7 time includes loading the scorer/SAE from cache)")
    for q, t, peak, after, cache, clen in rows:
        print(f"  Q{q}: {t:6.1f} s, peak allocated {peak:.2f} GiB, allocated after {after:.2f} GiB, "
              f"model._cache {cache:.3f} GiB (max_cache_len {clen})")
    print(f"  total {sum(r[1] for r in rows):.1f} s")


def q4steps_facts(tokenizer, model):
    section("q4steps")
    prompt = "Generate a paraphrase of the sentence 'Professor Hung-yi Lee is one of the best teachers in the domain of machine learning'. Just response with one sentence."
    input_ids = tokenizer(prompt, return_tensors="pt")
    # identical to hw3.py:211-221 plus logits/scores output
    generation_params = {
        "do_sample": True,
        "max_length": 30 + len(input_ids.input_ids[0]),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "attention_mask": input_ids.attention_mask.to(DEVICE),
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_logits": True,
    }
    for name, kw in [("top_k=2", dict(top_k=2)), ("top_p=0.6", dict(top_p=0.6)), ("top_p=0.999", dict(top_p=0.999)), ("top_k=200", dict(top_k=200))]:
        torch.manual_seed(0)
        top1, kept, steps, greedy_steps = [], [], 0, 0
        for _ in range(20):
            o = model.generate(input_ids=input_ids.input_ids.to(DEVICE), **kw, **generation_params)
            for lg, sc in zip(o.logits, o.scores):
                p = torch.softmax(lg[0].float(), -1)
                top1.append(float(p.max()))
                n = int(torch.isfinite(sc[0]).sum())
                kept.append(n)
                steps += 1
                greedy_steps += n == 1
        top1, kept = np.array(top1), np.array(kept)
        print(f"{name}: {steps} sampling steps over 20 sentences; top-1 prob mean {top1.mean():.3f} median {np.median(top1):.3f}; "
              f"steps with top-1 > 0.6: {(top1 > 0.6).sum()} ({(top1 > 0.6).mean():.1%}); "
              f"steps where only 1 token survives filtering: {greedy_steps} ({greedy_steps / steps:.1%}); "
              f"kept tokens per step mean {kept.mean():.2f} median {np.median(kept):.0f} max {kept.max()}")
        print("   kept-token histogram:", {int(k): int(v) for k, v in zip(*np.unique(kept, return_counts=True))})


def _sae():
    from sae_lens import SAE

    sae, _, _ = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id="layer_20/width_16k/canonical")
    return sae.to(DEVICE)


Q7_PROMPTS = {"a": "Time travel offers me the opportunity to correct past errors, but it comes with its own set of risks.",
              "b": "I accept that my decisions shape my future, and though mistakes are inevitable, they define who I become.",
              "c": "Time travel will become a reality as technology continues to advance."}


def _sae_stats(sae, tokenizer, model, prompt, idx):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        x = model(ids, output_hidden_states=True).hidden_states[idx].float()
        fa = sae.encode(x)
        rec = sae.decode(fa)
    toks = tokenizer.convert_ids_to_tokens(ids[0].tolist())
    v = fa[0, :, 10004].cpu().numpy()
    fvu = float(((rec - x)[0, 1:] ** 2).sum() / ((x[0, 1:] - x[0, 1:].mean(0)) ** 2).sum())
    l0 = float((fa[0, 1:] > 0).sum(-1).float().mean())
    return toks, v, fvu, l0


def ptit_facts(tokenizer, model):
    from transformers import AutoModelForCausalLM

    section("ptit")
    sae = _sae()
    t0 = time.time()
    pt = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", torch_dtype=hw3.DTYPE, device_map="cuda", attn_implementation="eager").eval()
    print(f"loaded google/gemma-2-2b (pt) in {time.time() - t0:.1f}s")
    for key, prompt in Q7_PROMPTS.items():
        for idx in [20, 21, 24]:
            for mname, m in [("it", model), ("pt", pt)]:
                toks, v, fvu, l0 = _sae_stats(sae, tokenizer, m, prompt, idx)
                print(f"prompt {key} hidden_states[{idx}] {mname}: FVU {fvu:.3f}, L0 {l0:.1f}; feature 10004 max {v.max():.4f} at {toks[int(v.argmax())]!r}, "
                      f"max excl <bos> {v[1:].max():.4f}")
                print("    " + ", ".join(f"{t}:{a:.2f}" for t, a in zip(toks, v)))
    del pt
    torch.cuda.empty_cache()


def rescale26_facts(tokenizer, model):
    section("rescale26")
    sae = _sae()
    ids = tokenizer(Q7_PROMPTS["c"], return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
        h25, h26 = hs[25][0].float(), hs[26][0].float()
        scaled = h26 * (h25.norm(dim=-1, keepdim=True) / h26.norm(dim=-1, keepdim=True))
        rows = [("hidden_states[25]", h25), ("hidden_states[26] (as hw3.py)", h26), ("hidden_states[26] rescaled to [25]'s per-token norm", scaled)]
        toks = tokenizer.convert_ids_to_tokens(ids[0].tolist())
        for name, x in rows:
            v = sae.encode(x)[:, 10004].cpu().numpy()
            print(f"{name}: " + ", ".join(f"{t}:{a:.2f}" for t, a in zip(toks, v)))
        print("per-token norm [25]:", [round(float(n)) for n in h25.norm(dim=-1)])
        print("per-token norm [26]:", [round(float(n)) for n in h26.norm(dim=-1)])
        print("cosine([25],[26]) per token:", [round(float(c), 3) for c in torch.nn.functional.cosine_similarity(h25, h26, dim=-1)])


# ---------------------------------------------------------------------------
# Review of ch01: what the Colab TODO does without max_new_tokens / do_sample, chat_template errors
def review_ch01_facts(tokenizer, model):
    import warnings

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    section("review_ch01")
    sm = AutoModelForSequenceClassification.from_pretrained(hw3.SCORING_MODEL_ID)
    st = AutoTokenizer.from_pretrained(hw3.SCORING_MODEL_ID)
    question = "Please tell me about the key differences between supervised learning and unsupervised learning. Answer in 200 words."
    prompt_t = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)

    def run(prompt, label, **kw):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)  # as hw3.py:77 (double <bos>)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = model.generate(input_ids, **kw)
        new = out[0, input_ids.shape[1]:]
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        resp = (text.split("model\n")[-1] if prompt is prompt_t else text.split(question.split(" ")[-1])[-1]).strip("\n").strip()
        score = hw3.calculate_coherence(question, resp, sm, st)
        print(f"--- {label}: generate({', '.join(f'{k}={v}' for k, v in kw.items())})")
        print(f"  new tokens {len(new)}, last token {tokenizer.convert_ids_to_tokens(int(new[-1]))!r}, coherence {score:.4f}")
        for x in w:
            print(f"  python warning: {x.category.__name__}: {x.message}")
        print(f"  response: {resp!r}")
        return new

    for prompt, name in [(prompt_t, "with template"), (question, "without template")]:
        model._cache = None  # start every run from a fresh KV cache (see R10)
        ref = run(prompt, f"{name}, hw3.py", max_new_tokens=512, do_sample=False)
        model._cache = None
        a = run(prompt, f"{name}, no do_sample", max_new_tokens=512)
        print(f"  identical to hw3.py: {torch.equal(ref, a)}")
        model._cache = None
        run(prompt, f"{name}, Colab TODO without max_new_tokens", do_sample=False)
    print("generation_config.max_length", model.generation_config.max_length, "max_new_tokens", model.generation_config.max_new_tokens)

    for label, chat in [("system role", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": question}]),
                        ("two user turns", [{"role": "user", "content": "Hi"}, {"role": "user", "content": question}])]:
        try:
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            print(f"{label}: no error")
        except Exception as e:
            print(f"{label}: {type(e).__module__}.{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# ch00b (KV cache chapter): speed, prefill vs decode, cache layout, masking, numerics
def kvcache_facts(tokenizer, model):
    from transformers import DynamicCache, HybridCache

    section("kvcache")
    cfg = model.config
    gib = 2**30
    question = "Please tell me about the key differences between supervised learning and unsupervised learning. Answer in 200 words."
    prompt_t = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt_t, return_tensors="pt").input_ids.to(DEVICE)  # Q1 with template, as hw3.py (32 tokens)
    T = ids.shape[1]

    def sync_time(f):
        torch.cuda.synchronize()
        t0 = time.time()
        r = f()
        torch.cuda.synchronize()
        return r, time.time() - t0

    def fresh(n):
        return HybridCache(config=cfg, max_batch_size=1, max_cache_len=n, device=DEVICE, dtype=hw3.DTYPE)

    # 1. generate with and without cache (fresh cache each time; warm-up first)
    model.generate(ids, max_new_tokens=8, do_sample=False)
    model._cache = None
    out_c, t_c = sync_time(lambda: model.generate(ids, max_new_tokens=512, do_sample=False, output_logits=True, return_dict_in_generate=True))
    model._cache = None
    out_n, t_n = sync_time(lambda: model.generate(ids, max_new_tokens=512, do_sample=False, use_cache=False))
    gen_c, gen_n = out_c.sequences[0, T:], out_n[0, T:]
    k = next((i for i in range(min(len(gen_c), len(gen_n))) if gen_c[i] != gen_n[i]), None)
    print(f"[1] Q1 prompt ({T} tokens), greedy, max_new_tokens=512:")
    print(f"  use_cache=True : {len(gen_c)} new tokens in {t_c:.2f} s ({len(gen_c) / t_c:.1f} tok/s)")
    print(f"  use_cache=False: {len(gen_n)} new tokens in {t_n:.2f} s ({len(gen_n) / t_n:.1f} tok/s)")
    print(f"  identical tokens: {torch.equal(gen_c, gen_n)}; first differing new token: {k}"
          + ("" if k is None else f" ({tokenizer.convert_ids_to_tokens(int(gen_c[k]))!r} vs {tokenizer.convert_ids_to_tokens(int(gen_n[k]))!r})"))
    N = len(gen_c)
    print(f"  tokens pushed through the model: with cache {T} + {N - 1} = {T + N - 1}; "
          f"without cache sum_(i=0..{N - 1}) ({T}+i) = {sum(T + i for i in range(N))}")

    # 2. the same sequence in one forward pass vs the step-by-step logits from generate
    seq = out_c.sequences[:, :-1]
    with torch.no_grad():
        full = model(seq, use_cache=False).logits[0, T - 1:].float()
    steps = torch.stack(out_c.logits)[:, 0].float()
    d = (full - steps).abs().amax(-1)
    agree = (full.argmax(-1) == gen_c).sum().item()
    bad = [i for i in range(N) if full[i].argmax() != gen_c[i]]
    for i in bad:
        top = torch.topk(steps[i], 2)
        print(f"  step {i}: generate picked {tokenizer.convert_ids_to_tokens(int(gen_c[i]))!r}, one-pass argmax "
              f"{tokenizer.convert_ids_to_tokens(int(full[i].argmax()))!r}; generate's top-2 logits "
              f"{[(tokenizer.convert_ids_to_tokens(int(j)), round(float(v), 4)) for v, j in zip(top.values, top.indices)]}")
    print(f"[2] one forward over all {seq.shape[1]} tokens vs generate's per-step logits ({N} steps): "
          f"max |diff| {d.max():.4f}, mean of per-step max {d.mean():.4f}, steps with diff 0: {(d == 0).sum().item()}; "
          f"argmax agrees with the generated token at {agree}/{N} steps")

    # 3. prefill vs decode timing with an explicit HybridCache; no-cache forward at growing lengths
    n_dec = 64
    pos = torch.arange(T, device=DEVICE)
    with torch.no_grad():
        cache = fresh(T + n_dec)
        model(ids, past_key_values=cache, cache_position=pos, use_cache=True)  # warm-up
        cache = fresh(T + n_dec)
        o, t_pre = sync_time(lambda: model(ids, past_key_values=cache, cache_position=pos, use_cache=True))
        nxt = o.logits[:, -1:].argmax(-1)
        dec = []
        for i in range(n_dec):
            o, t = sync_time(lambda: model(nxt, past_key_values=cache, cache_position=torch.tensor([T + i], device=DEVICE), use_cache=True))
            nxt = o.logits[:, -1:].argmax(-1)
            dec.append(t)
    print(f"[3] prefill {T} tokens: {t_pre * 1000:.1f} ms; decode 1 token with cache: median {np.median(dec) * 1000:.1f} ms "
          f"(min {min(dec) * 1000:.1f}, max {max(dec) * 1000:.1f}) over {n_dec} steps")
    with torch.no_grad():
        for L in [32, 64, 128, 256, T + N - 1]:
            x = out_c.sequences[:, :L]
            model(x, use_cache=False)
            ts = [sync_time(lambda: model(x, use_cache=False))[1] for _ in range(5)]
            print(f"    no-cache forward of {x.shape[1]:4d} tokens (= one generation step without cache): median {np.median(ts) * 1000:.1f} ms")

    # 3b. long sequences: one decode step with a nearly full cache vs one no-cache forward of the same length
    filler = out_c.sequences[0].repeat(20)[None]  # real tokens, repeated, only used for timing
    with torch.no_grad():
        for L in [512, 1024, 2048, 4096]:
            x = filler[:, :L]
            model(x, use_cache=False)
            ts = [sync_time(lambda: model(x, use_cache=False))[1] for _ in range(3)]
            cache = fresh(L + 1)
            model(x, past_key_values=cache, cache_position=torch.arange(L, device=DEVICE), use_cache=True)
            nx = x[:, -1:]
            model(nx, past_key_values=cache, cache_position=torch.tensor([L], device=DEVICE), use_cache=True)
            td = [sync_time(lambda: model(nx, past_key_values=cache, cache_position=torch.tensor([L], device=DEVICE), use_cache=True))[1]
                  for _ in range(5)]
            print(f"    length {L:4d}: no-cache forward median {np.median(ts) * 1000:.1f} ms; one decode step with {L} cached tokens median "
                  f"{np.median(td) * 1000:.1f} ms; peak allocated {torch.cuda.max_memory_allocated() / gib:.2f} GiB so far")
            del cache
    # 3c. force 1024 new tokens with and without cache
    for uc in [True, False]:
        model._cache = None
        torch.cuda.reset_peak_memory_stats()
        o, t = sync_time(lambda: model.generate(ids, min_new_tokens=1024, max_new_tokens=1024, do_sample=False, use_cache=uc))
        print(f"    generate exactly {o.shape[1] - T} new tokens, use_cache={uc}: {t:.1f} s; peak allocated {torch.cuda.max_memory_allocated() / gib:.2f} GiB")

    # 4. what the cache looks like
    model._cache = None
    model.generate(ids, max_new_tokens=512, do_sample=False)
    c = model._cache
    print(f"[4] model._cache after generate: {type(c).__name__}, max_cache_len {c.max_cache_len}, layers {len(c.key_cache)}, "
          f"is_sliding[:4] {c.is_sliding[:4].tolist()}")
    for li in [0, 1]:
        kc = c.key_cache[li]
        filled = int(kc[0, 0].any(-1).sum())
        print(f"  layer {li} ({'sliding' if c.is_sliding[li] else 'global'}): key {tuple(kc.shape)} value {tuple(c.value_cache[li].shape)} {kc.dtype}; "
              f"slots holding non-zero keys: {filled}; slots after them all zero: {bool((kc[0, :, filled:] == 0).all())}")
    nbytes = sum(t.numel() * t.element_size() for t in c.key_cache + c.value_cache)
    print(f"  total {nbytes:,} bytes = {nbytes / gib:.4f} GiB; per slot {nbytes // c.max_cache_len:,} bytes")
    big = fresh(8192)
    nb = sum(t.numel() * t.element_size() for t in big.key_cache + big.value_cache)
    print(f"  HybridCache(max_cache_len=8192): layer 0 {tuple(big.key_cache[0].shape)}, layer 1 {tuple(big.key_cache[1].shape)}; "
          f"total {nb:,} bytes = {nb / gib:.3f} GiB")
    del big
    try:
        dc = DynamicCache()
        with torch.no_grad():
            o = model(ids, past_key_values=dc, use_cache=True)
        pk = o.past_key_values
        print(f"  forward with past_key_values=DynamicCache(): returned {type(pk).__name__}; "
              f"layer 0 key {tuple(pk.key_cache[0].shape) if hasattr(pk, 'key_cache') else None}")
    except Exception as e:
        print(f"  forward with past_key_values=DynamicCache(): {type(e).__name__}: {e}")
    with torch.no_grad():
        o = model(ids, use_cache=True)
    print(f"  forward(use_cache=True) without a cache argument returns {type(o.past_key_values).__name__} "
          f"with max_cache_len {o.past_key_values.max_cache_len} (= prompt length)")

    # 5. cached k/v of a prefix do not depend on later tokens
    pc = tokenizer("Time travel will become a reality as technology continues to advance.", return_tensors="pt").input_ids.to(DEVICE)
    ca, cb = fresh(32), fresh(32)
    with torch.no_grad():
        model(pc, past_key_values=ca, cache_position=torch.arange(pc.shape[1], device=DEVICE), use_cache=True)
        model(pc[:, :6], past_key_values=cb, cache_position=torch.arange(6, device=DEVICE), use_cache=True)
    dk = max(float((ca.key_cache[i][:, :, :6].float() - cb.key_cache[i][:, :, :6].float()).abs().max()) for i in range(26))
    dv = max(float((ca.value_cache[i][:, :, :6].float() - cb.value_cache[i][:, :, :6].float()).abs().max()) for i in range(26))
    print(f"[5] prompt c ({pc.shape[1]} tokens) vs its first 6 tokens: max |diff| of cached keys at positions 0-5 over 26 layers {dk:.4f}, values {dv:.4f}")

    # 6. unwritten slots get exactly zero attention
    cache = fresh(T + 512)
    with torch.no_grad():
        o = model(ids, past_key_values=cache, cache_position=torch.arange(T, device=DEVICE), use_cache=True, output_attentions=True)
        a = o.attentions
        o2 = model(o.logits[:, -1:].argmax(-1), past_key_values=cache, cache_position=torch.tensor([T], device=DEVICE),
                   use_cache=True, output_attentions=True)
    print(f"[6] prefill attentions: {len(a)} x {tuple(a[0].shape)}; weight on slots >= {T} summed over all layers/heads/rows: "
          f"{sum(float(x[..., T:].float().sum()) for x in a)}; rows sum to 1: max |sum-1| {max(float((x.float().sum(-1) - 1).abs().max()) for x in a):.2e}")
    a2 = o2.attentions
    print(f"    decode step 1: {tuple(a2[0].shape)}; weight on slots >= {T + 1}: {sum(float(x[..., T + 1:].float().sum()) for x in a2)}; "
          f"min weight on slot {T} (the new token itself) over layers/heads {min(float(x[..., T].min()) for x in a2):.4f}")
    print(f"    fp16 mask value finfo(float16).min = {torch.finfo(torch.float16).min}")

    # 7. where does a different cache length first change the numbers? (Q1 without template, decode step 2)
    q = tokenizer(question, return_tensors="pt").input_ids.to(DEVICE)
    feats = {}
    for n in [535, 544]:
        cache = fresh(n)
        acts = []
        hooks = [model.model.layers[li].self_attn.register_forward_hook(
            lambda m, i, out, li=li: acts.append((li, out[0].detach().float().clone(), None if out[1] is None else out[1].detach().float().clone())))
            for li in range(26)]
        with torch.no_grad():
            o = model(q, past_key_values=cache, cache_position=torch.arange(q.shape[1], device=DEVICE), use_cache=True)
            t = o.logits[:, -1:].argmax(-1)
            for i in range(3):
                acts.clear()
                o = model(t, past_key_values=cache, cache_position=torch.tensor([q.shape[1] + i], device=DEVICE), use_cache=True,
                          output_attentions=True)
                t = o.logits[:, -1:].argmax(-1)
        for h in hooks:
            h.remove()
        feats[n] = acts[:]
    L = q.shape[1] + 3
    first = None
    for (li, oa, wa), (_, ob, wb) in zip(feats[535], feats[544]):
        dw = float((wa[..., :L] - wb[..., :L]).abs().max())
        do = float((oa - ob).abs().max())
        if first is None and (dw > 0 or do > 0):
            first = (li, dw, do)
    print(f"[7] Q1 without template, 3rd decode step, cache 535 vs 544: first layer whose self_attn differs: "
          f"{None if first is None else first[0]} (max |diff| attention weights on written slots "
          f"{first[1] if first else 0}, attention output {first[2] if first else 0})")
    model._cache = None


# ---------------------------------------------------------------------------
# ch02 review: prompt segments, forward-only cache, single <bos>, the 'Ver' token
def review_ch02_facts(tokenizer, model):
    section("review_ch02")

    def prompts(bos):
        hist, rows = [], []
        for u in hw3.Q2_TURNS:
            hist.append({"role": "user", "content": u})
            p = tokenizer.apply_chat_template(hist, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(p, return_tensors="pt", add_special_tokens=(bos == 2)).to(DEVICE)
            with torch.no_grad():
                fwd = model(**inputs)
            probs = torch.softmax(fwd.logits[:, -1, :].float(), -1)
            tp, ti = torch.topk(probs, 10)
            model._cache = None  # fresh generate cache, as in a standalone --q 2
            out = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            new = out[0][inputs.input_ids.shape[1]:]
            resp = tokenizer.decode(new, skip_special_tokens=True)
            rows.append((inputs, fwd, list(zip(ti[0].tolist(), tp[0].tolist())), new, resp))
            hist.append({"role": "assistant", "content": resp})
        return rows

    dbl = prompts(2)
    ids = [r[0].input_ids[0].tolist() for r in dbl]
    for r in range(1, 3):
        prefix = ids[r][:len(ids[r - 1])] == ids[r - 1]
        added = tokenizer.convert_ids_to_tokens(ids[r][len(ids[r - 1]):])
        print(f"round {r + 1}: previous prompt is a token prefix: {prefix}; {len(added)} added tokens: {added}")
    pkv = dbl[0][1].past_key_values
    print(f"forward without cache returns {type(pkv).__name__}, max_cache_len {getattr(pkv, 'max_cache_len', None)}, prompt {ids[0].__len__()}")

    one = prompts(1)
    for r, (d, o) in enumerate(zip(dbl, one), 1):
        print(f"--- round {r}: single <bos> prompt tokens {o[0].input_ids.shape[1]} (double {d[0].input_ids.shape[1]})")
        for (di, dp), (oi, op) in zip(d[2], o[2]):
            print(f"  double {tokenizer.decode([di])!r:>12} {dp:.4f}   single {tokenizer.decode([oi])!r:>12} {op:.4f}")
        print(f"  generated: double {tokenizer.convert_ids_to_tokens(d[3].tolist())} {d[4]!r}; single {tokenizer.convert_ids_to_tokens(o[3].tolist())} {o[4]!r}")

    ver = tokenizer.encode("Ver", add_special_tokens=False)
    print(f"'Ver' ids {ver}; Vermilion -> {tokenizer.tokenize('Vermilion')}")
    x = torch.cat([dbl[2][0].input_ids, torch.tensor([ver], device=DEVICE)], 1)
    with torch.no_grad():
        probs = torch.softmax(model(x).logits[0, -1].float(), -1)
    tp, ti = torch.topk(probs, 5)
    print("after round-3 prompt + 'Ver', top-5 next:", [(tokenizer.decode([i]), round(p, 4)) for i, p in zip(ti.tolist(), tp.tolist())])
    model._cache = None
    out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=8, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    print("greedy continuation:", tokenizer.convert_ids_to_tokens(out[0, x.shape[1]:].tolist()))


# ---------------------------------------------------------------------------
# ch03 prep: special tokens, round trips, byte fallback, spaces and case
def pre_ch03_facts(tokenizer, model):
    section("pre_ch03")
    sent = hw3.build_parser().get_default("sentence") if hasattr(hw3, "build_parser") else None
    sent = sent or "I love taking a Machine Learning course by Professor Hung-yi Lee, What about you?"
    ids = tokenizer.encode(sent, add_special_tokens=False)
    ids_bos = tokenizer.encode(sent)
    print(f"default sentence: {len(sent)} chars, {len(ids)} tokens; with add_special_tokens=True {len(ids_bos)} tokens, first {ids_bos[:2]} {tokenizer.convert_ids_to_tokens(ids_bos[:2])}")
    print(f"tokenize() == convert_ids_to_tokens(encode()): {tokenizer.tokenize(sent) == tokenizer.convert_ids_to_tokens(ids)}")
    print(f"decode(encode(s)) == s: {tokenizer.decode(ids) == sent}; convert_tokens_to_string == s: {tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)) == sent}")
    for i in [692, 4747, 23533, 235336, 235248, 18809, 42599]:
        print(f"  id {i}: token {tokenizer.convert_ids_to_tokens(i)!r}, decode {tokenizer.decode([i])!r}")
    for s in ["Machine", "machine", " machine", " Machine", "I love", " I love", "a  b", "a   b", "Hello\nworld", "\tx",
              "🙂", "👍🏽", "龘", "é", "naïve", "ChatGPT", "transformers", "Transformers", "unbelievable", "Lee,", "Lee ,"]:
        e = tokenizer.encode(s, add_special_tokens=False)
        print(f"  {s!r:>16} -> {list(zip(tokenizer.convert_ids_to_tokens(e), e))}; roundtrip {tokenizer.decode(e) == s}")
    vocab = tokenizer.get_vocab()
    print(f"vocab {len(vocab)}; tokens starting with '▁': {sum(t.startswith('▁') for t in vocab)}; byte tokens <0x..>: {sum(t.startswith('<0x') and t.endswith('>') and len(t) == 6 for t in vocab)}")
    print(f"single-char CJK tokens (U+4E00..U+9FFF): {sum(len(t) == 1 and 0x4E00 <= ord(t) <= 0x9FFF for t in vocab)}")
    print(f"unk id {tokenizer.unk_token_id}; any <unk> in the strings above: {any(3 in tokenizer.encode(s, add_special_tokens=False) for s in ['🙂', '龘', '𠀀'])}")


# ---------------------------------------------------------------------------
# ch03 review: convert_tokens_to_ids, leading space, digits in the vocabulary, embedding rows
def review_ch03_facts(tokenizer, model):
    import re

    section("review_ch03")
    for t in ["_love", "▁love", "love", "_Machine"]:
        print(f"convert_tokens_to_ids({t!r}) -> {tokenizer.convert_tokens_to_ids(t)}")
    for s in [" you and you", "you and you"]:
        e = tokenizer.encode(s, add_special_tokens=False)
        print(f"{s!r} -> {list(zip(tokenizer.convert_ids_to_tokens(e), e))}")
    e = tokenizer.encode("I love you")
    print(f"'I love you' with <bos>: {len(e)} tokens {tokenizer.convert_ids_to_tokens(e)}")
    vocab = tokenizer.get_vocab()
    digit = [t for t in vocab if re.search(r"[0-9]", t)]
    print(f"tokens containing an ASCII digit: {len(digit)}; examples {sorted(digit, key=len)[:12]}")
    print(f"'▁' + digit tokens: {[t for t in vocab if re.fullmatch(r'▁[0-9]+', t)]}; multi-digit tokens: {len([t for t in vocab if re.fullmatch(r'▁?[0-9]{2,}', t)])}")
    W = model.get_input_embeddings().weight.float()
    cos = lambda a, b: torch.nn.functional.cosine_similarity(W[a], W[b], dim=0).item()
    pairs = [("you", "▁you"), ("Machine", "▁Machine"), ("machine", "▁machine"), ("Machine", "machine"), ("Orange", "▁Orange"), ("you", "▁Machine")]
    for a, b in pairs:
        print(f"embedding cosine {a!r} vs {b!r}: {cos(tokenizer.convert_tokens_to_ids(a), tokenizer.convert_tokens_to_ids(b)):.4f}")
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, W.shape[0], (2, 10000), generator=g).to(W.device)
    rc = torch.nn.functional.cosine_similarity(W[idx[0]], W[idx[1]], dim=1)
    print(f"10000 random id pairs (seed 0): cosine mean {rc.mean():.4f}, std {rc.std():.4f}, 99th pct {rc.quantile(0.99):.4f}")


# ---------------------------------------------------------------------------
# ch04 prep: what top-k / top-p keep on real distributions, and one sentence step by step
def pre_ch04_facts(tokenizer, model):
    from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper

    section("pre_ch04")
    warpers = {"top_k=2": [TopKLogitsWarper(2)], "top_k=200": [TopKLogitsWarper(200)],
               "top_p=0.6 (+ default top_k=50)": [TopKLogitsWarper(50), TopPLogitsWarper(0.6)],
               "top_p=0.999 (+ default top_k=50)": [TopKLogitsWarper(50), TopPLogitsWarper(0.999)],
               "top_p=0.999, top_k=0": [TopPLogitsWarper(0.999)], "top_k=1": [TopKLogitsWarper(1)],
               "top_p=0 (+ default top_k=50)": [TopKLogitsWarper(50), TopPLogitsWarper(0.0)]}

    def show(label, ids):
        with torch.no_grad():
            logits = model(ids).logits[:, -1, :].float()  # generate() also filters fp32 logits (utils.py:3267)
        p = torch.softmax(logits, -1)
        tp, ti = torch.topk(p, 10)
        print(f"--- {label}: raw top-10 " + ", ".join(f"{tokenizer.decode([i])!r} {v:.4f}" for i, v in zip(ti[0].tolist(), tp[0].tolist())))
        for name, ws in warpers.items():
            sc = LogitsProcessorList(ws)(ids, logits.clone())
            keep = torch.isfinite(sc[0])
            q = torch.softmax(sc, -1)[0]
            order = torch.argsort(q, descending=True)[: min(int(keep.sum()), 5)]
            print(f"  {name:34s} keeps {int(keep.sum()):3d} tokens, raw mass {p[0, keep].sum():.4f}; renormalized: "
                  + ", ".join(f"{tokenizer.decode([i])!r} {q[i]:.4f}" for i in order.tolist()))

    hist = [{"role": "user", "content": hw3.Q2_TURNS[0]}]
    q2p = tokenizer.apply_chat_template(hist, tokenize=False, add_generation_prompt=True)
    show("Q2 round 1 (as hw3.py, double <bos>)", tokenizer(q2p, return_tensors="pt").input_ids.to(DEVICE))
    prompt = "Generate a paraphrase of the sentence 'Professor Hung-yi Lee is one of the best teachers in the domain of machine learning'. Just response with one sentence."
    input_ids = tokenizer(prompt, return_tensors="pt")
    show("Q4 prompt, first new token", input_ids.input_ids.to(DEVICE))

    params = {"do_sample": True, "max_length": 30 + len(input_ids.input_ids[0]), "pad_token_id": tokenizer.pad_token_id,
              "eos_token_id": tokenizer.eos_token_id, "bos_token_id": tokenizer.bos_token_id,
              "attention_mask": input_ids.attention_mask.to(DEVICE), "use_cache": True,
              "return_dict_in_generate": True, "output_scores": True, "output_logits": True}

    def steps(label, o):
        new = o.sequences[0, len(input_ids.input_ids[0]):]
        print(f"--- {label}: {len(new)} new tokens; decoded {tokenizer.decode(new, skip_special_tokens=True)!r}")
        for i, (t, lg, sc) in enumerate(zip(new.tolist(), o.logits, o.scores)):
            p = torch.softmax(lg[0].float(), -1)
            q = torch.softmax(sc[0], -1)
            top = int(p.argmax())
            print(f"  step {i:2d}: picked {tokenizer.convert_ids_to_tokens(t)!r:16} raw p {p[t]:.4f} renorm {q[t]:.4f} | kept {int(torch.isfinite(sc[0]).sum()):2d} | raw top-1 {tokenizer.convert_ids_to_tokens(top)!r} {p[top]:.4f}")

    # reproduce hw3.py --q 4 --seed 0: 20 top-k samples, then top-p on the same random stream
    model._cache = None
    torch.manual_seed(0)
    outs = [model.generate(input_ids=input_ids.input_ids.to(DEVICE), top_k=2, **params) for _ in range(20)]
    steps("top_k=2, sentence 0 (seed 0)", outs[0])
    steps("top_p=0.6, sentence 0 (after the 20 top-k samples)", model.generate(input_ids=input_ids.input_ids.to(DEVICE), top_p=0.6, **params))
    last = [tokenizer.convert_ids_to_tokens(int(o.sequences[0, -1])) for o in outs]
    lens = [o.sequences.shape[1] - len(input_ids.input_ids[0]) for o in outs]
    print(f"top_k=2 20 sentences: new-token counts {lens}; last token {last}")


# ---------------------------------------------------------------------------
# ch04 review: hw3.py-order runs with variations (eos [1, 107], top_k=0), first tokens, replace() effects
def review_ch04_facts(tokenizer, model):
    from collections import Counter

    section("review_ch04")
    prompt = "Generate a paraphrase of the sentence 'Professor Hung-yi Lee is one of the best teachers in the domain of machine learning'. Just response with one sentence."
    input_ids = tokenizer(prompt, return_tensors="pt")
    n0 = len(input_ids.input_ids[0])
    base = {"do_sample": True, "max_length": 30 + n0, "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id, "bos_token_id": tokenizer.bos_token_id,
            "attention_mask": input_ids.attention_mask.to(DEVICE), "use_cache": True,
            "return_dict_in_generate": True, "output_scores": True}

    def run(k, p, extra_p=None, eos=None):
        """Same order as hw3.py --q 4 --seed 0 (standalone): 20 top-k, then 20 top-p, one random stream."""
        params = dict(base)
        if eos is not None:
            params["eos_token_id"] = eos
        model._cache = None
        torch.manual_seed(0)
        res = {}
        for method in ["top-k", "top-p"]:
            kw = {"top_k": k} if method == "top-k" else {"top_p": p, **(extra_p or {})}
            res[method] = [model.generate(input_ids=input_ids.input_ids.to(DEVICE), **kw, **params) for _ in range(20)]
        return res

    def sent(o):
        t = tokenizer.decode(o.sequences[0, n0:], skip_special_tokens=True)
        return t.replace(" ,", ",").replace(" 's", "'s").replace(" .", ".").strip()

    def summary(label, outs):
        rows = []
        for o in outs:
            new = o.sequences[0, n0:].tolist()
            rows.append((len(new), tokenizer.convert_ids_to_tokens(new[-1]), new.index(107) if 107 in new else None))
        print(f"--- {label}: new-token counts {[r[0] for r in rows]}; reached 30: {sum(r[0] == 30 for r in rows)}")
        print(f"    last token {Counter(r[1] for r in rows)}; first <end_of_turn> step (None = never) {[r[2] for r in rows]}")
        raw = [tokenizer.decode(o.sequences[0, n0:], skip_special_tokens=True) for o in outs]
        print(f"    first new token {Counter(tokenizer.convert_ids_to_tokens(int(o.sequences[0, n0])) for o in outs)}")
        print(f"    raw text containing ' ,' / \" 's\" / ' .': {sum(' ,' in r for r in raw)} / {sum(chr(32) + chr(39) + 's' in r for r in raw)} / {sum(' .' in r for r in raw)}")

    for k, p in [(2, 0.6), (200, 0.999)]:
        ref = run(k, p)
        summary(f"hw3.py order k={k}", ref["top-k"])
        summary(f"hw3.py order p={p} (+ default top_k=50)", ref["top-p"])
        for o in ref["top-k"] + ref["top-p"]:
            new = o.sequences[0, n0:].tolist()
            if len(new) == 30:
                print(f"    30-token sentence: {sent(o)!r}")
        print(f"    sentence 0: k {sent(ref['top-k'][0])!r}; p {sent(ref['top-p'][0])!r}")
        e = run(k, p, eos=[1, 107])
        summary(f"eos_token_id=[1, 107], k={k}", e["top-k"])
        summary(f"eos_token_id=[1, 107], p={p}", e["top-p"])
        z = run(k, p, extra_p={"top_k": 0})
        same_k = sum(sent(a) == sent(b) for a, b in zip(ref["top-k"], z["top-k"]))
        same_p = sum(sent(a) == sent(b) for a, b in zip(ref["top-p"], z["top-p"]))
        print(f"--- top_p={p} with top_k=0 at hw3.py:231: top-k group identical {same_k}/20, top-p group identical {same_p}/20; "
              f"distinct top-p sentences {len(set(map(sent, ref['top-p'])))} -> {len(set(map(sent, z['top-p'])))}")
        for i, (a, b) in enumerate(zip(ref["top-p"], z["top-p"])):
            if sent(a) != sent(b):
                print(f"    {i}: {sent(a)!r} -> {sent(b)!r}")

    # top_k=200 vs top_p=0.999, top_k=0 from the same seed: per-step candidate sets
    outs = {}
    for name, kw in [("top_k=200", {"top_k": 200}), ("top_p=0.999,top_k=0", {"top_p": 0.999, "top_k": 0})]:
        model._cache = None
        torch.manual_seed(0)
        outs[name] = [model.generate(input_ids=input_ids.input_ids.to(DEVICE), **kw, **base) for _ in range(20)]
    a, b = outs.values()
    same = sum(torch.equal(x.sequences, y.sequences) for x, y in zip(a, b))
    diffs, mass, kept = [], [], []
    for x, y in zip(a, b):
        if not torch.equal(x.sequences, y.sequences):
            continue
        for sx, sy in zip(x.scores, y.scores):
            px, py = torch.softmax(sx[0], -1), torch.softmax(sy[0], -1)
            diffs.append(float((px - py).abs().max()))
            only = torch.isfinite(sy[0]) & ~torch.isfinite(sx[0])
            mass.append(float(py[only].sum()))
            kept.append((int(torch.isfinite(sx[0]).sum()), int(torch.isfinite(sy[0]).sum())))
    print(f"--- top_k=200 vs top_p=0.999,top_k=0 (each from seed 0): identical sentences {same}/20; {len(diffs)} steps compared")
    print(f"    kept per step: top_k=200 median {np.median([k[0] for k in kept]):.0f}; top_p=0.999,top_k=0 median {np.median([k[1] for k in kept]):.0f} min {min(k[1] for k in kept)} max {max(k[1] for k in kept)}")
    print(f"    steps where top_p keeps fewer than 200: {sum(k[1] < 200 for k in kept)}")
    print(f"    renormalized prob max |diff| over steps: max {max(diffs):.4f} median {np.median(diffs):.5f}; prob mass on tokens only top-p keeps: max {max(mass):.4f} median {np.median(mass):.5f}")

    # what the '-' at step 9 of k=2 sentence 0 would have become
    ids = tokenizer(prompt + " \n\nHe is highly regarded as a top", return_tensors="pt", add_special_tokens=True).input_ids.to(DEVICE)
    model._cache = None
    o = model.generate(torch.cat([ids, torch.tensor([[tokenizer.convert_tokens_to_ids("-")]], device=DEVICE)], 1),
                       attention_mask=None, max_new_tokens=6, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    print(f"--- after '... as a top' + '-', greedy: {tokenizer.convert_ids_to_tokens(o[0, ids.shape[1]:].tolist())}")


# ---------------------------------------------------------------------------
# ch06 prep: left padding vs single sentences, pad positions, word-position vectors, t-SNE settings
def pre_ch06_facts(tokenizer, model):
    import sklearn
    from sklearn.manifold import TSNE

    section("pre_ch06")
    sentences = ["I ate a fresh apple.", "Apple released the new iPhone.", "I peeled an orange and ate it.",
                 "The Orange network has great coverage.", "Microsoft announced a new update.", "Banana is my favorite fruit."]
    labels = ["Apple(f)", "Apple(c)", "Orange(f)", "Orange(t)", "MS(c)", "Banana(f)"]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    H = out.hidden_states[-1].float()
    mask = inputs.attention_mask.bool()
    print("attention_mask rows:", inputs.attention_mask.tolist())
    cos = torch.nn.functional.cosine_similarity
    for i, s in enumerate(sentences):
        single = tokenizer(s, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            h1 = model(**single, output_hidden_states=True).hidden_states[-1][0].float()
        hb = H[i][mask[i]]
        d = (hb - h1).abs().max().item()
        print(f"{labels[i]:10s} pads {int((~mask[i]).sum())}: batch vs alone, real-token hidden max|diff| {d:.4f}, "
              f"min per-token cosine {cos(hb, h1, dim=-1).min():.6f}; mean-pooled (pad incl.) vs alone cosine {cos(H[i].mean(0), h1.mean(0), dim=0):.6f}")
    pads = H[~mask]
    print(f"pad hidden states: {pads.shape[0]} vectors, norms {[round(x, 2) for x in pads.norm(dim=-1).tolist()]}, "
          f"max|diff| between any pad vector and the first {(pads - pads[0]).abs().max():.4f}")
    with torch.no_grad():
        e = model.get_input_embeddings()(torch.tensor([[tokenizer.pad_token_id]], device=DEVICE)).float()[0, 0]
    print(f"pad embedding row norm (x sqrt(2304) scaling not applied) {e.norm():.4f}")

    # last-layer vector at the key word's own position
    words = ["▁apple", "Apple", "▁orange", "▁Orange", "Microsoft", "Banana"]
    vecs = []
    for i, w in enumerate(words):
        toks = tokenizer.convert_ids_to_tokens(inputs.input_ids[i])
        j = toks.index(w)
        vecs.append(H[i, j])
        print(f"{labels[i]:10s} key token {w!r} at position {j}")
    V = torch.stack(vecs)
    M = cos(V[:, None], V[None], dim=-1)
    print("cosine of key-word vectors (rows/cols " + ", ".join(labels) + "):")
    for i in range(6):
        print("  " + " ".join(f"{M[i, j]:.3f}" for j in range(6)))

    print(f"sklearn {sklearn.__version__}")
    emb = H.mean(dim=1).cpu().numpy()
    t = TSNE(n_components=2, perplexity=2, random_state=42)
    t.fit_transform(emb)
    print(f"TSNE(perplexity=2, random_state=42): init={t.init!r} learning_rate={t.learning_rate!r} -> {t.learning_rate_:.3f}, "
          f"max_iter={getattr(t, 'max_iter', None)}, n_iter_={t.n_iter_}, kl_divergence_={t.kl_divergence_:.4f}, metric={t.metric!r}")
    for perp in [5, 6, 10]:
        try:
            TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(emb)
            print(f"perplexity={perp}: ok")
        except Exception as ex:
            print(f"perplexity={perp}: {type(ex).__name__}: {ex}")


# ---------------------------------------------------------------------------
# ch07 prep: hw3.q6's own matrix vs one full forward, vs generate(); the tick labels it draws
def pre_ch07_facts(tokenizer, model):
    import argparse

    section("pre_ch07")
    got = {}
    orig = hw3.plot_attention
    hw3.plot_attention = lambda m, toks, title, fn: got.update(m=m, toks=toks, fn=fn)
    try:
        model._cache = None
        hw3.q6(tokenizer, model, argparse.Namespace(layer_idx=10, head_idx=7))
    finally:
        hw3.plot_attention = orig
    A, labels = got["m"], got["toks"]
    print(f"matrix from hw3.q6: shape {A.shape}; {len(labels)} tick labels (repr): {labels}")
    blank = [i for i, t in enumerate(labels) if not t.strip() or t in ("\n", "\n\n")]
    print(f"labels that are only whitespace/newlines (drawn blank or as a bare line): positions {blank} -> {[labels[i] for i in blank]}")

    ids = tokenizer("Google ", return_tensors="pt").input_ids.to(DEVICE)
    model._cache = None
    gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    new = gen[0, ids.shape[1]:].tolist()
    print(f"generate() greedy 20 tokens: {new}")
    seq = gen[:, :ids.shape[1] + 19]  # the 22 tokens that the q6 loop actually fed to the model
    print(f"rows of the q6 matrix = these 22 tokens: {tokenizer.convert_ids_to_tokens(seq[0].tolist())}")
    with torch.no_grad():
        out = model(seq, output_attentions=True)
    F = out.attentions[10][0, 7].float().cpu().numpy()
    print(f"one full forward over the same 22 tokens, layer 10 head 7: shape {F.shape}; max |q6 loop - full forward| {np.abs(A - F).max():.5f}; "
          f"row argmax identical: {bool((A.argmax(1) == F.argmax(1)).all())}")
    print(f"column 0 (<bos>) mean over rows 1..21: loop {A[1:, 0].mean():.4f}, full forward {F[1:, 0].mean():.4f}")

    # replica of hw3.py:313-358 that keeps every layer's attention (same calls, same cache size)
    from transformers import HybridCache

    def loop(extra_slots=0):
        inp = tokenizer("Google ", return_tensors="pt")
        nxt, am = inp.input_ids.to(DEVICE), inp.attention_mask.to(DEVICE)
        cp = torch.arange(am.shape[1], device=DEVICE)
        cache = HybridCache(config=model.config, max_batch_size=1, max_cache_len=20 + nxt.size(1) - 1 + extra_slots, device=DEVICE, dtype=hw3.DTYPE)
        rows, toks = [], []
        for _ in range(20):
            with torch.no_grad():
                o = model(nxt, attention_mask=am, cache_position=cp, use_cache=True, past_key_values=cache, output_attentions=True)
            rows.append(torch.stack([a[0].float() for a in o.attentions]))  # (26, 8, q, cache_len)
            nxt = o.logits[:, -1, :].argmax(dim=-1)
            toks.append(nxt.item())
            am = torch.cat([am, torch.ones(1, 1, device=DEVICE)], dim=-1)
            nxt = nxt.unsqueeze(0)
            cache = o.past_key_values
            cp = cp[-1:] + 1
        return torch.cat(rows, dim=2).cpu().numpy(), toks

    R, toks = loop()
    print(f"replica: tokens identical to generate(): {toks == new}; layer 10 head 7 identical to hw3.q6's matrix: {np.array_equal(R[10, 7], A)}")
    Fall = torch.stack([a[0].float() for a in out.attentions]).cpu().numpy()  # (26, 8, 22, 22)
    for name, X in [("q6 cache (22 slots, as hw3.py)", R), ("one extra slot (23)", loop(1)[0][..., :22])]:
        d = np.abs(X - Fall).max(axis=(1, 3))  # (26 layers, 22 rows)
        bad_rows = sorted({int(r) for l in range(26) for r in np.where(d[l] > 0.01)[0]})
        print(f"--- {name}: rows with max|diff| > 0.01 in any layer: {bad_rows}")
        print("    max|diff| per layer, rows 0-20 / row 21: " + ", ".join(f"L{l}:{d[l, :21].max():.3f}/{d[l, 21]:.3f}" for l in range(26)))
    r21 = R[10, 7, 21]
    print(f"q6 row 21, layer 10 head 7: {np.round(r21, 4).tolist()}")
    print(f"full forward row 21, layer 10 head 7: {np.round(Fall[10, 7, 21], 4).tolist()}")


# ---------------------------------------------------------------------------
# ch07 review: what the rolled sliding cache holds, L0 H0 last row, label fix, (22, 23) plot, bad indices
def review_ch07_facts(tokenizer, model):
    import matplotlib
    import seaborn as sns
    from transformers import HybridCache

    matplotlib.use("Agg")
    section("review_ch07")

    def loop(slots):
        inp = tokenizer("Google ", return_tensors="pt")
        nxt, am = inp.input_ids.to(DEVICE), inp.attention_mask.to(DEVICE)
        cp = torch.arange(am.shape[1], device=DEVICE)
        cache = HybridCache(config=model.config, max_batch_size=1, max_cache_len=slots, device=DEVICE, dtype=hw3.DTYPE)
        rows, gen = [], []
        for _ in range(20):
            with torch.no_grad():
                o = model(nxt, attention_mask=am, cache_position=cp, use_cache=True, past_key_values=cache, output_attentions=True)
            rows.append(torch.stack([a[0].float() for a in o.attentions]))
            nxt = o.logits[:, -1, :].argmax(dim=-1)
            gen.append(nxt.item())
            am = torch.cat([am, torch.ones(1, 1, device=DEVICE)], dim=-1)
            nxt = nxt.unsqueeze(0)
            cache = o.past_key_values
            cp = cp[-1:] + 1
        return torch.cat(rows, dim=2).cpu().numpy(), cache, gen, inp

    A22, c22, gen, inp = loop(22)
    A23, c23, _, _ = loop(23)
    for L in (0, 10):
        k22, k23 = c22.key_cache[L][0].float(), c23.key_cache[L][0].float()  # (4, slots, 256)
        same = [j for j in range(22) if torch.allclose(k22[:, j], k23[:, j + 1] if j < 22 else k22[:, j], atol=1e-2)]
        print(f"layer {L} (sliding) after the last step, 22-slot cache: slot 20 all zero: {bool(k22[:, 20].abs().max() == 0)}; "
              f"slot j holds what the 23-slot cache has at j+1, for j in {same[:3]}...{same[-3:]} ({len(same)} slots)")
        print(f"    slot 21 equals 23-slot slot 21 (the new token): {torch.allclose(k22[:, 21], k23[:, 21], atol=5e-2)}; "
              f"<bos> key (23-slot slot 0) found anywhere in 22-slot cache: {any(torch.allclose(k22[:, j], k23[:, 0], atol=1e-2) for j in range(22))}")
    k22, k23 = c22.key_cache[1][0].float(), c23.key_cache[1][0].float()
    print(f"layer 1 (global): slots 0-20 equal to the 23-slot cache: {all(torch.allclose(k22[:, j], k23[:, j], atol=1e-2) for j in range(21))}")
    print(f"layer 0 head 0 row 21: 22-slot argmax {int(A22[0, 0, 21].argmax())} (weight {A22[0, 0, 21].max():.4f}); "
          f"23-slot argmax {int(A23[0, 0, 21, :22].argmax())} (weight {A23[0, 0, 21, :22].max():.4f}); "
          f"22-slot weight on slot 20 (the empty one) {A22[0, 0, 21, 20]:.4f}")
    print(f"layer 10 head 7 row 21, weight on the empty slot 20: {A22[10, 7, 21, 20]:.4f}")

    # R7 fix from ch07 7.8, verbatim
    input_ids, generated_tokens = inp, gen
    fed_ids = input_ids.input_ids[0].tolist() + generated_tokens[:-1]   # 3 + 19 = 22 個
    tokens = tokenizer.convert_ids_to_tokens(fed_ids)
    print(f"ch07 R7 fix: {len(tokens)} labels {tokens[:4]} ... {tokens[-2:]}")

    # what happens if hw3.py:320 is changed to 23 slots: matrix (22, 23), labels still 22
    m = A23[10, 7]
    labels = tokenizer.tokenize(hw3.Q6_PROMPT if hasattr(hw3, "Q6_PROMPT") else "Google " + tokenizer.decode(gen, skip_special_tokens=True))
    print(f"23-slot matrix shape {m.shape}, hw3.py labels {len(labels)}")
    import matplotlib.pyplot as plt
    try:
        plt.figure()
        sns.heatmap(m, xticklabels=labels, yticklabels=labels, cmap="viridis", annot=False)
        ax = plt.gca()
        print(f"sns.heatmap with 23 columns and 22 labels: no error; x tick labels drawn {len(ax.get_xticklabels())}, y {len(ax.get_yticklabels())}")
    except Exception as e:
        print(f"sns.heatmap with 23 columns and 22 labels: {type(e).__name__}: {e}")
    plt.close("all")


# ---------------------------------------------------------------------------
SECTIONS = ["env", "model", "attn", "tok", "q1", "q2", "q4", "q5", "q6", "q7"]


def main():
    todo = sys.argv[1:] or SECTIONS
    if "env" in todo:
        env()
    if "attn" in todo:
        attn_default()
    if set(todo) - {"env", "attn"}:
        t0 = time.time()
        tokenizer, model = hw3.load_model()
        print(f"load_model {time.time() - t0:.1f}s")
        fn = {"model": model_facts, "tok": lambda t, m: tok_facts(t), "q1": q1_facts, "q2": q2_facts, "q4": q4_facts,
              "q5": q5_facts, "q6": q6_facts, "q7": q7_facts, "shapes": shapes_facts, "perq": perq_facts,
              "q4steps": q4steps_facts, "ptit": ptit_facts, "rescale26": rescale26_facts,
              "review_ch01": review_ch01_facts, "kvcache": kvcache_facts, "review_ch02": review_ch02_facts, "pre_ch03": pre_ch03_facts, "review_ch03": review_ch03_facts, "pre_ch04": pre_ch04_facts, "review_ch04": review_ch04_facts, "pre_ch06": pre_ch06_facts, "pre_ch07": pre_ch07_facts, "review_ch07": review_ch07_facts}
        for s in todo:
            if s in fn:
                t0 = time.time()
                fn[s](tokenizer, model)
                print(f"[{s} took {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
