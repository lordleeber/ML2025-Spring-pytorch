"""Measure every number the HW03 textbook cites (needs GPU + HF access to Gemma).

Run from the repo root:
    .venv/bin/python docs/tools/hw03_facts.py [env model tok q1 q2 q4 q5 q6 q7]

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
              "q5": q5_facts, "q6": q6_facts, "q7": q7_facts}
        for s in todo:
            if s in fn:
                t0 = time.time()
                fn[s](tokenizer, model)
                print(f"[{s} took {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
