"""Take apart hw3.py's self-BLEU on the sentences recorded in docs/HW03/logs (CPU only, no model).

Run from the repo root:
    .venv/bin/python docs/tools/hw03_selfbleu.py

Sentences are parsed back out of the hw3.py logs (multi-line sentences included) and the
self-BLEU is recomputed with hw3.compute_self_bleu, so every number can be checked against
the score the log itself printed.
"""

import re
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "HW03"))
import hw3  # noqa: E402
from nltk.translate.bleu_score import SmoothingFunction, brevity_penalty, modified_precision, sentence_bleu  # noqa: E402

LOGS = ROOT / "docs" / "HW03" / "logs"


def parse(path):
    """Return {'top-k': [...], 'top-p': [...]} and the printed scores, from a hw3.py Q4 log."""
    t = path.read_text()
    # nltk's warnings go to stderr and land in the middle of the printed sentences in the logs; cut them out
    t = re.sub(r"[^\n]*UserWarning: \n(?:.*\n)*?  warnings\.warn\(_msg\)\n", "", t)
    t = t[t.index("===== Top-K Sampling Output ====="):]
    k_part, rest = t.split("===== Top-P Sampling Output =====")
    p_part, score_part = rest.split("self-BLEU Score for top_k", 1)
    out = {}
    for name, part in [("top-k", k_part), ("top-p", p_part)]:
        body = part.split("\n", 2)[2]  # skip the header line and the blank line after it
        sents, i = [], 0
        while True:
            start = body.find(f"{i}. ") if i == 0 else body.find(f"\n{i}. ")
            if start < 0:
                break
            start += len(f"{i}. ") + (0 if i == 0 else 1)
            nxt = body.find(f"\n{i + 1}. ", start)
            sents.append(body[start:nxt if nxt >= 0 else len(body)].rstrip("\n"))
            i += 1
        out[name] = sents
    scores = re.findall(r"self-BLEU Score for top_[kp] \([kp]=[^)]*\): ([0-9.]+)", "self-BLEU Score for top_k" + score_part)
    return out, [float(s) for s in scores]


def breakdown(hyp, ref):
    h, r = hyp.split(), ref.split()
    ps = [modified_precision([r], h, n) for n in range(1, 5)]
    bp = brevity_penalty(len(r), len(h))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = sentence_bleu([r], h)
    return ps, bp, b


def main():
    for name in ["run_q4_seed0.txt", "run_q4_k200_p0999_seed0.txt", "run_seed0.txt"]:
        groups, printed = parse(LOGS / name)
        print(f"\n===== {name}: printed self-BLEU {printed} =====")
        for (g, sents), pr in zip(groups.items(), printed):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                sb = hw3.compute_self_bleu(sents)
            pairs = [(i, j) for i in range(len(sents)) for j in range(len(sents)) if i != j]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bl = {(i, j): sentence_bleu([sents[j].split()], sents[i].split()) for i, j in pairs}
                sm = SmoothingFunction().method1
                sb_s = sum(sum(sentence_bleu([sents[j].split()], sents[i].split(), smoothing_function=sm) for j in range(len(sents)) if j != i) / (len(sents) - 1) for i in range(len(sents))) / len(sents)
            zero = sum(v < 1e-10 for v in bl.values())  # nltk returns ~1e-78, not 0, when an n-gram order has no overlap
            one = sum(abs(v - 1) < 1e-12 for v in bl.values())
            asym = sum(abs(bl[(i, j)] - bl[(j, i)]) > 1e-12 for i, j in pairs if i < j)
            msgs = Counter(str(x.message).split("\n")[1] for x in w)
            print(f"--- {g}: {len(sents)} sentences, {len(set(sents))} distinct; recomputed self-BLEU {sb:.4f} (log {pr:.4f}); "
                  f"with SmoothingFunction().method1 {sb_s:.4f}")
            print(f"    {len(pairs)} ordered pairs: BLEU < 1e-10 (effectively 0) in {zero}, = 1 in {one}; asymmetric unordered pairs {asym}/{len(pairs) // 2}")
            print(f"    warnings raised (always): {dict(msgs)}")
            print(f"    most common sentence x{Counter(sents).most_common(1)[0][1]}: {Counter(sents).most_common(1)[0][0]!r}")

    groups, _ = parse(LOGS / "run_q4_seed0.txt")
    k = groups["top-k"]
    print("\n===== pair breakdowns (run_q4_seed0.txt top-k) =====")
    for i, j in [(0, 2), (2, 0), (0, 1), (0, 0)]:
        ps, bp, b = breakdown(k[i], k[j])
        print(f"hyp {i} {k[i]!r}\nref {j} {k[j]!r}")
        print(f"  lengths {len(k[i].split())}/{len(k[j].split())}; p1..p4 " + ", ".join(f"{p.numerator}/{p.denominator}" for p in ps)
              + f"; BP {bp:.4f}; BLEU {b:.4f}")
    print("\nsplit() keeps punctuation attached:", k[0].split()[-2:])


if __name__ == "__main__":
    main()
