# HW3 - Understanding LLM / Transformers

Local port of the [official Colab](https://colab.research.google.com/drive/1Ku_p27ml8QJ-Rd7FilZQvDA9axe68V7o)
([slides](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data//hw3.pdf)).
Model: [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) (gated — accept the license first).

## Run

From the repo root (with the shared venv):

```bash
export HF_TOKEN=hf_...            # or `hf auth login`
python HW03/hw3.py                # all questions Q1–Q7
python HW03/hw3.py --q 3          # a single question
```

Figures are written to `HW03/outputs/`.

## Questions & useful flags

| Q | What | Flags |
|---|------|-------|
| 1 | Chat template vs. plain prompt, coherence score | `--max-new-tokens` |
| 2 | Multi-turn chat, top-10 next-token probabilities | `--interactive` (default replays the 3 turns from the slides) |
| 3 | Tokenization | `--sentence` |
| 4 | Top-k / top-p sampling + self-BLEU | `--top-k 2/200/1`, `--top-p 0.6/0.999/0`, `--seed` |
| 5 | t-SNE of sentence embeddings | |
| 6 | Attention map | `--layer-idx`, `--head-idx` |
| 7 | Gemma Scope SAE feature 10004 | `--sae-layer-idx`, `--token-idx 1 2 3` |

Examples for Q4:

```bash
python HW03/hw3.py --q 4 --top-k 200 --top-p 0.999
python HW03/hw3.py --q 4 --top-k 1 --top-p 0 --num-samples 1
```
