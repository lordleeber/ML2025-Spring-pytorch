# ML2025 Spring (PyTorch, local)

NTU 李宏毅 [Machine Learning 2025 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2025-spring.php) homeworks, ported from Colab to run locally.

## Environment

One shared venv at the repo root, managed by [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # creates .venv from pyproject.toml / uv.lock
source .venv/bin/activate
```

- PyTorch comes from the CUDA 12.8 index (required for Blackwell / sm_120 GPUs).
- `transformers` is pinned to the version the TAs specify. If a later HW needs an
  incompatible version, give that HW its own `pyproject.toml` + venv instead of bumping the root.
- Gated models (e.g. Gemma) need a Hugging Face token: `export HF_TOKEN=hf_...`
  or run `hf auth login` once.

## Homeworks

| HW | Topic | Code |
|----|-------|------|
| 3 | Understanding Transformer (Gemma-2-2b-it) | [HW03](HW03/) |
