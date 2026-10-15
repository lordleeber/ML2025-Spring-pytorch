# Share of each 1-shot training sample's tokens that the loss covers with --completion-only-loss
# (the final answer only), over all 7,473 samples; the default collator covers 100%. CPU only.
# usage: HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python docs/tools/hw06_loss_share.py
import sys, statistics as st
sys.path.insert(0, 'HW06')
import hw6
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
tok = AutoTokenizer.from_pretrained(hw6.MODEL_ID); tok.add_special_tokens({'pad_token': '[PAD]'})
train = hw6.load_jsonlines('HW06/data/gsm8k_train.jsonl')
ds, _ = hw6.format_train_data(train, tok, 1)
col = DataCollatorForCompletionOnlyLM(hw6.RESPONSE_TEMPLATE, tokenizer=tok)
r = []; tot = ans = 0
for row in ds:
    ids = tok(row['text'])['input_ids']; b = col([{'input_ids': ids}])['labels'][0]
    k = int((b != -100).sum()); r.append(k / len(ids)); tot += len(ids); ans += k
r.sort()
print(f'1-shot, all {len(ds)} samples: tokens {tot}, answer tokens {ans} = {ans/tot:.1%}; per-sample share median {st.median(r):.1%}, min {r[0]:.1%}, max {r[-1]:.1%}, 10th pct {r[len(r)//10]:.1%}, 90th pct {r[9*len(r)//10]:.1%}')
