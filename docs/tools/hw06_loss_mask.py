# Which tokens does the training loss cover? Build real training samples the way hw6.py does
# (format_train_data, n-shot), run them through trl's default collator (what the notebook uses)
# and through DataCollatorForCompletionOnlyLM (hw6.py --completion-only-loss), and count the tokens
# whose label is not -100. CPU only, no model needed.
# usage: HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python docs/tools/hw06_loss_mask.py [n_shot ...]
import sys

sys.path.insert(0, 'HW06')
import hw6  # noqa: E402
from transformers import AutoTokenizer, DataCollatorForLanguageModeling  # noqa: E402
from trl import DataCollatorForCompletionOnlyLM  # noqa: E402

tok = AutoTokenizer.from_pretrained(hw6.MODEL_ID)
tok.add_special_tokens({'pad_token': '[PAD]'})
train = hw6.load_jsonlines('HW06/data/gsm8k_train.jsonl')[:200]
default = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
completion = DataCollatorForCompletionOnlyLM(hw6.RESPONSE_TEMPLATE, tokenizer=tok)
for n in [int(a) for a in sys.argv[1:]] or [1, 8]:
    ds, _ = hw6.format_train_data(train, tok, n)
    tot = {'all': 0, 'default': 0, 'completion': 0}
    for row in ds:
        ids = tok(row['text'])['input_ids']
        a = default([{'input_ids': ids}])['labels'][0]
        b = completion([{'input_ids': ids}])['labels'][0]
        tot['all'] += len(ids)
        tot['default'] += int((a != -100).sum())
        tot['completion'] += int((b != -100).sum())
    print(f'{n}-shot, {len(ds)} samples: tokens {tot["all"]}, in loss (default) {tot["default"]}, '
          f'in loss (completion only) {tot["completion"]} = {tot["completion"] / tot["all"]:.1%}')
ids = tok(ds[0]['text'])['input_ids']
b = completion([{'input_ids': ids}])['labels'][0]
print('completion-only labels of sample 0 decode to:', repr(tok.decode([t for t in b.tolist() if t != -100])[:200]))
print('sample 0 answer field starts:', repr(train[0]['answer'][:60]))
