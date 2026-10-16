# Data and prompt facts for chapter 1 (CPU only): sizes and formats of the five data files, what the
# self-instruct answers changed, AILuminate's categories (counts only, no prompt text), and the exact
# text the model sees for one training sample and one test question, rendered by hw6.py's own functions.
# usage: HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python docs/tools/hw06_data_facts.py
import collections
import csv
import random
import re
import statistics as st
import sys

sys.path.insert(0, 'HW06')
import hw6  # noqa: E402  (seeds Python's random with 42 at import, as the notebook does)
from transformers import AutoTokenizer  # noqa: E402

D = 'HW06/data/'
tr = hw6.load_jsonlines(D + 'gsm8k_train.jsonl')
si = hw6.load_jsonlines(D + 'gsm8k_train_self-instruct.jsonl')
pu = hw6.load_jsonlines(D + 'gsm8k_test_public.jsonl')
pr = hw6.load_jsonlines(D + 'gsm8k_test_private.jsonl')
print('sizes: train', len(tr), 'self-instruct', len(si), 'public', len(pu), 'private', len(pr))
print('keys: train', list(tr[0]), 'private', list(pr[0]))
trq = {x['question']: x for x in tr}
print('self-instruct questions found in train:', sum(x['question'] in trq for x in si))
print('public questions found in train:', sum(x['question'] in trq for x in pu))
for name, data in [('train', tr), ('self-instruct', si)]:
    calc = sum('<<' in x['answer'] for x in data)
    lines = st.median(x['answer'].count('\n') + 1 for x in data)
    words = st.median(len(x['answer'].split()) for x in data)
    print(f'{name}: answers with <<calc>> {calc}, with #### {sum("####" in x["answer"] for x in data)}, '
          f'median lines {lines}, median words {words}')
same_final = sum(hw6.extract_ans_from_response(x['answer']) == hw6.extract_ans_from_response(trq[x['question']]['answer'])
                 for x in si)
print('self-instruct final answer equals the original one:', same_final, '/', len(si))
nonint = [x['answer'].split('####')[-1].strip() for x in pu if not re.fullmatch(r'-?\d+', x['answer'].split('####')[-1].strip())]
print('public answers that are not plain integers:', nonint)
print('--- train[0]'); print(tr[0]['answer'])
print('--- self-instruct entry for the same question'); print(next(x for x in si if x['question'] == tr[0]['question'])['answer'])
rows = list(csv.DictReader(open(D + 'ailuminate_test.csv')))
print('ailuminate rows', len(rows), 'columns', list(rows[0]), 'unique prompts', len({r['prompt_text'] for r in rows}))
print('hazard', collections.Counter(r['hazard'] for r in rows).most_common())
print('persona', collections.Counter(r['persona'] for r in rows), 'locale', collections.Counter(r['locale'] for r in rows))
print('ailuminate prompt words: median', st.median(len(r['prompt_text'].split()) for r in rows))

tok = AutoTokenizer.from_pretrained(hw6.MODEL_ID)
tok.add_special_tokens({'pad_token': '[PAD]'})
ds, maxlen = hw6.format_train_data(tr, tok, 1)
lens = [len(tok(x['text'])['input_ids']) for x in ds]
print('1-shot training samples: max tokens', maxlen, 'median', st.median(lens), '(tokenizer adds <|begin_of_text|>:', tok(ds[0]['text'])['input_ids'][0] == tok.bos_token_id, ')')
print('--- training sample 0 as stored in the dataset (system header cut off)'); print(ds[0]['text'])
random.seed(1126)  # what Trainer's set_seed(1126) leaves behind for the evaluation
test = hw6.nshot_chats(tr, 1, pu[0]['question'], None, 'test')
print('--- public question 0 as the generator sees it after training (Python random reseeded with 1126)')
print(tok.apply_chat_template(test, tokenize=False, add_generation_prompt=True))
