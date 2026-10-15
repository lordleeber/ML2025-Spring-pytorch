# Chapter 5 numbers from docs/tools/hw06_judge.jsonl (verdicts only, no response text): Safety Rate
# per hazard and per persona for the base model vs the 1-epoch model (greedy, Llama Guard 3 8B), and
# how often each judge says unsafe on the same 1,440 responses (6 sets x 240, sampled decoding).
# Rows are matched by position, not by release_prompt_id (12 ids appear twice).
import collections
import json

rows = collections.defaultdict(list)
for line in open('docs/tools/hw06_judge.jsonl'):
    r = json.loads(line)
    rows[(r['judge'], r['tag'])].append(r)


def rate(judge, tags, key):
    c = collections.defaultdict(lambda: [0, 0])
    for t in tags:
        for r in rows[(judge, t)]:
            c[r[key]][0] += r['safe']
            c[r[key]][1] += 1
    return c


base, ep1 = ['greedy_base'], ['greedy_ep1_ref1', 'greedy_ep1_ref2', 'greedy_ep1_ref5']
for key in ['hazard', 'persona']:
    b, e = rate('guard8b', base, key), rate('guard8b', ep1, key)
    print(f'--- Guard 8B, greedy, by {key}: base safe/n | 1 epoch (3 runs) safe/n')
    for k in sorted(b, key=lambda k: -b[k][1]):
        print(f'{k:8} {b[k][0]:3}/{b[k][1]:3} = {b[k][0] / b[k][1]:.3f} | {e[k][0]:3}/{e[k][1]:3} = {e[k][0] / e[k][1]:.3f}')
sets = ['base_s0', 'base_s1', 'base_s2', 'ref1', 'ref2', 'ref3']
print('--- unsafe verdicts on the same 1,440 responses (sampled, base s0-s2 + ref1-ref3)')
for j in ['guard1b', 'guard8b', 'shieldgemma2b']:
    n = sum(len(rows[(j, t)]) for t in sets)
    u = sum(not r['safe'] for t in sets for r in rows[(j, t)])
    print(f'{j:14} unsafe {u:4} / {n} = {u / n:.3f}')
cats = collections.Counter()
for t in sets:
    for r in rows[('guard8b', t)]:
        if not r['safe']:
            cats[r['verdict'].split('\n')[-1].strip()] += 1
print('--- Guard 8B categories on its unsafe verdicts:', cats.most_common(8))
