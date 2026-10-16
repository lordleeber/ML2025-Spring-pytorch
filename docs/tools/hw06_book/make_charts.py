"""Fill the <!-- CHART:name --> ... <!-- /CHART:name --> blocks of the HW06 book from the recorded numbers.

Safety Rates come from docs/tools/hw06_judge_summary.jsonl and Accuracy from docs/tools/hw06_runs.jsonl.
The per-checkpoint GSM8K accuracies of ref3 were recorded on 2026-10-11 in HW06/PLAN.md and
docs/HW06/FACTS.md only (their log files were lost when the machine rebooted), so they are listed here.
usage: python docs/tools/hw06_book/make_charts.py <page.html>...
"""
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(__file__))
from charts import dot_rows, legend, line_chart  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..')
C = {'base': '#8b949e', 'ep1': '#3987e5', 'ep3': '#d95926', 'lr': '#199e70', 'shot': '#c98500',
     'reg': '#d55181', 'self': '#008300', 'fixed': '#9085e9',
     'j8b': '#9085e9', 'j1b': '#d55181', 'jsg': '#c98500'}

safety, acc = {}, {}
for line in open(os.path.join(ROOT, 'docs/tools/hw06_judge_summary.jsonl')):
    r = json.loads(line)
    safety[(r['judge'], r['tag'])] = r['safety_rate']
for line in open(os.path.join(ROOT, 'docs/tools/hw06_runs.jsonl')):
    r = json.loads(line)
    if r.get('gsm8k_public_acc') is not None:
        acc[r['run']] = r['gsm8k_public_acc']

CKPTS = [187, 374, 561, 748, 935, 1122, 1309, 1496, 1683, 1868]
CKPT_ACC = [0.288, 0.341, 0.280, 0.333, 0.288, 0.348, 0.333, 0.439, 0.371, 0.341]  # FACTS.md, 2026-10-11


def ckpt_acc():
    s = [{'name': 'acc', 'label': '1 epoch（ref3）', 'values': CKPT_ACC, 'color': C['ep1']},
         {'name': 'base', 'label': '原模型', 'values': [0.318] * 10, 'color': C['base'], 'dash': '6 5'}]
    svg = line_chart(s, 10, 0.2, 0.5, [0.2, 0.3, 0.4, 0.5], xlabel='checkpoint（訓練步數）', ylabel='GSM8K public Accuracy',
                     aria='1 epoch 訓練中 10 個 checkpoint 的 GSM8K public Accuracy，取樣解碼、種子 0，原模型 0.318 以虛線表示',
                     xticks=[(i, str(c)) for i, c in enumerate(CKPTS)], fmt='{:.3f}', xname=lambda i: f'checkpoint-{CKPTS[i]}')
    return legend(s) + svg


def ckpt_safety():
    s = [{'name': j, 'label': lab, 'values': [safety[(j, f'ref3_c{c}')] for c in CKPTS], 'color': C[k]}
         for j, lab, k in [('guard8b', 'Guard 8B', 'j8b'), ('guard1b', 'Guard 1B', 'j1b'), ('shieldgemma2b', 'ShieldGemma', 'jsg')]]
    s.append({'name': 'base8b', 'label': '原模型（8B）', 'values': [safety[('guard8b', 'base_s0')]] * 10,
              'color': C['base'], 'dash': '6 5'})
    svg = line_chart(s, 10, 0.6, 1.0, [0.6, 0.7, 0.8, 0.9, 1.0], xlabel='checkpoint（訓練步數）', ylabel='Safety Rate',
                     aria='同樣 10 個 checkpoint 的 Safety Rate，三個審查模型各一條線；原模型以 Guard 8B 判的 0.850 用虛線表示',
                     xticks=[(i, str(c)) for i, c in enumerate(CKPTS)], fmt='{:.3f}', xname=lambda i: f'checkpoint-{CKPTS[i]}')
    return legend(s) + svg


MEDIUM = [('原模型', 'base', ['base1126_t256']), ('1 epoch（預設）', 'ep1', ['ep1g_s0', 'ep1g_s1', 'ep1g_s2']),
          ('lr 1e-4', 'lr', ['lr1e-4_s0', 'lr1e-4_s1', 'lr1e-4_s2']), ('lr 1e-5', 'lr', ['lr1e-5_s0', 'lr1e-5_s1', 'lr1e-5_s2']),
          ('5-shot', 'shot', ['shot5_s0', 'shot5_s1', 'shot5_s2']), ('8-shot', 'shot', ['shot8_s0', 'shot8_s1', 'shot8_s2'])]


def medium_rows(get):
    rows = []
    for label, k, runs in MEDIUM:
        v = [get(r) for r in runs]
        rows.append({'label': label, 'color': C[k], 'values': v, 'mean': st.mean(v) if len(v) > 1 else None})
    return rows


def medium_acc():
    return dot_rows(medium_rows(lambda r: acc[r]), 0.2, 0.45, [0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
                    xlabel='GSM8K public Accuracy（greedy；每個點是一個種子，直線是平均）',
                    aria='Medium 提示各設定的 GSM8K public Accuracy，三個種子與平均')


def medium_safety():
    return dot_rows(medium_rows(lambda r: safety[('guard8b', r)]), 0.75, 0.95, [0.75, 0.8, 0.85, 0.9, 0.95],
                    xlabel='Safety Rate（Guard 8B；greedy；每個點是一個種子，直線是平均）',
                    aria='Medium 提示各設定的 Safety Rate（Guard 8B），三個種子與平均')


def loss_curve():
    d = json.load(open(os.path.join(ROOT, 'docs/tools/hw06_loss_curve.json')))
    h = d['log_history']
    s = [{'name': 'loss', 'label': 'loss', 'values': [x['loss'] for x in h], 'color': C['ep1']}]
    return line_chart(s, len(h), 0.75, 1.0, [0.75, 0.8, 0.85, 0.9, 0.95, 1.0], xlabel='訓練步數', ylabel='這一段 187 步的平均 loss',
                      aria='1 epoch 訓練中每 187 步的平均 loss，種子 0，從 0.981 降到 0.802',
                      xticks=[(i, str(x['step'])) for i, x in enumerate(h)], fmt='{:.4f}', xname=lambda i: f"第 {h[i]['step']} 步")


STRONG = [('原模型', 'base', ['base1126_t256']), ('1 epoch（預設）', 'ep1', ['ep1g_s0', 'ep1g_s1', 'ep1g_s2']),
          ('5 epochs', 'ep3', ['ep5b_s0', 'ep5b_s1', 'ep5b_s2']),
          ('weight decay 1e-2', 'reg', ['wd1e-2_s0', 'wd1e-2_s1', 'wd1e-2_s2']),
          ('LoRA dropout 0.2', 'reg', ['drop0.2_s0', 'drop0.2_s1', 'drop0.2_s2']),
          ('固定 8-shot', 'fixed', ['fixed8_s0', 'fixed8_s1', 'fixed8_s2']),
          ('self-instruct', 'self', ['selfinst_s0', 'selfinst_s1', 'selfinst_s2']),
          ('只算答案，1-shot', 'shot', ['co1_s0', 'co1_s1', 'co1_s2']),
          ('只算答案，8-shot', 'shot', ['co8_s0', 'co8_s1', 'co8_s2']),
          ('lr 1e-3（對照）', 'lr', ['lr1e-3_s0', 'lr1e-3_s1', 'lr1e-3_s2'])]


def strong_rows(get):
    rows = []
    for label, k, runs in STRONG:
        v = [get(r) for r in runs]
        rows.append({'label': label, 'color': C[k], 'values': v, 'mean': st.mean(v) if len(v) > 1 else None})
    return rows


def strong_acc():
    return dot_rows(strong_rows(lambda r: acc[r]), 0.2, 0.5, [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                    xlabel='GSM8K public Accuracy（greedy；每個點是一個種子，直線是平均）', label_w=170,
                    aria='Strong 提示與對照各設定的 GSM8K public Accuracy，三個種子與平均')


def strong_safety():
    return dot_rows(strong_rows(lambda r: safety[('guard8b', r)]), 0.55, 0.95, [0.55, 0.65, 0.75, 0.85, 0.95],
                    xlabel='Safety Rate（Guard 8B；greedy；每個點是一個種子，直線是平均）', label_w=170,
                    aria='Strong 提示與對照各設定的 Safety Rate（Guard 8B），三個種子與平均')


def tradeoff():
    """Scatter: mean GSM8K accuracy (x) vs mean Guard 8B safety (y) per setting; base model as crosshair."""
    import html
    W, H, ML, MR, MT, MB = 900, 460, 64, 24, 20, 50
    x0, x1, y0, y1 = 0.2, 0.5, 0.6, 0.95
    pw, ph = W - ML - MR, H - MT - MB
    X = lambda v: ML + pw * (v - x0) / (x1 - x0)  # noqa: E731
    Y = lambda v: MT + ph * (1 - (v - y0) / (y1 - y0))  # noqa: E731
    pts = [(lab, k, st.mean(acc[r] for r in runs), st.mean(safety[('guard8b', r)] for r in runs), dx, dy)
           for lab, k, runs, dx, dy in TRADEOFF]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="各設定的 GSM8K Accuracy（橫軸）對 Safety Rate（Guard 8B，縱軸），三個種子平均；虛線是原模型">']
    for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        out.append(f'<line x1="{X(t):.1f}" y1="{MT}" x2="{X(t):.1f}" y2="{MT + ph}" stroke="#2f2d39" stroke-width="1"/>')
        out.append(f'<text class="s-mono" x="{X(t):.1f}" y="{MT + ph + 18}" text-anchor="middle">{t}</text>')
    for t in [0.6, 0.7, 0.8, 0.9]:
        out.append(f'<line x1="{ML}" y1="{Y(t):.1f}" x2="{ML + pw}" y2="{Y(t):.1f}" stroke="#2f2d39" stroke-width="1"/>')
        out.append(f'<text class="s-mono" x="{ML - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    bx, by = acc['base1126_t256'], safety[('guard8b', 'base1126_t256')]
    out.append(f'<line x1="{X(bx):.1f}" y1="{MT}" x2="{X(bx):.1f}" y2="{MT + ph}" stroke="#8b949e" stroke-width="1.5" stroke-dasharray="6 5"/>')
    out.append(f'<line x1="{ML}" y1="{Y(by):.1f}" x2="{ML + pw}" y2="{Y(by):.1f}" stroke="#8b949e" stroke-width="1.5" stroke-dasharray="6 5"/>')
    out.append(f'<text class="s-sm" x="{X(bx) + 6:.1f}" y="{MT + 12}">原模型 {bx:.3f}／{by:.3f}</text>')
    out.append(f'<text class="s-sm" x="{ML + pw / 2:.1f}" y="{H - 8}" text-anchor="middle">GSM8K public Accuracy（greedy，3 種子平均）→ 越右越會算</text>')
    out.append(f'<text class="s-sm" x="16" y="{MT + ph / 2:.1f}" text-anchor="middle" transform="rotate(-90 16 {MT + ph / 2:.1f})">Safety Rate（Guard 8B）→ 越上越安全</text>')
    for lab, k, x, y, dx, dy in pts:
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="6" fill="{C[k]}" stroke="#1b1a22" stroke-width="2" data-tip="{html.escape(lab)}：Accuracy {x:.3f}、Safety {y:.3f}"/>')
        out.append(f'<text class="s-lbl" x="{X(x) + dx:.1f}" y="{Y(y) + dy:.1f}">{html.escape(lab)}</text>')
    out.append('</svg>')
    return '\n'.join(out)


TRADEOFF = [('1 epoch（預設）', 'ep1', ['ep1g_s0', 'ep1g_s1', 'ep1g_s2'], 10, 18),
            ('lr 1e-4', 'lr', ['lr1e-4_s0', 'lr1e-4_s1', 'lr1e-4_s2'], -62, -8),
            ('lr 1e-5', 'lr', ['lr1e-5_s0', 'lr1e-5_s1', 'lr1e-5_s2'], -70, 4),
            ('8-shot（隨機）', 'shot', ['shot8_s0', 'shot8_s1', 'shot8_s2'], 10, 4),
            ('5 epochs', 'ep3', ['ep5b_s0', 'ep5b_s1', 'ep5b_s2'], 10, 4),
            ('固定 8-shot', 'fixed', ['fixed8_s0', 'fixed8_s1', 'fixed8_s2'], 10, -8),
            ('self-instruct', 'self', ['selfinst_s0', 'selfinst_s1', 'selfinst_s2'], 10, 4),
            ('只算答案 8-shot', 'shot', ['co8_s0', 'co8_s1', 'co8_s2'], 10, 16),
            ('lr 1e-3', 'lr', ['lr1e-3_s0', 'lr1e-3_s1', 'lr1e-3_s2'], 10, 4)]


CHARTS = {'tradeoff': tradeoff, 'strong_acc': strong_acc, 'strong_safety': strong_safety, 'loss_curve': loss_curve, 'ckpt_acc': ckpt_acc, 'ckpt_safety': ckpt_safety, 'medium_acc': medium_acc, 'medium_safety': medium_safety}

for page in sys.argv[1:]:
    s = open(page).read()

    def rep(m):
        return f'<!-- CHART:{m.group(1)} -->\n{CHARTS[m.group(1)]()}\n<!-- /CHART:{m.group(1)} -->'

    s2 = re.sub(r'<!-- CHART:(\w+) -->.*?<!-- /CHART:\1 -->', rep, s, flags=re.S)
    open(page, 'w').write(s2)
    print(page, 'changed' if s2 != s else 'unchanged')
