#!/usr/bin/env python3
"""每章查驗：標籤平衡、pre 內無裸 < > &、listing 逐字 diff、結尾三段、字數。
用法: verify_book.py <src_root> <html...>
"""
import html as H
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(sys.argv[1])
VOID = {'meta', 'link', 'br', 'hr', 'img', 'input', 'wbr', 'source', 'col', 'area', 'base'}
# SVG self-closing elements are parsed as startendtag, fine.


class Bal(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.err = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1][0] != tag:
            self.err.append(f'line {self.getpos()[0]}: </{tag}> vs open {self.stack[-1] if self.stack else None}')
            # try recover
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i][0] == tag:
                    del self.stack[i:]
                    break
        else:
            self.stack.pop()


fail = False
for p in sys.argv[2:]:
    f = Path(p)
    src = f.read_text(encoding='utf-8')
    probs = []
    # 1) tag balance (on source without inlined style/script)
    body = re.sub(r'<(style|script)[^>]*>.*?</\1>', '', src, flags=re.S)
    b = Bal()
    b.feed(body)
    probs += b.err
    if b.stack:
        probs.append(f'unclosed: {b.stack[:5]}')
    if not src.rstrip().endswith('</html>'):
        probs.append('not ending with </html>')
    # 2) bare chars in pre
    for m in re.finditer(r'<pre[^>]*>(.*?)</pre>', body, re.S):
        inner = re.sub(r'</?code[^>]*>', '', m.group(1))
        if '<' in inner or '>' in inner:
            probs.append(f'bare < or > in pre near: {inner[:60]!r}')
        for a in re.finditer(r'&(?!(?:[a-z]+|#\d+|#x[0-9a-f]+);)', inner):
            probs.append(f'bare & in pre near: {inner[max(0,a.start()-20):a.start()+20]!r}')
    # 3) listing verbatim diff
    n_list = 0
    for m in re.finditer(r'<figure class="listing"><figcaption>(.*?)</figcaption>\s*<pre[^>]*><code>(.*?)</code></pre>', body, re.S):
        cap, code = H.unescape(re.sub('<[^>]+>', '', m.group(1))), H.unescape(m.group(2))
        cm = re.match(r'\s*([\w./-]+):(\d+)(?:[–-](\d+))?', cap)
        if not cm:
            probs.append(f'listing caption unparsable: {cap!r}')
            continue
        fn, a, z = cm.group(1), int(cm.group(2)), int(cm.group(3) or cm.group(2))
        # docs/HWxx/chNN.html quotes HWxx/<file>; fall back to a repo-root path
        sp = next((c for c in (ROOT / Path(p).parent.name / fn, ROOT / fn) if c.exists()), ROOT / fn)
        if not sp.exists():
            probs.append(f'listing file missing: {fn}')
            continue
        want = sp.read_text(encoding='utf-8').split('\n')[a - 1:z]
        got = code.rstrip('\n').split('\n')
        n_list += 1
        if got != want:
            for i, (g, w) in enumerate(zip(got, want)):
                if g != w:
                    probs.append(f'{fn}:{a+i} mismatch\n   got : {g!r}\n   want: {w!r}')
                    break
            else:
                probs.append(f'{fn}:{a}-{z} line count {len(got)} vs {len(want)}')
    # 4) endings (chapters only)
    if re.fullmatch(r'ch\d+\.html', f.name):
        if 'class="recap"' not in body:
            probs.append('missing recap')
        q = body.count('class="quiz"')
        if q < 3:
            probs.append(f'quiz count {q} < 3')
        if '延伸閱讀' not in body:
            probs.append('missing 延伸閱讀')
    # 5) svg quick checks
    for i, s in enumerate(re.findall(r'<svg\b.*?</svg>', body, re.S), 1):
        head = re.match(r'<svg[^>]*>', s).group(0)
        for k in ('viewBox', 'role="img"', 'aria-label'):
            if k not in head:
                probs.append(f'svg#{i} missing {k}')
        bad = re.findall(r'</?(b|i|code|span|div|p|em|strong|br|small|sub|sup)\b', s)
        if bad:
            probs.append(f'svg#{i} has HTML-only tags {set(bad)}')
        for t in re.findall(r'<text\b[^>]*>', s):
            if 'class="' not in t and 'fill=' not in t:
                probs.append(f'svg#{i} text without class/fill: {t[:60]}')
    # 6) CJK char count in main
    mm = re.search(r'<main>(.*)</main>', body, re.S)
    txt = re.sub(r'<pre.*?</pre>|<svg.*?</svg>', '', mm.group(1) if mm else '', flags=re.S)
    cjk = len(re.findall(r'[一-鿿]', re.sub('<[^>]+>', '', txt)))
    print(f'{f.name}: listings={n_list} cjk={cjk} problems={len(probs)}')
    for pr in probs:
        print('  -', pr)
    fail |= bool(probs)
sys.exit(1 if fail else 0)
