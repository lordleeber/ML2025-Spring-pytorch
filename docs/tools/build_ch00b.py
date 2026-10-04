"""Fill {{SRC file a b}} / {{HW3 a b}} placeholders with verbatim, HTML-escaped source lines.

Used once to build docs/HW03/ch00b.html from a draft; kept so the transformers excerpts
(which verify_book.py cannot check: the cloud has no .venv) can be re-verified locally:
    python3 docs/tools/build_ch00b.py --check docs/HW03/ch00b.html
"""
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TF = ROOT / ".venv/lib/python3.12/site-packages/transformers"


def lines(path, a, b):
    return "\n".join(path.read_text(encoding="utf-8").split("\n")[a - 1:b])


def fill(src):
    src = re.sub(r"\{\{SRC (\S+) (\d+) (\d+)\}\}", lambda m: html.escape(lines(TF / m[1], int(m[2]), int(m[3])), quote=False), src)
    return re.sub(r"\{\{HW3 (\d+) (\d+)\}\}", lambda m: html.escape(lines(ROOT / "HW03/hw3.py", int(m[1]), int(m[2])), quote=False), src)


def check(page):
    """Every pre.py preceded by a <!-- src: file a b --> marker must match those transformers lines."""
    s = Path(page).read_text(encoding="utf-8")
    bad = 0
    for m in re.finditer(r'<!-- src: (\S+) (\d+) (\d+) -->\s*<pre class="py">(.*?)</pre>', s, re.S):
        want = lines(TF / m[1], int(m[2]), int(m[3]))
        got = html.unescape(m[4])
        ok = got == want
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {m[1]}:{m[2]}-{m[3]}")
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        sys.exit(1 if check(sys.argv[2]) else 0)
    src = Path(sys.argv[1]).read_text(encoding="utf-8")
    # keep a machine-checkable marker in front of each transformers excerpt
    src = re.sub(r'<pre class="py">\{\{SRC (\S+) (\d+) (\d+)\}\}</pre>', r'<!-- src: \1 \2 \3 --><pre class="py">{{SRC \1 \2 \3}}</pre>', src)
    Path(sys.argv[2]).write_text(fill(src), encoding="utf-8")
