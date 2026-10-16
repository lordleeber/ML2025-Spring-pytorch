"""Refill every figure.listing's <code> from the repo source named in its figcaption
("file.py:a–b — ..."), so quotes are verbatim (trailing spaces included). Usage:
python fill_listings.py <repo_root> <src_dir_rel> <page.html>..."""
import html, re, sys, os
root, src = sys.argv[1], sys.argv[2]
for page in sys.argv[3:]:
  s = open(page).read()
  def rep(m):
    cap = m.group(1)
    mm = re.match(r'\s*([\w/.\-]+\.py):(\d+)(?:[–-](\d+))?', html.unescape(re.sub(r'<[^>]+>', '', cap)))
    if not mm:
      return m.group(0)
    f, a, b = mm.group(1), int(mm.group(2)), int(mm.group(3) or mm.group(2))
    path = os.path.join(root, f) if os.path.exists(os.path.join(root, f)) else os.path.join(root, src, f)
    lines = open(path).read().split('\n')[a - 1:b]
    return m.group(0)[:m.start(3) - m.start(0)] + html.escape('\n'.join(lines), quote=False) + m.group(0)[m.end(3) - m.start(0):]
  s2 = re.sub(r'<figure class="listing"><figcaption>(.*?)</figcaption>\s*<pre([^>]*)><code>(.*?)</code></pre>', rep, s, flags=re.S)
  open(page, 'w').write(s2)
  print(page, 'changed' if s2 != s else 'unchanged')
