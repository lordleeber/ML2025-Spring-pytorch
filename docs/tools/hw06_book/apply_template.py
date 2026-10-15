"""Put the HW06 book template (fonts link + style + script) into index.html, then copy index.html's
blocks verbatim into every other page. Old inline-style / inline-js blocks are removed."""
import re, sys, glob, os
d, css, js = sys.argv[1], open(sys.argv[2]).read(), open(sys.argv[3]).read()
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600'
         '&amp;family=Noto+Sans+TC:wght@400;500;700&amp;family=Noto+Serif+TC:wght@600;700&amp;display=swap">')
def strip_old(s):
    s = re.sub(r'<link rel="preconnect"[^>]*>\n?', '', s)
    s = re.sub(r'<link rel="stylesheet" href="https://fonts.googleapis.com[^>]*>\n?', '', s)
    s = re.sub(r'<style id="(inline-style|book-style)">.*?</style>\n?', '', s, flags=re.S)
    s = re.sub(r'<script id="(inline-js|book-js)">.*?</script>\n?', '', s, flags=re.S)
    return s.replace('<!-- INLINE-ASSETS -->\n', '').replace('<!-- INLINE-ASSETS -->', '')
idx = os.path.join(d, 'index.html')
s = strip_old(open(idx).read())
head = FONTS + '\n<style id="book-style">\n' + css + '</style>\n'
s = s.replace('</head>', head + '</head>', 1)
s = s.replace('</body>', '<script id="book-js">\n' + js + '</script>\n</body>', 1)
open(idx, 'w').write(s)
# copy the exact blocks from index.html into the other pages
s = open(idx).read()
head_block = s[s.index('<link rel="preconnect"'):s.index('</style>') + len('</style>\n')]
js_block = s[s.index('<script id="book-js">'):s.index('</script>', s.index('<script id="book-js">')) + len('</script>\n')]
for p in sorted(glob.glob(os.path.join(d, '*.html'))):
    if p == idx: continue
    t = strip_old(open(p).read())
    t = t.replace('</head>', head_block + '</head>', 1).replace('</body>', js_block + '</body>', 1)
    open(p, 'w').write(t)
    print('updated', os.path.basename(p))
