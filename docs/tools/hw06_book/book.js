/* HW06 讀碼教材 — 頁面腳本（定義在 index.html，其他頁逐字複製；沿用 ML2022 HW14 書的 book.js）
   pre.py 上色、pre.shell 上色、data-hot 行強調（行號取 figcaption 的起始行）、«N» 圓形標記、本章目錄、圖表 hover。 */
(function () {
  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  var PY_KW = new Set(('False None True and as assert async await break class continue def del elif else except ' +
    'finally for from global if import in is lambda nonlocal not or pass raise return try while with yield').split(' '));
  var PY_BUILTIN = new Set(('int float str list dict set tuple len range enumerate zip print open super isinstance ' +
    'max min sum abs round sorted any all bool type object self').split(' '));
  var PY_RE = new RegExp(
    '(#[^\\n]*)' +                                                   // 1 comment
    '|("""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\'|[rbfRBF]{0,2}"(?:\\\\.|[^"\\\\\\n])*"|[rbfRBF]{0,2}\'(?:\\\\.|[^\'\\\\\\n])*\')' + // 2 string
    '|(@[A-Za-z_][\\w.]*)' +                                         // 3 decorator
    '|\\b([A-Za-z_]\\w*)(?=\\s*\\()' +                               // 4 call
    '|\\b(\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?)\\b' +                  // 5 number
    '|\\b([A-Za-z_]\\w*)\\b', 'g');                                  // 6 word

  function highlightPy(text) {
    var out = '', last = 0, m;
    PY_RE.lastIndex = 0;
    while ((m = PY_RE.exec(text)) !== null) {
      out += esc(text.slice(last, m.index));
      last = PY_RE.lastIndex;
      if (m[1]) out += '<span class="tok-cmt">' + esc(m[1]) + '</span>';
      else if (m[2]) out += '<span class="tok-str">' + esc(m[2]) + '</span>';
      else if (m[3]) out += '<span class="tok-pre">' + esc(m[3]) + '</span>';
      else if (m[4]) {
        var cls = PY_KW.has(m[4]) ? 'tok-kw' : (PY_BUILTIN.has(m[4]) ? 'tok-type' : 'tok-fn');
        out += '<span class="' + cls + '">' + esc(m[4]) + '</span>';
      } else if (m[5]) out += '<span class="tok-num">' + esc(m[5]) + '</span>';
      else if (m[6]) out += PY_KW.has(m[6]) ? '<span class="tok-kw">' + esc(m[6]) + '</span>' : esc(m[6]);
    }
    return out + esc(text.slice(last));
  }

  function highlightShell(text) {
    return text.split('\n').map(function (line) {
      var m = line.match(/^(\s*)\$\s(.*)$/);
      if (m) {
        var cmd = m[2], cmt = '', k = cmd.search(/\s#\s/);
        if (k >= 0) { cmt = cmd.slice(k); cmd = cmd.slice(0, k); }
        return esc(m[1]) + '<span class="sh-prompt">$</span> <span class="sh-cmd">' + esc(cmd) + '</span>' +
          (cmt ? '<span class="sh-cmt">' + esc(cmt) + '</span>' : '');
      }
      return esc(line);
    }).join('\n');
  }

  function parseHot(spec) {
    var set = new Set();
    spec.split(',').forEach(function (part) {
      var m = part.trim().match(/^(\d+)(?:-(\d+))?$/);
      if (m) for (var i = +m[1]; i <= +(m[2] || m[1]); i++) set.add(i);
    });
    return set;
  }

  /* 行號 = figcaption 的起始行號 + 區塊內第幾行。跨行的 span（多行字串）在行尾收、下一行重開。 */
  function wrapHot(html, spec, offset) {
    var hot = parseHot(spec), carry = null;
    return html.split('\n').map(function (line, i) {
      var s = (carry || '') + line;
      var opens = s.match(/<span[^>]*>/g) || [], closes = s.match(/<\/span>/g) || [];
      if (opens.length > closes.length) { carry = opens[opens.length - 1]; s += '</span>'; } else carry = null;
      return hot.has(i + 1 + offset) ? '<span class="hot-line">' + s + '</span>' : s;
    }).join('\n');
  }

  document.querySelectorAll('pre').forEach(function (pre) {
    if (pre.classList.contains('diagram')) return;
    var target = pre.querySelector('code') || pre;
    var text = target.textContent;
    if (pre.classList.contains('shell')) target.innerHTML = highlightShell(text);
    else if (pre.classList.contains('py')) target.innerHTML = highlightPy(text);
    if (/«\d+»/.test(target.innerHTML)) target.innerHTML = target.innerHTML.replace(/«(\d+)»/g, '<span class="mk">$1</span>');
    if (pre.dataset.hot) {
      var cap = pre.closest('figure.listing') && pre.closest('figure.listing').querySelector('figcaption');
      var m = cap && cap.textContent.match(/:(\d+)/);
      target.innerHTML = wrapHot(target.innerHTML, pre.dataset.hot, m ? parseInt(m[1], 10) - 1 : 0);
    }
  });

  var main = document.querySelector('main'), header = document.querySelector('header.chap-header');
  if (main && header) {
    var hs = main.querySelectorAll('h2');
    if (hs.length >= 3) {
      var box = document.createElement('nav');
      box.className = 'toc-box';
      box.setAttribute('aria-label', '本章目錄');
      var html = '<p class="toc-box-title">本章目錄</p><ol>';
      hs.forEach(function (h, i) {
        if (!h.id) h.id = 'sec-' + (i + 1);
        html += '<li><a href="#' + h.id + '">' + esc(h.textContent) + '</a></li>';
      });
      box.innerHTML = html + '</ol>';
      header.insertAdjacentElement('afterend', box);
    }
  }
  /* 圖表的 hover：figure.fig 裡任何帶 data-tip 的元素，滑過時在 .viz-tip 顯示它的內容 */
  document.querySelectorAll('figure.fig').forEach(function (fig) {
    if (!fig.querySelector('[data-tip]')) return;
    var tip = fig.querySelector('.viz-tip');
    if (!tip) { tip = document.createElement('div'); tip.className = 'viz-tip'; fig.appendChild(tip); }
    fig.addEventListener('mousemove', function (e) {
      var t = e.target.closest && e.target.closest('[data-tip]');
      if (!t) { tip.style.display = 'none'; return; }
      tip.textContent = t.getAttribute('data-tip');
      tip.style.display = 'block';
      var r = fig.getBoundingClientRect(), x = e.clientX - r.left + 14;
      if (x + tip.offsetWidth > r.width - 8) x = e.clientX - r.left - tip.offsetWidth - 14;
      tip.style.left = Math.max(4, x) + 'px';
      tip.style.top = (e.clientY - r.top + 14) + 'px';
    });
    fig.addEventListener('mouseleave', function () { tip.style.display = 'none'; });
  });
})();
