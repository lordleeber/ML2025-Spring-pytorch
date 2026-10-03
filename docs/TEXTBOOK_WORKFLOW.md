# HW 教材的製作流程

這份文件說明 docs/HWxx/ 底下每一本 HTML 教材怎麼做。本機 session 和雲端 session 開工前都應該先讀。

這份流程沿用姊妹 repo `lordleeber/ML2022-Spring-pytorch` 的 `docs/TEXTBOOK_WORKFLOW.md`，那邊的 HW01、HW09 已經照它做完。下面只寫這個 repo 的版本，以及兩邊不同的地方。

使用的 skill 有三個：
- `completed-repo-to-html-textbook`：整體流程與共用樣式。
- `incremental-html-textbook`：章節骨架與元件用法。
- `cold-read`：冷讀。

它們都在公開的 [poyilee1030/mySkills](https://github.com/poyilee1030/mySkills) 裡，用的時候放到 repo 的 `.claude/skills/` 底下。`.claude/` 已列在 .gitignore，不 commit。

## 這個 repo 和 ML2022 的差別

- **base branch 是 `main`**（ML2022 是 master）。
- **全 repo 共用一個 venv**，放在根目錄的 `.venv`，用 uv 管理（`pyproject.toml`、`uv.lock`）。
  - 指令一律在 **repo 根目錄**下，例如 `.venv/bin/python HW03/hw3.py`。
  - 不要 `cd` 進 HW 目錄再跑。
- **作業原始碼的形式**：每個 HW 是一支腳本加上 `--q` 旗標選題。原版 Colab notebook 與投影片也放在 HW 目錄裡，例如 `HW03/hw3_colab.ipynb`、`HW03/hw3.pdf`。
- **需要 HF 授權**：Gemma 是 gated 模型，本機已登入，雲端沒有，也不需要。

## 分工（採 HW09 的「一次量完」模式）

**本機 session（有 GPU、有模型）**
- 用 `docs/tools/hwxx_facts.py` 一次把全書的事實量完，寫進 `docs/HWxx/FACTS.md`。
- 真實輸出的圖放到 `docs/HWxx/img/`，逐字的執行紀錄放到 `docs/HWxx/logs/`。
- commit、push 到 main，再給使用者一份雲端 prompt。
- PR 回來後審稿：見下面「本機審 PR」。

**雲端 session（沒有 GPU，不跑任何程式）**
- clone mySkills 到 `.claude/skills/`。
- Phase 0 先寫 `docs/HWxx/index.html` 與 `outline.html`，寫完就停，等使用者核可。
- 之後每次只寫被指定的章。
  - 數字只引用 FACTS 或 logs 裡有的。
  - 缺的數字標成 `<!-- TODO(本機實測): 要量什麼 -->`，不估、不編。
- 每章寫完依序做：
  1. `verify_book.py`；
  2. 冷讀（2 支 subagent）；
  3. `inline_assets.py`；
  4. 把新名詞補進 FACTS；
  5. 開 branch、開 PR。branch 名稱由雲端自己取，例如 `claude/hw03-ch01-xxxx`。

**本機審 PR**
1. 用 `git merge-base` 確認 PR 是從哪個 commit 開始的；如果不是最新的 main，先把 main merge 進來。
2. 每一頁跑 `python3 docs/tools/verify_book.py . docs/HWxx/<page>.html`。這支工具會依 HTML 所在的資料夾找原始碼：`docs/HW03/x.html` 對應 `HW03/<file>`。
3. 檢查每一個 `href="x.html#id"` 的錨點都存在。
4. 用 grep 核對交叉引用與程式行號。
5. 在本機量出 TODO 的數字填回去，寫進 FACTS 的「chNN 審稿補測」。
6. commit 並 push 到 PR 的 branch，再由使用者 merge。

## 沿用 ML2022 學到的教訓

- **「改 X 會怎樣」這類題目，一律實際跑一次**：照題目字面修改再跑，不靠推論。
- **避免 FACTS 衝突**：上一章的 PR merge 之後，才追加下一章的「實測」。
- **每次 commit 前用 `git status -sb` 確認目前的 branch**。
- **stdout 和 stderr 混在一起的輸出**，用 `script -qc "<指令>" <log>` 模擬終端機來抓；接到管線時順序會改變。
- **實驗用的額外套件**裝在專案 .venv 之外，例如 `uv pip install --target <暫存目錄>`；不要改 pyproject.toml。
- **樣式**：用 mySkills 新版共用資產（`assets/style.css`、`enhance.js`）。同一本書內要一致，寫進 prompt。
- **每本書的固定結構**：
  - ch00 要有「模型總覽」一節（架構圖、各層形狀、參數量、定義在哪裡）；
  - 目錄頁要有「原版 Colab vs 本 repo」的導論；
  - 各章遇到原版的寫法有問題時，加「現在的做法」或「讀 code 不盡信文件」框。

## 雲端 prompt 的寫法

每一份 prompt 都包含以下幾段：
- repo 是 `lordleeber/ML2025-Spring-pytorch`，base 是 main，以及預期的 HEAD hash。
- 「不需要環境、不要跑程式」：數字只引用 FACTS 與 logs，缺的標 TODO。
- clone mySkills 的指令；`figure.listing` 的 `data-hot` 寫原始碼行號；指令區塊用哪一種 class。
- 這次寫哪幾章、寫完就停，以及範圍（檔名:行號）。
- 前面章節答應本章要講的事（grep「第 N 章」整理出來），以及已經講過、只要回指的內容。
- 本章的主線，以及容易寫錯的地方。
- 流程：verify → 冷讀 → inline_assets.py → 補 FACTS 名詞 → 開 PR → 回報。回報包含改了哪些檔、TODO 清單、需要本機核對的地方、前面章節要修的地方。
