# HW 教材的製作流程

這份文件說明 docs/HWxx/ 底下每一本 HTML 教材怎麼做。本機 session 和雲端 session 開工前都應該先讀。

這份流程沿用姊妹 repo `lordleeber/ML2022-Spring-pytorch` 的 `docs/TEXTBOOK_WORKFLOW.md`，那邊到 2026-10-10 已經做完 HW01–04、06、07、09、10、13、14 共十本。下面只寫這個 repo 的版本，以及兩邊不同的地方；HW03 之後 ML2022 改變的做法整理在最後一節「ML2022 HW02–HW14 的做法（2026-10-10 同步）」，**HW06 起以那一節為準**。

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

## 分工（HW03 採 HW09 的「一次量完」模式）

> HW03 用這套雲端分工完成。HW06 起預設改成「本機 session 整本依序寫完」，見最後一節；下面這套仍可用，但要使用者指定。

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

## ML2022 HW02–HW14 的做法（2026-10-10 同步）

ML2022 在 HW03 之後又寫了八本書，做法有幾處改變。完整的教訓（每本書一節）在 [ML2022 的 TEXTBOOK_WORKFLOW.md](https://github.com/lordleeber/ML2022-Spring-pytorch/blob/master/docs/TEXTBOOK_WORKFLOW.md)，這裡只摘要會影響這個 repo、特別是 LLM 作業的部分。

### 流程
- **整本由同一個本機 session 依序寫完**，不走雲端分工（使用者決定，HW02 起）。
  - Phase 0：讀程式、建參照版、跑 baseline、檢查量法，寫大綱（outline.html）與事實清單（FACTS.md），**等使用者核可**才開始寫章。
  - 每章寫完 → `verify_book.py` →（字數用 cjk 計數檢查，每章 3,000+ 中文字）→ 回報。
- **推送只在使用者明講「推」時**：一章一個 commit、一次推一個；「繼續」只代表寫下一章。使用者授權「每章寫完驗證就推、接著寫下一章」時，照授權的字面範圍做。
- **冷讀**：ML2022 HW03 之後各書都由使用者決定不冷讀；預設仍是要跑，開書時問一次。
- **長任務每 30 分鐘回報一次**（session 內的 cron）。
- **後面的章推翻前面章節的說法時**，先問使用者，再做點狀修改並重驗那幾章。

### skill 版本（開新書前先對照）
- 現行的 `completed-repo-to-html-textbook` 已經改成：**不寫附錄**（名詞對照、速查併入最後的總結章）；**不用共用的 style.css／enhance.js／inline_assets.py**，每本書自己設計樣板、寫在 index.html。HW03 是舊版 skill 的產物（有 appendix、用共用樣式），新書不要照抄它的結構。
- 這個 repo 的 `.claude/skills/` 是舊版，開新書前先換成 mySkills 的最新版。
- ML2022 各書的樣板與工具可以參考：`docs/tools/hw10_book/`（`make_charts.py` 依 `<!-- CHART:名稱 -->` 放 SVG 圖、`fill_listings.py` 依 figcaption 的「檔名:行號」從原始碼重填程式碼）。配色每本書重新設計。
- 渲染可以在本機看：Windows 的 Chrome headless 截 WSL 裡的頁面（`chrome.exe --headless=new --user-data-dir=<scratch> --window-size=1200,N --screenshot=<path> file://wsl.localhost/...`）。
- 樣板要 `font-variant-ligatures: none`（JetBrains Mono 會把 `!=` 畫成 ≠）。
- 資料圖照 dataviz skill，附 hover 與同數字的表格。

### 拆檔與參照版
- **notebook 先做「參照版」**：程式格原樣串成腳本（只拿掉 `!` 指令、補上跑不起來的 import），拆檔後的程式與實驗工具都對它做逐位元比對（權重 `torch.equal`、輸出 `diff`）。
- **被刪掉的 API 原樣抄出來**，不換成新 API（預設值不同就會改變結果；HW07 的 `transformers.AdamW` 抄成 `legacy_adamw.py`）。
- **只為了 print 的程式也可能改變結果**：建模型、torchsummary 等都會用掉全域亂數，拆檔時保留原位置。
- **實驗工具連進度條一起複製**：`tqdm.auto` 會多抽一次亂數；工具用 `tqdm(..., disable=True)` 包住同樣的迴圈。
- **引用實驗工具的行號後就不要改那幾行**：要加功能就加在別處，或用包裝腳本 monkeypatch 後 `runpy` 執行。
- **環境**：作業本身需要的套件版本照 README（不相容時給該 HW 自己的 `pyproject.toml`＋venv）；只為了實驗或量測多裝的套件，裝在 .venv 之外（`uv pip install --target <暫存目錄>`），不改 pyproject。舊套件跑不起來時用 `sitecustomize.py` shim，不改套件原始碼。

### 可重現性與量法
- **GPU 上固定種子不等於可重現**（HW07：BERT 訓練跑兩次 100 步就分岔）。拆檔比對、工具驗證都在 `torch.use_deterministic_algorithms(True)`（加 `CUBLAS_WORKSPACE_CONFIG=:4096:8`）下做，用一個不改原檔的 wrapper（ML2022 的 `docs/tools/hw07_det.py`）。
- **先量種子雜訊**：每個比較至少 3 個種子；差距小於雜訊不下結論。
- **量法本身要先查**：印出的指標和「最佳 checkpoint 在整個評估集一次算完」的真實值比；評估工具本身（第三方套件、審查模型）也要先驗證、量它的偏差。
- **一律量「交出去的東西」**：存成檔案再讀回來評估。
- **「全部一樣」的數字要先懷疑程式。**
- **任何會跑 forward 的檢查，先 `eval()`**；量 train／eval 差別時每種模式各用一個新載入的模型。
- **「同一個 checkpoint、只改評估」的實驗很便宜，先做**；解釋要先寫下可檢驗的預測再跑。
- **對照組要和被比較的設定同一套訓練方式**，參數量也要控制；結果出來後發現分不開的，問使用者加跑。
- **「改 X 會怎樣」的題目照樣實測**；沒實測的答案在題目裡註明「本書沒有實際跑」。

### GPU 與背景工作
- 使用者核可實驗規格後，規格內的實驗可以**一組一組依序**自己排；規格外的加跑先問。
- **背景佇列只起一個**：寫成腳本檔、用 `setsid nohup` 起，起完用 `ps -eo pid,ppid,cmd` 確認只有一個。**不要用 `pgrep -f`／`pkill -f`**（會比對到自己的 shell，ML2022 犯過兩次），要停自己的佇列用 `ps -eo pid,args | awk` 精確比對取 PID。
- **計時排在所有實驗之後**，GPU 上不能有其他工作，前後記 `nvidia-smi --query-compute-apps`；並行時量到的時間在 FACTS 註明不可用。
- checkpoint 只放 scratchpad 或 gitignore 的目錄，**數字一律即時寫進 FACTS 與 `docs/tools/hwNN_*runs.jsonl`**（scratchpad 可能在 session 中途被清空）。
- `os._exit(0)` 前先 `sys.stdout.flush()`。

### 書的固定結構
- ch00 有「模型總覽」一節（架構圖、各層形狀、參數量、定義在哪個檔案）。
- 目錄頁有「原版 vs 現在」的導論；各章遇到過時的寫法時加「現在的做法」框。
- 最後一章是總結章：三級題庫、名詞對照、速查。

