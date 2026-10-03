# HW03 教材事實清單（維護筆記，不進教材）

> **這份檔案是什麼**：docs/HW03/ 這本教材背後的事實清單。教材裡的每一個數字、每一段逐字輸出，都要能在這裡、`docs/HW03/logs/` 或 repo 原始碼找到出處。這份檔案本身不是教材，HTML 不會連到它。
>
> **寫作分工**：本機（有 GPU）已在 2026-10-04 把全書需要的數字一次量完。
> - 量到的數字寫在這裡；逐字輸出放在 `docs/HW03/logs/`。
> - 雲端 session 沒有 GPU、沒有環境、也沒有 Gemma 授權，不跑程式，只引用這裡與 logs 的內容寫章。
> - 這裡找不到的，標 `<!-- TODO(本機實測): 要量什麼 -->`，PR 回來後由本機補。
>
> **重現方法**（需要 GPU 與 HF 授權）：在 repo 根目錄執行 `.venv/bin/python docs/tools/hw03_facts.py [env attn model tok q1 q2 q4 q5 q6 q7]`。
> - 輸出整理成下面各節，原始輸出在 `logs/facts_*.txt`。
> - 逐字執行紀錄是用 `script -qc "<指令>" <log>` 抓的，再把 `\r` 覆寫的進度條只留最後狀態，存成 `logs/run_*.txt`。

## 版本與原始碼位置

- 教材對應的程式碼：`HW03/hw3.py`，在 commit `f02fbdf` 加入，PR #1 merge 成 `807b402`，之後沒有改過。
- 作業原始材料（都已 commit 在 repo 裡）：
  - 官方原版 Colab：`HW03/hw3_colab.ipynb`，39 格，沒有輸出。
  - 作業投影片：`HW03/hw3.pdf`，35 頁。
  - 原始連結：https://colab.research.google.com/drive/1Ku_p27ml8QJ-Rd7FilZQvDA9axe68V7o
- 課程頁：https://speech.ee.ntu.edu.tw/~hylee/ml/2025-spring.php 。
  - 作業名稱：HW3 "Understanding transformers"。
  - 助教：傅啟恩、李冠儀、許景淯。
  - 期限 2025/04/04。
- 計分方式（投影片 p.4、p.32）：
  - Coding & Answer Question 8%：7 大題，都在 Gradescope 作答，選擇題與填空題；
  - Paper reading 2%：讀 3 篇論文，答 Problem 8–9。
  - **Gradescope 題目的選項不在投影片與 Colab 裡**，教材不要編造選項，也不要寫成「標準答案」。
  - 投影片 p.34 禁止分享答案，所以教材的定位是「理解每一題在量什麼，以及本機實測的數字」。

## 全書約定

- **樣式**：用 mySkills 新版共用資產，即 `completed-repo-to-html-textbook/assets/style.css`、`enhance.js`。
  - 指令塊：要讀者貼上執行的用 `<pre class="shell cmd">`，輸出用 `<pre class="shell">`，Python 用 `<pre class="py">`。
- **listing**：`data-hot` 寫**原始碼行號**，與 figcaption 的 `檔名:起–迄` 同一套。
  - 這本書的 listing 檔名寫 `hw3.py:起–迄`。
  - `verify_book.py` 會從 `docs/HW03/` 對應到 `HW03/hw3.py`；引用原版 notebook 的程式不要用 listing（它不是行號可對的檔案），改用一般 `<pre class="py">`，並註明「原版 Colab 第 N 格」。
- **檢查**：`python3 docs/tools/verify_book.py . docs/HW03/chNN.html`。
- **指令一律在 repo 根目錄下跑**，例如 `.venv/bin/python HW03/hw3.py --q 4`。這個 repo 全部 HW 共用根目錄的 `.venv`。
- **圖**：
  - 機制圖照 skill 規定手繪 inline SVG。
  - **結果圖**直接用 `<img src="img/xxx.png">` 引用真的輸出，不要手畫，例如 top-10 機率長條圖、t-SNE、attention heatmap、SAE activation。
  - 圖檔清單見最後一節。
- **每本書的固定結構**（使用者指定）：
  - ch00 要有「模型總覽」：Gemma-2-2b-it 的架構 SVG、每層 tensor 形狀、參數量（手算加上 PyTorch 印出的數字）、模型從哪裡來（HF `google/gemma-2-2b-it`，由 `AutoModelForCausalLM` 載入，見 hw3.py:49–59）。
  - 全書開頭有「原版 Colab vs 本 repo」段落。
  - 原版寫法有問題的地方，加「現在的做法」或「讀 code 不盡信文件」框。

## 環境（實測 2026-10-04）

- **軟體**：
  - Python 3.12.3、torch 2.11.0+cu128、transformers **4.47.0**（助教指定，Colab 第 4 格：`!pip install transformers==4.47.0`，「please do not alter the version」）。
  - tokenizers 0.21.4、accelerate 1.15.0、huggingface_hub 0.36.2。
  - sae-lens 5.11.0、transformer-lens 2.15.4（sae-lens 帶進來的）。
  - nltk 3.10.3、scikit-learn 1.9.1、numpy 1.26.4、matplotlib 3.11.2、seaborn 0.13.2。
- **GPU**：NVIDIA RTX PRO 4000 Blackwell，sm_120，23.9 GiB，CUDA 12.8，WSL2。
  - Blackwell 需要 cu128 的 wheel，所以 pyproject 指定 PyTorch 的 cu128 index。
- **依賴管理**：用 uv。
  - `pyproject.toml` 與 `uv.lock` 在 repo 根目錄，`uv sync` 會建立 `.venv`。
  - `sae-lens<6`：v6 改了 `SAE.from_pretrained` 的回傳值與 `cfg.hook_layer`，所以釘在 5.x。原版 Colab 只寫 `!pip install sae-lens`，沒有指定版本。
- **HF 模型與授權**：
  - `google/gemma-2-2b-it` 是 gated 模型：要先在 HF 網頁接受授權，再用 read token 登入（投影片 p.5–11）。
  - 登入方式：`.venv/bin/hf auth login`，或設環境變數 `HF_TOKEN`。hw3.py:509–512 有 `HF_TOKEN` 就呼叫 `login()`。
  - 在 WSL 裡，token 存在 WSL 的 `~/.cache/huggingface/token`，Windows 那邊的登入不算。
- **HF 快取實際大小**：gemma-2-2b-it 5.251 GB；gemma-scope-2b-pt-res 0.302 GB（只抓了 layer_20/width_16k 那一個 SAE）；cross-encoder/ms-marco-MiniLM-L-6-v2 0.092 GB。
- **執行時間**：`.venv/bin/python HW03/hw3.py --seed 0`（全部 7 題）real **45.7 s**，模型都已在快取裡。
  - `load_model` 約 2 s。
  - Q4 的 40 次取樣約 25 s，是最慢的一題。
- **stderr 的固定雜訊**：
  - transformers 4.47 的 HybridCache 會印兩條 deprecation：`The 'batch_size' argument/attribute of HybridCache is deprecated ...`，來自 transformers 內部，與 hw3.py 無關；hw3.py 自己用的是 `max_batch_size`。
  - nltk 的 `UserWarning: The hypothesis contains 0 counts of N-gram overlaps`，Q4 算 BLEU 時出現。
  - 都可以忽略。

## 程式結構（HW03/hw3.py，521 行）

- hw3.py:1–13：模組 docstring，列出與 Colab 的差異。
- hw3.py:34–38：常數 `MODEL_ID`、`SCORING_MODEL_ID`、`DEVICE="cuda"`、`DTYPE=torch.float16`、`OUT_DIR`。
- 共用函式：
  - `save_fig` :41–46；
  - `load_model` :49–59，`attn_implementation="eager"` 在 :56。
- Q1：
  - `calculate_coherence` :65–70；
  - `generate_text_from_prompt` :73–85，TODO 在 :79–81；
  - `q1` :88–108。
- Q2：`Q2_TURNS` :114–118；`q2` :121–165。
- Q3：`q3` :171–182，TODO 在 :174–179。
- Q4：
  - `compute_self_bleu` :188–197；
  - `q4` :200–250，TODO 在 :203–206、:230–232；
  - `generation_params` :211–221。
- Q5：`q5` :256–295。
- Q6：`plot_attention` :301–309；`q6` :312–366，TODO 在 :339–344。
- Q7：
  - `load_sae` :372–380；
  - `get_dashboard_html` :383–385；
  - `get_max_activation` :388–406；
  - `plot_token_activations` :409–429；
  - `plot_layer_activations` :432–453；
  - `q7` :456–478。
- 主程式：`QUESTIONS` :481；`main` :484–517，argparse 旗標在 :486–503。
- **CLI 旗標與預設值**：
  - 選題：`--q 1..7`（可多個，預設全部）、`--seed`（預設 None，即不設）。
  - Q1：`--max-new-tokens 512`。
  - Q2：`--interactive`。
  - Q3：`--sentence`。
  - Q4：`--top-k 2`、`--top-p 0.6`、`--num-samples 20`。
  - Q6：`--layer-idx 10`、`--head-idx 7`。
  - Q7：`--sae-layer-idx 24`、`--token-idx 1`（nargs+）。
- 各題 TODO 的填法（本 repo 的解答）：
  - Q1：`model.generate(input_ids, max_new_tokens=..., do_sample=False)`；
  - Q3：`tokenizer.encode(sentence, add_special_tokens=False)` 加 `tokenizer.convert_ids_to_tokens(token_ids)`；
  - Q4：`model.generate(input_ids=..., top_p=top_p, **generation_params)`；
  - Q6：`outputs.logits[:, -1, :]` 與 `outputs.attentions`；
  - Q7.1：印出 Neuronpedia 網址。

## 原版 Colab vs 本 repo（「原版 vs 本 repo」段落與「現在的做法」框的素材）

- **Colab 專屬的格子拿掉**：`!nvidia-smi`、`!pip install`、`login("your_hf_token")`（改讀 `HF_TOKEN` 或 `hf auth login` 的快取）、`IPython.display.IFrame`。
- **一支腳本取代整本 notebook**：用 `--q` 選題，每題是一個函式 `q1`…`q7`。模型只載入一次，所有題目共用。
- **畫圖**：`plt.show()` 改成 `save_fig()`，存到 `HW03/outputs/`（已 gitignore），並 `matplotlib.use("Agg")`。圖檔名帶上參數，例如 `q6_attention_L10_H7.png`、`q2_round3_top_tokens.png`。
- **Q2**：
  - 原版是 `while True: input("You: ")` 互動式聊天；本 repo 預設照投影片 p.14 的三句話自動跑完，`--interactive` 才互動。
  - 本 repo 另外印出 top-10 機率表。
  - 原版 softmax 在 fp16 上做；本 repo 先 `.float()` 再 softmax（hw3.py:144）。實測兩者到小數第 4 位最多差 0.0002（round 3 的 Green：fp32 0.5125、fp16 0.5127），不影響排名。
  - 原版的 `sns.barplot(..., palette=...)` 沒給 `hue`，seaborn 0.13 會警告；本 repo 加了 `hue=top_tokens, legend=False`。
- **Q4**：
  - 原版建了一個 `kv_cache = HybridCache(...)` 卻從來沒用（`model.generate` 自己管 cache），本 repo 刪掉。
  - 次數 20 改成 `--num-samples`。
- **Q5**：原版 `hidden_states.mean(dim=1).cpu().numpy()`；本 repo 多一個 `.float()`，因為 fp16 也能轉 numpy，但 t-SNE 用 fp32 比較穩。
- **Q6**：
  - 原版 `.detach().cpu().numpy()`；本 repo 多 `.float()`。
  - 原版 `model.eval()` 寫在 Q6 格子裡；本 repo 在 `load_model` 統一呼叫。
- **Q7**：
  - 原版在每個函式裡 `sae.to(device)`；本 repo 在 `load_sae` 一次 `.to(DEVICE)`。
  - 本 repo 加 `@torch.no_grad()`（原版沒有，所以原版會建 autograd graph，浪費記憶體）。
  - 原版 `.cpu().detach().numpy()`；本 repo改成 `.float().cpu().numpy()`。
  - 本 repo 把每個 token 的數值印成表格，並且 `--token-idx` 可以一次給多個。
- **`attn_implementation="eager"`**：
  - 原版沒寫。
  - 實測 transformers 4.47 不寫時，Gemma2 預設就是 `eager`（logs/facts_env_model_tok.txt 的 attn 段），所以行為相同；本 repo 寫明是為了 Q6 一定拿得到 attention weights。
  - 背景：Gemma 2 有 attention logit soft-capping（`attn_logit_softcapping = 50.0`）。transformers 4.47 的 `Gemma2PreTrainedModel._check_and_enable_sdpa`（`.venv/lib/python3.12/site-packages/transformers/models/gemma2/modeling_gemma2.py:553–562`）把預設的 sdpa 換成 eager，註解寫的是「SDPA reduces the model performance on Gemma2 because of the logits softcapping」。同檔 :185–188 是 eager 路徑裡的 soft-capping：`attn_weights / 50 → tanh → × 50`。
- **刻意保留不改的「怪地方」**（docstring hw3.py:11–12 說明：保留是為了讓結果和助教參考答案一致）：
  - Q1、Q2 的雙重 `<bos>`；
  - Q4 的 top-p 沒有關掉預設的 top-k=50；
  - Q5 的 mean pooling 包含 padding；
  - Q6 的 heatmap 標籤錯位；
  - Q7 的 hidden_states 索引差一。
  - 每一條的實測見下面的「repo 問題清單」。

## 投影片與程式不一致的地方（「讀 code 不盡信文件」素材）

1. **Q4 prompt**：
   - 投影片 p.18 寫 `'Professor Lee is one of the best teachers ...'`；
   - 程式（Colab 與 hw3.py:208）是 `'Professor Hung-yi Lee is one of the best teachers ...'`。
   - 以程式為準。
2. **Q3 的參考表（p.17）有兩個錯字**：
   - `Token: you, token index: 692`：692 其實是 `▁you`（前面有空白記號）。不帶空白的 `you` 是 4747。
   - `Token: ?, token index: 23533`：實際 `?` 是 **235336**；23533 解碼出來是 `▁Operation`。
   - 實測輸出見 logs/run_seed0.txt 的 Q3 段。
3. **Q4 投影片 p.18 說要比較** k=2 vs k=200、p=0.6 vs p=0.999；p.19 的填空題問 k=1 與 p=0 的生成句。程式預設只跑 `top_k=2`、`top_p=0.6`，其他值要自己改。
   - 本 repo 用旗標改，例如 `--top-k 200 --top-p 0.999`。
4. **Q7.4–7.6 投影片 p.26 說「layer 24」**：
   - 程式傳的是 `outputs.hidden_states[24]`，等於 **第 23 個 decoder block（0 起算）的輸出**，見 repo 問題 5。
   - 而且 SAE 是在 block 20 的輸出上訓練的，拿去編碼別層的 hidden state，本來就是「把量尺用在它沒校正過的地方」。
5. **Q2 投影片 p.14 的對話範本**寫 `Model 1st output : xxxx`。實測模型三輪依序回 `Indigo`、`Orange`、`Green`，見下面 Q2 實測。

## repo 問題清單（原版就有、本 repo 刻意保留的行為；每條都實測過）

1. **雙重 `<bos>`（Q1、Q2）**：
   - `apply_chat_template(tokenize=False)` 產生的字串已經以 `<bos>` 開頭，接著 `tokenizer(prompt)` 又在前面加一個。
   - 實測 Q1 的 input_ids 前 6 個是 `[2, 2, 106, 1645, 108, 5958]` → `['<bos>', '<bos>', '<start_of_turn>', 'user', '\n', 'Please']`，共 32 個 token。
   - 用 `apply_chat_template(tokenize=True)` 是 31 個，只有一個 `<bos>`。
   - **影響（實測，greedy、max_new_tokens=512）**：
     - 雙 `<bos>`：生成 240 個 token，coherence **6.0734**；
     - 單 `<bos>`（`add_special_tokens=False`）：生成 161 個 token，coherence **6.1188**，回答內容也不同。
     - 完整文字見 logs/facts_q1_q2.txt。
   - 正確寫法：`tokenizer(prompt, add_special_tokens=False)`，或直接 `apply_chat_template(..., tokenize=True, return_tensors="pt")`。
2. **Q4 的 top-p 其實是「top-k=50 加上 top-p」**：
   - `model.generate(top_p=0.6, do_sample=True)` 沒指定 top_k，就用 `GenerationConfig` 的類別預設 **top_k=50**（實測 effective config：`top_k=50 top_p=0.6 temperature=1.0`）。
   - **影響（實測）**：
     - p=0.6 時，加不加 `top_k=0`，5 個 seed 的 self-BLEU 完全相同（因為 60% 機率質量本來就落在前 50 個 token 內）；
     - p=0.999 時就有差：seed 0 是 0.1257（預設 top_k=50）vs 0.1343（`top_k=0`），seed 3 是 0.1371 vs 0.1329。
3. **Q4 的 `eos_token_id` 被覆蓋**：
   - `generation_params` 傳 `eos_token_id=tokenizer.eos_token_id`，也就是 **1**（`<eos>`）；
   - 模型自己的 `generation_config.eos_token_id` 是 `[1, 107]`，107 是 `<end_of_turn>`。
   - 所以 Q4 遇到 `<end_of_turn>` 不會停，會繼續寫，直到 `<eos>` 或 30 個新 token 的上限。
   - 實測 seed 0、top_k=2 的第 0 句原始 decode：`' \n\nProfessor Lee is highly regarded as a leading expert in machine learning education. \n<end_of_turn><eos>'`。這句模型自己在 `<end_of_turn>` 後面接著吐了 `<eos>`，才停下來。
   - 有些句子會一路寫到 30 個 token 上限，後面接 `Here's why this paraphrase works:` 之類的贅詞，見 Q4 實測。
4. **Q4 沒有用 chat template**：
   - prompt 是純文字（hw3.py:208–209），所以每句開頭都是 `' \n\n'`，結尾是 `' \n'`；`.strip()` 會把它們去掉。
   - greedy 結果：`' \n\nProfessor Lee is highly regarded as a leading expert in machine learning education. \n'`。
5. **Q7 的 hidden_states 索引差一**：
   - `outputs.hidden_states` 有 **27** 個：
     - `[0]` 是 embedding × √2304（實測 `torch.equal` 為 True）；
     - `[i]`（1 ≤ i ≤ 25）是 `layers[i-1]` 的輸出（實測 1、20、21、25 都 True）；
     - `[26]` **不是** `layers[25]` 的原始輸出，而是經過最後 `model.norm` 的結果（實測 `norm(output of layers[25])` 為 True）。
   - SAE 的 `cfg.hook_name = 'blocks.20.hook_resid_post'`，也就是 block 20 的輸出，對應 `hidden_states[21]`。
   - 但程式用 `hidden_states[sae.cfg.hook_layer]` = `hidden_states[20]` = **block 19 的輸出**。
   - **影響（實測，feature 10004）**：
     - prompt a 的最大值：[20] 是 **58.1038**（hw3.py 印的），[21] 是 71.2747；
     - 重建誤差 FVU：[20] 0.463，[21] 0.279。在正確的層上 SAE 重建得好很多。
   - 同理，Q7.7 圖的 x 軸「Layer」其實是 hidden_states 的索引 0–26，不是 block 編號；而且 26 是 norm 之後的值。
6. **Q7.2–7.3：prompt b 的最大值來自 `<bos>`，不是任何一個字**：
   - hw3.py 印 `max_activation for prompt_b: 31.740745544433594`。
   - 實測 prompt b 只有 `<bos>` 這個位置非零（31.74），其餘 21 個 token 全是 0。
   - prompt a 的最大值 58.10 在 `▁travel`；a 的 `<bos>` 也是 31.74。
   - `<bos>` 位置的 hidden state 範數特別大（下面「模型」一節），SAE 對它的輸出不代表語意。
   - 「比較兩句的最大值」時，b 的 31.74 是 `<bos>` 帶來的假訊號。
7. **Q6 heatmap 的列／欄標籤錯位一格**：
   - 矩陣的 22 列，依序是實際餵進模型的 `<bos>`、`Google`、`▁`，加上前 19 個生成的 token；
   - 標籤用的卻是 `tokenizer.tokenize(full_text)`：沒有 `<bos>`，而且多了第 20 個生成的 token（`▁`）。
   - 結果每一列、每一欄的標籤都往前錯一格，對照表見 Q6 實測。
   - 例如 hw3 圖上 y 軸「Google」那一列（第 0 列）幾乎全部注意力都在第 0 欄，那其實是 **`<bos>` 注意自己**。
   - 正確標籤的版本：img/exp_q6_attention_true_labels.png。
8. **Q5 的 mean pooling 包含 padding**：
   - Gemma tokenizer 的 `padding_side='left'`，6 句話 padding 到 9 個 token。各句的 pad 數：2、2、0、1、2、2。
   - `hidden_states.mean(dim=1)` 連 `<pad>` 位置一起平均。pad 位置一樣有 hidden state，範數約 108.9。
   - **影響**：餘弦相似度數字有變，但「最近鄰」關係不變（見 Q5 實測）。只排除 pad 的 t-SNE 圖：img/exp_q5_tsne_masked.png。
9. **`--num-samples 1` 會除以零**：
   - `compute_self_bleu` 對每句拿「其他句」當 reference，只有 1 句時 references 是空的，`sum([])/len([])` 丟出 ZeroDivisionError。
   - 實測 log：logs/run_q4_k1_p0_n1_seed0.txt。
   - 原版也一樣，因為原版寫死 20 次，所以碰不到。
   - HW03/README.md 已改成建議 `--num-samples 2`；hw3.py 沒改。
10. **`--seed` 只能讓「同一條指令」重現**：
    - `.venv/bin/python HW03/hw3.py --q 4 --seed 0` 的 top-k 20 句，與 facts 腳本以 seed 0 重跑的結果逐句相同。
    - 但 `hw3.py --seed 0`（全部題目，Q1–Q3 先跑）的 Q4 和只跑 `--q 4 --seed 0` 不同：top-k 第 13 句不一樣，self-BLEU 是 0.2020 vs 0.2029。
    - 原因推測是前面幾題的 GPU 運算改變了狀態，屬推論，沒有追。
    - 教材不要宣稱「設了 seed 就一定一樣」。

## 模型（ch00 模型總覽用；logs/facts_env_model_tok.txt）

- **config**：
  - `Gemma2ForCausalLM`；26 層 decoder；hidden_size 2304；intermediate_size 9216；
  - num_attention_heads 8、num_key_value_heads 4（GQA，2 個 query head 共用一組 K/V）、head_dim 256；
  - vocab_size 256000；max_position_embeddings 8192；sliding_window 4096；
  - attn_logit_softcapping 50.0；final_logit_softcapping 30.0；query_pre_attn_scalar 256；
  - rope_theta 10000；rms_norm_eps 1e-6；hidden_activation `gelu_pytorch_tanh`；
  - **tie_word_embeddings True**：lm_head 與 embed_tokens 共用同一塊記憶體（實測 `data_ptr` 相同）。
  - config.json 原本的 torch_dtype 是 bfloat16；本作業用 float16 載入，跟原版 Colab 一樣。
- **Sliding 與 global 交替**：`is_sliding = not bool(layer_idx % 2)`，偶數層（0, 2, …, 24）是 sliding window（4096），奇數層是 global。
  - 實測 layer 10 是 sliding、layer 25 是 global。
  - 本作業的序列都遠短於 4096，所以兩種層的行為其實一樣。
- **參數量（PyTorch 數的）**：
  - 總計 **2,614,341,888**；
  - embed_tokens 589,824,000（= 256000 × 2304，佔 22.6%）；
  - 非 embedding 2,024,517,888。
- **每層 77,865,984**：
  - attention：q_proj (2048, 2304)、k_proj (1024, 2304)、v_proj (1024, 2304)、o_proj (2304, 2048)，合計 14,155,776；
    - 2048 = 8 heads × 256；1024 = 4 KV heads × 256。
  - MLP：gate_proj、up_proj 都是 (9216, 2304)，down_proj (2304, 9216)，各 21,233,664，合計 63,700,992。
  - 4 個 RMSNorm 各 2304：input、post_attention、pre_feedforward、post_feedforward，「三明治」norm。
  - 26 層合計 2,024,515,584；加上最後的 norm 2,304，等於非 embedding 的總數。
- **記憶體**：fp16 權重 4.87 GiB；載入後 `torch.cuda.memory_allocated()` 也是 4.87 GiB。
- **`print(model.model.layers[0])` 逐字**：在 logs/facts_env_model_tok.txt，包含 `Gemma2RotaryEmbedding`、`PytorchGELUTanh`。
- **hidden state 的 L2 範數**（句子 "Time travel will become a reality."，表中數字是「非 `<bos>` token 平均 / `<bos>`」）：

  | hidden_states 索引 | 非 `<bos>` 平均 | `<bos>` |
  |---|---|---|
  | [0] | 85.9 | 200.0 |
  | [1] | 80.1 | 367.9 |
  | [5] | 91.6 | 1030.1 |
  | [10] | 184.7 | 1902.2 |
  | [15] | 291.7 | 2849.9 |
  | [20] | 397.2 | 3731.5 |
  | [21] | 430.3 | 3807.3 |
  | [24] | 591.7 | 4157.9 |
  | [25] | 740.2 | 4244.0 |
  | [26]（norm 後） | 104.4 | 98.7 |

  - `<bos>` 的範數從第 5 層起大約是其他 token 的 10 倍。
  - 這和 Q6 的「大家都注意 `<bos>`」（attention sink），以及 Q7 的「`<bos>` 有大 activation」是同一件事的不同面向。
  - 這句屬推論，教材要標明是推論。

## Tokenizer（logs/facts_env_model_tok.txt）

- **基本資訊**：`GemmaTokenizerFast`，len 256000，`padding_side='left'`。
- **特殊 token**：`<pad>`=0、`<eos>`=1、`<bos>`=2、`<unk>`=3、`<start_of_turn>`=106、`<end_of_turn>`=107、`'\n'`=108、`'▁'`=235248。
- **chat_template 全文**：見 log。重點：
  - 開頭 `{{ bos_token }}`；
  - system role 直接 `raise_exception('System role not supported')`；
  - user 與 assistant 必須交替；
  - assistant 的 role 名稱改寫成 `model`；
  - 每則訊息是 `'<start_of_turn>' + role + '\n' + content | trim + '<end_of_turn>\n'`；
  - `add_generation_prompt` 時補上 `'<start_of_turn>model\n'`。
- **Q1 的 prompt_with_template**（repr）：`'<bos><start_of_turn>user\nPlease tell me about ... Answer in 200 words.<end_of_turn>\n<start_of_turn>model\n'`。
- **純文字問題**：23 個 token，開頭 `['<bos>', 'Please', '▁tell']`。
- **各種字串的切法**（`encode(add_special_tokens=False)`）：
  - `'Google '` → `[('Google', 12583), ('▁', 235248)]`；`'Google'` → 只有 12583。
  - `'Hung-yi'` → Hung 42599、`-` 235290、yi 12636；`' Hung-yi Lee'` → ▁Hung 18809、-、yi、▁Lee 9201。
    - 前面有沒有空白，`Hung` 的 id 不同：42599 vs 18809。
  - `'機器學習'` → 機器 100697、學習 88900；`'李宏毅'` → 李 236454、宏 238805、毅 239597，三個字各一個 token。
  - `'2025'` → 2、0、2、5，每個數字一個 token；`'12345'` 同理，切成 5 個。
  - `'unsupervised'` → un 549、supervised 122612。
  - Q7 的句子 `'Time travel will become a reality as technology continues to advance.'` → Time 2282、▁travel 5056、▁will 877、▁become 3831、▁a 476、▁reality 10546、▁as 685、▁technology 6178、▁continues 11614、▁to 577、▁advance 8484、. 235265。
    - 加上 `<bos>` 共 13 個 token。
- **Q3 實測**（logs/run_seed0.txt）：18 個 token。
  - I 235285、▁love 2182、▁taking 4998、▁a 476、▁Machine 13403、▁Learning 14715、▁course 3205、▁by 731、▁Professor 11325、
  - **▁Hung 18809**、**- 235290**、**yi 12636**、
  - ▁Lee 9201、, 235269、▁What 2439、▁about 1105、▁you 692、? 235336。
  - 投影片 p.17 的四個空格答案：(1) 18809、(2) `-`、(3) `yi`、(4) 12636。這屬於作業答案，教材是否直接寫出，由使用者決定；預設不寫，只示範怎麼跑出來。
  - `▁` 是 SentencePiece 表示「前面有空白」的記號；`I` 在句首，前面沒有空白，所以沒有 `▁`。

## Q1 實測（chat template；logs/facts_q1_q2.txt、logs/run_seed0.txt）

- **coherence 評分模型**：`cross-encoder/ms-marco-MiniLM-L-6-v2`。
  - 22,713,601 個參數，num_labels 1，`model_max_length` 512，**跑在 CPU**（程式沒有 `.to("cuda")`）。
  - 它輸出的是 (question, answer) 這一對的相關性 logit，沒有上下限，不是機率。
- **greedy（`do_sample=False`）可以完全重現**：全程跑兩次，數字一樣。

  | 條件 | 新 token 數 | 停在 | 秒數 | 回答字數 | 評分器 token 數 | coherence |
  |---|---|---|---|---|---|---|
  | 有 template（雙 `<bos>`，即 hw3.py） | 240 | `<end_of_turn>` | 6.8 s | 172 | 285 | **6.0734** |
  | 有 template，單 `<bos>`（實驗） | 161 | `<end_of_turn>` | 3.9 s | 127 | 204 | 6.1188 |
  | 沒有 template（hw3.py） | 138 | `<end_of_turn>` | 3.3 s | 87 | 189 | **5.0104** |

  - 512 個新 token 的上限三種都沒碰到。評分器 512 token 的截斷也都沒碰到。
  - 「Answer in 200 words」三種都沒做到：172、127、87 字。
  - 短版本（`--max-new-tokens 128`）2026-10-03 跑過一次：有 template 6.0589、沒有 template 4.0062，回答被截斷，所以分數不同。教材以 512（預設）為準。
- **回答的抽取方式**：
  - 有 template：`decode(skip_special_tokens=True)` 會把 `<start_of_turn>` 拿掉，留下 `'user\n...words.\nmodel\n<回答>'`，所以用 `split("model\n")[-1]` 取回答。
  - 沒 template：用問題最後一個字 `'words.'` 切開（hw3.py:105）。
  - 兩種都依賴「回答裡不會出現 `model\n` 或 `words.`」這個假設；這次的回答剛好都沒有。
- **兩種回答的形狀**（全文見 log）：
  - 有 template：散文段落，有「Think of it like teaching a child」這類比喻，最後是 `**Key differences:**` 條列。
  - 沒 template：直接 `**Supervised Learning:**` 條列，Labeled data / Goal / Examples。
  - 沒 template 的那次，模型最後也吐了 `<end_of_turn>` 才停：IT 模型就算沒套 template，仍然會用對話格式收尾。

## Q2 實測（多輪對話；logs/run_seed0.txt、logs/facts_q1_q2.txt）

- **每輪 prompt 的 token 數**：round 1 是 27、round 2 是 59、round 3 是 82。每輪都是雙 `<bos>`，見問題 1。
- **round 3 的完整 prompt**（印出的字串，只有一個 `<bos>`；雙 `<bos>` 是 tokenize 時才多出來的）：
  ```
  <bos><start_of_turn>user
  Name a color in a rainbow, please just answer in a word without any emoji.<end_of_turn>
  <start_of_turn>model
  Indigo<end_of_turn>
  <start_of_turn>user
  That's great! Now, could you tell me another color that I can find in a rainbow?<end_of_turn>
  <start_of_turn>model
  Orange<end_of_turn>
  <start_of_turn>user
  Could you continue and name yet another color from the rainbow?<end_of_turn>
  <start_of_turn>model
  ```
- **top-10 下一個 token**（fp32 softmax，即 hw3.py 印的；括號是 logit）：

  | round 1 | | round 2 | | round 3 | |
  |---|---|---|---|---|---|
  | Indigo | 0.3068 (24.125) | Orange | 0.4887 (25.312) | Green | 0.5125 (24.625) |
  | Green | 0.2624 (23.969) | Green | 0.2168 | Yellow | 0.2659 |
  | Red | 0.2012 | Violet | 0.1689 | Red | 0.0978 |
  | Orange | 0.1320 | Red | 0.0255 | Violet | 0.0786 |
  | Violet | 0.0409 | Purple | 0.0175 | Purple | 0.0212 |
  | Blue | 0.0226 | Blue | 0.0115 | Blue | 0.0063 |
  | orange | 0.0080 | Yellow | 0.0113 | Gold | 0.0044 |
  | Purple | 0.0056 | orange | 0.0063 | Scarlet | 0.0021 |
  | green | 0.0042 | Teal | 0.0059 | Ver | 0.0012 |
  | indigo | 0.0036 | ' Orange' | 0.0056 | Magenta | 0.0008 |

  - top-10 機率合計：0.9873、0.9580、0.9906。
  - token id：Indigo 173209、Green 13936、Red 5200、Orange 36565、Violet 107712、Blue 10716、Yellow 33157、Purple 48876。
  - round 2 的 `' Orange'`（id 20723）是「前面帶空白」的另一個 token，跟 `Orange`（36565）不同。
- **觀察**：
  - 已經講過的顏色機率大幅下降：Indigo 在 round 2、3 都掉出前 10；Orange 在 round 3 也掉出前 10。
  - Green 在 round 1 是第 2 名（0.2624），round 3 變成第 1 名。
  - Yellow 在 round 1 不在前 10，round 3 是第 2 名。
- **生成**（greedy，max_new_tokens 200）：
  - 每輪都是 4 個 token，例如 `['Indigo', '▁', '\n', '<end_of_turn>']`。
  - decode 後是 `'Indigo \n'`，印出來是 `Chatbot: Indigo `。
  - 存進 chat_history 的就是 `'Indigo \n'`，但 chat_template 的 `| trim` 會把空白去掉，所以下一輪 prompt 裡是 `Indigo<end_of_turn>`。
- **圖**：img/q2_round{1,2,3}_top_tokens.png（seaborn 橫條圖，coolwarm 色）。

## Q4 實測（取樣；logs/facts_q4_q7.txt、logs/run_q4_*.txt）

- **生成設定**：
  - prompt 32 個 token（含 `<bos>`），`max_length = 30 + 32 = 62`，也就是最多 30 個新 token。
  - generation_config：top_k 50、top_p 1.0、temperature 1.0。
- **self-BLEU 的定義**（hw3.py:188–197）：
  - 每一句當 hypothesis，其他 19 句當 reference，各算一次 `sentence_bleu`，取平均，再對 20 句取平均。
  - 用空白切詞；預設 4-gram 等權，沒有 smoothing。
  - 只要沒有任何 4-gram 重疊，該對就是 0，所以會出現 nltk 的 0 counts 警告。
  - 越高代表 20 句越像，也就是多樣性越低；完全相同的兩句 BLEU = 1.0。
- **hw3.py 實際執行的數字**（同一次執行裡 top-k 先跑 20 次、top-p 再跑 20 次，共用同一個亂數流）：

  | 指令 | top-k self-BLEU | top-p self-BLEU |
  |---|---|---|
  | `--q 4 --seed 0`（預設 k=2、p=0.6） | 0.2029 | 0.5542 |
  | `--seed 0`（全部題目一起跑） | 0.2020 | 0.5542 |
  | `--q 4 --seed 0 --top-k 200 --top-p 0.999` | 0.1343 | 0.0625 |
  | `--q 4 --top-k 1 --top-p 0 --num-samples 2` | 1.0000 | 1.0000 |
  | `--q 4 --seed 0 --top-k 1 --top-p 0 --num-samples 1` | ZeroDivisionError（問題 9） | |

- **5 個 seed 的分布**（facts 腳本：每個設定各自 `torch.manual_seed(seed)` 之後取 20 句；括號是 20 句裡不同句子的數量）：

  | seed | k=2 | k=200 | p=0.6 | p=0.999 | p=0.6, k=0 | p=0.999, k=0 |
  |---|---|---|---|---|---|---|
  | 0 | 0.2029 (20) | 0.1343 (19) | 0.5013 (9) | 0.1257 (20) | 0.5013 (9) | 0.1343 (19) |
  | 1 | 0.2552 (19) | 0.1403 (20) | 0.5349 (10) | 0.1403 (20) | 0.5349 (10) | 0.1403 (20) |
  | 2 | 0.2740 (18) | 0.0423 (20) | 0.4284 (13) | 0.0662 (20) | 0.4284 (13) | 0.0423 (20) |
  | 3 | 0.2529 (17) | 0.1329 (20) | 0.7719 (5) | 0.1371 (20) | 0.7719 (5) | 0.1329 (20) |
  | 4 | 0.2037 (19) | 0.0859 (20) | 0.5104 (9) | 0.0859 (20) | 0.5104 (9) | 0.0859 (20) |

  - **結論穩定**：每個 seed 都是 k=2 > k=200，而且 p=0.6 > p=0.999。
  - **數值波動大**：p=0.6 在 0.43–0.77 之間，所以單次的數字不要寫成定值。
  - **p=0.6 比 k=2 更不多樣**（實測）。可能的解釋（**推論，沒有逐步量機率**）：k=2 每一步都在前兩名之間抽；p=0.6 時，只要第 1 名的機率就超過 0.6，那一步等於 greedy。
  - **「碰到 30 token 上限」的句數**：k=2 有 0–4 句，k=200 有 3–7 句，p=0.6 是 0 句，p=0.999 有 3–7 句。分布越平，越容易寫出贅詞，例如 `Here's why this paraphrase works:`。
  - **兩組設定抽出來一模一樣**：`top_k=200` 與 `top_p=0.999, top_k=0` 在 5 個 seed 下 self-BLEU 完全相同，seed 0 的 20 句也逐句相同。
    - 推論：兩者保留的 token 集合在這個 prompt 上幾乎相同，同一個亂數流就抽出同樣的結果。
- **k=1 與 p=0**：
  - 3 個 seed 各抽 3 次，都只得到一種句子：`Professor Lee is highly regarded as a leading expert in machine learning education.`，跟 greedy 相同。
  - p=0 時 transformers 的 TopP 仍至少保留 1 個 token（`min_tokens_to_keep=1`），所以等於 greedy。
- **逐字的 20 句**：
  - seed 0 各設定：logs/facts_q4_q7.txt；
  - hw3.py 執行的：logs/run_q4_seed0.txt、logs/run_q4_k200_p0999_seed0.txt、logs/run_seed0.txt。
  - k=200 和 p=0.999 會出現 `Dr. Hung-yi Lee, renowned for ...`、`Here I would like an accurate paraphrase, ...` 這類偏離指令的句子；k=2 和 p=0.6 幾乎都是 `Professor Lee is highly regarded ...` 的變體。

## Q5 實測（t-SNE；logs/facts_q4_q7.txt）

- **輸入**：6 句，padding 後 input_ids 是 (6, 9)，左邊補 pad。
  - 實際 token 數：7、7、9、8、7、7。
  - 例：`['<pad>', '<pad>', '<bos>', 'I', '▁ate', '▁a', '▁fresh', '▁apple', '.']`。
- **hidden_states**：`hidden_states[-1]` 是 (6, 9, 2304) fp16，經過最後 norm（見問題 5）。
  - 第 0 句各位置的範數：pad 108.9、108.9；`<bos>` 98.7；其餘 106.6–150.4。
- **餘弦相似度**（hw3.py 的作法，mean 包含 pad）：

  | | Apple(f) | Apple(c) | Orange(f) | Orange(t) | MS(c) | Banana(f) |
  |---|---|---|---|---|---|---|
  | Apple (fruit) | 1 | 0.757 | **0.902** | 0.793 | 0.757 | 0.897 |
  | Apple (company) | 0.757 | 1 | 0.703 | 0.851 | **0.924** | 0.761 |
  | Orange (fruit) | 0.902 | 0.703 | 1 | 0.788 | 0.704 | 0.859 |
  | Orange (telecom) | 0.793 | 0.851 | 0.788 | 1 | **0.868** | 0.822 |
  | Microsoft (company) | 0.757 | 0.924 | 0.704 | 0.868 | 1 | 0.764 |
  | Banana (fruit) | 0.897 | 0.761 | 0.859 | 0.822 | 0.764 | 1 |

  - 排除 pad（masked mean）：Apple(f)–Orange(f) 0.924、Apple(c)–MS 0.915、Orange(t)–MS 0.869、Apple(f)–Apple(c) 0.728。
  - 同時排除 pad 和 `<bos>`：0.926、0.915、0.871、0.732。
  - **同一個字 Apple 的兩個意思，相似度（0.757）低於 Apple(c) 和 Microsoft（0.924）**：句子表示反映的是語境，不是字面。
- **t-SNE**（`perplexity=2, random_state=42`；6 個點，perplexity 必須小於樣本數）：
  - hw3 版本座標：Apple(f) (−150.2, −112.6)、Apple(c) (50.1, 97.7)、Orange(f) (−154.1, −148.4)、Orange(t) (10.7, 38.3)、MS (43.0, 64.4)、Banana (−121.0, −87.4)。
  - 水果 3 個在左下，公司或電信 3 個在右上。
  - random_state 0、1、42、123 的最近鄰關係都一樣：Apple(f)↔Orange(f)、Apple(c)↔MS、Orange(t)→MS、Banana→Apple(f)。
  - t-SNE 座標的絕對值與軸向沒有意義，換 random_state 或排除 pad，座標就整個變了（masked 版的座標在 log），但分群關係不變。
- **圖**：img/q5_tsne.png（hw3.py 輸出）；img/exp_q5_tsne_masked.png（排除 pad 的實驗版）。

## Q6 實測（attention；logs/facts_q4_q7.txt、logs/run_seed0.txt）

- **輸入**：prompt `"Google "` → input_ids `[2, 12583, 235248]` = `['<bos>', 'Google', '▁']`，3 個 token。
  - `total_tokens = 20 + 3 - 1 = 22` = HybridCache 的 `max_cache_len`。
- **每一步的 attentions**：
  - `len(outputs.attentions)` = 26，每層一個。
  - 第 0 步形狀 (1, 8, **3**, 22)：一次餵 3 個 token，所以有 3 列 query。
  - 之後每步 (1, 8, **1**, 22)。
  - 欄數一律是 22，也就是 cache 的長度。還沒寫入的位置被 mask，值為 0：實測上三角最大值 0.0。
  - 每列和介於 0.99970 與 1.00026 之間（fp16 誤差）。
- **20 步共 3 + 19 = 22 列**：第 20 個生成的 token（`▁`）沒有再餵回模型，所以沒有它的列。
- **生成結果**（greedy）：
  - full_text = `'Google \n\n**Google** is a multinational technology company that specializes in internet-related services and products. '`。
  - 生成的 id：109, 688, 12583, 688, 603, 476, 97824, 6178, 3277, 674, 82044, 575, 9574, 235290, 10451, 3545, 578, 3773, 235265, 235248。
- **標籤錯位對照**（問題 7）：

  | 列 | hw3.py 的標籤 | 實際的 token |
  |---|---|---|
  | 0 | Google | `<bos>` |
  | 1 | ▁ | Google |
  | 2 | \n\n | ▁ |
  | 3 | ** | \n\n |
  | 4 | Google | ** |
  | 5 | ** | Google |
  | 6 | ▁is | ** |
  | … | 每列往前錯一格 | … |
  | 20 | . | ▁products |
  | 21 | ▁ | . |

- **layer 10、head 7**（hw3 預設，sliding 層）：
  - 每列注意 `<bos>`（第 0 欄）的比例：1.0, 0.89, 0.777, 0.678, 0.6, 0.406, 0.383, 0.332, 0.276, 0.487, 0.205, 0.391, 0.517, 0.673, 0.229, 0.032, 0.08, 0.042, 0.139, 0.331, 0.314, 0.059。
  - 第 1 列以後平均 0.373；對角線（注意自己）平均 0.084。
  - 每列最大值所在的欄：0,0,0,0,0,0,0,0,7,0,9,0,0,0,13,14,14,14,14,13,0,6。
    - `▁internet` 之後的 `-`、`related`、`▁services`、`▁and`，最注意的是第 14 欄 `▁in` 或第 13 欄 `▁specializes`。
- **對照組**：
  - layer 0、head 0：每列最注意的欄多半是「前一個 token」（argmax 0,0,1,2,3,0,5,6,…），是 previous-token head 的樣子；`<bos>` 平均 0.198。
  - layer 25、head 0（global 層）：每一列最注意的都是 `<bos>`，平均 0.763。
- **各層對 `<bos>` 的平均注意力**（句子 "Google is a multinational technology company"，8 個 head 平均，第 1 列以後）：
  - layer 0–25 依序：0.57, 0.78, 0.62, 0.62, 0.57, 0.55, 0.66, 0.63, 0.63, 0.63, 0.57, 0.55, 0.53, 0.63, 0.61, 0.6, 0.6, 0.69, 0.79, 0.7, 0.75, 0.72, 0.91, 0.76, 0.66, 0.61。
  - layer 10 的 8 個 head：0.38, 0.2, 0.69, 0.76, 0.56, 0.88, 0.53, 0.56。
  - 這種「大量注意力流向第一個 token」的現象叫 attention sink（Xiao et al. 2023, *Efficient Streaming Language Models with Attention Sinks*, arXiv:2309.17453）。
- **圖**：img/q6_attention_L10_H7.png（hw3.py 輸出，標籤錯位）；img/exp_q6_attention_true_labels.png（同一個矩陣，標籤改成實際 token）。

## Q7 實測（Gemma Scope SAE；logs/facts_q4_q7.txt、logs/run_seed0.txt、logs/run_q7_layer21_tok123.txt）

- **SAE 來源**：
  - release `gemma-scope-2b-pt-res-canonical`，sae_id `layer_20/width_16k/canonical`，載入約 1.1 s。
  - `SAE.from_pretrained` 在 sae-lens 5.x 回傳 `(sae, cfg_dict, sparsity)`，這裡 sparsity 是 None。
- **cfg_dict 重點**：
  - architecture `jumprelu`；d_in 2304；d_sae 16384；dtype float32；
  - model_name **`gemma-2-2b`**：**預訓練（pt）版，不是本作業用的 -it 版**；
  - hook_name `blocks.20.hook_resid_post`；hook_layer 20；
  - dataset_path `monology/pile-uncopyrighted`；context_size 1024；prepend_bos True；
  - neuronpedia_id `gemma-2-2b/20-gemmascope-res-16k`。
  - 完整的 `print(sae, cfg_dict, sparsity)` 輸出在 logs/run_seed0.txt。
- **參數**：
  - W_enc (2304, 16384)、W_dec (16384, 2304) 各 37,748,736；b_enc、threshold 各 16384；b_dec 2304。
  - 合計 75,532,544，float32。
- **JumpReLU**：
  - 每個 feature 有自己的門檻 `threshold`：最小 4.516、中位數 7.100、最大 30.226；feature 10004 的門檻是 7.1234。
  - 小於門檻的輸出 0，所以表裡看不到 0 到 7 之間的值。
  - 編碼輸出是 float32：fp16 的 hidden state 進去，float32 出來。
- **L0**（每個 token 有幾個 feature 非零，排除 `<bos>`）：
  - 在 hidden_states[20] 上約 58–64 個（三句依序 58.4、63.5、60.9）；在正確的 [21] 上約 88–91 個（89.0、91.0、88.2）。
- **重建誤差 FVU**（排除 `<bos>`）：[20] 是 0.458–0.463，[21] 是 0.279–0.296。
- **Neuronpedia 上的 feature 10004**（2026-10-04 用 API `https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/10004` 抓的，原始 JSON 在 logs/neuronpedia_feature_10004.json）：
  - explanations（5 條，自動產生）：`words related to time travel.`、`concepts related to time travel and movement across dimensions or timelines.`、`words related to time travel and its consequences`、`Time, space, and dimensional travel`、`travel through dimensions`。
  - `frac_nonzero` = **0.0035**：activation density，也就是 0.35% 的 token 會讓它非零。
  - `maxActApprox` = 125.441。最大的例子是 `...with the power to travel` 的 `▁travel`（125.44）。
  - 正向 logit 影響最大的 token：`▁dimension` 1.077、`▁dimensional` 0.983、`▁space` 0.921、`▁dimensions` 0.917、`▁Dimension` 0.917、`dimension`、`dimensional`、`▁portal` 0.893、`▁tele` 0.889、`▁Time` 0.874。
  - 投影片 p.24 的 Q7.1 要選 3 個敘述；選項在 Gradescope 上，這裡沒有。
- **Q7.2–7.3**（hw3.py 印的）：
  - `max_activation for prompt_a: 58.10378646850586`、`max_activation for prompt_b: 31.740745544433594`。
  - prompt a：22 個 token，14 個非零。
    - 最大值 58.10 在 `▁travel`，其次是 `▁to` 39.07、`.` 35.94、`<bos>` 31.74、`,` 24.10。
    - 完整的逐 token 值在 log。
  - prompt b：22 個 token，**只有 `<bos>` 非零（31.74）**，其他全 0（問題 6）。
  - 直方圖：img/q7_max_activation_a.png、img/q7_max_activation_b.png（50 bins，b 幾乎全部堆在 0）。
- **Q7.4–7.6**：layer 24，句子 "Time travel will become a reality as technology continues to advance."。
  - hw3.py 印出的 `feature_acts shape: torch.Size([13, 16384])`。
  - 各 token 的值：`<bos>` 22.5960、Time 10.8022、▁travel **74.1713**、▁will 24.2492、▁become 20.4319、▁a 19.4009、▁reality 13.2850、▁as 18.8962、▁technology 0、▁continues 0、▁to 0、▁advance 7.2241、`.` 46.4142。
  - 圖：img/q7_token_activations_L24.png。
  - 對照 `--sae-layer-idx 21`（block 20 的輸出，SAE 本來的位置）：`<bos>` 30.95、Time 0、▁travel 71.23、▁will 13.53、▁become 9.62、▁a 17.89、▁reality 0、▁as 10.52、`.` 29.18，其餘 0。圖：img/var_q7_token_activations_L21.png。
- **Q7.7–7.9**：同一句，feature 10004 在 hidden_states 索引 0–26 的值。

  | token | 非零的索引範圍與數值 |
  |---|---|
  | `<bos>` | 9–25 都非零，從 9.6 漲到 33.6（索引 18），再降到 15.3（25）；0–8 與 26 是 0 |
  | Time（token_idx 1，hw3 預設） | 11–17 是 8.6、13.9、15.9、15.3、12.1、11.1、8.7；18–23 是 0；24、25 是 10.8、10.2；26 是 0 |
  | ▁travel（2） | 1、2 是 7.7、8.2；3 是 0；4–25 大致上升（中間有小回落，例如 12→13 是 21.9→20.0），到 25 是 78.9；索引 20 是 58.1、21 是 71.2；26 掉到 11.8 |
  | ▁will（3） | 零星：10 是 7.5、17 是 8.6，19–25 從 10.9 升到 24.2 |
  | ▁technology、▁continues、▁to | 全部 27 個索引都是 0 |
  | `.` | 19 起非零，升到 55.8（25） |

  - 索引 26（經過最後 norm，範數只剩約 100）大多掉到 0 或很小，因為 SAE 的門檻是照 block 20 未 norm 的尺度訓練的。這句屬推論。
  - `Time` 這個 token 的 hidden state 範數，依索引 0–26：72, 78, 82, 88, 85, 90, 125, 127, 225, 242, 380, 515, 654, 700, 683, 681, 724, 659, 571, 518, 539, 546, 568, 561, 566, 540, 117。
  - 圖：img/q7_layer_activations_tok1.png（hw3 預設）、img/var_q7_layer_activations_tok2.png（▁travel）、img/var_q7_layer_activations_tok3.png（▁will）。
- **給教材的提醒**：
  - SAE 只在 block 20 的輸出上訓練，拿去編碼其他層的 hidden state，數值沒有「這一層有沒有這個概念」的嚴格意義。這是作業設計的簡化，教材要明講。
  - 投影片 p.27 的提示（「the lower/deeper layers tend to process complex information」）要在這個前提下讀。
  - 而且 SAE 是在 **pt** 模型上訓練，這裡套在 **it** 模型上（cfg_dict 的 model_name 是 `gemma-2-2b`）。pt 的 SAE 用在 it 模型上效果如何，本機**沒有量**，也還沒查證 Gemma Scope 論文的說法。教材提到時標 `<!-- TODO(本機實測): pt SAE 用在 it 模型的影響 -->`，不要寫成定論。

## 圖檔清單（docs/HW03/img/，14 張）

- **hw3.py 實際輸出**：
  - 來源指令：`.venv/bin/python HW03/hw3.py --seed 0`，2026-10-04，從 `HW03/outputs/` 複製。
  - 檔案：q2_round1_top_tokens.png、q2_round2_top_tokens.png、q2_round3_top_tokens.png、q5_tsne.png、q6_attention_L10_H7.png、q7_max_activation_a.png、q7_max_activation_b.png、q7_token_activations_L24.png、q7_layer_activations_tok1.png。
- **hw3.py 換參數的輸出**（`--q 7 --sae-layer-idx 21 --token-idx 1 2 3`）：var_q7_token_activations_L21.png、var_q7_layer_activations_tok2.png、var_q7_layer_activations_tok3.png。
- **facts 腳本的實驗圖**（不是 hw3.py 的輸出，教材要標明是「修正後的對照」）：exp_q5_tsne_masked.png、exp_q6_attention_true_labels.png。
- **Q1、Q3、Q4 沒有圖**：純文字輸出。

## 逐字紀錄（docs/HW03/logs/）

- run_seed0.txt：`.venv/bin/python HW03/hw3.py --seed 0`，全部 7 題，stdout 與 stderr 都有，real 45.7 s。
- run_q4_seed0.txt：`--q 4 --seed 0`。
- run_q4_k200_p0999_seed0.txt：`--q 4 --seed 0 --top-k 200 --top-p 0.999`。
- run_q4_k1_p0_n1_seed0.txt：`--q 4 --seed 0 --top-k 1 --top-p 0 --num-samples 1`，以 ZeroDivisionError 結束。
- run_q4_k1_p0_n2.txt：`--q 4 --top-k 1 --top-p 0 --num-samples 2`。
- run_q7_layer21_tok123.txt：`--q 7 --sae-layer-idx 21 --token-idx 1 2 3`。
- facts_env_model_tok.txt、facts_q1_q2.txt、facts_q4_q7.txt：`docs/tools/hw03_facts.py` 的輸出。
- neuronpedia_feature_10004.json：Neuronpedia API 的原始回應。
- log 裡的絕對路徑 `/home/valtec/poyi/GitHubLL/ML2025-Spring-pytorch/` 是本機 repo 位置。教材引用時改寫成相對路徑，例如 `HW03/outputs/...`。

## 給大綱的建議（本機量完後的觀察，雲端可以調整）

- 7 題可以自然分成 4 個區塊，大致對應投影片 p.4 的分類：
  - 對話格式：Q1–Q2；
  - token 與生成：Q3–Q4；
  - 表示與注意力：Q5–Q6；
  - 可解釋性：Q7。
- 「repo 問題清單」的 10 條，分別是各章「讀 code 不盡信文件」框的主角：
  - 問題 1（雙 `<bos>`）放 Q1／Q2 章；
  - 問題 2、3、4、9、10 放 Q4 章；
  - 問題 8 放 Q5 章；
  - 問題 7 放 Q6 章；
  - 問題 5、6 放 Q7 章。
- 論文閱讀（Problem 8–9）不在程式範圍內，教材可以只在目錄頁提一句，或放附錄的延伸閱讀。
