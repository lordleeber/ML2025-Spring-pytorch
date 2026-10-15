# HW06 讀碼教材 — 事實清單

每一章動筆前先讀這份清單，敘述不得與之矛盾。新查證推翻舊事實時，改這裡並註明日期。
原始數字在 `docs/tools/hw06_runs.jsonl`（訓練與 Accuracy）、`hw06_judge.jsonl`（逐題判決，沒有回答原文）、`hw06_judge_summary.jsonl`（Safety Rate）；
來龍去脈在 `HW06/PLAN.md`。下面的「取樣」指 notebook 預設的解碼（`do_sample=True, temperature=0.6, top_p=0.9`），「greedy」指 `--greedy`。

## 環境與模型（第 0 章）
- GPU：NVIDIA RTX PRO 4000 Blackwell 24 GB（sm_120），驅動 596.71，WSL2。
- `HW06/.venv`（`HW06/pyproject.toml`）：Python 3.12、torch 2.11.0+cu128、transformers 4.48.3、accelerate 1.3.0、peft 0.14.0、trl 0.15.2、datasets 3.3.2、bitsandbytes 0.45.3、huggingface_hub 0.28.1、tokenizers 0.21.0、safetensors 0.5.3。
- Colab（2025-03）的版本來自 notebook 的 pip 輸出：transformers 4.48.3、accelerate 1.3.0、torch 2.5.1+cu124、huggingface-hub 0.28.1、tokenizers 0.21.0、safetensors 0.5.3、Python 3.11；`pip install -U` 裝了 datasets 3.3.2、trl 0.15.2、bitsandbytes 0.45.3。peft 沒印，推定 0.14.0（0.15.0 在 2025-03-19 發布）。torch 2.5.1+cu124 沒有 sm_120 kernel，所以換成 2.11.0+cu128。
- bitsandbytes 0.45.3 在 sm_120 上可用：4-bit NF4 載入、生成、LoRA forward／backward、`PagedAdamW32bit` 一步都正常。
- 官方 `meta-llama/Llama-3.2-1B-Instruct` 是 `gated=manual`（人工審核）。模型從 `alpindale/Llama-3.2-1B-Instruct`（revision `f92201d8185818a9d079b3b52efdab4b68bdd17f`）下載，8 個檔（model.safetensors、config、generation_config、3 個 tokenizer 檔、LICENSE.txt、USE_POLICY.md）與官方 revision `9213176726f574b556790deb65791e0c5aa438b6` 的記錄一致；model.safetensors 2,471,645,608 bytes、sha256 `1ff795ff6a07…`。`unsloth/Llama-3.2-1B-Instruct` 權重相同，但 tokenizer 與 config 改過，沒有用。
- 在 HF 快取建 `models--meta-llama--…` 條目、`refs/main` 寫官方 revision、檔案 symlink 到鏡像 blob；以 `HF_HUB_OFFLINE=1` 執行，程式裡的 `meta-llama/...` 不改。
- 模型總覽（transformers 實際載入）：16 層、hidden 2048、32 個 attention 頭、8 個 KV 頭（GQA）、參數 1,235,814,400、詞彙 128,256。bf16 greedy 回答 Natalia 那題得到 72。
- 審查模型（都和官方記錄逐檔一致）：Llama Guard 3 1B（`alpindale/Llama-Guard-3-1B`，官方 revision `acf7aafa60f0…`）、Llama Guard 3 8B（`jy622/Llama-Guard-3-8B`，官方 `7327bd9f6efb…`，32 層）、ShieldGemma 2B（官方 `google/shieldgemma-2b` revision `d1dffc9c8c92…`；tokenizer.json 是 LFS 檔，比 sha256 `3f289bc0…`）。
- Colab 2025 的執行輸出：`global_step=1868`、`train_loss=0.8524938824345264`、`train_runtime 8991.5782` 秒（T4）。投影片的時間估計：Simple 3 h 微調＋2 h 推論。

## 拆檔與驗證（第 0、6 章）
- `HW06/hw6.py` 308 行左右（以 commit 為準），預設值重現 notebook；新增選項：`--data-dir`、`--out-dir`、`--name`、`--seed`、`--student-id`、`--train-data`、`--train-n-shot`、`--test-n-shot`、`--fixed-shots-seed`、`--skip-train`、`--lr`、`--epochs`、`--scheduler`、`--warmup-ratio`、`--weight-decay`、`--lora-dropout`、`--adapter`、`--max-new-tokens`、`--greedy`；另存 `responses.json`。
- 參照版 `docs/tools/hw06_make_ref.py`：只把 8 行 `!` 指令註解掉：nvidia-smi、5 個 wget、pip、huggingface-cli login（數過 notebook 與產生的腳本，都是 8 行）。
- 原版不可重現：ref1／ref2 的 10 個 checkpoint 全部不同、504 筆有 416 筆不同；train_loss 0.85292／0.85347；GSM8K public（取樣）0.333／0.356。原因：LoRA A 在 `get_peft_model`（cell [12]）初始化，早於 SFTTrainer 的 `set_seed(1126)`，而 torch 的啟動種子每個 process 隨機。`hw06_nondet_probe.py` 證明同一 batch 的 forward／backward 在決定性模式下逐位元相同（4-bit、bf16、有無 LoRA）。
- `docs/tools/hw06_det.py`：決定性模式＋`torch.manual_seed(HW06_TORCH_SEED, 預設 0)`；ref3＝ref4 逐位元相同（train_loss 0.852828219820141、GSM8K public 0.348）。
- 拆檔驗證：split1（hw6.py）＝ref5（notebook，同一天跑）：10 個 checkpoint 相同、504 筆 0 筆不同，GSM8K public 0.326（`docs/tools/hw06_compare_split1_ref5.txt`）。
- 跨日：ref3（10-11）對 ref5（10-12）：checkpoint 相同、504 筆有 407 筆不同，從第 7 筆開始。排除：垃圾回收（`hw06_gc_probe.py`）、`pipeline.model` 那一行（split2＝split1）、import 順序（torch 數值旗標不變、logits 相同）、驅動與檔案。追蹤（`hw06_trace.py`）：notebook 與 hw6.py 在 10-12 的前 10 次呼叫的 RNG 狀態、輸入、輸出全部相同。
- 原因（10-13 **實測確認**：`hw06_fixdate.py` 把日期固定成 2026-10-11 重跑 hw6.py，和 10-11 的 ref3 逐位元相同，10 個 checkpoint 相同、504 筆 0 筆不同、GSM8K public 0.348；`docs/tools/hw06_compare_ref3_date1011.txt`）：Llama 3.2 的 chat template 在 system header 寫 `Today Date: {strftime_now("%d %b %Y")}`；評估 prompt 帶當天日期，訓練資料切掉了 system header。`docs/tools/hw06_fixdate.py` 固定日期。

## 資料（第 1 章）
- `gsm8k_train.jsonl` 7,473 題；`gsm8k_train_self-instruct.jsonl` 7,472 題（問題全部出自原訓練集；全部有 `####`；只有 103 題保留 `<<…>>`，原版 7,378 題有）。
- `gsm8k_test_public.jsonl` 132 題（有答案、不在訓練集）；`gsm8k_test_private.jsonl` 132 題（只有 question）。
- `ailuminate_test.csv` 240 列，欄位 release_prompt_id、prompt_text、hazard、persona、locale、prompt_hash；全部 en_US；persona unskilled 135、skilled 105；hazard 15 類：cse 30、ipv 30、vcr 22、dfm 22、iwp 19、ncr 19、src 18、hte 18、ssh 15、prv 15、sxc_prn 15、spc_fin 10、spc_lgl 4、spc_ele 2、spc_hlt 1。**12 題重複**（同 id、同文字，如第 8／190 列），不同題目 228 題。
- sha256：train `00af26ed…`、self-instruct `cc752d35…`、public `c6745c2b…`、private `7c569596…`、ailuminate `5c6976d3…`。
- 交出去的 list：264 個 GSM8K 答案（public 132＋private 132）＋240 個 AILuminate 回答＝504 筆。

## QLoRA 與訓練（第 2、3 章）
- 4-bit NF4、double quant、compute dtype bf16；4-bit 載入佔 1.03 GB。
- LoRA r=8、alpha=16、bias none、7 個投影層（up、down、gate、k、q、v、o）；可訓練 5,636,096＝0.454%（peft 印的 all params 1,241,450,496）；LoRA 張量 224 個（16 層 × 7 × 2）。B 初始為 0，第一步 A 的梯度為 0。
- `[PAD]` 讓 tokenizer 變 128,257，embedding 沒 resize（128,256）。
- trl 0.15.2 `_prepare_peft_model`：model 已是 PeftModel 就直接回傳（2025 版因此跳過 `prepare_model_for_kbit_training` 與 bf16 轉型）；2026 版傳原始模型，會做這兩步。
- 1 epoch 1868 步（batch 1 × grad-accum 4）；每 187 步存一個 checkpoint、共 10 個；本機 train_runtime 約 1,033–1,046 s（決定性模式；正式計時見 H 組）。3 epochs 5604 步。
- 1 epoch 的 train_loss 0.852828（種子 0）；Colab 0.852494。3 epochs train_loss 平均約 0.628，最後一段 log 0.42。
- **loss 算整段文字**（2026-10-14 查證）：trl 0.15.2 的 SFTTrainer 沒給 collator 時用 `DataCollatorForLanguageModeling(mlm=False)`（sft_trainer.py 第 210–211 行），labels＝input_ids，只遮 pad。`docs/tools/hw06_loss_mask.py`（輸出 `hw06_loss_mask.txt`，前 200 個訓練樣本）：預設算進 loss 的 token 1-shot 74,935、5-shot 217,647、8-shot 322,008（皆 100%）；`DataCollatorForCompletionOnlyLM`（`--completion-only-loss`）只剩最後一段回答 21,214 個 token＝1-shot 28.3%、5-shot 9.7%、8-shot 6.6%。沒有 instruction_template 時，它把最後一個 assistant 標頭之前全部遮掉（迴圈找到符合後不 break）。

## 生成與 Accuracy（第 4 章）
- `pipeline.model = PeftModel.from_pretrained(sft_model, adapter_path)`：設在函式上，但會就地把 checkpoint 載入 sft_model 的 LoRA 層，generator 用的確實是該 checkpoint（`hw06_adapter_probe.txt`）。
- 抽取：`split('####')[-1]`、去掉 `, $ % g`、字串相等；ref1 的 88 題錯誤裡 86 題是數字錯、2 題沒有 `####`；第 95 題 `originally` → `oriinally`。
- 截斷：原模型（取樣、種子 0）15 題生成滿 256 token、21 題沒有 `####`；微調後 checkpoint-187 是 2／2、1868 是 4／4。
- 取樣的 GSM8K public：原模型種子 0／1／2 0.318／0.227／0.280（平均 0.275）；1 epoch ref1／ref2／ref3 0.333／0.356／0.348（平均 0.346）。
- 512 token（取樣）：原模型 0.220／0.280／0.258（平均 0.253）；ref3 checkpoint-1868 0.394／0.364／0.341（平均 0.366）。「多給 token 原模型會變好」的預測錯了。
- greedy：原模型 0.303；1 epoch ref1／ref2／ref5 0.386／0.386／0.341（平均 0.371）；3 epochs s0／s1／s2 0.318／0.333／0.258（平均 0.303）。

## Safety Rate 與審查模型（第 5 章）
- Llama Guard：生成 `safe`／`unsafe\nS<k>`；1B 模板要 list 形式的 content，13 類；8B 用字串 content，14 類（多 S14 Code Interpreter Abuse）。ShieldGemma：model card 的 `guideline=` chat template、4 條 Prompt-Response 政策原文、任一 P(Yes) > 0.5 算 unsafe。
- 取樣、3 種子平均：原模型 8B 0.854、1B 0.662、ShieldGemma 0.931；1 epoch 0.887、0.753、0.921；3 epochs 0.843、0.699、0.903。
- greedy、3 種子平均：原模型 0.863、0.696、0.933；1 epoch 0.905、0.767、0.935；3 epochs 0.842、0.721、0.918。
- 一致率（6 組 × 240＝1,440 對）：1B–8B 0.780、κ 0.364；1B–ShieldGemma 0.729、κ 0.162；8B–ShieldGemma 0.866、κ 0.273；三個都判 unsafe 41 對。嚴格程度 1B > 8B > ShieldGemma。
- 原模型 vs 1 epoch：開頭就拒答 102 vs 110／123／113 題；字數中位數 180 vs 93／56／120；原模型最常見開頭 "I can provide"（37 題）。
- 投影片基準線（2025 public）：Simple 0.280／0.558、Medium 0.379／0.642、Strong 0.455／0.725（Accuracy／Safety）。審查模型不同，不能直接比。

- **max_new_tokens 與審查（D 組，greedy）**：GSM8K public：原模型 256／512／1024 token 0.303／0.326／0.326；1 epoch（ref1／ref2／ref5）512 token 0.402／0.371／0.341、1024 token 0.402／0.371／（見 runs.jsonl），平均和 256 token 的 0.371 相同——多給 token 只對原模型有幫助。
  Safety 8B：原模型 256／512／1024 0.863／0.846／0.871；1 epoch 512 0.900／0.933／0.887、1024 0.900／0.933／0.896。
  **Guard 1B 對截斷很敏感**：原模型 256 → 512 token，1B 的 Safety 0.696 → 0.829。原模型的 AILuminate 回答在 256 token 有 130／240 題生成滿上限；1B 從 unsafe 翻成 safe 的 36 題裡 35 題是 256 token 時被截斷的回答（給足長度後回答寫完、結尾常有提醒或拒絕）。8B 只翻 6 題成 safe、10 題成 unsafe。`max_new_tokens` 也是 Safety 量法的一部分。

## 遺忘曲線（第 7 章）
- ref3 的 10 個 checkpoint（取樣、種子 0）GSM8K public：0.288、0.341、0.280、0.333、0.288、0.348、0.333、0.439、0.371、0.341；Safety 8B：0.871、0.858、0.871、0.879、0.875、0.921、0.887、0.875、0.883、0.904。
- 學習率（B 組，greedy，種子 0／1／2）：
  - lr 1e-4：GSM8K public 0.341／0.402／0.356（平均 0.366）；Safety 8B 0.917／0.912／0.912（0.914）、1B 0.762／0.738／0.787（0.762）、ShieldGemma 0.950／0.938／0.954（0.947）。
  - lr 1e-5：GSM8K public 0.295／0.303／0.326（平均 0.308）；Safety 8B 0.892／0.879／0.883（0.885）、1B 0.742／0.738／0.713（0.731）、ShieldGemma 0.929／0.933／0.933（0.932）。
  - lr 1e-5 幾乎沒學到數學（原模型 greedy 0.303），Safety 8B 卻比原模型（0.863）高：微調後變安全不是學會數學的副產品。

- **few-shot 數（C 組，greedy，訓練＝測試）**：5-shot 最長樣本 1,701 token、訓練 1,519 s；8-shot 最長 2,406 token。
  - 5-shot：GSM8K public 0.326／0.364／0.333（平均 0.341）；Safety 8B 0.854／0.842／0.850（0.849）、1B 0.700／0.696／0.708（0.701）、ShieldGemma 0.925／0.900／0.908（0.911）。
  - 8-shot：GSM8K public 0.258／0.318／0.265（平均 0.280）；Safety 8B 0.808／0.779／0.808（**0.799**）、1B 0.613／0.654／0.633（0.633）、ShieldGemma 0.904／0.892／0.875（0.890）。
  - **目前最明顯的遺忘**：8-shot 三個種子、三個審查模型都低於原模型（greedy 8B 0.863、1B 0.696、ShieldGemma 0.933），Accuracy 也低於原模型（0.303）。推測（未驗證）：loss 若算整段文字，8-shot 每個樣本有 9 段 GSM8K 式的 assistant 回答，訓練訊號是 1-shot 的好幾倍；要在第 3 章查 loss 的範圍時一起驗證。

- **weight decay／LoRA dropout（E 組，greedy，1 epoch，種子 0／1／2）**：

  | | GSM8K public | Safety 8B | 1B | ShieldGemma |
  |---|---|---|---|---|
  | weight decay 1e-2 | 0.386／0.371／0.333（0.363） | 0.883／0.908／0.875（0.889） | 0.736 | 0.939 |
  | weight decay 1e-4 | 0.394／0.379／0.348（0.374） | 0.863／0.900／0.892（0.885） | 0.740 | 0.939 |
  | dropout 0.1 | 0.379／0.379／0.348（0.369） | 0.863／0.900／0.883（0.882） | 0.731 | 0.929 |
  | dropout 0.2 | 0.379／0.379／0.364（0.374） | 0.887／0.900／0.917（0.901） | 0.742 | 0.926 |

  全部和 1 epoch 預設（0.371、8B 0.905、1B 0.767、ShieldGemma 0.935）在種子雜訊內；1 epoch 本來就沒有遺忘，這兩個提示沒有東西可以保護。

- **self-instruct 訓練資料（F 組前半，greedy，1 epoch，種子 0／1／2；1868 步）**：GSM8K public 0.485／0.455／0.492（平均 **0.477**，全書最高，超過投影片 Strong 基準線 0.455）；Safety 8B 0.871／0.854／0.892（0.872）、1B 0.692／0.642／0.746（0.693）、ShieldGemma 0.921／0.908／0.938（0.922）。數學大幅上升，安全維持在原模型水準（greedy 原模型 0.863／0.696／0.933），但沒有原版資料那種「微調後更安全」（1 epoch 預設 0.905／0.767／0.935）。

## 規格內其餘實驗
- C、D、E、F、G、I、H 組：排程中（`scratchpad/queue_spec.sh`），完成後逐項補進來。
