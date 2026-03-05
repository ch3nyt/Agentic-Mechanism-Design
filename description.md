# LLM Multi-Agent Auction Experiment (In-Context RL)

## 實驗目標

驗證以下假說：**LLM Agent 在無顯式強化學習演算法的情況下，純粹依靠上下文中的歷史記錄（In-Context RL），是否能自主學習出第一價格拍賣的貝氏納許均衡（BNE）出價策略。**

理論預測：在 $N$ 個 Agent、私人估值 $v \sim U(0, 100)$ 的對稱第一價格拍賣中，BNE 最佳出價為：

$$b^*(v) = \frac{N-1}{N} \cdot v$$

---

## 檔案架構

```
.
├── api.env           # API 金鑰與實驗參數設定
├── prompts.py        # System Prompt 與字串模板（文字與邏輯解耦）
├── environment.py    # 拍賣機制與結算引擎
├── agent.py          # LLM Agent 大腦 + ICRL 記憶體
├── main.py           # 實驗主迴圈（Orchestrator）
├── analysis.ipynb    # 數據分析與視覺化
└── results.csv       # 實驗原始數據（由 main.py 生成）
```

---

## 各模組詳細說明

### 1. `api.env` — 設定檔

集中管理所有可調參數，無需修改程式碼即可調整實驗設定：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `OPENAI_API_KEY` | *(必填)* | OpenAI API 金鑰 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 使用的 LLM 模型 |
| `N_AGENTS` | `3` | 競標者數量 |
| `N_ROUNDS` | `50` | 實驗輪次 |
| `VALUE_LOW` / `VALUE_HIGH` | `0` / `100` | 估值均勻分配範圍 |
| `MEMORY_WINDOW` | `10` | Agent 記憶體視窗（最近幾輪歷史）|
| `TEMPERATURE` | `0.7` | LLM 生成溫度 |
| `RESULTS_FILE` | `results.csv` | 輸出 CSV 檔案名稱 |

---

### 2. `environment.py` — 拍賣裁判與數學結算中心

**角色**：絕對理性的裁判，不偏袒任何 Agent。

**核心功能**：
- `generate_values()` — 從 $U(\text{low}, \text{high})$ 為 $N$ 個 Agent 隨機生成私人估值，各 Agent 只知道自己的估值（私有資訊）。
- `resolve(values, bids)` — 收集所有報價，找出最高出價者。若有平局則隨機抽籤。計算各 Agent 的利潤：
  - 得標者：$\pi = v_{\text{winner}} - b_{\text{winner}}$
  - 未得標者：$\pi = 0$
- `bne_bid(value)` — 回傳理論 BNE 出價 $b^*(v) = \frac{N-1}{N} \cdot v$，供分析時比較。

**設計原則**：與 Agent 完全解耦，未來可替換為第二價格拍賣或其他機制，只需修改此檔案。

---

### 3. `agent.py` — LLM 大腦與 ICRL 記憶體

**角色**：封裝 LLM API 調用與歷史記憶的載體。這是 In-Context RL 的核心實作。

**核心功能**：

**記憶體緩衝區（Memory Buffer）**：
- 每個 Agent 維護一個 `history` 列表，儲存每輪的 `{round, value, bid, won, payoff, winning_bid}`。
- `memory_window` 參數控制每次傳入 Prompt 的歷史輪數（預設最近 10 輪），避免 Context 過長。

**API 調用流程**：
1. `decide_bid(value, round_num)` — 組裝 Prompt（System + 歷史 + 本輪估值），呼叫 OpenAI API，解析回傳的出價。
2. `update_memory(...)` — 結算後由 `main.py` 呼叫，將本輪結果寫入記憶體。

**出價解析（Robust JSON Parser）**：
優先序：嚴格 JSON → 正則表達式提取嵌入 JSON → 提取第一個數字 → 回退至 `value/2`。確保 LLM 偶發的格式錯誤（Hallucination）不會中斷實驗。

**Persona 支援**：
- `"rational"` — 要求利潤最大化，鼓勵學習並壓低出價。
- `"irrational"` — 模擬衝動型競標者，偏好激進出價以求得標。
- 可在 `prompts.py` 新增更多 Persona 並在 `main.py` 的 `AGENT_PERSONAS` 列表中指定。

---

### 4. `prompts.py` — 提示詞設定檔

**角色**：將「文字設定」與「程式邏輯」解耦（Decouple），方便 A/B 測試。

**核心內容**：
- `SYSTEM_PROMPT_RATIONAL` — 理性 Agent 的系統提示，強調利潤最大化、從歷史學習。
- `SYSTEM_PROMPT_IRRATIONAL` — 衝動型 Agent 的系統提示，偏好勝利而非利潤。
- `format_history_entry(record)` — 將單輪記錄格式化為易讀的文字行。
- `format_bid_request(value, n_agents, round_num)` — 生成本輪的出價請求文字。
- `build_user_message(history, value, ...)` — 組裝完整的 User Message（歷史 + 本輪）。

**擴充方式**：只需在此檔案新增新的 Prompt 常數，並在 `agent.py` 的 `PERSONAS` dict 中登記即可。

---

### 5. `main.py` — 實驗主迴圈（Orchestrator）

**角色**：整個實驗的中央控制器，協調各模組之間的數據流。

**每輪執行流程**：
```
environment.generate_values()
  -> agent[i].decide_bid(values[i], round_num)  # 並行概念，順序執行
  -> environment.resolve(values, bids)
  -> agent[i].update_memory(...)
  -> csv_writer.writerow(...)
```

**CSV 輸出欄位**：

| 欄位 | 說明 |
|------|------|
| `round` | 輪次編號 |
| `agent_id` | Agent 編號 |
| `persona` | Agent 性格（rational/irrational） |
| `value` | 本輪私人估值 |
| `bid` | 提交的出價 |
| `bid_ratio` | `bid / value`（關鍵分析指標） |
| `bne_bid` | 本輪 BNE 理論最佳出價 |
| `won` | 是否得標（0/1） |
| `payoff` | 本輪利潤 |
| `winning_bid` | 本輪最高成交價 |
| `cumulative_profit` | 累積利潤 |

**容錯設計**：每輪結束後立即 `flush()` CSV，中斷實驗不會遺失已完成的輪次數據。

---

### 6. `analysis.ipynb` — 數據分析與視覺化

**角色**：將原始數據轉化為經濟學洞見。

**五個分析區塊**：

1. **Bid vs Value 散佈圖（全輪次）** — 點的顏色代表輪次（Viridis 色階），直觀看出隨時間的演化趨勢，並覆蓋 45° 誠實出價線與 BNE 理論線。
2. **收斂分析（前半段 vs 後半段）** — 將實驗分為早期與晚期，對比散佈圖的集中程度，驗證是否向 BNE 線收斂。
3. **Bid Ratio 時間序列** — 繪製每輪平均 `bid/value` 比率（含 5 輪滾動平均），觀察是否收斂至 $(N-1)/N$。
4. **累積利潤折線圖** — 各 Agent 的累積收益曲線，可比較不同 Persona 的長期表現。
5. **摘要統計表** — 彙整各 Agent 的平均 bid_ratio、與 BNE 的偏差、勝率、總利潤。

---

## 快速開始

### 安裝依賴

```bash
pip install openai python-dotenv
```

### 設定 API 金鑰

編輯 `api.env`，填入你的 OpenAI API 金鑰：

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 執行實驗

```bash
python main.py
```

### 分析結果

```bash
jupyter notebook analysis.ipynb
```

---

## 實驗設計延伸

| 想法 | 修改方式 |
|------|----------|
| 換成第二價格拍賣 | 修改 `environment.py` 的 `resolve()` 方法（得標者付第二高價） |
| 新增性格類型 | 在 `prompts.py` 新增 Prompt，在 `agent.py` 的 `PERSONAS` 登記 |
| 換成 Q-Learning Agent | 繼承或替換 `agent.py` 的 `decide_bid()`，其餘不變 |
| 更多 Agent | 修改 `api.env` 的 `N_AGENTS` |
| 非均勻估值分布 | 修改 `environment.py` 的 `generate_values()` |

---

## 理論背景

這 5 個模組組成了計算實驗經濟學（Computational Experimental Economics）的標準架構：「拍賣規則（`environment.py`）」與「代理人記憶（`agent.py`）」完全解耦，可獨立替換，便於跨模型比對研究。

**In-Context RL 的核心機制**：LLM 並未在參數層面進行學習（沒有梯度更新），但其 Context Window 充當了「外部記憶體」與「少樣本提示（Few-shot Prompt）」的角色，使 Agent 能在推理時動態調整策略——這正是 Transformer 架構的隱式歸納偏好（Inductive Bias）所賦予的能力。
