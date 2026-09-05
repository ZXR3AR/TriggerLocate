# TriggerLocate

在有毒 RAG 语料库中，基于模型行为偏差定位可疑后门触发词（trigger）。

本仓库对应论文工作：**RAG 系统中基于行为偏差的后门触发器定位方法**。给定一份可能被投毒的 RAG 知识库，方法先从语料中粗筛高频短语，再通过“有该短语 / 无该短语”的决策对比做精验证，按行为翻转程度给候选触发词排序。

## 仓库结构

```text
TriggerLocate/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── TriggerLocate_Experiment.py         # 主实验：粗筛 + 精验证 + 指标
│   └── TriggerLocate_Experiment_Count.py   # 同上，额外统计 LLM 调用次数
├── data/                                   # 评测语料（JSONL）
└── paper/                                  # 论文 LaTeX 源稿
```

## 环境要求

- Python 3.9+
- 本地 [Ollama](https://ollama.com)（脚本默认请求 `http://localhost:11434`，模型名 `llama2`）
- 建议有基本内存：会用 HuggingFace embedding + Chroma 建临时向量库


## 数据说明

语料为 JSONL（每行一条 JSON），由 nuScenes 风格场景描述构造，仅用于研究复现。

投毒文件名中的 `A-B` 表示：**前面是投毒条数，后面是语料总条数**。  
因此毒化率 = A / B。例如 `1-50` 为 1/50 = **2%**，`4-50` 为 4/50 = **8%**。

| 文件 | 说明 |
|------|------|
| `nusc_eval_clean_50.jsonl` | 干净对照集（50 条，无投毒） |
| `nusc_eval_backdoor_1-50.jsonl` | 投毒 1 条 / 共 50 条，毒化率 2% |
| `nusc_eval_backdoor_2-50.jsonl` | 投毒 2 条 / 共 50 条，毒化率 4% |
| `nusc_eval_backdoor_4-50.jsonl` | 投毒 4 条 / 共 50 条，毒化率 8% |
| `nusc_eval_backdoor_5-50.jsonl` | 投毒 5 条 / 共 50 条，毒化率 10% |
| `nusc_eval_backdoor_6-50.jsonl` | 投毒 6 条 / 共 50 条，毒化率 12% |
| `nusc_eval_backdoor_15-50.jsonl` | 投毒 15 条 / 共 50 条，毒化率 30% |
| `nusc_eval_backdoor_purple_laundry_basket_4-50.jsonl` | 触发物变为 purple laundry basket（8%） |
| `nusc_eval_backdoor_checkered_picnic_blanket_4-50.jsonl` | 触发物变为 checkered picnic blanket（8%） |

默认 ground-truth 触发（只用于算指标，不参与检测逻辑）是：

- `gray trash bin`
- `gray trash`
- `trash bin`

换触发物语料时，请同步修改脚本中的 `GROUND_TRUTH`。

### 脚本说明

打开 `TriggerLocate_Experiment.py`，主要改这三处：

1. `POISON_CORPORA`：语料路径  
   例：`"06": "data/nusc_eval_backdoor_4-50.jsonl"`
2. `RUN_LIST`：`(poison_key, ablation_key)`  
   例：`("06", "full")`
3. `GROUND_TRUTH`：当前语料对应的真实触发短语(GT)

内置 ablation：

- `full`：动词过滤 + 第二阶段验证 + 短语级候选
- `no_verb`：关闭动词过滤
- `stage1_only`：只做粗筛
- `word_level`：只出单词，不出双词/三词短语

## 注意

- 需要本机已启动 Ollama，否则请求 `localhost:11434` 会失败。
- 评测数据是研究用构造样本，不是 nuScenes 官方完整数据集。

## 引用

若使用本仓库中的代码或语料，请引用对应论文：

---

# TriggerLocate (English)

Locate suspicious backdoor trigger words in a poisoned RAG corpus using behavioral deviation.

This repository accompanies the paper **Trigger Localization for Backdoors in RAG Systems via Behavioral Deviation**. Given a possibly poisoned RAG knowledge base, the method first mines frequent phrases from the corpus, then verifies each candidate by comparing model decisions with and without that phrase, and ranks candidates by how often the decision flips.

## Repository Layout

```text
TriggerLocate/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── TriggerLocate_Experiment.py         # main experiment: screening + verification + metrics
│   └── TriggerLocate_Experiment_Count.py   # same pipeline, plus LLM call counting
├── data/                                   # evaluation corpora (JSONL)
└── paper/                                  # LaTeX source of the paper
```

## Requirements

- Python 3.9+
- Local [Ollama](https://ollama.com) (the scripts call `http://localhost:11434` by default, model name `llama2`)
- Enough memory to build a temporary HuggingFace embedding + Chroma vector store

## Data

Each corpus file is JSONL (one JSON object per line), constructed from nuScenes-style scene descriptions for research reproduction only.

In poisoned filenames, `A-B` means: **A = number of poisoned entries, B = total entries**.  
Poison rate = A / B. For example, `1-50` is 1/50 = **2%**, and `4-50` is 4/50 = **8%**.

| File | Description |
|------|------|
| `nusc_eval_clean_50.jsonl` | Clean control set (50 entries, no poison) |
| `nusc_eval_backdoor_1-50.jsonl` | 1 poisoned / 50 total, poison rate 2% |
| `nusc_eval_backdoor_2-50.jsonl` | 2 poisoned / 50 total, poison rate 4% |
| `nusc_eval_backdoor_4-50.jsonl` | 4 poisoned / 50 total, poison rate 8% |
| `nusc_eval_backdoor_5-50.jsonl` | 5 poisoned / 50 total, poison rate 10% |
| `nusc_eval_backdoor_6-50.jsonl` | 6 poisoned / 50 total, poison rate 12% |
| `nusc_eval_backdoor_15-50.jsonl` | 15 poisoned / 50 total, poison rate 30% |
| `nusc_eval_backdoor_purple_laundry_basket_4-50.jsonl` | Trigger object changed to purple laundry basket (8%) |
| `nusc_eval_backdoor_checkered_picnic_blanket_4-50.jsonl` | Trigger object changed to checkered picnic blanket (8%) |

Default ground-truth triggers (used only for evaluation, not inside the detector):

- `gray trash bin`
- `gray trash`
- `trash bin`

When switching to a different trigger-object corpus, update `GROUND_TRUTH` in the script.

### Script Notes

Edit these three places in `TriggerLocate_Experiment.py`:

1. `POISON_CORPORA`: path to a corpus  
   e.g. `"06": "data/nusc_eval_backdoor_4-50.jsonl"`
2. `RUN_LIST`: `(poison_key, ablation_key)`  
   e.g. `("06", "full")`
3. `GROUND_TRUTH`: the true trigger phrases (GT) for the current corpus

Built-in ablations:

- `full`: verb filter + stage-2 verification + phrase-level candidates
- `no_verb`: verb filter off
- `stage1_only`: coarse screening only
- `word_level`: unigrams only, no bigram/trigram phrases

## Notes

- Ollama must be running locally; otherwise requests to `localhost:11434` will fail.
- The evaluation files are constructed research samples, not the official full nuScenes dataset.

## Citation

If you use the code or corpora in this repository, please cite the corresponding paper:
