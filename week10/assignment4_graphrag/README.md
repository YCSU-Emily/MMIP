# Week 10 Assignment 4 — GraphRAG

**Course:** MMIP
**Assignment:** GraphRAG (20 Points)
**Paper:** *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*

---

## 1. Project Overview

本專案選擇研究論文 *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*，實作一套以 Knowledge Graph、Community Detection 和 Community Summarization 為核心的 GraphRAG 問答系統。

傳統 Vector RAG 主要透過文字向量相似度檢索相關段落，對於需要跨主題、跨段落整合資訊的 Global Sensemaking 問題可能有所限制。

本專案將論文內容轉換為 Knowledge Graph，利用實體與關係建立知識結構，並透過 Community Detection 和 Community Summarization 建立高層次的知識表示。

最後，結合本地端大型語言模型進行問答，並建立 24 題 Benchmark，與 Vector RAG Baseline 進行對照評估。

> This project is a GraphRAG-inspired implementation based on the core concepts of the research paper. It is not a complete reproduction of the official Microsoft GraphRAG framework.

## 2. Selected Research Paper

**Title:** From Local to Global: A Graph RAG Approach to Query-Focused Summarization

**Authors:** Darren Edge et al.

**arXiv:** https://arxiv.org/abs/2404.16130

**Main Concepts:**
- Knowledge Graph Construction
- Entity and Relationship Extraction
- Community Detection
- Community Summarization
- Map-Reduce Query Processing
- Global Sensemaking

## 3. System Architecture

本專案主要分為五個階段。

### Stage 1 — Paper Parsing and Knowledge Extraction

1. 讀取 PDF 論文。
2. 將論文內容轉換成文字。
3. 使用 Qwen3.5-4B 輔助抽取 Entities 和 Relationships。
4. 對抽取結果進行清理、語意合併與證據檢查。

### Stage 2 — Knowledge Graph Construction

使用 NetworkX 建立有向多重圖（MultiDiGraph）。

- Nodes: 論文中的實體與概念。
- Edges: 實體之間的關係。
- Evidence: 保留原始論文頁碼與相關證據。

### Stage 3 — Community Detection and Summarization

1. 將 Knowledge Graph 投影為加權無向圖。
2. 使用 Leiden Algorithm 進行 Community Detection。
3. 使用 Qwen3.5-4B 為各 Community 生成摘要。
4. 保存 Community Assignments、Community Reports 和 Summaries。

### Stage 4 — GraphRAG Retrieval and Answer Generation

結合圖結構資訊與語意檢索進行問答。

主要流程：

1. Entity Retrieval：根據問題尋找相關實體。
2. Relationship Retrieval：取得相關實體之間的關係。
3. Community Retrieval：尋找相關 Community 與摘要。
4. Evidence Retrieval：回到原始論文 chunks 取得支持證據。
5. Answer Generation：將檢索結果提供給 Qwen3.5-4B 產生回答。

回答要求根據檢索到的原文證據作答，並附上論文頁碼引用。

### Stage 5 — Benchmark Evaluation

建立 24 題與研究論文相關的 Question-Answer Benchmark，分別使用：

- GraphRAG
- Vector RAG Baseline

兩種方法使用相同的 Benchmark 和 Qwen3.5-4B 生成模型，進行逐題比較。

## 4. Implementation Environment

| Component | Configuration |
|---|---|
| Operating System | Ubuntu Linux |
| Python | 3.11.17 |
| Environment | Conda: mmipWeek10 |
| GPU | NVIDIA RTX 5090 |
| Deep Learning Framework | PyTorch |
| LLM | Qwen3.5-4B |
| Embedding Model | BAAI BGE-M3 |
| Knowledge Graph | NetworkX MultiDiGraph |
| Community Detection | Leiden Algorithm |
| Vector Database | ChromaDB |
| Visualization | PyVis |
| Evaluation | Custom Benchmark Evaluation |

模型使用本地 Hugging Face Cache 載入，避免每次推論時重新下載模型。

## 5. Knowledge Graph Results

論文解析與知識抽取後，經過清理及語意合併，得到以下結果：

| Item | Result |
|---|---:|
| Original Paper Pages | 26 |
| Extracted Entities | 75 |
| Relationships | 82 |
| Communities | 20 |
| Community Summaries | 20 |
| Paper Chunks | 95 |

### 5.1 Graph Construction

- Graph Type: NetworkX MultiDiGraph
- Directed Graph: Yes
- Multigraph: Yes
- Nodes: 75
- Edges: 82

使用 MultiDiGraph 保留實體之間可能存在的多種關係。

### 5.2 Community Detection

使用 Leiden Algorithm 進行社群偵測。

- Number of Communities: 20
- Modularity: 0.595168
- Community Summarization: 20/20

Community Summarization 完成全部 20 個社群的摘要生成。

## 6. Benchmark Dataset

本專案建立 24 題與研究論文相關的 Question-Answer Benchmark。

| Difficulty | Number of Questions |
|---|---:|
| Easy | 5 |
| Medium | 16 |
| Hard | 3 |
| **Total** | **24** |

每題包含：

- Question ID
- Topic
- Difficulty
- Question
- Reference Answer
- Reference Pages
- Keywords

題目涵蓋以下主題：

- GraphRAG Motivation and Architecture
- Knowledge Graph Construction
- Community Detection and Summarization
- Global Search and Map-Reduce
- Adaptive Benchmarking
- Evaluation Metrics
- Experimental Design and Results
- Statistical Analysis

Benchmark File:

`results/benchmark.json`

## 7. Experimental Setup

### 7.1 GraphRAG

GraphRAG 使用 Knowledge Graph、Relationships、Community Summaries 及原始論文 chunks 進行檢索。

| Parameter | Value |
|---|---:|
| Top Entities | 8 |
| Top Relationships | 10 |
| Top Communities | 5 |
| Top Evidence Chunks | 7 |

### 7.2 Vector RAG Baseline

Vector RAG 將 95 個論文 chunks 建立 BGE-M3 Embeddings，並儲存於 ChromaDB。

查詢時使用語意相似度檢索 Top-7 chunks，再將結果提供給 Qwen3.5-4B 生成回答。

### 7.3 Fair Comparison

兩種方法使用：

- 相同的 24 題 Benchmark。
- 相同的 Qwen3.5-4B Generation Model。
- 相同的生成設定。
- 相同的引用格式要求。

由於兩種方法的檢索上下文組成不同，因此 Prompt Tokens 和 Generation Time 也可能有所差異。

## 8. Evaluation Metrics

本專案使用以下指標比較 GraphRAG 和 Vector RAG。

| Metric | Description |
|---|---|
| Reference Page Hit Rate | 是否至少檢索到一個參考頁面 |
| Retrieval Page Recall | 檢索到的參考頁面比例 |
| Retrieval Page Precision | 檢索結果中屬於參考頁面的比例 |
| Keyword Coverage | 回答涵蓋 Benchmark 關鍵字的比例 |
| Citation Valid Rate | 引用頁碼通過程式有效性檢查的比例 |
| Citation Precision | 引用有效性精確率 |
| Retrieval Time | 平均檢索時間 |
| Generation Time | 平均回答生成時間 |
| Prompt Tokens | 平均生成提示詞 Token 數 |

**Note:** Keyword Coverage 是詞彙比對的代理指標，不等同於完整語意正確率。Citation Valid Rate 和 Citation Precision 主要反映程式化引用驗證結果，不代表已完成人工逐句證據驗證。

## 9. Experimental Results

正式評估共完成 24 題，GraphRAG 與 Vector RAG 各完成 24 次問答生成。

### 9.1 Overall Results

| Metric | GraphRAG | Vector RAG |
|---|---:|---:|
| Completed Questions | 24 | 24 |
| Reference Page Hit Rate | 91.67% | 91.67% |
| Retrieval Page Recall | **80.42%** | 79.58% |
| Retrieval Page Precision | **25.65%** | 24.26% |
| Keyword Coverage | **66.53%** | 58.28% |
| Citation Valid Rate | **87.50%** | 83.33% |
| Citation Precision | **87.50%** | 83.33% |
| Mean Retrieval Time | 0.0076 s | 0.0083 s |
| Mean Generation Time | 5.5264 s | 4.5074 s |
| Mean Prompt Tokens | 5149.54 | 2398.29 |

### 9.2 Discussion

**GraphRAG Performance**

GraphRAG 在 Retrieval Page Recall、Retrieval Page Precision、Keyword Coverage 和 Citation Valid Rate 上略優於 Vector RAG。

這表示透過 Knowledge Graph 與 Community Summaries，可以在本次 Benchmark 中提供較完整的相關資訊，並提高回答的關鍵字覆蓋率。

**Vector RAG Performance**

Vector RAG 的平均 Generation Time 和 Prompt Tokens 較低。

這代表其檢索上下文較精簡，因此生成模型需要處理的 Token 數量較少。

**Overall Observation**

在本次 24 題實驗中，GraphRAG 在多數檢索及回答代理指標上略有優勢，但生成成本較高。

兩種方法各有優點，不能僅根據目前的平均指標宣稱 GraphRAG 在所有情況下都優於 Vector RAG。

## 10. Limitations

本專案目前仍有以下限制：

1. Knowledge Graph 僅包含選定論文抽取及清理後的 75 個實體與 82 個關係，圖譜規模有限。
2. Leiden Community Detection 的結果受到目前圖譜結構與關係品質影響。
3. Retrieval Page Precision 約為 25%，表示仍有改善檢索精確度的空間。
4. Q04、Q16 兩種方法都沒有命中指定的參考頁面。
5. Q10、Q23 兩種方法都沒有產生有效引用。
6. Keyword Coverage 為詞彙比對指標，無法完整代表回答的語意正確性。
7. Citation Valid Rate 不等同於人工驗證每個回答主張的事實正確性。
8. 目前只使用 24 題 Benchmark，尚未進行統計顯著性檢定。
9. 本實作是參考論文概念建立的 GraphRAG 系統，並非官方 Microsoft GraphRAG 的完整重現。

## 11. Project Structure

```text
week10/assignment4_graphrag/
│
├── README.md
│
├── papers/
│   └── graphrag_paper.pdf
│
├── src/
│   ├── parse_paper.py
│   ├── extract_knowledge.py
│   ├── clean_knowledge.py
│   ├── semantic_cleaning.py
│   ├── build_graph.py
│   ├── detect_communities.py
│   ├── summarize_communities.py
│   ├── repair_community_summaries.py
│   ├── build_paper_chunks.py
│   ├── build_vector_index.py
│   ├── test_vector_search.py
│   ├── query_engine.py
│   ├── answer_generator.py
│   └── evaluate_rag.py
│
└── results/
    ├── benchmark.json
    ├── paper_text.txt
    ├── paper_metadata.json
    ├── extracted_knowledge.json
    ├── cleaned_knowledge.json
    ├── semantic_knowledge.json
    ├── cleaning_report.json
    ├── semantic_review_report.json
    ├── knowledge_graph.graphml
    ├── knowledge_graph.json
    ├── knowledge_graph.html
    ├── community_assignments.json
    ├── community_report.json
    ├── community_summaries.json
    ├── community_summaries_repaired.json
    ├── paper_chunks.json
    ├── chroma_vector_db/
    ├── evaluation_graphrag.json
    ├── evaluation_vector.json
    └── final_evaluation.log
```

## 12. How to Run

### 12.1 Activate Environment

```bash
conda activate mmipWeek10
cd ~/MMIP/week10/assignment4_graphrag
```

### 12.2 Run GraphRAG Query

```bash
python src/answer_generator.py \
  --question "How does GraphRAG use community summaries to answer global questions?"
```

**Note:** The current answer generator checks CUDA availability and uses GPU execution. A compatible NVIDIA GPU and CUDA environment are required.

### 12.3 Run Benchmark Evaluation

```bash
python -u src/evaluate_rag.py --methods both --overwrite
```

**Warning:** `--overwrite` replaces existing evaluation result files. Back up the current results before rerunning the full experiment.

For a partial test:

```bash
python -u src/evaluate_rag.py --methods both --limit 2 --overwrite
```

### 12.4 Evaluation Outputs

The formal evaluation results are saved in:

```text
results/evaluation_graphrag.json
results/evaluation_vector.json
results/final_evaluation.log
```

## 13. Final Conclusion

本專案完成了從論文解析、知識抽取、Knowledge Graph 建構、Community Detection、Community Summarization，到 GraphRAG Retrieval 與 LLM Answer Generation 的完整流程。

此外，建立 24 題與論文相關的 Benchmark，並使用相同題目比較 GraphRAG 與 Vector RAG，完成兩種方法各 24 題的正式評估。

實驗結果顯示，GraphRAG 在本次 Benchmark 的多項檢索及回答代理指標上略優於 Vector RAG；Vector RAG 則具有較低的生成時間與 Token 使用量。

本專案展示了如何利用 Knowledge Graph 與 Community Summarization，補足單純向量檢索在跨主題資訊整合上的不足，也提供了後續改善檢索品質、引用可靠性及語意評估的基礎。
# Week 10 Assignment 4 — GraphRAG
