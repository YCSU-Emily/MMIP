# VideoRAG: Evidence-Grounded Multimodal Video Question Answering

## 1. Project Overview

This project implements a multimodal Video Retrieval-Augmented Generation (VideoRAG) system for video-based question answering.

The system uses two self-recorded Wi-Fi troubleshooting videos:

- A speech video containing spoken instructions.
- A silent video containing visual information and on-screen text.

Instead of directly asking a language model to answer questions about an entire video, the system first converts video content into timestamped textual evidence. It then indexes the evidence in a vector database and retrieves relevant information according to the user's question.

The retrieved evidence is passed to a local large language model to generate answers. Each answer is associated with its corresponding evidence ID and video timestamp.

The system also supports multi-turn conversations, allowing users to ask follow-up questions without repeating the original topic.

## 2. Project Objectives

The main objectives are:

- Build a multimodal video knowledge base using speech transcripts and visual OCR.
- Convert video content into timestamped and searchable evidence records.
- Generate semantic embeddings using BGE-M3.
- Store and retrieve video evidence using ChromaDB.
- Improve retrieval quality through modality-aware retrieval and Reciprocal Rank Fusion (RRF).
- Implement evidence-first answer generation using Qwen2.5-3B-Instruct.
- Reduce unsupported answers through evidence validation and source-grounding checks.
- Support context-aware multi-turn conversations.
- Provide an interactive command-line interface for video question answering.
- Save conversation sessions as structured JSON reports.

## 3. Dataset

### 3.1 Self-Recorded Videos

The dataset consists of two self-recorded videos about smartphone Wi-Fi troubleshooting.

| Video | Modality | Duration |
|---|---|---:|
| `wifi_troubleshooting_speech.mp4` | Speech + video | 74.945 seconds |
| `wifi_troubleshooting_silent.mp4` | Silent video + visual OCR | 26.177 seconds |

### 3.2 Multimodal Evidence Extraction

The system extracts two types of textual evidence.

**Speech Transcript**

Speech content is converted into timestamped transcript segments.

Each speech record contains:

- Record ID
- Video ID
- Start time
- End time
- Transcript text
- Source file
- Language and transcription model metadata

**Visual OCR**

The silent video is processed to extract readable text from selected frames.

The OCR pipeline uses Tesseract with:

- Traditional Chinese (`chi_tra`)
- English (`eng`)

Each visual record contains:

- Record ID
- Video ID
- Frame timestamp
- Extracted text
- Frame source path
- OCR metadata

### 3.3 Corpus Statistics

The generated video corpus contains 22 evidence records.

| Evidence Type | Number of Records |
|---|---:|
| Speech Transcript | 8 |
| Visual OCR | 14 |
| **Total** | **22** |

The records are stored in both JSON and JSONL formats:

```text
outputs/
├── video_corpus.json
└── video_corpus.jsonl
```

## 4. System Architecture

```text
              Self-Recorded Videos
                       |
             +---------+---------+
             |                   |
        Speech Video        Silent Video
             |                   |
       Speech Transcript     Frame Extraction
             |                   |
             |                  OCR
             |                   |
             +---------+---------+
                       |
              Evidence Construction
                       |
              Timestamped Corpus
                       |
                BGE-M3 Embedding
                 (CPU Inference)
                       |
                 ChromaDB Store
                       |
                User Question
                       |
              Multi-Query Retrieval
                       |
             Modality-Aware Search
                       |
             Reciprocal Rank Fusion
                       |
              Evidence Matching
                       |
              Evidence Extraction
                       |
             Qwen2.5-3B-Instruct
                       |
             Source-Grounded Answer
                       |
             Traditional Chinese
                       |
               Interactive CLI
                       |
              JSON Session Report
```

## 5. Models and Technologies

| Component | Technology | Purpose |
|---|---|---|
| Text Embedding | BGE-M3 | Semantic embedding and query-evidence matching |
| Vector Database | ChromaDB | Persistent vector storage and retrieval |
| Similarity Metric | Cosine Similarity | Semantic similarity comparison |
| Language Model | Qwen2.5-3B-Instruct | Evidence-grounded answer generation |
| OCR | Tesseract | Extract text from video frames |
| Chinese Conversion | OpenCC | Convert generated responses into Traditional Chinese |
| Deep Learning Framework | PyTorch | Model inference |
| Model Framework | Hugging Face Transformers | Load and run the language model |

### Model Execution Configuration

- BGE-M3 is configured to run on CPU.
- Qwen2.5-3B-Instruct was tested on CUDA GPU.
- Models are loaded from local Hugging Face cache directories.

The current tested configuration is not a fully CPU-only inference pipeline.

## 6. Retrieval Pipeline

### 6.1 Modality-Aware Retrieval

The retriever searches the two evidence modalities independently:

1. Speech transcript records.
2. Visual OCR records.

The current retriever uses a configurable number of candidates per modality.

### 6.2 Reciprocal Rank Fusion

The retrieved results are combined using Reciprocal Rank Fusion (RRF).

The RRF constant is set to 60.

The final ranking considers the retrieval rank from each modality rather than relying only on a single retrieval result.

### 6.3 Multi-Query Retrieval

The system generates multiple query variants to improve evidence coverage.

The retrieved candidates are fused and ranked, with the interactive pipeline selecting up to eight final evidence records.

### 6.4 Query-Evidence Matching

BGE-M3 is also used to compare retrieved evidence with the current query variants.

The system selects up to two relevant query-evidence matches for subsequent answer generation.

## 7. Evidence-First Answer Generation

The answer generation pipeline uses a two-stage approach.

### Stage 1: Evidence Fact Extraction

The language model independently extracts factual information from each retrieved evidence record.

The extracted facts retain their original evidence IDs and timestamps.

### Stage 2: Answer Generation

Qwen2.5-3B-Instruct generates an answer using the extracted evidence facts.

The system attempts relevant query matches in ranked order.

### Source-Grounding Validation

Generated answers are checked against their original evidence.

The system also includes:

- Rejection of empty or invalid model outputs.
- Source-consistency validation.
- Duplicate-answer filtering.
- Safe fallback responses when the evidence does not support an answer.

This design aims to reduce unsupported statements and keep answers grounded in the video content.

## 8. Multi-Turn Conversation

The project implements a rule-based `ConversationManager`.

It maintains:

- The current conversation topic.
- User questions.
- Rewritten follow-up queries.
- Conversation history.

For example:

**Turn 1**

User: My phone cannot connect to Wi-Fi. What should I do?

**Turn 2**

User: What if it still does not work?

The conversation manager identifies the second question as a follow-up and incorporates the previous topic into the retrieval query.

Users can also enter:

- `reset`: Clear the current conversation history.
- `exit`: End the interactive session.

## 9. Interactive Command-Line Interface

The interactive application is implemented in:

```text
src/chat_videorag_cli.py
```

Run the application from the Assignment 2 directory:

```bash
cd ~/MMIP/week10/assignment2_videorag

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmipWeek10

python src/chat_videorag_cli.py
```

After initialization, users can enter questions directly.

Example:

```text
你：我的手機 Wi-Fi 無法連線，怎麼辦？

VideoRAG：
如果 Wi-Fi 已經開啟但手機仍然無法連線，
可以先關閉 Wi-Fi，等待幾秒鐘後再重新開啟。
[證據 ID: speech_0002，影片時間：20.00–33.00 秒]

你：那如果還是不行呢？
```

The application automatically saves each completed conversation turn.

## 10. Output Files

The project stores intermediate data, vector database files, experiment reports, and interactive session records under `outputs/`.

```text
outputs/
├── video_corpus.json
├── video_corpus.jsonl
├── videorag_chroma_db/
├── interactive_sessions/
│   └── videorag_session_<timestamp>.json
├── v7_opencc_retry_multiturn_videorag_report.json
├── v7_opencc_retry_multiturn_videorag_answers.txt
└── v7_opencc_retry_multiturn_execution.log
```

The interactive session report records:

- User questions
- Rewritten queries
- Follow-up detection
- Retrieved evidence
- Extracted facts
- Generation attempts
- Final answers
- Evidence counts and answer statistics

## 11. Verification and Experimental Results

The interactive CLI was tested using a three-turn conversation about smartphone Wi-Fi troubleshooting.

The test included:

1. Asking how to troubleshoot a smartphone Wi-Fi connection.
2. Asking a follow-up question about what to do if the problem persists.
3. Asking how to verify whether the smartphone has successfully connected to Wi-Fi.

**Turn 1:** The system retrieved speech evidence `speech_0002` (20.00–33.00 seconds) and generated an answer suggesting that the user turn Wi-Fi off, wait a few seconds, and turn it back on.

**Turn 2:** The conversation manager successfully identified the follow-up question and incorporated the previous topic into the rewritten query. The system retrieved `speech_0005` (50.00–64.00 seconds) and generated a relevant answer.

**Turn 3:** The system retrieved four evidence records from both speech and visual OCR sources. Although the retrieved evidence provided information relevant to the question, multiple similar answers were accepted, resulting in redundant content.

The complete three-turn interaction was saved as a JSON session report under `outputs/interactive_sessions/`.

These results demonstrate that the interactive retrieval and multi-turn workflow is operational. However, evidence aggregation and answer redundancy reduction remain areas for improvement.

## 12. Current Limitations and Future Work

Current limitations:

- The current corpus contains only 22 evidence records from two self-recorded videos.
- OCR quality depends on the readability of the selected video frames.
- Follow-up detection currently uses rule-based cues.
- Multiple relevant evidence records can produce redundant answers.
- Qwen2.5-3B-Instruct was tested with CUDA GPU inference; CPU-only generation has not yet been established.
- The project does not currently include a standalone dependency manifest.

Future improvements:

- Expand the video dataset and increase evidence coverage.
- Improve frame sampling and visual evidence extraction.
- Improve query rewriting and follow-up detection.
- Introduce evidence aggregation to reduce repetitive answers.
- Improve answer evaluation using a larger set of test questions.
- Add a reproducible dependency and environment configuration.
- Investigate optimized CPU inference.

## 13. Project Structure

```text
assignment2_videorag/
├── src/
│   ├── build_video_corpus.py
│   ├── build_vector_store.py
│   ├── retriever.py
│   ├── conversation_manager.py
│   ├── evidence_query_matcher.py
│   ├── answer_generator.py
│   ├── answer_generator_evidence_first_v4.py
│   ├── chat_videorag_cli.py
│   └── ...
├── videos/
│   ├── speech/
│   └── silent/
├── transcripts/
│   ├── wifi_troubleshooting_speech.json
│   └── wifi_troubleshooting_silent_ocr.json
├── outputs/
│   ├── video_corpus.json
│   ├── video_corpus.jsonl
│   ├── videorag_chroma_db/
│   └── interactive_sessions/
├── backups/
└── README.md
```

## 14. Summary

This project demonstrates an evidence-grounded multimodal VideoRAG system that integrates speech transcripts, visual OCR, semantic retrieval, vector database storage, and local language model generation.

By combining modality-aware retrieval, Reciprocal Rank Fusion, evidence-first generation, source validation, and multi-turn conversation management, the system provides an interactive approach to answering questions based on video content.

The current implementation establishes a working end-to-end prototype and provides a foundation for further improvements in retrieval quality, answer reliability, and inference efficiency.
