# MMIP Week 7 Assignment A: LLM Deployment, Prompt Optimization, and Quantization

## 1. Project Overview

This project evaluates an open-source Large Language Model (LLM) through local deployment, instruction-following benchmarks, prompt optimization, and low-bit quantization.

The model used in this project is **Qwen2.5-3B-Instruct**. Experiments are conducted using FP16 and 4-bit NF4 quantization to investigate the trade-offs between response quality, inference latency, generation throughput, and GPU memory consumption.

The project includes a custom benchmark of 20 instruction-following questions covering multiple task categories, including summarization, translation, reasoning, information extraction, classification, code generation, debugging, and knowledge-based question answering.

### Objectives

- Deploy and run an open-source instruction-tuned LLM locally.
- Evaluate the model using 20 custom instruction-following questions.
- Investigate the effect of structured prompt optimization.
- Quantize the model to 4-bit NF4 and evaluate its performance.
- Compare FP16 and 4-bit models in terms of latency, generation throughput, keyword coverage, and GPU memory usage.

## 2. Model and Experimental Environment

| Component | Configuration |
|---|---|
| Model | Qwen/Qwen2.5-3B-Instruct |
| GPU | NVIDIA GeForce RTX 5090 |
| Python | 3.11.17 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.19.0 |
| bitsandbytes | 0.50.2 |
| Default precision | FP16 |
| Quantization method | 4-bit NF4 |
| Double quantization | Enabled |
| Quantized compute dtype | FP16 |
| Benchmark size | 20 questions |
| Maximum new tokens | 160 |
| Decoding strategy | Greedy decoding |

The experiments were conducted in the `mmipQuant` Conda environment.

## 3. Project Structure

```text
Assignment_A_LLM/
├── benchmarks/
│   └── instruction_benchmark.json
├── scripts/
│   ├── 01_run_baseline.py
│   ├── 02_run_instruction_benchmark.py
│   ├── 03_run_optimized_prompt.py
│   ├── 04_compare_prompt_results.py
│   ├── 05_test_4bit_quantization.py
│   ├── 06_run_4bit_instruction_benchmark.py
│   ├── 07_compare_fp16_vs_4bit.py
│   ├── 08_run_4bit_optimized_prompt.py
│   └── 09_run_fp16_optimized_memory.py
└── results/
    ├── baseline_results.json
    ├── fp16_instruction_results.json
    ├── fp16_instruction_results.csv
    ├── optimized_prompt_results.json
    ├── optimized_prompt_results.csv
    ├── prompt_comparison.json
    ├── 4bit_smoke_test.json
    ├── 4bit_instruction_results.json
    ├── 4bit_instruction_results.csv
    ├── fp16_vs_4bit_comparison.json
    ├── fp16_vs_4bit_per_question.csv
    ├── 4bit_optimized_prompt_results.json
    ├── 4bit_optimized_prompt_results.csv
    ├── fp16_optimized_memory_results.json
    └── fp16_optimized_memory_results.csv
```

The JSON files contain experiment metadata and detailed results. CSV files provide tabular results for further analysis.

## 4. Benchmark Dataset

The benchmark dataset is stored in:

`benchmarks/instruction_benchmark.json`

It contains 20 custom questions. Each question includes a task category, prompt, reference answer, and evaluation keywords.

The benchmark covers a variety of instruction-following tasks, such as:

- Instruction following
- Summarization and translation
- English text generation
- Mathematical and logical reasoning
- Information extraction and classification
- Structured output generation
- Code generation and debugging
- Concept explanation and knowledge-based question answering
- Uncertainty handling and data safety
- Causal analysis and creative generation

### Evaluation Metric

**Keyword Coverage** measures the proportion of predefined evaluation keywords that appear in a generated answer.

It is calculated as:

`Keyword Coverage = Matched Keywords / Total Evaluation Keywords`

This metric provides a simple, reproducible quantitative comparison. However, it is a string-matching proxy rather than a comprehensive measure of semantic correctness or answer quality.

## 5. Experimental Methods

### 5.1 FP16 Baseline

The original model is loaded in FP16 precision and evaluated using the original system prompt.

The baseline experiment measures inference latency, generated tokens, generation throughput, and keyword coverage.

### 5.2 Prompt Optimization

A structured system prompt is introduced to improve instruction adherence. It asks the model to identify the main task and constraints, follow the requested language and output format, acknowledge insufficient information, check mathematical or programming results, and provide only the requested final answer.

The optimized prompt is evaluated using the same 20-question benchmark.

### 5.3 4-bit NF4 Quantization

The model is loaded using bitsandbytes 4-bit NF4 quantization with double quantization enabled and FP16 computation.

The quantized model is evaluated with the original system prompt and the optimized system prompt. This allows the effects of quantization and prompt optimization to be investigated separately.

### 5.4 GPU Memory Measurement

GPU memory usage is measured using PyTorch CUDA memory statistics.

The measurements include:

- GPU memory allocated after model loading.
- Average per-question peak allocated memory.
- Maximum per-question peak allocated memory.

These measurements represent PyTorch-tracked GPU memory, not total GPU utilization reported by system monitoring tools.

## 6. How to Run

### 6.1 Activate the Environment

```bash
conda activate mmipQuant
cd ~/MMIP/Week7/Assignment_A_LLM
```

Install the required dependencies if they are not already available in your environment. The model may need to be downloaded from Hugging Face on its first run.

### 6.2 Run the FP16 Baseline

```bash
python scripts/01_run_baseline.py
```

### 6.3 Run the FP16 Instruction Benchmark

```bash
python scripts/02_run_instruction_benchmark.py
```

### 6.4 Run Prompt Optimization

```bash
python scripts/03_run_optimized_prompt.py
python scripts/04_compare_prompt_results.py
```

### 6.5 Test and Benchmark 4-bit Quantization

```bash
python scripts/05_test_4bit_quantization.py
python scripts/06_run_4bit_instruction_benchmark.py
python scripts/07_compare_fp16_vs_4bit.py
```

### 6.6 Run the Optimized Prompt with 4-bit Quantization

```bash
python scripts/08_run_4bit_optimized_prompt.py
```

### 6.7 Measure FP16 GPU Memory Usage

```bash
python scripts/09_run_fp16_optimized_memory.py
```

Each benchmark script saves its results to the `results/` directory. The comparison scripts analyze the corresponding experiment outputs.

## 7. Experimental Results

The following table compares the FP16 model with the optimized prompt against the 4-bit NF4 model using the same optimized prompt.

| Metric | FP16 + Optimized Prompt | 4-bit NF4 + Optimized Prompt |
|---|---:|---:|
| Completed questions | 20/20 | 20/20 |
| Model load time | 3.37 s | 4.17 s |
| GPU allocated after loading | 5.748 GiB | 1.923 GiB |
| Average peak allocated memory | 5.799 GiB | 2.017 GiB |
| Maximum peak allocated memory | 5.804 GiB | 2.021 GiB |
| Average latency per question | 0.7344 s | 1.1397 s |
| Overall generation throughput | 91.24 tokens/s | 60.98 tokens/s |
| Average keyword coverage | 70.00% | 67.92% |

### 7.1 Memory Efficiency

After model loading, GPU allocated memory decreased from 5.748 GiB with FP16 to 1.923 GiB with 4-bit NF4 quantization.

This represents an approximately **66.5% reduction in allocated GPU memory**.

Average per-question peak allocated memory decreased from 5.799 GiB to 2.017 GiB, corresponding to an approximately **65.2% reduction**.

These results demonstrate the memory-efficiency benefits of low-bit quantization for local LLM deployment.

### 7.2 Inference Performance

The FP16 model achieved an average latency of 0.7344 seconds per question, while the 4-bit model required 1.1397 seconds.

The measured generation throughput decreased from 91.24 tokens/s to 60.98 tokens/s after quantization.

Therefore, under the current experimental setup, 4-bit quantization reduced GPU memory consumption but did not improve inference speed.

### 7.3 Keyword Coverage

The FP16 model achieved 70.00% keyword coverage, compared with 67.92% for the 4-bit model.

The difference is relatively small, but keyword matching alone cannot establish whether the two models have equivalent semantic accuracy. A more comprehensive evaluation would require additional correctness criteria or human evaluation.

## 8. Discussion

The experiments reveal a trade-off between memory efficiency and inference performance.

First, structured prompt optimization reduced average latency in the FP16 benchmark from 0.8985 seconds to 0.7337 seconds per question. However, overall generation throughput and keyword coverage did not improve, so the result should not be interpreted as a general improvement in answer quality.

Second, 4-bit NF4 quantization substantially reduced GPU memory consumption. The reduction may be beneficial when deploying LLMs on devices with limited memory.

Nevertheless, the quantized model was slower in the current environment and achieved slightly lower keyword coverage. Quantization performance can depend on hardware, inference kernels, framework implementation, and generation characteristics, so these results should not be generalized to every deployment environment.

Finally, keyword coverage is a limited evaluation metric. Future work could include semantic evaluation, human assessment, repeated performance measurements, and additional quantization configurations.

## 9. Conclusion

This project implemented a local LLM evaluation workflow covering baseline inference, prompt optimization, 4-bit quantization, and GPU memory profiling.

The main finding is that 4-bit NF4 quantization reduced model-loading GPU allocated memory by approximately 66.5% compared with FP16 under the tested conditions. However, this memory reduction came with increased latency and lower generation throughput.

The results highlight the importance of evaluating memory consumption, inference performance, and answer quality together when selecting an LLM deployment configuration.

## 10. Reproducibility Notes

- All primary benchmark comparisons use the same 20-question dataset.
- Both optimized-prompt configurations use the same system prompt and maximum generation length.
- Greedy decoding is used to reduce variability from sampling.
- GPU memory measurements are based on PyTorch CUDA memory statistics.
- Initial model-download time and cached model-loading time are not directly comparable.
- Results may vary with software versions, GPU state, hardware, and runtime conditions.

