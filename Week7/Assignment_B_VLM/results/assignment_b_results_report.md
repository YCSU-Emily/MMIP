# Assignment B: Visual Multimodal Large Language Models

## 1. Objective

This experiment evaluates two open-source vision-language models (VLMs)
on a custom benchmark of 20 visual-understanding questions. The evaluation
considers inference success, inference latency, generated token count,
and preliminary human-assigned answer scores.

A commercial LLM comparison remains pending and is not included in the
measured results below.

## 2. Models and Experimental Setup

| Item | Qwen2.5-VL-3B | SmolVLM2-2.2B |
|---|---|---|
| Model ID | Qwen/Qwen2.5-VL-3B-Instruct | HuggingFaceTB/SmolVLM2-2.2B-Instruct |
| Benchmark questions | 20 | 20 |
| Successful inferences | 20 | 20 |
| Inference errors | 0 | 0 |
| Model load time | 2.980 s | 2.964 s |
| Mean inference time | 0.2049 s | 0.3434 s |
| Mean generated tokens | 13.95 | 4.75 |

The benchmark contains 20 images and English prompts covering shape and
color recognition, object counting, relative position, visual attributes,
and text recognition. Both models were evaluated using the same benchmark
prompts and images.

## 3. Quantitative Results

| Metric | Qwen2.5-VL-3B | SmolVLM2-2.2B |
|---|---:|---:|
| Successful inferences | 20/20 (100%) | 20/20 (100%) |
| Mean inference time | 0.2049 s | 0.3434 s |
| Mean generated tokens | 13.95 | 4.75 |
| Preliminary human score | 34/40 (85.0%) | 28/40 (70.0%) |
| Mean score per question | 1.70/2 | 1.40/2 |

Qwen's measured mean inference time was approximately 40.3%
lower than SmolVLM2's in this experiment. SmolVLM2 generated fewer tokens
on average, indicating shorter answers in this benchmark.

## 4. Preliminary Human Evaluation

Each answer was assigned a score from 0 to 2:

- 2: Correct and sufficiently complete.
- 1: Partially correct or missing requested information.
- 0: Incorrect on the core answer.

The resulting scores are preliminary manual judgments, not a standardized
accuracy metric. The scoring should be reviewed consistently against the
images, reference answers, and information explicitly requested by each
question.

## 5. Error Analysis

The observed answer examples indicate several useful error categories:

- Shape confusion, such as describing a square as a rectangle.
- Incorrect object counts.
- Answers that identify the correct color or shape but omit other requested
  information.
- Answers that provide only part of a spatial relationship or description.

These categories should be quantified from the reviewed answers before
making stronger claims about which error type is most common.

## 6. Commercial LLM Comparison

The commercial LLM comparison has not yet been completed because the
available ChatGPT free-tier image-upload/message allowance is limited.
No commercial-model scores are reported in this draft.

If a commercial model is evaluated later, the same images, prompts, and
0–2 scoring rubric should be used where possible. The report should record
the actual model name and the number of questions tested. If only a subset
is evaluated, it must be reported as a subset rather than a complete
20-question comparison.

Web-interface response time should not be treated as pure model inference
latency because it may include network and service overhead.

## 7. Conclusion

Both open-source VLMs successfully completed all 20 inference attempts.
In this preliminary evaluation, Qwen2.5-VL-3B had a lower measured mean
inference time and a higher manual score than SmolVLM2-2.2B. SmolVLM2-2.2B
produced shorter responses on average.

These findings apply only to the current benchmark and test environment.
The manual scoring requires consistent review, and the commercial LLM
comparison remains incomplete.
