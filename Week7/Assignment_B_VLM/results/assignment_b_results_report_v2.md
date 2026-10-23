# Assignment B: Visual Multimodal Large Language Models

## 1. Objective

This experiment compares two open-source vision-language models (VLMs)
on a custom benchmark of 20 visual-understanding questions. The evaluation
includes inference success, inference latency, generated token count,
and preliminary manual answer scoring.

The commercial LLM comparison has not yet been completed and is not
included in the measured results.

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

The benchmark uses the same 20 images and English prompts for both models.
Task categories include shape and color recognition, counting, spatial
relations, comparison, visual reasoning, and optical character recognition.

## 3. Quantitative Results

| Metric | Qwen2.5-VL-3B | SmolVLM2-2.2B |
|---|---:|---:|
| Successful inferences | 20/20 (100%) | 20/20 (100%) |
| Mean inference time | 0.2049 s | 0.3434 s |
| Mean generated tokens | 13.95 | 4.75 |
| Preliminary manual score | 34/40 (85.0%) | 30/40 (75.0%) |
| Mean score per question | 1.70/2 | 1.50/2 |

Qwen's measured mean inference time was approximately 40.3%
lower than SmolVLM2's in this experiment. SmolVLM2 generated fewer tokens
on average, which is consistent with its shorter answers in this benchmark.
The models' response lengths differ, so latency should be interpreted
alongside answer completeness.

## 4. Scoring Rubric

Each answer was scored from 0 to 2:

- 2: Correct and sufficiently complete for the question.
- 1: Partially correct or missing requested information.
- 0: Incorrect on the core answer.

Concise answers can receive full credit when the prompt specifies an order
that makes the answer unambiguous. For example, in Q14 the prompt lists
circle, square, and triangle in that order, and SmolVLM2 responds
"yellow, purple, green." In Q15 the prompt asks for the largest and then
the smallest square, and the answer "green, red" follows that order.
Both answers received 2 points under this rubric.

The scores are preliminary manual judgments rather than standardized
accuracy measurements.

## 5. Category-wise Performance

The following percentages are normalized rubric scores: total points
divided by the maximum possible points in each category.

| Category | Questions | Qwen2.5-VL-3B | SmolVLM2-2.2B |
| comparison | 2 | 100.0% | 100.0% |
| counting | 4 | 87.5% | 62.5% |
| ocr | 3 | 100.0% | 100.0% |
| shape_color | 3 | 66.7% | 83.3% |
| spatial_relation | 5 | 90.0% | 80.0% |
| visual_reasoning | 3 | 66.7% | 33.3% |

## 6. Error Analysis

The reviewed responses illustrate several types of mistakes:

- Shape confusion, such as describing a square as a rectangle.
- Incorrect object counts.
- Answers that give the right colors or shapes but omit requested details.
- Incomplete descriptions of spatial relationships or object arrangements.

In this benchmark, both models answered the OCR questions correctly under
the current manual rubric. Qwen2.5-VL-3B received higher normalized scores
in counting, comparison, and visual reasoning. These are observations from
a small custom benchmark and should not be generalized to all visual tasks.

## 7. Commercial LLM Comparison

The commercial LLM comparison remains pending because the available
ChatGPT free-tier image-upload/message allowance is limited. No commercial
model score has been fabricated or included.

For a fair follow-up comparison, use the same images, prompts, and scoring
rubric. Record the actual commercial model name and number of questions
tested. If only a subset is tested, report it explicitly as a subset.

Response time measured through a web interface may include network and
service overhead, so it should not be treated as directly equivalent to
local inference latency.

## 8. Conclusion

Both open-source VLMs completed all 20 inference attempts successfully.
Under the current preliminary rubric, Qwen2.5-VL-3B scored
85.0%, while SmolVLM2-2.2B scored 75.0%.
Qwen also had a lower measured mean inference time in this experiment.

The results are limited to the current 20-question benchmark and testing
environment. Manual scoring is not a standardized accuracy metric, and
the commercial LLM comparison remains incomplete.
