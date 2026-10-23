# DPO Fine-Tuning and Response Quality Evaluation

## 1. Overview

This project compares the response quality of a baseline instruction-tuned language model and a Direct Preference Optimization (DPO)-fine-tuned model.

The base model is `Qwen/Qwen2.5-3B-Instruct`. The DPO model uses a LoRA adapter saved in `models/dpo_lora_v1`.

The project covers preference-data preparation, DPO training, automated model evaluation, paired blind annotation, and statistical analysis.

## 2. Project Workflow

```text
Preference Data
      |
      v
Data Cleaning and Preparation
      |
      v
DPO + LoRA Fine-Tuning
      |
      v
Baseline and DPO Evaluation
      |
      v
Blind Annotation of 100 Paired Responses
      |
      v
Exact McNemar Tests
      |
      v
Results and Discussion
```

## 3. Repository Structure

```text
Assignment_C_DPO_PPO/
├── annotation/       # Manual annotations and blind mapping key
├── data/
│   ├── benchmark/    # Target and general evaluation questions
│   └── preference/   # Preference-training datasets
├── logs/             # Training and evaluation logs
├── models/           # LoRA adapters
├── results/          # Predictions and evaluation summaries
├── scripts/          # Data preparation, training, evaluation, analysis
└── README.md
```

## 4. Model and Training Configuration

### Base Model

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Fine-tuning method: DPO with LoRA
- Adapter directory: `models/dpo_lora_v1`

### Training Parameters

The training script, `scripts/train_dpo.py`, specifies the following settings:

| Parameter | Value |
|---|---:|
| Learning rate | 2e-5 (default) |
| Number of epochs | 3 |
| Per-device batch size | 8 |
| Gradient accumulation steps | 1 |
| DPO beta | 0.1 |
| Maximum sequence length | 256 |
| Warmup steps | 5 |

The script also accepts `--out` to specify the output directory and `--max_steps` to override the default step limit.

### Evaluation Configuration

The recorded evaluation summaries specify:

| Setting | Value |
|---|---|
| Precision | `torch.bfloat16` |
| Sampling | Disabled |
| Maximum new tokens | 256 |
| Target questions per model | 100 |
| General questions per model | 100 |
| Total questions per model | 200 |

Training `max_length=256` and generation `max_new_tokens=256` are different parameters.

## 5. Manual Blind Evaluation

A total of 100 target questions were manually annotated: 10 pilot questions and 90 additional questions.

The two model responses were presented as A/B for each question. The model mapping was stored in `annotation/target_blind_key.json` and used after the annotation-completeness audit.

Three binary criteria were used:

1. **Factual correctness:** Whether the core conclusion is correct and contains no important factual error.
2. **Fabricated or false detail:** Whether the response contains a clearly false or fabricated additional detail.
3. **Appropriate uncertainty:** Whether the response appropriately acknowledges uncertainty or handles a false premise instead of confidently inventing information.

The annotation files used in the final calculation are:

- `annotation/target_pilot_10.csv`
- `annotation/target_remaining_90.csv`
- `annotation/target_blind_key.json`

The original blind annotation file is retained at `annotation/target_blind_annotation.csv`.

## 6. Manual Evaluation Results

The following results combine the pilot and remaining questions.

| Metric | Baseline | DPO | Observed Difference |
|---|---:|---:|---:|
| Factual correctness | 84% | 90% | +6 percentage points |
| Responses containing fabricated or false details | 19% | 10% | -9 percentage points |
| Appropriate uncertainty handling | 98% | 95% | -3 percentage points |

### Paired Comparison

- **Factual correctness:** DPO performed better on 12 questions, while Baseline performed better on 6; 82 pairs were tied.
- **Fabricated or false details:** DPO had the better outcome on 15 questions, while Baseline had the better outcome on 6; 79 pairs were tied.
- **Appropriate uncertainty:** Baseline performed better on 4 questions, while DPO performed better on 1; 95 pairs were tied.

These percentages describe the annotated sample and do not independently establish statistical significance.

## 7. Statistical Significance Testing

Two-sided exact McNemar tests were performed on the paired binary labels, using a significance threshold of alpha = 0.05.

| Metric | Exact p-value | Significant at 0.05? |
|---|---:|---|
| Factual correctness | 0.237885 | No |
| Fabricated or false details | 0.078354 | No |
| Appropriate uncertainty handling | 0.375000 | No |

Although DPO showed a lower observed rate of responses containing fabricated or false details, the difference did not reach statistical significance (p = 0.078354).

None of the three tests reached the 0.05 significance threshold. The tests were not adjusted for multiple comparisons, so these findings should be treated as exploratory.

Run the statistical analysis from the project root:

```bash
python scripts/mcnemar_test.py
```

The script reads the annotation CSV files and blind mapping key without modifying them.

## 8. Automated Evaluation Artifacts

Recorded evaluation summaries and predictions are stored in:

- `results/baseline_256/evaluation_summary.json`
- `results/baseline_256/target_predictions.jsonl`
- `results/baseline_256/general_predictions.jsonl`
- `results/dpo_256/evaluation_summary.json`
- `results/dpo_256/target_predictions.jsonl`
- `results/dpo_256/general_predictions.jsonl`
- `results/final_general_summary.json`

The general benchmark summary includes strict and lenient scoring results. These automated scores are distinct from the manually annotated metrics and should be interpreted according to the corresponding evaluation scripts.

## 9. Environment and Reproducibility

The experiments were conducted in the Conda environment named `mmipVLM`.

Activate the existing environment and enter the project directory:

```bash
conda activate mmipVLM
cd ~/MMIP/Week7/Assignment_C_DPO_PPO
```

Inspect the training script's available arguments without starting a training run:

```bash
python scripts/train_dpo.py --help
```

This repository snapshot does not include a root-level `requirements.txt`, `environment.yml`, or `pyproject.toml`. Package versions should be checked in the original environment before attempting to reproduce training.

Reproduction may require access to the base model, preference data, trained adapter, and compatible software and hardware.

## 10. Limitations

- The manual evaluation covers 100 target questions and may not represent all prompts or use cases.
- Human annotation involves judgment and may be subject to annotation variability.
- None of the three paired statistical tests reached the 0.05 significance threshold.
- The tests were not adjusted for multiple comparisons.
- Automated benchmark scores and manual annotation results use different evaluation procedures and should not be conflated.
- Reproducing the experiment may require the original environment and model assets.

## 11. Conclusion

In this 100-question manual evaluation, DPO achieved higher factual correctness and a lower rate of responses containing fabricated or false details than the baseline. However, its appropriate-uncertainty rate was slightly lower.

None of the observed paired differences was statistically significant at the 0.05 level. The results therefore provide descriptive, exploratory evidence rather than a definitive demonstration of improvement.
