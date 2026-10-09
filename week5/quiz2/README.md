# Week 5 Quiz 2: Multimodal Deep Learning

## 1. Project Overview

This project implements a multimodal, multi-label image classification system.
It combines visual features extracted from an image backbone with textual
features extracted from a pretrained language model.

The project compares two image backbones while keeping the text backbone,
data split, training configuration, and test prediction threshold consistent.

### Objectives

- Prepare an image-text dataset from COCO 2017 validation data.
- Extract image and text features using pretrained backbones.
- Fuse multimodal features for 80-class multi-label classification.
- Train and evaluate a baseline model.
- Replace the image backbone and compare the results.

## 2. Dataset

The dataset is derived from COCO 2017 validation images, captions, and instance
annotations. It is an internal coursework split, not the official COCO test set.

| Property | Value |
|---|---:|
| Images with usable annotations | 4,952 |
| Training images | 3,466 |
| Validation images | 742 |
| Test images | 744 |
| Number of categories | 80 |
| Random seed for splitting | 42 |

Images without instance labels are excluded rather than treated as images with
no positive labels. One caption is selected per image using a deterministic
sorting rule. Splits are created by image ID, with no image overlap between
train, validation, and test.

### Expected raw-data layout

Download the COCO 2017 validation images and the COCO 2017 train/validation
annotations. Extract them so the following files and directory exist:

```text
data/raw/
├── val2017/
│   └── val2017/
│       └── <image files>
└── annotations/
    └── annotations/
        ├── captions_val2017.json
        └── instances_val2017.json
```

For example, if the ZIP archives are placed in `data/raw/`:

```bash
unzip data/raw/val2017.zip -d data/raw/
unzip data/raw/annotations_trainval2017.zip -d data/raw/annotations/
```

The archive contents must produce the directory structure shown above.

### Preprocessing

- Resize images to 224 × 224.
- Normalize images using ImageNet statistics.
- Tokenize captions with `distilbert/distilbert-base-uncased`.
- Limit text sequences to 64 tokens.
- Encode labels as 80-dimensional multi-hot vectors.
- Save train, validation, and test records as JSONL files.

The processed records store image paths relative to the project directory.
The raw images must therefore remain available when loading the dataset.

### Prepare the dataset

Run from the `week5/quiz2` directory:

```bash
python scripts/prepare_dataset.py
```

The script creates:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `data/processed/label_map.json`
- `data/processed/dataset_summary.json`

**Important:** The script intentionally stops if these output files already
exist, to avoid overwriting existing data. Do not rerun it over an existing
processed dataset. For a fresh reproduction, use a clean project copy or
preserve the existing files and prepare the data in a separate copy.

## 3. Model Architecture

Both experiments use a pretrained DistilBERT text encoder. The image and text
features are concatenated and passed to a trainable classification head.
The pretrained image and text encoders are frozen during training.

### Baseline: ResNet18 + DistilBERT

- Image backbone: pretrained ResNet18.
- Image feature dimension: 512.
- Text backbone: pretrained DistilBERT.
- Text feature dimension: 768.
- Fused feature dimension: 1,280.
- Classifier: `Linear(1280, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 80)`.

### Advanced experiment: ResNet50 + DistilBERT

- Image backbone: pretrained ResNet50.
- Image feature dimension: 2,048.
- Text backbone: pretrained DistilBERT.
- Text feature dimension: 768.
- Fused feature dimension: 2,816.
- Classifier: `Linear(2816, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 80)`.

## 4. Environment and Dependencies

The experiments were run in a Python 3.11 environment with an NVIDIA RTX 5090.
The recorded package versions include:

| Package | Version in the working environment |
|---|---|
| PyTorch | 2.14.1+cu130 |
| Torchvision | 0.29.1+cu130 |
| Transformers | 5.19.0 |
| scikit-learn | 1.9.1 |
| NumPy | 2.4.6 |
| Pillow | 12.3.0 |

Install the packages from the project directory:

```bash
pip install -r requirements.txt
```

The listed PyTorch and Torchvision version pins do not fully specify the
CUDA-specific package build. On another machine, install a PyTorch and
Torchvision build compatible with its operating system, Python version,
GPU, and driver before installing or resolving the remaining dependencies.

The pretrained ResNet weights and DistilBERT model must be available locally
or downloadable from their configured model sources.

## 5. Training Configuration

| Setting | Value |
|---|---|
| Loss function | `BCEWithLogitsLoss` |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 16 |
| Epochs | 5 |
| Random seed | 42 |
| Model selection metric | Validation Macro-F1 |
| Test prediction threshold | 0.5 |

Only the fusion classifier is trained. The best checkpoint is selected by
validation Macro-F1. The test set is reserved for final evaluation.

## 6. How to Run

Run commands from the `week5/quiz2` directory, with the required raw and
processed datasets available.

### Optional: smoke test

The smoke test checks a training batch, model output shapes, loss computation,
and classifier backpropagation. It does not run full training or save a model.

```bash
python scripts/train_multimodal.py --smoke-test
python scripts/train_multimodal_resnet50.py --smoke-test
```

### Train the baseline

```bash
python scripts/train_multimodal.py
```

### Evaluate the baseline

```bash
python scripts/evaluate_multimodal.py
```

### Train ResNet50 + DistilBERT

```bash
python scripts/train_multimodal_resnet50.py
```

### Evaluate ResNet50 + DistilBERT

```bash
python scripts/evaluate_multimodal_resnet50.py
```

Training and evaluation scripts protect existing output files and stop rather
than overwrite them. To reproduce the experiments from scratch, use a clean
project copy with the required data files in place.

## 7. Experimental Results

The following results were measured on the same 744-image test split using
the same prediction threshold of 0.5.

| Metric | ResNet18 + DistilBERT | ResNet50 + DistilBERT |
|---|---:|---:|
| Test Loss (lower is better) | 0.0725 | 0.0614 |
| Macro-F1 | 0.4260 | 0.5649 |
| Micro-F1 | 0.5703 | 0.6619 |
| Macro Precision | 0.7090 | 0.7804 |
| Micro Precision | 0.7859 | 0.8194 |
| Macro Recall | 0.3424 | 0.4748 |
| Micro Recall | 0.4475 | 0.5553 |

### Discussion

ResNet50 + DistilBERT outperforms ResNet18 + DistilBERT on all reported test
metrics in this experiment. Macro-F1 increases from approximately 0.4260 to
0.5649, while Micro-F1 increases from approximately 0.5703 to 0.6619.

These results indicate that the ResNet50 configuration performs better under
the current setup. They do not prove that backbone depth alone caused the
improvement, and the results should not be interpreted as official COCO
benchmark scores.

### Additional threshold analysis

Threshold analysis was also conducted for the baseline model. On the
validation set, threshold 0.2 produced the highest Macro-F1 among the tested
thresholds, while threshold 0.3 produced the highest Micro-F1. The main
backbone comparison above uses threshold 0.5 for both models to maintain a
consistent comparison.

## 8. Evaluation Metrics

- **Macro-F1:** Computes F1 for each class and averages the class scores.
- **Micro-F1:** Aggregates true positives, false positives, and false negatives
  across classes before computing F1.
- **Precision:** Measures the proportion of predicted positive labels that
  are correct.
- **Recall:** Measures the proportion of ground-truth positive labels that
  are recovered.
- **Test Loss:** Binary cross-entropy loss computed from model logits and
  multi-label targets.

## 9. Project Structure

Large raw datasets and model checkpoints are not included in this project
listing because they are excluded from Git.

```text
quiz2/
├── data/
│   └── processed/
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       ├── label_map.json
│       └── dataset_summary.json
├── results/
│   ├── baseline/
│   │   ├── history.json
│   │   ├── test_metrics.json
│   │   ├── per_class_metrics.csv
│   │   ├── threshold_analysis.csv
│   │   └── validation_threshold_analysis.csv
│   └── resnet50_distilbert/
│       ├── history.json
│       └── test_metrics.json
├── scripts/
│   ├── prepare_dataset.py
│   ├── test_multimodal_loader.py
│   ├── train_multimodal.py
│   ├── evaluate_multimodal.py
│   ├── analyze_per_class.py
│   ├── analyze_thresholds.py
│   ├── analyze_validation_thresholds.py
│   ├── train_multimodal_resnet50.py
│   └── evaluate_multimodal_resnet50.py
├── requirements.txt
└── README.md
```

## 10. Limitations

- This is a subset of COCO 2017 validation data split internally for coursework,
  not the official COCO test set.
- Both pretrained backbones are frozen; end-to-end fine-tuning was not tested.
- The results apply to the specified data split, training configuration,
  checkpoint selection metric, and prediction threshold.
- Model checkpoints are excluded from Git. Evaluation requires the matching
  trained checkpoint to be available locally.

## 11. Reproducibility Notes

The dataset split uses random seed 42, and both model configurations use the
same train/validation/test split. Training histories and evaluation metrics
are saved under their respective `results/` directories.

The processed JSONL records refer to the raw image files. Keep the expected
raw-data directory structure intact when running the data loader or training
scripts.
