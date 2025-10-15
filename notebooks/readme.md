# Notebooks

Reproducible workflows for CLIP and supervised training/inference.

## Contents

- `train-inference-CLIP.ipynb` — Zero-shot CLIP prompts and optional fine-tuning flow.
- `train-inference-supervised-resnet101.ipynb` — Supervised ResNet-101 training and inference.

## Run

```bash
jupyter lab
# or
jupyter notebook
```

## Data Assumptions

- Training data: `dataCLIP/Genus_species/*.jpg|*.png`.
- Evaluation images: `zeroCLIP/Genus_species_Family.*`.
- Notebooks assume the repository root as the working directory.