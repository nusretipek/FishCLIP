# FishCLIP

A compact project for fish family recognition using two approaches:
- Zero-shot CLIP (RN101) for prompt-based classification.
- Supervised ResNet-101 fine-tuning for fish family labels.

It includes simple training, inference, and data download utilities with minimal dependencies.

## Project Layout

```text
FishCLIP/
├─ fishTaxa.py                 # Shared fish taxa list/mappings
├─ requirements.txt            # Minimal dependencies
├─ dataCLIP/                   # Dataset: dataCLIP/Genus_species/*.jpg|png
├─ zeroCLIP/                   # Small evaluation set for quick checks
├─ download/                   # Data scraping utilities
│  ├─ download-images.py
│  └─ Fish.xlsx								 # Metadata for 50 fish species
├─ scripts/                    # Training and inference scripts
│  ├─ train-CLIP.py
│  ├─ inference-CLIP.py
│  ├─ train-supervised-resnet101.py
│  └─ inference-supervised-resnet101.py
├─ notebooks/                  # Notebook versions for the scripts
   ├─ train-inference-CLIP.ipynb
   └─ train-inference-supervised-resnet101.ipynb
```

## Install

```bash
git clone https://github.com/nusretipek/FishCLIP.git
```

Using pip:

```bash
pip install -r requirements.txt

# Optional: Jupyter for notebooks
pip install jupyter ipykernel
```

Using uv (optional, faster installer):

```bash
uv pip install -r requirements.txt

# Optional: Jupyter for notebooks
uv pip install jupyter ipykernel
```

## Quickstart

- CLIP training + periodic zero-shot eval:
```bash
python scripts/train-CLIP.py
```
- CLIP single-image inference (optionally pass a fine-tuned checkpoint):
```bash
python scripts/inference-CLIP.py --image zeroCLIP/Salmo_hucho_Salmonidae.jpg \
  --prompts Salmonidae Sphyraenidae Pomacanthidae --topk 3
# with checkpoint add: --model {MODEL_NAME}.pt
```
- Supervised ResNet-101 training + eval:
```bash
python scripts/train-supervised-resnet101.py
```
- Supervised single-image inference:
```bash
python scripts/inference-supervised-resnet101.py \
  --model {MODEL_NAME}.pt \
  --image zeroCLIP/Salmo_hucho_Salmonidae.jpg --topk 3
```

## Data Expectations

- Training data under `dataCLIP/Genus_species/*.jpg|*.png`.
- Evaluation images in `zeroCLIP/` (filenames like `Genus_species_Family.*`).
