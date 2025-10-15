# Scripts

Command-line tools for training and inference.

## CLIP

- Train (fine-tune CLIP RN101) with periodic zero-shot evaluation:
```bash
python scripts/train-CLIP.py
```
Notes:
- Uses images under `dataCLIP/Genus_species/` and evaluates on `zeroCLIP/`.
- Constants such as batch size, epochs, and prompts are defined in the script.

- Inference on a single image with prompts (optionally pass a checkpoint via `--model`):
```bash
python scripts/inference-CLIP.py \
  --image zeroCLIP/Salmo_hucho_Salmonidae.jpg \
  --prompts Salmonidae Sphyraenidae Pomacanthidae \
  --topk 3
# Optionally: --model RN101_15.pt
```

## Supervised ResNet-101

- Train classifier on fish families derived from `fishTaxa.py`:
```bash
python scripts/train-supervised-resnet101.py
```
Notes:
- Expects dataset at `dataCLIP/Genus_species/*.jpg|*.png`.
- Evaluates on images in `zeroCLIP/` (filenames like `Genus_species_Family.*`).

- Inference on a single image with a trained checkpoint:
```bash
python scripts/inference-supervised-resnet101.py \
  --model resnet101_supervised_15.pt \
  --image zeroCLIP/Salmo_hucho_Salmonidae.jpg \
  --topk 3
```

## Device and Dependencies

- Device is auto-detected: `cuda:0` if GPU is available, else CPU.
- Install dependencies first: `pip install -r requirements.txt` (or `uv pip install -r requirements.txt`).
