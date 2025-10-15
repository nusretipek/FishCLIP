"""
CLI inference for the supervised ResNet101 classifier trained on fish families.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import models, transforms


sys.path.append(str(Path(__file__).resolve().parent.parent))
from fishTaxa import taxaTuples


def buildFamilies() -> Tuple[List[str], dict, dict]:
    """
    Build family label lists and mappings from fishTaxa.taxaTuples.

    Returns:
        families: Sorted unique family names.
        familyToIdx: Mapping family -> index.
        idxToFamily: Mapping index -> family.
    """
    families: List[str] = sorted(list({species[4] for species in taxaTuples}))
    familyToIdx = {family: idx for idx, family in enumerate(families)}
    idxToFamily = {idx: family for family, idx in familyToIdx.items()}
    return families, familyToIdx, idxToFamily


def makeEvalTransform():
    """
    Create the evaluation transform matching the training settings.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )


def buildModel(numClasses: int, device: str) -> torch.nn.Module:
    """
    Build the ResNet101 model with the same classifier head as training.
    """
    model = models.resnet101(weights="IMAGENET1K_V1")
    numFeatures = model.fc.in_features 
    model.fc = nn.Sequential(
        nn.Linear(numFeatures, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, numClasses),
    )
    return model.to(device)


def loadWeights(model: torch.nn.Module, modelPath: str, device: str) -> None:
    """
    Load weights into the model. Accepts raw state_dict or dict with 'state_dict'.
    """       
    print(f"Loading checkpoint from: {modelPath}")
    ckpt = torch.load(modelPath, weights_only=False, map_location=device)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state, strict=False)
        print("Model weights loaded successfully.\n")
    except Exception as e:
        print(f"Warning: could not load some weights → {e}")
        sys.exit(1)


@torch.no_grad()
def predictImage(
    imagePath: str,
    model: torch.nn.Module,
    transform,
    device: str,
    idxToFamily: dict,
    topk: int = 3,
) -> List[Tuple[float, str]]:
    """
    Predict top-k families for a single image using the supervised classifier.

    Args:
        imagePath: Path to the image file.
        model: ResNet101 classifier.
        transform: Evaluation transform.
        device: Torch device string.
        idxToFamily: Mapping index -> family name.
        topk: Number of top predictions to return.
    """
    model.eval()
    img = Image.open(imagePath)
    img = ImageOps.exif_transpose(img)
    if img.mode == "P" and ("transparency" in img.info or img.info.get("transparency") is not None):
        img = img.convert("RGBA").convert("RGB")
    else:
        img = img.convert("RGB")
    img.load()

    image = transform(img).unsqueeze(0).to(device)
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1).squeeze(0)
    k = min(topk, probs.shape[-1])
    scores, indices = torch.topk(probs, k=k, largest=True, sorted=True)
    return [(float(scores[i]), idxToFamily[int(indices[i])]) for i in range(k)]


def main() -> None:
    """
    Parse CLI args, build model, load weights, and print predictions.
    """
    parser = argparse.ArgumentParser(description="Supervised ResNet101 family inference")
    parser.add_argument("--model", type=str, required=True, help="Path to trained weights (.pth)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to show")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    families, _familyToIdx, idxToFamily = buildFamilies()
    model = buildModel(numClasses=len(families), device=device)
    loadWeights(model, args.model, device)

    transform = makeEvalTransform()
    results = predictImage(args.image, model, transform, device, idxToFamily, topk=args.topk)

    print(f"Image: {Path(args.image).name}")
    for rank, (score, label) in enumerate(results, start=1):
        print(f"Top-{rank}: {label} ({score:.4f})")


if __name__ == "__main__":
    main()

