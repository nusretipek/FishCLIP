"""
Simple CLI inference for CLIP model.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import clip
import torch
from PIL import Image


MODEL_NAME: str = "RN101"


def loadModel(modelPath: Optional[str], device: str) -> Tuple[torch.nn.Module, any]:
    """
    Load CLIP model (optionally loading fine-tuned weights).

    Args:
        modelPath: Optional path to a checkpoint file (.pt or .pth).
        device: Torch device string ('cuda' or 'cpu').

    Returns:
        (model, preprocess)
    """
    model, preprocess = clip.load(MODEL_NAME, device=device, jit=False)

    if modelPath:
        print(f"Loading checkpoint from: {modelPath}")
        ckpt = torch.load(modelPath, weights_only=False, map_location=device)

        # Look for correct key
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        try:
            model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded successfully.\n")
        except Exception as e:
            print(f"Warning: could not load some weights → {e}")

    return model, preprocess


@torch.no_grad()
def clipPredict(
    imagePath: str,
    textTokens: torch.Tensor,
    texts: List[str],
    model: torch.nn.Module,
    preprocess,
    device: str,
    topk: int = 5,
) -> List[Tuple[float, str]]:
    """
    Run CLIP zero-shot prediction for a single image.

    Args:
        imagePath: Path to image file.
        textTokens: Tokenized prompts tensor.
        texts: Original prompts for label mapping.
        model: CLIP model instance.
        preprocess: CLIP image preprocessing transform.
        device: Torch device string.
        topk: Number of top predictions.

    Returns:
        List of (score, label) sorted by score descending.
    """
    model.eval()
    img = Image.open(imagePath)
    image = preprocess(img).unsqueeze(0).to(device)
    logitsPerImage, _ = model(image, textTokens)
    probs = logitsPerImage.softmax(dim=-1).squeeze(0)
    k = min(topk, len(texts))
    scores, idx = torch.topk(probs, k=k, largest=True, sorted=True)
    return [(float(scores[i]), texts[int(idx[i])]) for i in range(k)]


def main() -> None:
    """
    Parse CLI args, load model, and print predictions for a single image.
    """
    parser = argparse.ArgumentParser(description="CLIP inference compatible with train-CLIP.py")
    parser.add_argument("--model", type=str, default=None, help="Optional path to fine-tuned weights (.pth)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="One or more text prompts (space-separated)",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to show")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = loadModel(args.model, device)

    texts: List[str] = args.prompts
    tokens = clip.tokenize(texts).to(device)
    results = clipPredict(args.image, tokens, texts, model, preprocess, device, topk=args.topk)

    print(f"Image: {Path(args.image).name}")
    for rank, (score, label) in enumerate(results, start=1):
        print(f"Top-{rank}: {label} ({score:.4f})")


if __name__ == "__main__":
    main()

