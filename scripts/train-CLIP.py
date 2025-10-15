"""
CLIP training script.
"""

# import libraries
import os
import glob
import random
import sys
from pathlib import Path
from tabulate import tabulate
from PIL import Image, ImageOps

import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


# import taxa from parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from fishTaxa import taxaTuples

# constants
BATCH_SIZE: int = 16
EPOCHS: int = 15
LEARNING_RATE: float = 1e-5
WEIGHT_DECAY: float = 1e-3
BETA1: float = 0.9
BETA2: float = 0.98
MODEL_NAME: str = "RN101"
SEED: int = 0
ENABLE_REPRODUCIBILITY = True

# project-root-relative paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
EVAL_FOLDER: Path = ROOT_DIR / "zeroCLIP"
DATASET_ROOT: Path = ROOT_DIR / "dataCLIP"

# reproducibility settings
if ENABLE_REPRODUCIBILITY:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    
# select device
device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

# load pre-trained CLIP model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(MODEL_NAME, device=device, jit=False)


# helper functions
def makeCaptions(binom: str, common: str, cls: str, order: str, family: str, genus: str) -> List[str]:
    """
    Create simple caption candidates for a species.

    Args:
        binom: Scientific binomial name.
        common: Common name.
        cls: Taxonomic class.
        order: Taxonomic order.
        family: Taxonomic family.
        genus: Genus name.

    Returns:
        A list of caption strings.
    """
    return [
        f"{binom}",
        f"a fish from family {family}",
        f"a fish from order {order}",
        f"a fish from class {cls}",
    ]


def buildClipLists(root: Path = DATASET_ROOT) -> Tuple[List[str], List[List[str]]]:
    """
    Build parallel lists of image paths and caption options.

    Args:
        root: Dataset root containing subfolders per species.

    Returns:
        Tuple (imagePaths, textOptions) aligned by index.
    """
    root = Path(root)
    imagePaths, textOptions = [], []
    for species in taxaTuples:
        binom, common, cls, order, family, genus = species
        captions = makeCaptions(binom, common, cls, order, family, genus)
        folderPath = root / binom.replace(" ", "_")
        for img in sorted(folderPath.glob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                imagePaths.append(str(img))
                textOptions.append(captions)
    assert len(imagePaths) == len(textOptions)
    return imagePaths, textOptions


class CLIPDataset(Dataset):
    """
    Minimal dataset yielding (imageTensor, textTokens) pairs for CLIP.

    Attributes:
        imagePath: List of image file paths.
        textOptions: List of caption lists per image.
    """

    def __init__(self, listImagePath: List[str], listTxtOptions: List[List[str]]):
        self.imagePath: List[str] = listImagePath
        self.textOptions: List[List[str]] = listTxtOptions

    def __len__(self) -> int:
        return len(self.imagePath)

    def __getitem__(self, idx: int):
        imgPath = self.imagePath[idx]
        captionList = self.textOptions[idx]
        caption = random.choice(captionList)

        img = Image.open(imgPath)
        img = ImageOps.exif_transpose(img)
        if img.mode == "P" and ("transparency" in img.info or img.info.get("transparency") is not None):
            img = img.convert("RGBA").convert("RGB")
        else:
            img = img.convert("RGB")
        img.load()

        image = preprocess(img)
        title = clip.tokenize(caption)[0]
        return image, title


# build dataset and dataloader
imageData, textData = buildClipLists(DATASET_ROOT)
dataset = CLIPDataset(imageData, textData)
trainDataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# zero-shot evaluation prompts
prompts = [
    "Salmonidae",
    "Sphyraenidae",
    "Pomacanthidae",
    "Epinephelidae",
    "Moronidae",
    "Gymnotidae",]
    
textTokens = clip.tokenize(prompts).to(device)


@torch.no_grad()
def clipPredict(imagePath: str, textTokens: torch.Tensor, texts: List[str], model: torch.nn.Module,
                preprocess, device: str, topk: int = 5,) -> List[Tuple[float, str]]:
    """
    Run CLIP zero-shot prediction for a single image.

    Args:
        imagePath: Path to image file.
        textTokens: Tokenized prompt tensor.
        texts: Original prompt strings (for labels).
        model: CLIP model.
        preprocess: Image transform function.
        device: Device string (e.g., 'cuda:0' or 'cpu').
        topk: Number of top classes to return.

    Returns:
        List of (score, label) pairs sorted by score desc.
    """
    model.eval()
    img = Image.open(imagePath)
    image = preprocess(img).unsqueeze(0).to(device)
    logitsPerImage, _ = model(image, textTokens)
    probs = logitsPerImage.softmax(dim=-1).squeeze(0)
    k = min(topk, len(texts))
    scores, idx = torch.topk(probs, k=k, largest=True, sorted=True)
    return [(float(scores[i]), texts[int(idx[i])]) for i in range(k)]


def runEvaluation(imgFolder: Path) -> None:
    """
    Evaluate a folder of images using zero-shot CLIP prompts.

    Args:
        imgFolder: Folder containing images to evaluate.
    """
    if not imgFolder.exists():
        print(f"{imgFolder} does not exist.")
        return
    if not len(glob.glob(f"{str(imgFolder)}/*")):
        print(f"{imgFolder} is empty.")
        return

    rows = []
    for name in glob.glob(f"{str(imgFolder)}/*"):
        results = clipPredict(name, textTokens, prompts, model, preprocess, device, topk=3)
        nameSplit = os.path.basename(name).split(".")[0].split("_")
        row = [f"{nameSplit[0]} {nameSplit[1]}"] + [nameSplit[2]] + [f"{label} ({score:.3f})" for score, label in results]
        rows.append(row)

    headers = ["Image", "True Family", "Top-1", "Top-2", "Top-3"]
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"), "\n")


# initial evaluation (before fine-tuning)
print("=" * 50)
print("INITIAL EVALUATION (Before fine-tuning)")
print("=" * 50)
runEvaluation(EVAL_FOLDER)


# fine-tune CLIP model
model = model.float().to(device)
lossImg = nn.CrossEntropyLoss()
lossTxt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=1e-6, weight_decay=WEIGHT_DECAY)

# training loop
print("=" * 50)
print("FINE-TUNING CLIP MODEL")
print("=" * 50)
for epoch in range(EPOCHS):
    model.train()
    epochLossSum = 0.0
    epochCount = 0
    stepIdx = 0
    print(15 * "-", f"Epoch {epoch+1}", 15 * "-")
    for batch in trainDataloader:
        optimizer.zero_grad()

        # Unpack the batch: (images, texts)
        # images: float tensor (B, 3, H, W); texts: token ids (B, ctx_len)
        images, texts = batch

        # Move tensors to the selected device
        images = images.to(device)
        texts = texts.to(device)

        # Forward through CLIP
        logitsPerImage, logitsPerText = model(images, texts)
        logitsPerImage = logitsPerImage.contiguous()
        logitsPerText = logitsPerText.contiguous()

        # Ground-truth matches along the batch diagonal
        groundTruth = torch.arange(len(images), dtype=torch.long, device=device)
        totalLoss = (lossImg(logitsPerImage, groundTruth) + lossTxt(logitsPerText, groundTruth)) / 2

        # Backpropagation and optimizer step
        totalLoss.backward()
        optimizer.step()

        # Track running loss statistics
        bs = images.size(0)
        epochLossSum += float(totalLoss.detach()) * bs
        epochCount += bs
        
        # Verbose
        stepIdx += 1
        if stepIdx % 10 == 0:
            runningAvg = epochLossSum / max(1, epochCount)
            print(f"Epoch {epoch+1} | Step {stepIdx}/{len(trainDataloader)} | Running average step loss: {runningAvg:.4f}")
    epochAvg = epochLossSum / max(1, epochCount)
    print(f"Epoch {epoch+1} | Epoch average loss: {epochAvg:.4f}\n")
    runEvaluation(EVAL_FOLDER)


# save checkpoint
torch.save(
    {
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": totalLoss,
    },
    f"{MODEL_NAME}_{EPOCHS}.pt",
)
print(f"Training complete and model saved to: ./{MODEL_NAME}_{EPOCHS}.pt")
