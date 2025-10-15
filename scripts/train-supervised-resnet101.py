"""
ResNet-101 supervised training script.
"""

# import libraries
import os
import glob
import sys
from pathlib import Path
from typing import List, Tuple

from tabulate import tabulate
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# import taxa from parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from fishTaxa import taxaTuples


# constants
BATCH_SIZE: int = 16
EPOCHS: int = 15
LEARNING_RATE: float = 1e-5
WEIGHT_DECAY: float = 0.01
LABEL_SMOOTHING: float = 0.1
STEP_LOG_INTERVAL: int = 10
SEED: int = 0
ENABLE_REPRODUCIBILITY = True

# project-root-relative paths
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
EVAL_FOLDER: Path = ROOT_DIR / "zeroCLIP"
DATASET_ROOT: Path = ROOT_DIR / "dataCLIP"

# reproducibility settings
if ENABLE_REPRODUCIBILITY:
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


# build families and label mappings from taxa
families: List[str] = sorted(list({species[4] for species in taxaTuples}))
familyToIdx = {family: idx for idx, family in enumerate(families)}
idxToFamily = {idx: family for family, idx in familyToIdx.items()}
numClasses: int = len(families)

print(f"Number of unique families: {numClasses}")
print(f"Families: {families}\n")


# define transforms
trainTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

evalTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def buildDatasetLists(root: Path = DATASET_ROOT) -> Tuple[List[str], List[int]]:
    """
    Build parallel lists for image paths and numeric labels using taxa from fishTaxa.

    Args:
        root: Dataset root containing subfolders named by binomial (Genus_species).

    Returns:
        Tuple of (imagePaths, labels) aligned by index.
    """
    rootPath = Path(root)
    imagePaths: List[str] = []
    labels: List[int] = []

    for species in taxaTuples:
        binom, _common, _cls, _order, family, _genus = species
        familyIdx = familyToIdx[family]
        folderPath = rootPath / binom.replace(" ", "_")
        if not folderPath.exists():
            continue
        for img in sorted(folderPath.glob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                imagePaths.append(str(img))
                labels.append(familyIdx)

    assert len(imagePaths) == len(labels)
    return imagePaths, labels


class FishDataset(Dataset):
    """
    Dataset yielding (imageTensor, familyLabel) pairs for supervised training.

    Attributes:
        imagePaths: List of image file paths.
        labels: List of integer family labels.
        transform: Optional torchvision transform to apply to images.
    """

    def __init__(self, imagePaths: List[str], labels: List[int], transform=None):
        self.imagePaths: List[str] = imagePaths
        self.labels: List[int] = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.imagePaths)

    def __getitem__(self, idx: int):
        imgPath = self.imagePaths[idx]
        label = self.labels[idx]

        img = Image.open(imgPath)
        img = ImageOps.exif_transpose(img)
        if img.mode == "P" and ("transparency" in img.info or img.info.get("transparency") is not None):
            img = img.convert("RGBA").convert("RGB")
        else:
            img = img.convert("RGB")
        img.load()

        if self.transform:
            img = self.transform(img)

        return img, label


# build dataset and dataloader
imageData, labelData = buildDatasetLists(DATASET_ROOT)
dataset = FishDataset(imageData, labelData, transform=trainTransform)
trainDataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# model setup
model = models.resnet101(weights="IMAGENET1K_V1")
numFeatures = model.fc.in_features
model.fc = nn.Sequential(  
    nn.Linear(numFeatures, 1024),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(1024, numClasses),
)
model = model.to(device)


@torch.no_grad()
def predictImage(imagePath: str, model: torch.nn.Module, transform, device: str, topk: int = 5,) -> List[Tuple[float, str]]:
    """
    Predict top-k families for a single image using the current model.

    Args:
        imagePath: Path to the image file.
        model: The classification model.
        transform: The evaluation transform to apply.
        device: Device string (e.g., 'cuda:0' or 'cpu').
        topk: Number of top predictions to return.

    Returns:
        List of (score, familyLabel) pairs sorted by score desc.
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
    k = min(topk, numClasses)
    scores, indices = torch.topk(probs, k=k, largest=True, sorted=True)
    return [(float(scores[i]), idxToFamily[int(indices[i])]) for i in range(k)]


def runEvaluation(imgFolder: Path) -> None:
    """
    Evaluate images in a folder; expects filenames like Genus_species_Family.* for true family.

    Args:
        imgFolder: Folder containing images to evaluate.
    """
    if not imgFolder.exists():
        print(f"{imgFolder} does not exist.")
        return

    imgFiles = sorted(glob.glob(f"{str(imgFolder)}/*"))
    if not len(imgFiles):
        print(f"{imgFolder} is empty.")
        return

    rows = []
    evalResults = []
    for name in imgFiles:
        results = predictImage(name, model, evalTransform, device, topk=3)
        nameSplit = os.path.basename(name).split(".")[0].split("_")
        if len(nameSplit) >= 3:
            trueFamily = nameSplit[2]
        else:
            trueFamily = "Unknown"

        predFamilies = [label for _, label in results]
        row = [f"{nameSplit[0]} {nameSplit[1] if len(nameSplit) > 1 else ''}"] + [trueFamily]
        row += [f"{label} ({score:.3f})" for score, label in results]
        rows.append(row)

        if trueFamily in familyToIdx:
            isTop1 = predFamilies[0] == trueFamily
            isTop3 = trueFamily in predFamilies
            evalResults.append(
                {
                    "name": os.path.basename(name),
                    "true_family": trueFamily,
                    "pred_top1": predFamilies[0],
                    "pred_top3": predFamilies,
                    "top1_correct": isTop1,
                    "top3_correct": isTop3,
                },
            )

    headers = ["Image", "True Family", "Top-1", "Top-2", "Top-3"]
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"), "\n")

    if len(evalResults) > 0:
        correctTop1 = sum(r["top1_correct"] for r in evalResults)
        correctTop3 = sum(r["top3_correct"] for r in evalResults)
        total = len(evalResults)
    else:
        print("\nNo valid images found with families in the training set.\n")


# initial evaluation (before fine-tuning)
print("=" * 50)
print("INITIAL EVALUATION (Before fine-tuning)")
print("=" * 50)
runEvaluation(EVAL_FOLDER)


# training configuration
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# training loop
print("=" * 50)
print("FINE-TUNING SUPERVISED RESNET-101 MODEL")
print("=" * 50)
for epoch in range(EPOCHS):
    model.train()
    epochLossSum = 0.0
    epochCorrect = 0
    epochTotal = 0
    stepIdx = 0

    print(15 * "-", f"Epoch {epoch+1}", 15 * "-")
    for batch in trainDataloader:
        optimizer.zero_grad()
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        batchCorrect = (predicted == labels).sum().item()
        batchTotal = labels.size(0)

        epochLossSum += loss.item() * batchTotal
        epochCorrect += batchCorrect
        epochTotal += batchTotal

        stepIdx += 1
        if stepIdx % STEP_LOG_INTERVAL == 0:
            runningLoss = epochLossSum / epochTotal
            runningAcc = 100 * epochCorrect / epochTotal
            print(f"Step {stepIdx}/{len(trainDataloader)} | Loss: {runningLoss:.4f} | Acc: {runningAcc:.2f}%",)

    scheduler.step()
    epochAvgLoss = epochLossSum / epochTotal
    epochAcc = 100 * epochCorrect / epochTotal
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Average Loss: {epochAvgLoss:.4f}")
    print(f"  Training Accuracy: {epochAcc:.2f}%\n")

    print("-" * 30)
    print(f"EVALUATION AFTER EPOCH {epoch + 1}")
    print("-" * 30)
    runEvaluation(EVAL_FOLDER)


# save checkpoint
torch.save(model.state_dict(), f'resnet101_supervised_{EPOCHS}.pt')
print(f"Training complete and model saved to: ./resnet101_supervised_{EPOCHS}.pt")
