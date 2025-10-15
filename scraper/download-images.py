"""
Image downloader for fish species.

Requires icrawler, pillow, and tqdm.

Usage:
    python src/download-images.py
"""

import os
import re
import time
from pathlib import Path
from typing import Iterable, List, Tuple
import sys

from icrawler.builtin import GoogleImageCrawler
from PIL import Image
from tqdm import tqdm

# allow importing fishTaxa from the parent directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from fishTaxa import taxaTuples


dataRootEnv = os.getenv("DATA_ROOT", "dataCLIP")
dataRoot = Path(dataRootEnv)
if not dataRoot.is_absolute():
    dataRoot = ROOT_DIR / dataRoot
imagesSubdir = os.getenv("IMAGES_SUBDIR", "images")
maxPerSpecies = int(os.getenv("MAX_PER_SPECIES", "35"))
minImageDim = int(os.getenv("MIN_IMAGE_DIM", "224"))
sleepBetweenSec = float(os.getenv("SLEEP_BETWEEN_SEC", "5"))


def safeStem(name: str) -> str:
    """
    Normalize a string to a filesystem-friendly stem.

    Args:
        name: Raw name to normalize.

    Returns:
        Normalized stem containing only letters, digits, and underscores.
    """

    assert isinstance(name, str)
    return re.sub(r"[^A-Za-z0-9_]", "_", name.strip().replace(" ", "_"))


def buildQuery(binomial: str, common: str) -> str:
    """
    Build a search query for the image crawler.

    Args:
        binomial: Scientific name.
        common: Common name.

    Returns:
        Query string for the GoogleImageCrawler.
    """

    assert isinstance(binomial, str) and isinstance(common, str)
    return f'"{binomial}" OR "{common}" fish'


def isValidImage(path: Path) -> bool:
    """
    Verify that an image is decodable and above a minimum size.

    Args:
        p: Image path.

    Returns:
        True when loadable and min dimension meets MIN_IMAGE_DIM.
    """

    assert isinstance(path, Path)
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
        return min(w, h) >= minImageDim
    except Exception:
        return False


def downloadSpeciesImages(destinationDir: Path, taxaIterable: Iterable[Tuple[str, str, str, str, str, str]]) -> None:
    """
    Download images per species using GoogleImageCrawler.

    Args:
        dst: Target images root directory.
        taxa: Iterable of (binomial, common, class, order, family, genus).
    """

    assert isinstance(destinationDir, Path)
    destinationDir.mkdir(parents=True, exist_ok=True)
    for binomial, common, _, _, _, _ in tqdm(taxaIterable, desc="Taxa"):
        outDir = destinationDir / safeStem(binomial)
        outDir.mkdir(parents=True, exist_ok=True)
        crawler = GoogleImageCrawler(storage={"root_dir": str(outDir)})
        crawler.crawl(keyword=buildQuery(binomial, common), max_num=maxPerSpecies, overwrite=False)
        time.sleep(sleepBetweenSec)


def purgeAndCountValid(imagesRoot: Path) -> int:
    """
    Remove invalid images and return count of kept files.

    Args:
        dst: Images root directory.

    Returns:
        Total number of valid images kept.
    """

    assert isinstance(imagesRoot, Path)
    kept = 0
    for speciesDir in imagesRoot.iterdir():
        if not speciesDir.is_dir():
            continue
        for filePath in list(speciesDir.iterdir()):
            if filePath.is_file() and filePath.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                if not isValidImage(filePath):
                    try:
                        filePath.unlink()
                    except Exception:
                        pass
                else:
                    kept += 1
    return kept


def reportPerSpeciesCounts(imagesRoot: Path) -> None:
    """
    Print a short per-species image count report.

    Args:
        dst: Images root directory.
    """

    assert isinstance(imagesRoot, Path)
    for root, _, files in os.walk(str(imagesRoot)):
        if Path(root) == imagesRoot:
            continue
        count = sum(1 for f in files if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"})
        species = Path(root).name
        if count == maxPerSpecies:
            print(f"OK {species}: {count} images")
        else:
            print(f"WARN {species}: {count} images")


def main() -> None:
    """
    Orchestrate download, cleanup, and reporting.
    """

    root = dataRoot
    imagesRoot = root / imagesSubdir
    imagesRoot.mkdir(parents=True, exist_ok=True)
    downloadSpeciesImages(imagesRoot, taxaTuples)
    kept = purgeAndCountValid(imagesRoot)
    print(f"Kept {kept} images after validation.")
    reportPerSpeciesCounts(imagesRoot)


if __name__ == "__main__":
    main()
