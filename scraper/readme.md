# Download Utilities

`download-images.py` fetches web images per species using GoogleImageCrawler, validates them with Pillow, and keeps only decodable files above a minimum size.

## Usage

```bash
python download/download-images.py
```

Images are saved under:
- Root: `dataCLIP/` (configurable via `DATA_ROOT`)
- Subdir: `images/` (configurable via `IMAGES_SUBDIR`)
- Per-species folders: `dataCLIP/images/Genus_species/`

## Configuration (env vars)

- `DATA_ROOT` (default: `dataCLIP`)
- `IMAGES_SUBDIR` (default: `images`)
- `MAX_PER_SPECIES` (default: `35`)
- `MIN_IMAGE_DIM` (default: `224`)
- `SLEEP_BETWEEN_SEC` (default: `5`)

Dependencies: install via `pip install -r requirements.txt` (or `uv pip install -r requirements.txt`).
