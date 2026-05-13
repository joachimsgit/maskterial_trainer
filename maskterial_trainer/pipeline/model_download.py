"""Pretrained-model download + caching.

The M2F segmentation model needs the ~600 MB synthetic-pretrained checkpoint
from Zenodo as its starting point. We cache it under
~/.maskterial_trainer/models/ so it's a one-time download.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

PRETRAINED_M2F_URL = (
    "https://zenodo.org/records/15765516/files/"
    "SEG_M2F_Synthetic_Data.zip?download=1"
)
MODELS_DIR = Path.home() / ".maskterial_trainer" / "models"
PRETRAINED_M2F_PATH = MODELS_DIR / "pretrained_m2f_synthetic.pth"


ProgressCallback = Callable[[int, int, str], None]


def pretrained_m2f_available() -> bool:
    return PRETRAINED_M2F_PATH.exists() and PRETRAINED_M2F_PATH.stat().st_size > 0


def download_pretrained_m2f(progress: ProgressCallback | None = None) -> Path:
    """Download + extract the pretrained M2F backbone.

    Writes to `<target>.part` and renames atomically on success so a failed
    or aborted download cannot leave a truncated file that
    `pretrained_m2f_available()` would then accept as valid.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MODELS_DIR / "_pretrained_m2f.zip"
    part_path = PRETRAINED_M2F_PATH.with_suffix(
        PRETRAINED_M2F_PATH.suffix + ".part"
    )
    for stale in (zip_path, part_path):
        if stale.exists():
            stale.unlink()

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if progress is None:
            return
        downloaded = block_num * block_size
        if total_size > 0:
            downloaded = min(downloaded, total_size)
        progress(downloaded, max(total_size, 0), "downloading")

    try:
        urllib.request.urlretrieve(PRETRAINED_M2F_URL, zip_path, reporthook=_report)
    except Exception:
        if zip_path.exists():
            zip_path.unlink()
        raise

    if progress is not None:
        progress(0, 0, "extracting")

    try:
        with zipfile.ZipFile(zip_path) as zf:
            target_name = next(
                (n for n in zf.namelist() if n.endswith("model_final.pth")),
                None,
            )
            if target_name is None:
                raise FileNotFoundError(
                    "model_final.pth not found inside pretrained zip"
                )
            with zf.open(target_name) as src, open(part_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        # Only swap the final path in when extraction succeeded end-to-end.
        if PRETRAINED_M2F_PATH.exists():
            PRETRAINED_M2F_PATH.unlink()
        part_path.replace(PRETRAINED_M2F_PATH)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise
    finally:
        if zip_path.exists():
            zip_path.unlink()

    if progress is not None:
        size = PRETRAINED_M2F_PATH.stat().st_size
        progress(size, size, "done")

    return PRETRAINED_M2F_PATH
