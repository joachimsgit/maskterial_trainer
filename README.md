# MaskTerial Trainer

Desktop GUI for training [MaskTerial](https://github.com/Jaluus/MaskTerial)
classifiers (AMM, GMM) and segmentation models (M2F) on your own
2D-material microscope images, without writing any code.

The app walks you through the full pipeline in a single window:

1. **Images** — drop in microscope images, set a flatfield reference.
2. **Instance Masks** — annotate flakes via watershed or polygon tools.
3. **Semantic Masks** — assign each flake a class (1L, 2L, Bulk, …) by
   lassoing clusters on RGB-contrast scatter plots. Click any dot to
   preview the underlying flake.
4. **COCO Conversion** — export to COCO format with a stable 80/20 split.
5. **Train** — train AMM (deep), GMM (closed-form), or M2F (segmentation,
   GPU). AMM gets a CPU/CUDA picker; M2F downloads the pretrained
   backbone on demand.
6. **Evaluate** — per-class precision/recall/F1 + confusion matrix on the
   val split, plus a single-image inference preview with class-coloured
   contours.
7. **Export** — bundle masks, models, and an upload-ready README into a
   folder or zip for the inference website.

## Install

The trainer needs Python 3.10+ and a working MaskTerial install (which
itself needs PyTorch; segmentation training additionally needs
`detectron2` and a CUDA GPU).

### 1. Clone

```bash
git clone <your-fork-url> maskterial-trainer
cd maskterial-trainer
```

### 2. Create a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

Or use conda — anything that gives you a clean Python 3.10+.

### 3. Install the trainer

```bash
pip install -e .
```

This pulls the GUI's runtime deps (PySide6, numpy, opencv, matplotlib,
pycocotools, scikit-learn). It does **not** pull PyTorch or MaskTerial —
those are heavier and platform-specific, so install them next.

### 4. Install PyTorch

Pick the right wheel for your platform from
<https://pytorch.org/get-started/locally/>. Example for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

CPU-only is also fine for AMM/GMM training, just slower for AMM:

```bash
pip install torch
```

### 5. Install MaskTerial

Clone and editable-install the upstream repo:

```bash
git clone https://github.com/Jaluus/MaskTerial.git
pip install -e ./MaskTerial
```

### 6. (Optional) Install detectron2 for M2F segmentation training

Segmentation needs `detectron2` and a CUDA GPU. AMM/GMM work without it.
See the [detectron2 install
docs](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
— the right command depends on your CUDA + torch version.

## Run

```bash
maskterial-trainer
# or:
python -m maskterial_trainer
```

The window opens maximized. Pick or create a project folder
(any folder containing your microscope images) and step through the tabs
on the left.

## Where things live

| What | Where |
|---|---|
| Per-user config | `~/.maskterial_trainer/config.json` |
| Cached pretrained M2F backbone (~600 MB) | `~/.maskterial_trainer/models/` |
| Per-project annotations + outputs | inside the project folder (`coco/`, `outputs/`, `instance_masks/`, `semantic_masks/`, `splits/`) |
| Project metadata (classes, flatfield) | `<project>/project.json` |

Project folders are self-contained — you can move or copy one and the
trainer will pick it up.

## Architecture in one paragraph

The GUI process only imports PySide6, numpy, OpenCV, PIL, matplotlib,
and pycocotools — it stays launchable even when PyTorch or MaskTerial
aren't installed yet. Training, evaluation, and inference all run as
**subprocesses** under `python -m maskterial_trainer.runners.<name>` so
heavy dependencies (`torch`, `maskterial`, `detectron2`) are imported in
the child, never in the GUI. The runners stream `PROGRESS {json}`
lines to stdout; the GUI parses them to drive progress bars and ETAs.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the protocol and how to add
a new stage.

## Known issues

- **Validation data is intentionally not passed to AMM/GMM training**
  because upstream `maskterial.utils.data_loader.ContrastDataloader`
  overwrites `X_train` when test_* args are set. Evaluation runs as a
  separate step via [evaluate_classifier.py](maskterial_trainer/runners/evaluate_classifier.py).
- The single-class GMM path **synthesizes the background class at
  (0,0,0)** with a small fixed covariance (configurable via
  `synthetic_background_std` in
  [configs/GMM/default_config.json](maskterial_trainer/resources/configs/GMM/default_config.json)).
  AMM does not synthesize; if you only annotate one layer class for AMM,
  ensure your images have a clean background so the dataloader's
  auto-sampled class 0 survives denoising.

## License

TBD. Pick a license (MIT / Apache 2.0 / GPL) before publishing — note
that `MaskTerial` itself ships under its own license, which doesn't
constrain this project but does constrain any vendored copy.
