# Contributing

Notes for developers extending the MaskTerial Trainer.

## Repo layout

```
maskterial_trainer/
├── app.py              # entry point (creates QApplication, shows MainWindow)
├── __main__.py         # supports `python -m maskterial_trainer`
├── config.py           # per-user config under ~/.maskterial_trainer/
├── ui/                 # PySide6 widgets, one per workflow stage
├── pipeline/           # pure-Python project + data layer (no Qt deps below here)
├── runners/            # subprocess entry points for training / eval / inference
└── resources/configs/  # bundled JSON / YAML default configs
```

`ui/` may import `pipeline/`. `pipeline/` may import `runners/` modules
only at runtime (subprocess spawn) — never directly, so the heavy deps
stay out of the GUI process. `runners/` are leaf modules; they don't
import from `ui/`.

## The PROGRESS protocol

Every runner streams structured progress to stdout via
[runners/_progress.py](maskterial_trainer/runners/_progress.py). Each
line is either:

```
PROGRESS {"step": 42, "total": 100, "stage": "train", "message": "...", "loss": 0.123}
```

…or a plain log line that the GUI forwards verbatim to the log pane.

The GUI side (`TrainingRunner` / `EvaluationRunner` / `InferenceRunner`
in [pipeline/](maskterial_trainer/pipeline/)) wraps a `QProcess`,
buffers stdout, and emits `progress(dict)` / `log(str)` /
`finished(int)` signals to the view.

Stages used today: `loading`, `train`, `evaluate`, `finalizing`,
`load_model`, `load_image`, `classify`, `draw`, `done`, `error`.

## Adding a new pipeline stage

1. **Add a runner** at `maskterial_trainer/runners/<name>.py`:
   ```python
   from ._progress import emit_done, emit_error, emit_progress

   def main() -> int:
       # argparse, import heavy deps inside try/except,
       # emit_progress(...) periodically, emit_done(...) at the end.
       ...
   ```
   Heavy imports (`torch`, `maskterial`, `detectron2`, …) go inside a
   `try/except ImportError` block that calls `emit_error(...)` and
   returns `2` — this keeps the GUI launchable even when those deps
   aren't installed.

2. **Add a runner wrapper** as a `QObject` in `pipeline/<area>.py` (see
   `EvaluationRunner` / `InferenceRunner` for the pattern). It owns a
   `QProcess`, parses `PROGRESS ` lines, and emits Qt signals.

3. **Wire it into a view** in `ui/`. Use the existing views as a
   reference for the readiness / start / stop / progress / log layout.

4. **Register the stage** in `STAGES` in
   [ui/main_window.py](maskterial_trainer/ui/main_window.py) and add the
   view to the `QStackedWidget`.

## Coding conventions

- Python 3.10+. Use `from __future__ import annotations` at the top of
  every module so annotations stay strings.
- Prefer `pathlib.Path` to `os.path`.
- Keep `ui/` thin: business logic and file IO belong in `pipeline/`.
- Don't import `torch`, `maskterial`, or `detectron2` from anywhere
  other than `runners/` — the GUI must launch with only the deps in
  `requirements.txt` installed.
- New configs live as JSON / YAML under
  `maskterial_trainer/resources/configs/<MODEL>/` and are referenced via
  the helpers in `pipeline/training.py`.

## Running locally

```bash
pip install -r requirements-dev.txt
pip install -e .

# Launch
python -m maskterial_trainer

# Lint
ruff check maskterial_trainer
```

## Upstream MaskTerial coupling

The trainer only depends on the **installed `maskterial` Python
package**, not on any path inside its source tree. Error messages that
mention "the MaskTerial source tree" are install hints; nothing in the
runtime resolves files relative to a sibling directory.

If MaskTerial's API changes, the breakage points are:

| Used here | Upstream symbol |
|---|---|
| `runners/train_amm.py`, `train_gmm.py` | `maskterial.utils.data_loader.ContrastDataloader` |
| `runners/train_amm.py` | `maskterial.modeling.common.fcresnet.FCResNet` |
| `runners/train_segmentation.py` | `maskterial.utils.dataset_functions.setup_config`, `maskterial.utils.model_trainer.MaskTerial_Trainer` |
| `runners/evaluate_classifier.py`, `run_inference.py` | `maskterial.modeling.classification_models.{AMM,GMM}.*_head` |

Pin a known-good upstream commit in the install instructions if upstream
churn becomes a problem.
