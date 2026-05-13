"""Shared helpers for training runners spawned as subprocesses.

Runners write structured `PROGRESS {json}` lines to stdout; the GUI parses them
to drive the progress bar / ETA / log pane. Plain stdout lines are forwarded
to the log pane verbatim.
"""

from __future__ import annotations

import json
import sys

PROGRESS_PREFIX = "PROGRESS "


def emit_progress(
    step: int,
    total: int,
    *,
    loss: float | None = None,
    stage: str = "train",
    message: str = "",
) -> None:
    payload = {"step": int(step), "total": int(total), "stage": stage}
    if loss is not None:
        payload["loss"] = float(loss)
    if message:
        payload["message"] = message
    sys.stdout.write(PROGRESS_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


def emit_done(message: str = "Training complete.") -> None:
    emit_progress(1, 1, stage="done", message=message)


def emit_error(message: str) -> None:
    emit_progress(0, 0, stage="error", message=message)
