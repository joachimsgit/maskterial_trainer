from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Callable

from .project import IMAGE_EXTS, MaterialProject

AMM_FILES = ["meta_data.json", "loc.npy", "cov.npy", "model.pth"]
GMM_FILES = ["contrast_dict.json"]
M2F_FILES = ["model.pth", "config.yaml"]

EVAL_FILE = "evaluation.json"


def _outputs(project: MaterialProject, model: str) -> Path:
    return project.path / "outputs" / model


def has_amm(project: MaterialProject) -> bool:
    d = _outputs(project, "amm")
    return all((d / f).exists() for f in AMM_FILES)


def has_gmm(project: MaterialProject) -> bool:
    d = _outputs(project, "gmm")
    return all((d / f).exists() for f in GMM_FILES)


def has_segmentation(project: MaterialProject) -> bool:
    d = _outputs(project, "segmentation")
    return all((d / f).exists() for f in M2F_FILES)


def has_amm_eval(project: MaterialProject) -> bool:
    return (_outputs(project, "amm") / EVAL_FILE).exists()


def has_gmm_eval(project: MaterialProject) -> bool:
    return (_outputs(project, "gmm") / EVAL_FILE).exists()


def default_export_dir(project: MaterialProject) -> Path:
    return project.path.parent / f"{project.name}_export"


def export_project(
    project: MaterialProject,
    dest: Path,
    include_zip: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict:
    if dest.exists():
        if any(dest.iterdir()):
            raise FileExistsError(
                f"Destination is not empty: {dest}. "
                "Pick another folder or delete the existing one."
            )
    else:
        dest.mkdir(parents=True)

    def step(msg: str) -> None:
        if progress is not None:
            progress(msg)

    # 1. Raw images
    images_dest = dest / "images"
    images_dest.mkdir()
    raw_images = project.list_raw_images()
    step(f"Copying {len(raw_images)} image(s)…")
    for src in raw_images:
        shutil.copy2(src, images_dest / src.name)

    # 2. Instance masks
    instance_dest = dest / "instance_masks"
    instance_src = project.path / "instance_masks"
    n_inst = 0
    if instance_src.exists():
        instance_dest.mkdir()
        for f in instance_src.iterdir():
            if f.is_file() and f.suffix.lower() == ".png":
                shutil.copy2(f, instance_dest / f.name)
                n_inst += 1
    step(f"Copied {n_inst} instance mask(s).")

    # 3. Semantic masks
    semantic_dest = dest / "semantic_masks"
    semantic_src = project.path / "semantic_masks"
    n_sem = 0
    if semantic_src.exists():
        semantic_dest.mkdir()
        for f in semantic_src.iterdir():
            if f.is_file() and f.suffix.lower() == ".png":
                shutil.copy2(f, semantic_dest / f.name)
                n_sem += 1
    step(f"Copied {n_sem} semantic mask(s).")

    # 4. Models
    bundled_models: list[str] = []
    eval_dest = dest / "evaluation"

    if has_amm(project):
        amm_dest = dest / "amm"
        amm_dest.mkdir()
        amm_src = _outputs(project, "amm")
        for f in AMM_FILES:
            shutil.copy2(amm_src / f, amm_dest / f)
        bundled_models.append("amm")
        step("Bundled AMM model.")
        if has_amm_eval(project):
            eval_dest.mkdir(exist_ok=True)
            shutil.copy2(amm_src / EVAL_FILE, eval_dest / "amm.json")

    if has_gmm(project):
        gmm_dest = dest / "gmm"
        gmm_dest.mkdir()
        gmm_src = _outputs(project, "gmm")
        for f in GMM_FILES:
            shutil.copy2(gmm_src / f, gmm_dest / f)
        # Include the auxiliary loc/cov/meta as well — useful for re-training
        for extra in ["loc.npy", "cov.npy", "meta_data.json"]:
            src = gmm_src / extra
            if src.exists():
                shutil.copy2(src, gmm_dest / extra)
        bundled_models.append("gmm")
        step("Bundled GMM model.")
        if has_gmm_eval(project):
            eval_dest.mkdir(exist_ok=True)
            shutil.copy2(gmm_src / EVAL_FILE, eval_dest / "gmm.json")

    if has_segmentation(project):
        seg_dest = dest / "segmentation"
        seg_dest.mkdir()
        seg_src = _outputs(project, "segmentation")
        for f in M2F_FILES:
            shutil.copy2(seg_src / f, seg_dest / f)
        bundled_models.append("segmentation")
        step("Bundled segmentation model.")

    # 5. Project metadata
    project_json = project.path / "project.json"
    if project_json.exists():
        shutil.copy2(project_json, dest / "project.json")

    # 6. README
    (dest / "README.md").write_text(
        _build_readme(project, bundled_models, n_inst, n_sem, len(raw_images))
    )

    # 7. Optional zip
    zip_path: Path | None = None
    if include_zip:
        zip_path = dest.parent / f"{dest.name}.zip"
        step(f"Writing zip {zip_path.name}…")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in dest.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(dest.parent))

    step("Export complete.")
    return {
        "dest": str(dest),
        "zip": str(zip_path) if zip_path else None,
        "bundled_models": bundled_models,
        "n_images": len(raw_images),
        "n_instance_masks": n_inst,
        "n_semantic_masks": n_sem,
    }


def _build_readme(
    project: MaterialProject,
    bundled_models: list[str],
    n_instance: int,
    n_semantic: int,
    n_images: int,
) -> str:
    lines = [
        f"# {project.name} export",
        "",
        "Bundle produced by MaskTerial Trainer for upload to the inference website.",
        "",
        "## Contents",
        "",
        f"- `images/` — {n_images} original microscope image(s)",
        f"- `instance_masks/` — {n_instance} 16-bit PNG mask(s), pixel value = instance id",
        f"- `semantic_masks/` — {n_semantic} 8-bit PNG mask(s), pixel value = class id (0 = background)",
    ]
    if "amm" in bundled_models:
        lines.append(
            "- `amm/` — AMM classifier (`meta_data.json`, `loc.npy`, `cov.npy`, `model.pth`)"
        )
    if "gmm" in bundled_models:
        lines.append(
            "- `gmm/` — GMM classifier (`contrast_dict.json` is the upload payload; "
            "extras `loc.npy`/`cov.npy`/`meta_data.json` included for reference)"
        )
    if "segmentation" in bundled_models:
        lines.append(
            "- `segmentation/` — segmentation model (`model.pth`, `config.yaml`)"
        )
    lines.extend(
        [
            "- `evaluation/` — accuracy / per-class P/R/F1 / confusion matrix per model (if evaluated)",
            "- `project.json` — class names and colors used during annotation",
            "",
            "## Uploading to the inference website",
            "",
        ]
    )
    if "amm" in bundled_models:
        lines.append(
            "- **AMM** (`/upload/amm`): pick a model name, attach the four files in `amm/`."
        )
    if "gmm" in bundled_models:
        lines.append(
            "- **GMM** (`/upload/gmm`): pick a model name, attach `gmm/contrast_dict.json`."
        )
    if "segmentation" in bundled_models:
        lines.append(
            "- **Segmentation / M2F** (`/upload/m2f`): pick a model name, attach "
            "`segmentation/model.pth` and `segmentation/config.yaml`."
        )
    if not bundled_models:
        lines.append(
            "_No trained models were bundled — train at least one model before exporting._"
        )
    lines.append("")
    lines.append("## Classes")
    lines.append("")
    if project.classes:
        for c in project.classes:
            lines.append(f"- {c.id}: {c.name} ({c.color})")
    else:
        lines.append("_No classes defined._")
    lines.append("")
    return "\n".join(lines)
