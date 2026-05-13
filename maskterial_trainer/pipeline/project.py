from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_FILE = "project.json"

SUBDIRS = [
    "instance_masks",
    "semantic_masks",
    "coco",
    "splits",
    "outputs/segmentation",
    "outputs/gmm",
    "outputs/amm",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class MaterialClass:
    id: int
    name: str
    color: str  # hex


@dataclass
class MaterialProject:
    path: Path
    name: str
    flatfield_path: str | None = None
    flatfield_enabled: bool = False
    classes: list[MaterialClass] = field(default_factory=list)

    @classmethod
    def open_or_create(cls, path: Path) -> "MaterialProject":
        if (path / PROJECT_FILE).exists():
            return cls.open(path)
        return cls.create(path)

    @classmethod
    def create(cls, path: Path) -> "MaterialProject":
        path.mkdir(parents=True, exist_ok=True)
        for sub in SUBDIRS:
            (path / sub).mkdir(parents=True, exist_ok=True)
        project = cls(path=path, name=path.name)
        project.save()
        return project

    @classmethod
    def open(cls, path: Path) -> "MaterialProject":
        project_file = path / PROJECT_FILE
        if not project_file.exists():
            raise FileNotFoundError(f"No {PROJECT_FILE} in {path}")
        data = json.loads(project_file.read_text())
        classes = []
        for i, c in enumerate(data.get("classes", []), start=1):
            classes.append(
                MaterialClass(
                    id=int(c.get("id", i)),
                    name=c["name"],
                    color=c["color"],
                )
            )
        return cls(
            path=path,
            name=data.get("name", path.name),
            flatfield_path=data.get("flatfield_path"),
            flatfield_enabled=data.get("flatfield_enabled", False),
            classes=classes,
        )

    def save(self) -> None:
        data = {
            "name": self.name,
            "flatfield_path": self.flatfield_path,
            "flatfield_enabled": self.flatfield_enabled,
            "classes": [asdict(c) for c in self.classes],
        }
        (self.path / PROJECT_FILE).write_text(json.dumps(data, indent=2))

    @property
    def flatfield_full_path(self) -> Path | None:
        if self.flatfield_path is None:
            return None
        return self.path / self.flatfield_path

    def list_raw_images(self) -> list[Path]:
        excluded = {self.flatfield_path} if self.flatfield_path else set()
        return sorted(
            p for p in self.path.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and p.name not in excluded
        )

    def import_image(self, src: Path) -> Path:
        if src.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image type: {src.suffix}")
        dest = self.path / src.name
        if dest.resolve() == src.resolve():
            return dest
        if dest.exists():
            stem, suffix = src.stem, src.suffix
            i = 1
            while True:
                candidate = self.path / f"{stem}_{i}{suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
                i += 1
        shutil.copy2(src, dest)
        return dest

    def set_flatfield(self, src: Path) -> Path:
        if src.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image type: {src.suffix}")
        if self.flatfield_path:
            old = self.path / self.flatfield_path
            if old.exists():
                old.unlink()
        dest = self.path / f"flatfield{src.suffix.lower()}"
        shutil.copy2(src, dest)
        self.flatfield_path = dest.name
        self.flatfield_enabled = True
        self.save()
        return dest

    def clear_flatfield(self) -> None:
        if self.flatfield_path:
            old = self.path / self.flatfield_path
            if old.exists():
                old.unlink()
        self.flatfield_path = None
        self.flatfield_enabled = False
        self.save()

    def set_flatfield_enabled(self, enabled: bool) -> None:
        self.flatfield_enabled = bool(enabled and self.flatfield_path is not None)
        self.save()

    def add_class(self, name: str, color: str) -> MaterialClass:
        next_id = max((c.id for c in self.classes), default=0) + 1
        cls = MaterialClass(id=next_id, name=name, color=color)
        self.classes.append(cls)
        self.save()
        return cls

    def remove_class(self, class_id: int) -> bool:
        before = len(self.classes)
        self.classes = [c for c in self.classes if c.id != class_id]
        if len(self.classes) < before:
            self.save()
            return True
        return False

    def find_class(self, class_id: int) -> MaterialClass | None:
        for c in self.classes:
            if c.id == class_id:
                return c
        return None
