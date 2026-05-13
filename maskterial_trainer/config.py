from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".maskterial_trainer"
CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_PROJECTS_ROOT = Path.home() / "MaskTerialProjects"


@dataclass
class UserConfig:
    projects_root: str = str(DEFAULT_PROJECTS_ROOT)
    recent_projects: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> "UserConfig":
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            return cls(
                projects_root=data.get("projects_root", str(DEFAULT_PROJECTS_ROOT)),
                recent_projects=data.get("recent_projects", []),
            )
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))

    def add_recent(self, project_path: Path) -> None:
        s = str(project_path)
        if s in self.recent_projects:
            self.recent_projects.remove(s)
        self.recent_projects.insert(0, s)
        self.recent_projects = self.recent_projects[:10]
        self.save()
