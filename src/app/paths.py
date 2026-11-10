"""Locate repository files when the app is launched from a src checkout."""

from pathlib import Path

_SAMPLE_NAME = "DEMO-PCB-001.json"


def repo_root() -> Path | None:
    """Return the checkout root that holds ``assets/projects``, if any."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "assets" / "projects" / _SAMPLE_NAME).is_file():
            return parent
    return None


def sample_project_path() -> Path | None:
    root = repo_root()
    if root is None:
        return None
    return root / "assets" / "projects" / _SAMPLE_NAME
