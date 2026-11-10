"""JSON board project model."""

from model.schema import (
    STATUS_COLORS,
    BoardProject,
    Component,
    Fiducial,
    SchemaError,
    validate_project,
)
from model.store import ProjectStore

__all__ = [
    "STATUS_COLORS",
    "BoardProject",
    "Component",
    "Fiducial",
    "ProjectStore",
    "SchemaError",
    "validate_project",
]
