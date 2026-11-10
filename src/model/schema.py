"""Board project schema.

Aligned with ``docs/DATA_MODEL.md`` v0.1. Validation is hand-written; pydantic
is not a dependency.

Confirmed for this slice:

- D2: a pure-pixel board is valid. ``absolute_pos_mm`` and ``pixel_per_mm``
  may be omitted.
- D5: ``doubt`` stays in ``STATUS_COLORS`` and is accepted on load. The MVP
  shell does not have to offer a doubt control.
- D9/D12: ``Component.status`` is the only authoritative status. Bubble
  ``core_params`` are exactly ``name``, ``value``, and ``package``. A missing
  bubble key is stored as ``""`` on load. ``core_params.status`` is promoted
  to the top level and removed from the bubble keys.

Unknown ``status`` values are coerced to ``unchecked`` during
``BoardProject.from_dict`` (and logged). ``validate_project`` itself stays
strict: a raw document that still contains an unknown status, or core keys
other than the bubble trio, is reported.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

STATUS_COLORS = {
    "unchecked": "#E53935",
    "checked": "#43A047",
    "doubt": "#FDD835",
}
KNOWN_STATUSES = frozenset(STATUS_COLORS)
# D9/D12. Order is the bubble order.
CORE_PARAM_KEYS = ("name", "value", "package")
MAX_CORE_KEYS = len(CORE_PARAM_KEYS)


class SchemaError(ValueError):
    """One or more project-document problems. Nothing was partially applied."""

    def __init__(self, messages: list[str]):
        self.messages = list(messages)
        super().__init__("; ".join(self.messages) if self.messages else "invalid project")


@dataclass
class PointMm:
    x: float
    y: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}


@dataclass
class Fiducial:
    id: str
    pixel: tuple[float, float]
    mm: PointMm | None = None
    template_path: str | None = None

    def to_dict(self) -> dict:
        payload: dict = {"id": self.id, "pixel": [self.pixel[0], self.pixel[1]]}
        if self.mm is not None:
            payload["mm"] = self.mm.to_dict()
        if self.template_path:
            payload["template_path"] = self.template_path
        return payload


@dataclass
class Component:
    component_id: str
    base_pixel_box: tuple[float, float, float, float]
    status: str
    core_params: dict
    # D2: millimetres are optional. A pure-pixel component leaves this None.
    absolute_pos_mm: PointMm | None = None
    extended_params: dict = field(default_factory=dict)
    meta: dict | None = None
    geo_fingerprint: dict | None = None

    def to_dict(self) -> dict:
        payload: dict = {"component_id": self.component_id}
        if self.absolute_pos_mm is not None:
            payload["absolute_pos_mm"] = self.absolute_pos_mm.to_dict()
        payload["base_pixel_box"] = list(self.base_pixel_box)
        payload["status"] = self.status
        payload["core_params"] = dict(self.core_params)
        payload["extended_params"] = dict(self.extended_params)
        if self.meta is not None:
            payload["meta"] = copy.deepcopy(self.meta)
        if self.geo_fingerprint is not None:
            payload["geo_fingerprint"] = copy.deepcopy(self.geo_fingerprint)
        return payload


@dataclass
class BoardProject:
    board_id: str
    global_map_path: str
    components: list[Component]
    schema_version: str = "0.1"
    name: str | None = None
    global_ref_path: str | None = None
    map_source: str | None = None
    topology_db: str | None = None
    pixel_per_mm: PointMm | None = None
    fiducials: list[Fiducial] = field(default_factory=list)
    meta: dict | None = None

    def to_dict(self) -> dict:
        payload: dict = {
            "schema_version": self.schema_version,
            "board_id": self.board_id,
        }
        if self.name:
            payload["name"] = self.name
        payload["global_map_path"] = self.global_map_path
        if self.map_source:
            payload["map_source"] = self.map_source
        if self.global_ref_path:
            payload["global_ref_path"] = self.global_ref_path
        if self.topology_db:
            payload["topology_db"] = self.topology_db
        if self.pixel_per_mm is not None:
            payload["pixel_per_mm"] = self.pixel_per_mm.to_dict()
        payload["fiducials"] = [item.to_dict() for item in self.fiducials]
        payload["components"] = [component.to_dict() for component in self.components]
        if self.meta is not None:
            payload["meta"] = copy.deepcopy(self.meta)
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> BoardProject:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SchemaError([f"invalid JSON: {exc}"]) from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> BoardProject:
        if not isinstance(data, dict):
            raise SchemaError(["project must be a JSON object"])
        working = copy.deepcopy(data)
        _normalize_project(working)
        errors = validate_project(working)
        if errors:
            raise SchemaError(errors)
        return _build_project(working)


def validate_project(data: dict) -> list[str]:
    """Return hard-error messages. An empty list means the dict is acceptable.

    This function does **not** promote ``core_params.status`` and does **not**
    coerce unknown statuses. Call ``BoardProject.from_dict`` to load documents.
    Global-map file existence is not checked here: the sample map is not in
    this slice, and path-root resolution is still an open decision.
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["project must be a JSON object"]

    _require_str(data, "schema_version", errors, required=False)
    _require_str(data, "board_id", errors, required=True)
    _require_str(data, "name", errors, required=False)
    _check_path(data.get("global_map_path"), "global_map_path", errors, required=True)
    if data.get("global_ref_path"):
        _check_path(data.get("global_ref_path"), "global_ref_path", errors, required=True)
    if "map_source" in data and data["map_source"] is not None and not isinstance(data["map_source"], str):
        errors.append("map_source must be a string")
    if data.get("topology_db"):
        _check_path(data.get("topology_db"), "topology_db", errors, required=True)

    if "pixel_per_mm" in data and data["pixel_per_mm"] is not None:
        _check_point(data["pixel_per_mm"], "pixel_per_mm", errors)

    fiducials = data.get("fiducials", [])
    if fiducials is None:
        fiducials = []
    if not isinstance(fiducials, list):
        errors.append("fiducials must be an array")
    else:
        seen_fiducials: set[str] = set()
        for index, item in enumerate(fiducials):
            _check_fiducial(item, index, seen_fiducials, errors)

    if "components" not in data:
        errors.append("components is required")
    elif not isinstance(data["components"], list):
        errors.append("components must be an array")
    else:
        seen_ids: set[str] = set()
        for index, item in enumerate(data["components"]):
            _check_component(item, index, seen_ids, errors)

    if "meta" in data and data["meta"] is not None and not isinstance(data["meta"], dict):
        errors.append("meta must be an object")
    return errors


def resolve_project_relative(project_file: Path, relative: str, repo_root: Path) -> Path:
    """Resolve a project-relative path and reject targets outside ``repo_root`` (D11)."""
    if not isinstance(relative, str) or not relative or "\\" in relative or relative.startswith("/"):
        raise ValueError(f"path must be a relative '/' path: {relative!r}")
    project_file = Path(project_file).resolve()
    repo_root = Path(repo_root).resolve()
    target = (project_file.parent / relative).resolve()
    try:
        target.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {relative}") from exc
    return target


def _normalize_project(data: dict) -> None:
    data.setdefault("schema_version", "0.1")
    components = data.get("components")
    if not isinstance(components, list):
        return
    for index, component in enumerate(components):
        if isinstance(component, dict):
            _promote_status(component, index)
            _coerce_status(component, index)
            _canonicalize_core(component, index)


def _promote_status(component: dict, index: int) -> None:
    core = component.get("core_params")
    if not isinstance(core, dict) or "status" not in core:
        return
    legacy = core.pop("status")
    top = component.get("status")
    if top in (None, ""):
        component["status"] = legacy
        logger.info(
            "component[%s] promoted core_params.status to top-level status",
            index,
        )
        return
    if top != legacy:
        logger.info(
            "component[%s] ignored core_params.status=%r because top-level status=%r",
            index,
            legacy,
            top,
        )


def _canonicalize_core(component: dict, index: int) -> None:
    """Force bubble keys to name, value, package. Extra keys are left for validation."""
    core = component.get("core_params")
    if not isinstance(core, dict):
        return
    canonical: dict = {}
    for key in CORE_PARAM_KEYS:
        if key in core:
            canonical[key] = core[key]
        else:
            canonical[key] = ""
            logger.info(
                "component[%s] core_params.%s missing; using an empty placeholder",
                index,
                key,
            )
    for key, value in core.items():
        if key not in CORE_PARAM_KEYS:
            canonical[key] = value
    component["core_params"] = canonical


def _coerce_status(component: dict, index: int) -> None:
    status = component.get("status")
    if isinstance(status, str) and status not in KNOWN_STATUSES:
        logger.warning(
            "component[%s] status %r is unknown; treating as unchecked",
            index,
            status,
        )
        component["status"] = "unchecked"


def _build_project(data: dict) -> BoardProject:
    fiducials = [_build_fiducial(item) for item in data.get("fiducials", [])]
    components = [_build_component(item) for item in data["components"]]
    pixel_per_mm = None
    if data.get("pixel_per_mm") is not None:
        pixel_per_mm = PointMm(x=data["pixel_per_mm"]["x"], y=data["pixel_per_mm"]["y"])
    return BoardProject(
        board_id=data["board_id"],
        global_map_path=data["global_map_path"],
        components=components,
        schema_version=str(data.get("schema_version", "0.1")),
        name=data.get("name") or None,
        global_ref_path=data.get("global_ref_path") or None,
        map_source=data.get("map_source") or None,
        topology_db=data.get("topology_db") or None,
        pixel_per_mm=pixel_per_mm,
        fiducials=fiducials,
        meta=copy.deepcopy(data["meta"]) if data.get("meta") is not None else None,
    )


def _build_fiducial(data: dict) -> Fiducial:
    mm = None
    if data.get("mm") is not None:
        mm = PointMm(x=data["mm"]["x"], y=data["mm"]["y"])
    pixel = data["pixel"]
    return Fiducial(
        id=data["id"],
        pixel=(pixel[0], pixel[1]),
        mm=mm,
        template_path=data.get("template_path") or None,
    )


def _build_component(data: dict) -> Component:
    absolute = None
    if data.get("absolute_pos_mm") is not None:
        absolute = PointMm(x=data["absolute_pos_mm"]["x"], y=data["absolute_pos_mm"]["y"])
    box = data["base_pixel_box"]
    extended = data.get("extended_params") or {}
    meta = copy.deepcopy(data["meta"]) if data.get("meta") is not None else None
    fingerprint = copy.deepcopy(data["geo_fingerprint"]) if data.get("geo_fingerprint") is not None else None
    return Component(
        component_id=data["component_id"],
        base_pixel_box=(box[0], box[1], box[2], box[3]),
        status=data["status"],
        core_params={key: data["core_params"][key] for key in CORE_PARAM_KEYS},
        absolute_pos_mm=absolute,
        extended_params=dict(extended),
        meta=meta,
        geo_fingerprint=fingerprint,
    )


def _require_str(data: dict, key: str, errors: list[str], *, required: bool) -> None:
    if key not in data or data[key] in (None, ""):
        if required:
            errors.append(f"{key} is required")
        return
    if not isinstance(data[key], str):
        errors.append(f"{key} must be a string")


def _check_path(value: object, field: str, errors: list[str], *, required: bool) -> None:
    if value in (None, ""):
        if required:
            errors.append(f"{field} is required")
        return
    if not isinstance(value, str):
        errors.append(f"{field} must be a string path")
        return
    if "\\" in value:
        errors.append(f"{field} must use '/' separators")
        return
    if value.startswith("/"):
        errors.append(f"{field} must be a relative path")
        return


def _check_point(value: object, field: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object with x and y")
        return
    for axis in ("x", "y"):
        number = value.get(axis)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            errors.append(f"{field}.{axis} must be a number")


def _check_fiducial(item: object, index: int, seen: set[str], errors: list[str]) -> None:
    prefix = f"fiducials[{index}]"
    if not isinstance(item, dict):
        errors.append(f"{prefix} must be an object")
        return
    fid = item.get("id")
    if not isinstance(fid, str) or not fid:
        errors.append(f"{prefix}.id is required")
    elif fid in seen:
        errors.append(f"duplicate fiducial id {fid}")
    else:
        seen.add(fid)
    pixel = item.get("pixel")
    if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
        errors.append(f"{prefix}.pixel must be [x, y]")
    else:
        for axis, number in zip(("x", "y"), pixel, strict=True):
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                errors.append(f"{prefix}.pixel {axis} must be a number")
    if "mm" in item and item["mm"] is not None:
        _check_point(item["mm"], f"{prefix}.mm", errors)
    if item.get("template_path"):
        _check_path(item.get("template_path"), f"{prefix}.template_path", errors, required=True)


def _check_component(item: object, index: int, seen: set[str], errors: list[str]) -> None:
    prefix = f"components[{index}]"
    if not isinstance(item, dict):
        errors.append(f"{prefix} must be an object")
        return
    component_id = item.get("component_id")
    if not isinstance(component_id, str) or not component_id:
        errors.append(f"{prefix}.component_id is required")
    elif component_id in seen:
        errors.append(f"duplicate component_id {component_id}")
    else:
        seen.add(component_id)

    if "absolute_pos_mm" in item and item["absolute_pos_mm"] is not None:
        _check_point(item["absolute_pos_mm"], f"{prefix}.absolute_pos_mm", errors)

    box = item.get("base_pixel_box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        errors.append(f"{prefix}.base_pixel_box must be [x, y, w, h]")
    else:
        numeric = True
        for number in box:
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                numeric = False
                errors.append(f"{prefix}.base_pixel_box values must be numbers")
                break
        if numeric and not (float(box[2]) > 0 and float(box[3]) > 0):
            errors.append(f"{prefix}.base_pixel_box width and height must be > 0")

    status = item.get("status")
    if not isinstance(status, str) or not status:
        errors.append(f"{prefix}.status is required at the top level")
    elif status not in KNOWN_STATUSES:
        errors.append(f"{prefix}.status must be unchecked, checked, or doubt")

    core = item.get("core_params")
    if not isinstance(core, dict):
        errors.append(f"{prefix}.core_params must be an object")
    else:
        extra = [key for key in core if key not in CORE_PARAM_KEYS]
        missing = [key for key in CORE_PARAM_KEYS if key not in core]
        if extra or missing:
            errors.append(
                f"{prefix}.core_params must be exactly name, value, package"
                + (f"; extra {extra}" if extra else "")
                + (f"; missing {missing}" if missing else "")
            )
        if "status" in core:
            errors.append(
                f"{prefix}.core_params.status is not allowed; use top-level status"
            )
        for key, value in core.items():
            if not isinstance(key, str) or not key:
                errors.append(f"{prefix}.core_params keys must be non-empty strings")
            if isinstance(value, (dict, list)):
                errors.append(f"{prefix}.core_params.{key} must not be nested")

    if "extended_params" in item and item["extended_params"] is not None:
        if not isinstance(item["extended_params"], dict):
            errors.append(f"{prefix}.extended_params must be an object")
    if "meta" in item and item["meta"] is not None and not isinstance(item["meta"], dict):
        errors.append(f"{prefix}.meta must be an object")
    if "geo_fingerprint" in item and item["geo_fingerprint"] is not None and not isinstance(item["geo_fingerprint"], dict):
        errors.append(f"{prefix}.geo_fingerprint must be an object")
