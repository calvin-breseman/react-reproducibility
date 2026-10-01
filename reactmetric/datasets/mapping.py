"""Explicit dataset column/unit mappings used by the unified loader.

Built-in presets live in this module as Python objects (the published-package
API). Users can also load/save YAML for custom datasets. ``specify_mapping``
updates a named mapping in the registry and optionally persists it to YAML; it
is not part of the load call path.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

MappingLike = Union[str, "DatasetMapping", Dict[str, Any]]


@dataclass(frozen=True)
class DatasetMapping:
    """How a tabular trajectory file maps onto REACT's TrajectorySet.

    Attributes:
        name: Registry key, e.g. ``\"eindhoven\"`` or a user-defined name.
        id: Column holding the track / object identifier.
        time: Optional time column (sorted ascending per track).
        coords: Position column names, in axis order.
        dt: Nominal frame interval in seconds when not inferred from ``time``.
        position_scale: Multiply raw coordinate values (e.g. 1e-3 for mm -> m).
        time_scale: Multiply raw time values (e.g. 1e-3 for ms -> s).
        min_length: Drop tracks shorter than this many frames.
        format: ``\"tabular\"`` (csv/parquet) or ``\"whitespace_ethucy\"``.
        notes: Free-form documentation for the mapping.
    """

    name: str
    id: str
    coords: Tuple[str, ...]
    dt: float
    time: Optional[str] = None
    position_scale: float = 1.0
    time_scale: float = 1.0
    min_length: int = 20
    format: str = "tabular"
    notes: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **overrides: Any) -> DatasetMapping:
        """Return a copy with selected fields replaced."""
        return replace(self, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["coords"] = list(self.coords)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetMapping:
        payload = dict(data)
        coords = payload.get("coords")
        if coords is not None:
            payload["coords"] = tuple(coords)
        extras = payload.pop("extras", {}) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = {k: payload.pop(k) for k in list(payload) if k not in known}
        extras = {**extras, **unknown}
        return cls(extras=extras, **payload)

    def to_yaml(self, path: str) -> str:
        """Write this mapping to a YAML file; returns ``path``."""
        _dump_yaml(self.to_dict(), path)
        return path

    @classmethod
    def from_yaml(cls, path: str) -> DatasetMapping:
        """Load a mapping definition from a YAML file."""
        return cls.from_dict(_load_yaml(path))


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "YAML mapping I/O requires PyYAML: pip install 'reactmetric[datasets]'"
        ) from e
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"mapping YAML must be a mapping/object; got {type(data)}")
    return data


def _dump_yaml(data: Dict[str, Any], path: str) -> None:
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "YAML mapping I/O requires PyYAML: pip install 'reactmetric[datasets]'"
        ) from e
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# Built-in presets: explicit, documented, and the single source of truth for
# default dataset column/unit assumptions.
_BUILTIN: Dict[str, DatasetMapping] = {
    "eindhoven": DatasetMapping(
        name="eindhoven",
        id="object_identifier",
        time="time_ms",
        coords=("x_position_mm", "y_position_mm"),
        dt=0.1,
        position_scale=1e-3,
        time_scale=1e-3,
        min_length=20,
        format="tabular",
        notes="Zenodo Eindhoven Centraal: mm positions, ms timestamps, 10 Hz.",
    ),
    "tracks": DatasetMapping(
        name="tracks",
        id="person_track_id",
        time="timestamp",
        coords=("middle_of_waist_x", "middle_of_waist_y"),
        dt=0.1,
        position_scale=1.0,
        time_scale=1.0,
        min_length=20,
        format="tabular",
        notes=(
            "Standard AI Tracks public sample schema: person_track_id + datetime "
            "timestamp + planar waist position (meters) at 10 Hz. Load a single "
            "local CSV/parquet; calib/eval is done later with TrajectorySet.split."
        ),
        extras={
            "coords_3d": (
                "middle_of_waist_x",
                "middle_of_waist_y",
                "middle_of_waist_z",
            )
        },
    ),
    "ethucy": DatasetMapping(
        name="ethucy",
        id="ped_id",
        time="frame",
        coords=("x", "y"),
        dt=0.4,
        position_scale=1.0,
        time_scale=1.0,
        min_length=20,
        format="whitespace_ethucy",
        notes="ETH/UCY whitespace rows: frame ped_id x y (or raw obsmat x z y).",
    ),
}


class MappingRegistry:
    """Named dataset mappings: builtins + user overrides."""

    def __init__(self, builtins: Optional[Dict[str, DatasetMapping]] = None):
        self._maps: Dict[str, DatasetMapping] = {
            k: deepcopy(v) for k, v in (builtins or _BUILTIN).items()
        }

    def __getattr__(self, name: str) -> DatasetMapping:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._maps[name]
        except KeyError as e:
            raise AttributeError(
                f"unknown mapping {name!r}; available: {sorted(self._maps)}"
            ) from e

    def __contains__(self, name: object) -> bool:
        return name in self._maps

    def __iter__(self) -> Iterable[str]:
        return iter(self._maps)

    def names(self) -> List[str]:
        return sorted(self._maps)

    def get(self, name: str) -> DatasetMapping:
        if name not in self._maps:
            raise KeyError(f"unknown mapping {name!r}; available: {self.names()}")
        return self._maps[name]

    def resolve(self, mapping: MappingLike) -> DatasetMapping:
        if isinstance(mapping, DatasetMapping):
            return mapping
        if isinstance(mapping, str):
            return self.get(mapping)
        if isinstance(mapping, dict):
            return DatasetMapping.from_dict(mapping)
        raise TypeError(f"mapping must be str, DatasetMapping, or dict; got {type(mapping)}")


mappings = MappingRegistry()


def specify_mapping(
    name: str,
    mapping: Optional[MappingLike] = None,
    *,
    path: Optional[str] = None,
    persist: Optional[str] = None,
    **overrides: Any,
) -> DatasetMapping:
    """Update a named mapping in the registry (and optionally persist to YAML).

    This is the configuration surface for column/unit assumptions. It is not
    the dataset load path.

    Examples:
        # replace / register a mapping object
        rm.datasets.specify_mapping(
            "eindhoven",
            rm.datasets.mappings.eindhoven.with_updates(dt=0.1),
        )

        # patch fields on an existing preset
        rm.datasets.specify_mapping("tracks", dt=0.1, min_length=30)

        # load overrides from YAML into the registry
        rm.datasets.specify_mapping("my_store", path="store.yaml")

        # update registry and write YAML
        rm.datasets.specify_mapping(
            "my_store",
            id="pid",
            coords=("x", "y"),
            dt=0.1,
            persist="store.yaml",
        )
    """
    if mapping is not None and path is not None:
        raise ValueError("pass either mapping=... or path=..., not both")

    if path is not None:
        base = DatasetMapping.from_yaml(path)
        if overrides:
            base = base.with_updates(**overrides)
        resolved = base.with_updates(name=name)
    elif mapping is not None:
        resolved = mappings.resolve(mapping)
        if overrides:
            resolved = resolved.with_updates(**overrides)
        resolved = resolved.with_updates(name=name)
    elif name in mappings:
        resolved = mappings.get(name).with_updates(**overrides)
        resolved = resolved.with_updates(name=name)
    else:
        if "id" not in overrides or "coords" not in overrides or "dt" not in overrides:
            raise ValueError(
                "new mappings require id=, coords=, and dt= (or mapping=/path=)"
            )
        coords = overrides.pop("coords")
        if not isinstance(coords, Sequence) or isinstance(coords, (str, bytes)):
            raise TypeError("coords must be a sequence of column names")
        resolved = DatasetMapping(
            name=name,
            id=overrides.pop("id"),
            coords=tuple(coords),
            dt=float(overrides.pop("dt")),
            **overrides,
        )

    mappings._maps[name] = resolved
    if persist is not None:
        resolved.to_yaml(persist)
    return resolved


def reset_mappings() -> None:
    """Restore built-in presets (useful in tests)."""
    mappings._maps = {k: deepcopy(v) for k, v in _BUILTIN.items()}
