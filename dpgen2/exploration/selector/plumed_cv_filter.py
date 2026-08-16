from pathlib import (
    Path,
)
from typing import (
    Dict,
    List,
    Sequence,
)

import numpy as np
from dargs import (
    Argument,
)
from dflow.python import (
    FatalError,
)


class PlumedCVFilter:
    """Select frames in a union of PLUMED CV regions.

    Each region maps field names from ``#! FIELDS`` to a lower-inclusive,
    upper-exclusive interval. A frame is selected when it matches every
    interval in any one region.
    """

    @staticmethod
    def args() -> List[Argument]:
        return [
            Argument(
                "regions",
                list,
                optional=False,
                doc=(
                    "A list of PLUMED field-to-[lower, upper] mappings. "
                    "Fields within a region are ANDed; regions are ORed."
                ),
            )
        ]

    def __init__(self, regions: List[Dict[str, Sequence[float]]]):
        if (
            not isinstance(regions, list)
            or not regions
            or any(not isinstance(region, dict) or not region for region in regions)
        ):
            raise ValueError("PLUMED CV regions must be a non-empty list of dicts")
        self.regions = []
        for region in regions:
            normalized = {}
            for field, bounds in region.items():
                if not isinstance(field, str) or not field:
                    raise ValueError("PLUMED field names must be non-empty strings")
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError(f"invalid interval for PLUMED field {field!r}")
                try:
                    normalized[field] = (float(bounds[0]), float(bounds[1]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid interval for PLUMED field {field!r}"
                    ) from exc
                if normalized[field][0] >= normalized[field][1]:
                    raise ValueError(f"invalid interval for PLUMED field {field!r}")
                if not np.all(np.isfinite(normalized[field])):
                    raise ValueError(f"non-finite interval for PLUMED field {field!r}")
            self.regions.append(normalized)

    def get_selected_ids(
        self,
        files: List[Path],
        nframes: List[int],
    ) -> List[List[int]]:
        if len(files) != len(nframes):
            raise FatalError("PLUMED outputs and trajectories have different lengths")
        selected = []
        for file, expected_nframes in zip(files, nframes):
            fields, values = self._read(file)
            if len(values) != expected_nframes:
                raise FatalError(
                    f"PLUMED output {file} has {len(values)} rows, expected "
                    f"{expected_nframes}; PRINT STRIDE must match the trajectory stride"
                )
            field_idx = {field: idx for idx, field in enumerate(fields)}
            keep = np.zeros(expected_nframes, dtype=bool)
            for region in self.regions:
                in_region = np.ones(expected_nframes, dtype=bool)
                for field, (lower, upper) in region.items():
                    if field not in field_idx:
                        raise FatalError(
                            f"PLUMED field {field!r} is missing from {file}"
                        )
                    column = values[:, field_idx[field]]
                    in_region &= (column >= lower) & (column < upper)
                keep |= in_region
            selected.append(np.flatnonzero(keep).tolist())
        return selected

    @staticmethod
    def _read(file: Path):
        fields = None
        rows = []
        try:
            with open(file, encoding="utf8") as handle:
                for line_number, line in enumerate(handle, 1):
                    words = line.split()
                    if not words:
                        continue
                    if words[:2] == ["#!", "FIELDS"]:
                        new_fields = words[2:]
                        if fields is not None and fields != new_fields:
                            raise FatalError(
                                f"inconsistent PLUMED FIELDS headers in {file}"
                            )
                        fields = new_fields
                    elif words[0].startswith("#"):
                        continue
                    else:
                        if fields is None:
                            raise FatalError(
                                f"PLUMED numeric row precedes FIELDS header in {file}"
                            )
                        try:
                            rows.append([float(value) for value in words])
                        except ValueError as exc:
                            raise FatalError(
                                f"invalid PLUMED numeric row {line_number} in {file}"
                            ) from exc
        except OSError as exc:
            raise FatalError(f"cannot read PLUMED output {file}: {exc}") from exc
        if fields is None:
            raise FatalError(f"PLUMED FIELDS header is missing from {file}")
        if not fields or len(fields) != len(set(fields)):
            raise FatalError(f"PLUMED FIELDS must be non-empty and unique in {file}")
        if any(len(row) != len(fields) for row in rows):
            raise FatalError(f"PLUMED row width does not match FIELDS in {file}")
        values = np.asarray(rows, dtype=float).reshape((-1, len(fields)))
        if not np.all(np.isfinite(values)):
            raise FatalError(f"non-finite PLUMED values in {file}")
        if "time" not in fields:
            raise FatalError(f"PLUMED time field is missing from {file}")
        time = values[:, fields.index("time")]
        if len(time) > 1 and np.any(np.diff(time) <= 0):
            raise FatalError(f"PLUMED time must be strictly increasing in {file}")
        return fields, values
