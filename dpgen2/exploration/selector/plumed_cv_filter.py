import csv
import json
from pathlib import (
    Path,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
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

    Each region maps exact field labels from ``#! FIELDS`` to a
    lower-inclusive, upper-exclusive interval. Labels are matched by name, not
    column position, and are otherwise arbitrary. Fields within a region are
    ANDed; regions are ORed. A region may use the legacy bare mapping or the
    named form ``{"name": ..., "conditions": {...}}``.
    """

    @staticmethod
    def args() -> List[Argument]:
        return [
            Argument(
                "regions",
                list,
                optional=False,
                doc=(
                    "A list of exact PLUMED FIELDS-label-to-[lower, upper] "
                    "mappings or named regions with name, conditions, and "
                    "optional weight. Labels are matched by name, not column "
                    "position. Fields within a region are ANDed; regions are "
                    "ORed."
                ),
            ),
            Argument(
                "sampling",
                dict,
                [
                    Argument("mode", str, optional=False),
                    Argument("field", str, optional=True, default=None),
                    Argument("n_bins", int, optional=True, default=10),
                    Argument("grid", dict, optional=True, default=None),
                    Argument("within_bin", str, optional=True, default="random"),
                    Argument("seed", int, optional=True, default=0),
                    Argument("min_frame_gap", int, optional=True, default=0),
                ],
                optional=True,
                default=None,
                doc=(
                    "Optional final candidate sampling. mode is random, uniform, "
                    "grid, or report. If omitted, one or two common CV fields are "
                    "covered uniformly. report keeps the report's selection policy."
                ),
            ),
            Argument(
                "time_alignment",
                dict,
                [
                    Argument("start", float, optional=True, default=0.0),
                    Argument("step", float, optional=False),
                    Argument("atol", float, optional=True, default=1e-8),
                ],
                optional=True,
                default=None,
                doc="Optional expected PLUMED time = start + frame * step.",
            ),
        ]

    def __init__(
        self,
        regions: List[Dict],
        sampling: Optional[Dict] = None,
        time_alignment: Optional[Dict] = None,
    ):
        if (
            not isinstance(regions, list)
            or not regions
            or any(not isinstance(region, dict) or not region for region in regions)
        ):
            raise ValueError("PLUMED CV regions must be a non-empty list of dicts")

        self.regions = []
        self.region_names = []
        self.region_weights = []
        for region_idx, region in enumerate(regions):
            if "conditions" in region:
                unknown = set(region) - {"name", "conditions", "weight"}
                if unknown:
                    raise ValueError(
                        f"unknown named PLUMED region keys: {sorted(unknown)}"
                    )
                conditions = region["conditions"]
                name = region.get("name", f"region_{region_idx}")
                weight = region.get("weight", 1.0)
            else:
                conditions = region
                name = f"region_{region_idx}"
                weight = 1.0
            if not isinstance(name, str) or not name:
                raise ValueError("PLUMED region names must be non-empty strings")
            if name in self.region_names:
                raise ValueError(f"duplicate PLUMED region name {name!r}")
            try:
                weight = float(weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid weight for PLUMED region {name!r}") from exc
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError(f"invalid weight for PLUMED region {name!r}")
            if not isinstance(conditions, dict) or not conditions:
                raise ValueError(f"PLUMED region {name!r} conditions must be a dict")

            normalized = {}
            for field, bounds in conditions.items():
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
            self.region_names.append(name)
            self.region_weights.append(weight)

        self.cv_fields = sorted({field for region in self.regions for field in region})
        self.sampling_inferred = sampling is None
        if sampling is None:
            sampling = self._default_sampling()
        self.sampling = self._normalize_sampling(sampling)
        self.time_alignment = self._normalize_time_alignment(time_alignment)

    def _default_sampling(self):
        common_fields = sorted(
            set.intersection(*(set(region) for region in self.regions))
        )
        if len(common_fields) == 1:
            return {
                "mode": "uniform",
                "field": common_fields[0],
                "n_bins": 10,
                "within_bin": "max_deviation",
            }
        if len(common_fields) == 2:
            return {
                "mode": "grid",
                "grid": {field: 10 for field in common_fields},
                "within_bin": "max_deviation",
            }
        raise ValueError(
            "sampling must be explicit unless regions share exactly one or two CV fields"
        )

    def _normalize_sampling(self, sampling: Optional[Dict]):
        if sampling is None:
            return None
        if not isinstance(sampling, dict):
            raise ValueError("PLUMED CV sampling must be a dict")
        mode = sampling.get("mode")
        if mode not in {"random", "uniform", "grid", "report"}:
            raise ValueError(
                "PLUMED CV sampling mode must be random, uniform, grid, or report"
            )
        if mode == "report":
            return None
        seed = sampling.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("PLUMED CV sampling seed must be a non-negative integer")
        min_frame_gap = sampling.get("min_frame_gap", 0)
        if (
            not isinstance(min_frame_gap, int)
            or isinstance(min_frame_gap, bool)
            or min_frame_gap < 0
        ):
            raise ValueError("PLUMED CV min_frame_gap must be non-negative")
        normalized = {
            "mode": mode,
            "seed": seed,
            "min_frame_gap": min_frame_gap,
            "within_bin": "random",
        }
        if mode == "random":
            return normalized

        within_bin = sampling.get("within_bin", "random")
        if within_bin not in {"random", "max_deviation"}:
            raise ValueError("PLUMED CV within_bin must be random or max_deviation")
        if mode == "uniform":
            field = sampling.get("field")
            n_bins = sampling.get("n_bins", 10)
            if not isinstance(field, str) or not field:
                raise ValueError("uniform PLUMED CV sampling requires a field")
            if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins <= 0:
                raise ValueError("PLUMED CV sampling n_bins must be positive")
            grid = {field: n_bins}
            normalized.update({"field": field, "n_bins": n_bins})
        else:
            grid = sampling.get("grid")
            if not isinstance(grid, dict) or len(grid) != 2:
                raise ValueError("grid PLUMED CV sampling requires exactly two fields")
            if any(
                not isinstance(field, str)
                or not field
                or not isinstance(n_bins, int)
                or isinstance(n_bins, bool)
                or n_bins <= 0
                for field, n_bins in grid.items()
            ):
                raise ValueError("PLUMED CV grid fields and bin counts are invalid")
            grid = dict(grid)
        if any(field not in region for region in self.regions for field in grid):
            raise ValueError("every grid field must bound every PLUMED CV region")
        normalized.update({"grid": grid, "within_bin": within_bin})
        return normalized

    @staticmethod
    def _normalize_time_alignment(time_alignment: Optional[Dict]):
        if time_alignment is None:
            return None
        if not isinstance(time_alignment, dict):
            raise ValueError("PLUMED CV time_alignment must be a dict")
        unknown = set(time_alignment) - {"start", "step", "atol"}
        if unknown:
            raise ValueError(f"unknown PLUMED time_alignment keys: {sorted(unknown)}")
        try:
            start = float(time_alignment.get("start", 0.0))
            step = float(time_alignment["step"])
            atol = float(time_alignment.get("atol", 1e-8))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid PLUMED CV time_alignment") from exc
        if not np.all(np.isfinite([start, step, atol])) or step <= 0 or atol < 0:
            raise ValueError("invalid PLUMED CV time_alignment")
        return {"start": start, "step": step, "atol": atol}

    def get_selected_ids(
        self,
        files: List[Path],
        nframes: List[int],
    ) -> List[List[int]]:
        outputs = self._load_outputs(files, nframes)
        selected = []
        for fields, values in outputs:
            masks = self._region_masks(fields, values)
            selected.append(np.flatnonzero(np.logical_or.reduce(masks)).tolist())
        return selected

    def select_candidate_ids(
        self,
        files: List[Path],
        nframes: List[int],
        candidate_ids: List[List[int]],
        max_nframes: Optional[int],
        max_devi_f: Optional[List[np.ndarray]] = None,
    ) -> List[List[int]]:
        """Filter model-deviation candidates and apply configured sampling."""
        selected, _, _ = self.select_candidate_ids_with_audit(
            files, nframes, candidate_ids, max_nframes, max_devi_f
        )
        return selected

    def select_candidate_ids_with_audit(
        self,
        files: List[Path],
        nframes: List[int],
        candidate_ids: List[List[int]],
        max_nframes: Optional[int],
        max_devi_f: Optional[List[np.ndarray]] = None,
    ):
        """Return sampled IDs plus selected-frame records and aggregate counts."""
        if self.sampling is None:
            raise ValueError("PLUMED CV sampling is not configured")
        if len(candidate_ids) != len(files):
            raise FatalError("candidate IDs and trajectories have different lengths")
        if max_nframes is not None and max_nframes < 0:
            raise ValueError("max_nframes must be non-negative")

        outputs = self._load_outputs(files, nframes)
        masks_by_traj = [
            self._region_masks(fields, values) for fields, values in outputs
        ]
        candidates = self._eligible_candidates(candidate_ids, nframes, masks_by_traj)
        limit = (
            len(candidates)
            if max_nframes is None
            else min(max_nframes, len(candidates))
        )
        rng = np.random.default_rng(self.sampling["seed"])
        rejected_by_gap = set()
        picked = []
        if self.sampling["mode"] == "random":
            self._pick_candidates(
                candidates,
                limit,
                "random",
                max_devi_f,
                rng,
                picked,
                rejected_by_gap,
            )
        else:
            picked, rejected_by_gap = self._sample_grid(
                candidates,
                outputs,
                masks_by_traj,
                limit,
                max_devi_f,
                rng,
            )
        records, summary = self._make_audit(
            candidate_ids,
            candidates,
            picked,
            outputs,
            masks_by_traj,
            max_devi_f,
            limit,
            rejected_by_gap,
            self.sampling["mode"],
        )
        return self._group_candidates(picked, len(files)), records, summary

    def audit_candidate_ids(
        self,
        files: List[Path],
        nframes: List[int],
        candidate_ids: List[List[int]],
        selected_ids: List[List[int]],
        max_devi_f: Optional[List[np.ndarray]] = None,
    ):
        """Audit filtering followed by the report's existing sampling policy."""
        outputs = self._load_outputs(files, nframes)
        masks_by_traj = [
            self._region_masks(fields, values) for fields, values in outputs
        ]
        candidates = self._eligible_candidates(candidate_ids, nframes, masks_by_traj)
        eligible_set = set(candidates)
        picked = []
        for traj_idx, frame_ids in enumerate(selected_ids):
            for frame_idx in sorted(set(frame_ids)):
                candidate = (traj_idx, int(frame_idx))
                if candidate not in eligible_set:
                    raise FatalError("selected frame is outside PLUMED CV candidates")
                picked.append(candidate)
        records, summary = self._make_audit(
            candidate_ids,
            candidates,
            picked,
            outputs,
            masks_by_traj,
            max_devi_f,
            len(picked),
            set(),
            "report",
        )
        return records, summary

    def _eligible_candidates(self, candidate_ids, nframes, masks_by_traj):
        candidates = []
        for traj_idx, frame_ids in enumerate(candidate_ids):
            for frame_idx in sorted(set(frame_ids)):
                if (
                    not isinstance(frame_idx, (int, np.integer))
                    or frame_idx < 0
                    or frame_idx >= nframes[traj_idx]
                ):
                    raise FatalError("candidate frame index is outside the trajectory")
                if any(mask[frame_idx] for mask in masks_by_traj[traj_idx]):
                    candidates.append((traj_idx, int(frame_idx)))
        return candidates

    def _sample_grid(
        self,
        candidates: List[Tuple[int, int]],
        outputs,
        masks_by_traj,
        limit: int,
        max_devi_f: Optional[List[np.ndarray]],
        rng,
    ):
        sampling = self.sampling
        assert sampling is not None
        within_bin = sampling["within_bin"]
        if within_bin == "max_deviation" and max_devi_f is None:
            raise FatalError("max_deviation sampling requires force model deviations")

        buckets = [dict() for _ in self.regions]
        for candidate in candidates:
            traj_idx, frame_idx = candidate
            fields, values = outputs[traj_idx]
            for region_idx, mask in enumerate(masks_by_traj[traj_idx]):
                if not mask[frame_idx]:
                    continue
                cell = self._cell_key(region_idx, fields, values[frame_idx])
                buckets[region_idx].setdefault(cell, []).append(candidate)

        capacities = [
            len({item for items in region.values() for item in items})
            for region in buckets
        ]
        region_quotas = self._weighted_quotas(capacities, self.region_weights, limit)
        picked = []
        picked_set = set()
        rejected_by_gap = set()
        grid_sizes = tuple(sampling["grid"].values())
        for region, quota in zip(buckets, region_quotas):
            available = {
                cell: [item for item in items if item not in picked_set]
                for cell, items in region.items()
            }
            available = {cell: items for cell, items in available.items() if items}
            cell_quotas = self._cell_quotas(available, quota, grid_sizes)
            for cell, count in cell_quotas.items():
                before = len(picked)
                self._pick_candidates(
                    available[cell],
                    count,
                    within_bin,
                    max_devi_f,
                    rng,
                    picked,
                    rejected_by_gap,
                )
                picked_set.update(picked[before:])

        if len(picked) < limit:
            remaining = [item for item in candidates if item not in picked_set]
            self._pick_candidates(
                remaining,
                limit - len(picked),
                within_bin,
                max_devi_f,
                rng,
                picked,
                rejected_by_gap,
            )
        return picked[:limit], rejected_by_gap

    def _pick_candidates(
        self,
        candidates,
        count,
        mode,
        max_devi_f,
        rng,
        picked,
        rejected_by_gap,
    ):
        sampling = self.sampling
        assert sampling is not None
        if count <= 0:
            return
        initially_picked = len(picked)
        picked_set = set(picked)
        candidates = list(
            dict.fromkeys(item for item in candidates if item not in picked_set)
        )
        if mode == "random":
            order = rng.permutation(len(candidates))
            ordered = [candidates[ii] for ii in order]
        else:
            if max_devi_f is None:
                raise FatalError(
                    "max_deviation sampling requires force model deviations"
                )
            ordered = sorted(
                candidates,
                key=lambda item: (
                    -self._deviation_value(max_devi_f, item),
                    item[0],
                    item[1],
                ),
            )
        min_frame_gap = sampling["min_frame_gap"]
        for candidate in ordered:
            if len(picked) - initially_picked >= count:
                break
            if min_frame_gap and any(
                candidate[0] == other[0]
                and abs(candidate[1] - other[1]) < min_frame_gap
                for other in picked
            ):
                rejected_by_gap.add(candidate)
                continue
            picked.append(candidate)

    @staticmethod
    def _deviation_value(max_devi_f, candidate):
        traj_idx, frame_idx = candidate
        try:
            value = float(max_devi_f[traj_idx][frame_idx])
        except (IndexError, TypeError, ValueError) as exc:
            raise FatalError(
                "force model deviations do not match trajectories"
            ) from exc
        if not np.isfinite(value):
            raise FatalError("force model deviations must be finite")
        return value

    @classmethod
    def _weighted_quotas(cls, capacities, weights, total):
        if not capacities or len(set(weights)) == 1:
            return cls._balanced_quotas(capacities, total)
        quotas = [0] * len(capacities)
        for _ in range(total):
            available = [
                idx for idx, capacity in enumerate(capacities) if quotas[idx] < capacity
            ]
            if not available:
                break
            idx = max(
                available,
                key=lambda ii: (weights[ii] / (quotas[ii] + 1), -ii),
            )
            quotas[idx] += 1
        return quotas

    @staticmethod
    def _balanced_quotas(capacities: List[int], total: int) -> List[int]:
        quotas = [0] * len(capacities)
        active = [idx for idx, capacity in enumerate(capacities) if capacity]
        if total <= 0 or not active:
            return quotas
        if total <= len(active):
            if total == 1:
                selected = [active[len(active) // 2]]
            else:
                selected = [
                    active[round(idx * (len(active) - 1) / (total - 1))]
                    for idx in range(total)
                ]
            for idx in selected:
                quotas[idx] = 1
            return quotas
        for idx in active:
            quotas[idx] = 1
        total -= len(active)
        while total:
            progressed = False
            for idx, capacity in enumerate(capacities):
                if quotas[idx] < capacity:
                    quotas[idx] += 1
                    total -= 1
                    progressed = True
                    if not total:
                        break
            if not progressed:
                break
        return quotas

    @classmethod
    def _cell_quotas(cls, buckets, total, grid_sizes):
        cells = sorted(buckets)
        if not cells or total <= 0:
            return {}
        if total <= len(cells):
            return {cell: 1 for cell in cls._spread_cells(cells, total, grid_sizes)}
        extra = cls._balanced_quotas(
            [len(buckets[cell]) - 1 for cell in cells], total - len(cells)
        )
        return {cell: 1 + increment for cell, increment in zip(cells, extra)}

    @staticmethod
    def _spread_cells(cells, total, grid_sizes):
        if total == 1:
            center = tuple((size - 1) / 2 for size in grid_sizes)
            return [
                min(
                    cells,
                    key=lambda cell: sum(
                        ((value - middle) / max(size - 1, 1)) ** 2
                        for value, middle, size in zip(cell, center, grid_sizes)
                    ),
                )
            ]
        chosen = [cells[0]]
        while len(chosen) < total:
            best = None
            best_distance = -1.0
            for cell in cells:
                if cell in chosen:
                    continue
                distance = min(
                    sum(
                        ((left - right) / max(size - 1, 1)) ** 2
                        for left, right, size in zip(cell, other, grid_sizes)
                    )
                    for other in chosen
                )
                if distance > best_distance:
                    best = cell
                    best_distance = distance
            chosen.append(best)
        return chosen

    def _cell_key(self, region_idx, fields, row):
        sampling = self.sampling
        assert sampling is not None
        field_idx = {field: idx for idx, field in enumerate(fields)}
        region = self.regions[region_idx]
        cell = []
        for field, n_bins in sampling["grid"].items():
            lower, upper = region[field]
            value = row[field_idx[field]]
            cell.append(
                min(int((value - lower) / (upper - lower) * n_bins), n_bins - 1)
            )
        return tuple(cell)

    @staticmethod
    def _group_candidates(candidates, ntraj):
        grouped = [[] for _ in range(ntraj)]
        for traj_idx, frame_idx in sorted(candidates):
            grouped[traj_idx].append(frame_idx)
        return grouped

    def _make_audit(
        self,
        candidate_ids,
        candidates,
        picked,
        outputs,
        masks_by_traj,
        max_devi_f,
        requested,
        rejected_by_gap,
        mode,
    ):
        grid = None if self.sampling is None else self.sampling.get("grid")
        within_bin = (
            "" if self.sampling is None else self.sampling.get("within_bin", "")
        )
        seed = "" if self.sampling is None else self.sampling["seed"]
        records = []
        for traj_idx, frame_idx in sorted(picked):
            fields, values = outputs[traj_idx]
            row = values[frame_idx]
            field_idx = {field: idx for idx, field in enumerate(fields)}
            matched = [
                idx
                for idx, mask in enumerate(masks_by_traj[traj_idx])
                if mask[frame_idx]
            ]
            cells = []
            if grid is not None:
                for region_idx in matched:
                    cell = self._cell_key(region_idx, fields, row)
                    labels = ",".join(
                        f"{field}={value}" for field, value in zip(grid, cell)
                    )
                    cells.append(f"{self.region_names[region_idx]}:{labels}")
            record = {
                "traj_idx": traj_idx,
                "frame_idx": frame_idx,
                "time": float(row[field_idx["time"]]),
                "max_devi_f": (
                    ""
                    if max_devi_f is None
                    else self._deviation_value(max_devi_f, (traj_idx, frame_idx))
                ),
                "region_names": ";".join(self.region_names[idx] for idx in matched),
                "cell_or_bin": ";".join(cells),
                "sampling_mode": mode,
                "within_bin": within_bin,
                "seed": seed,
            }
            for field in self.cv_fields:
                record[f"cv_{field}"] = float(row[field_idx[field]])
            records.append(record)

        eligible_set = set(candidates)
        picked_set = set(picked)
        per_region = {}
        for region_idx, name in enumerate(self.region_names):
            eligible_region = {
                candidate
                for candidate in eligible_set
                if masks_by_traj[candidate[0]][region_idx][candidate[1]]
            }
            selected_region = eligible_region & picked_set
            eligible_cells = set()
            selected_cells = set()
            if grid is not None:
                for candidate in eligible_region:
                    fields, values = outputs[candidate[0]]
                    cell = self._cell_key(region_idx, fields, values[candidate[1]])
                    eligible_cells.add(cell)
                    if candidate in selected_region:
                        selected_cells.add(cell)
            per_region[name] = {
                "eligible": len(eligible_region),
                "selected": len(selected_region),
                "nonempty_cells": len(eligible_cells),
                "selected_cells": len(selected_cells),
            }

        trust_candidates = sum(len(set(frame_ids)) for frame_ids in candidate_ids)
        summary = {
            "trust_candidates": trust_candidates,
            "cv_eligible_candidates": len(eligible_set),
            "requested": requested,
            "selected": len(picked_set),
            "underfilled_quota": max(requested - len(picked_set), 0),
            "min_frame_gap_rejects": len(rejected_by_gap),
            "sampling_mode": mode,
            "sampling_inferred": self.sampling_inferred,
            "within_bin": within_bin,
            "seed": seed,
            "min_frame_gap": (
                0 if self.sampling is None else self.sampling["min_frame_gap"]
            ),
            "interval_semantics": "lower-inclusive, upper-exclusive",
            "cv_fields": self.cv_fields,
            "time_alignment": self.time_alignment,
            "per_region": per_region,
            "regions": [
                {
                    "name": name,
                    "weight": weight,
                    "conditions": {
                        field: list(bounds) for field, bounds in region.items()
                    },
                }
                for name, weight, region in zip(
                    self.region_names, self.region_weights, self.regions
                )
            ],
        }
        return records, summary

    @staticmethod
    def write_audit(out_path: Path, records, summary):
        out_path.mkdir(exist_ok=True)
        base_fields = [
            "traj_idx",
            "frame_idx",
            "time",
            "max_devi_f",
            "region_names",
            "cell_or_bin",
            "sampling_mode",
            "within_bin",
            "seed",
        ]
        fieldnames = base_fields + [f"cv_{field}" for field in summary["cv_fields"]]
        with (out_path / "cv_selection.csv").open(
            "w", newline="", encoding="utf8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        with (out_path / "cv_selection_summary.json").open(
            "w", encoding="utf8"
        ) as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _load_outputs(self, files: List[Path], nframes: List[int]):
        if len(files) != len(nframes):
            raise FatalError("PLUMED outputs and trajectories have different lengths")
        outputs = []
        for file, expected_nframes in zip(files, nframes):
            fields, values = self._read(file)
            if len(values) != expected_nframes:
                raise FatalError(
                    f"PLUMED output {file} has {len(values)} rows, expected "
                    f"{expected_nframes}; PRINT STRIDE must match the trajectory stride"
                )
            if self.time_alignment is not None:
                time = values[:, fields.index("time")]
                expected_time = (
                    self.time_alignment["start"]
                    + np.arange(expected_nframes) * self.time_alignment["step"]
                )
                if not np.allclose(
                    time,
                    expected_time,
                    rtol=0.0,
                    atol=self.time_alignment["atol"],
                ):
                    raise FatalError(
                        f"PLUMED time in {file} does not match configured frame times"
                    )
            outputs.append((fields, values))
        return outputs

    def _region_masks(self, fields, values):
        field_idx = {field: idx for idx, field in enumerate(fields)}
        masks = []
        for region in self.regions:
            in_region = np.ones(len(values), dtype=bool)
            for field, (lower, upper) in region.items():
                if field not in field_idx:
                    raise FatalError(f"PLUMED field {field!r} is missing")
                column = values[:, field_idx[field]]
                in_region &= (column >= lower) & (column < upper)
            masks.append(in_region)
        return masks

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
