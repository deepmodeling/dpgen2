from pathlib import (
    Path,
)
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
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
            ),
            Argument(
                "sampling",
                dict,
                [
                    Argument("mode", str, optional=False),
                    Argument("field", str, optional=True, default=None),
                    Argument("n_bins", int, optional=True, default=10),
                    Argument("within_bin", str, optional=True, default="random"),
                    Argument("seed", int, optional=True, default=0),
                ],
                optional=True,
                default=None,
                doc=(
                    "Optional final candidate sampling. mode is random or uniform; "
                    "uniform uses field and n_bins, then random or max_deviation "
                    "within each bin."
                ),
            ),
        ]

    def __init__(
        self,
        regions: List[Dict[str, Sequence[float]]],
        sampling: Optional[Dict] = None,
    ):
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
        self.sampling = self._normalize_sampling(sampling)

    def _normalize_sampling(self, sampling: Optional[Dict]):
        if sampling is None:
            return None
        if not isinstance(sampling, dict):
            raise ValueError("PLUMED CV sampling must be a dict")
        mode = sampling.get("mode")
        if mode not in {"random", "uniform"}:
            raise ValueError("PLUMED CV sampling mode must be random or uniform")
        seed = sampling.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("PLUMED CV sampling seed must be a non-negative integer")
        normalized = {"mode": mode, "seed": seed}
        if mode == "uniform":
            field = sampling.get("field")
            n_bins = sampling.get("n_bins", 10)
            within_bin = sampling.get("within_bin", "random")
            if not isinstance(field, str) or not field:
                raise ValueError("uniform PLUMED CV sampling requires a field")
            if any(field not in region for region in self.regions):
                raise ValueError(
                    f"uniform sampling field {field!r} must bound every region"
                )
            if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins <= 0:
                raise ValueError("PLUMED CV sampling n_bins must be positive")
            if within_bin not in {"random", "max_deviation"}:
                raise ValueError("PLUMED CV within_bin must be random or max_deviation")
            normalized.update(
                {"field": field, "n_bins": n_bins, "within_bin": within_bin}
            )
        return normalized

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

        limit = (
            len(candidates)
            if max_nframes is None
            else min(max_nframes, len(candidates))
        )
        if limit == len(candidates):
            return self._group_candidates(candidates, len(files))
        rng = np.random.default_rng(self.sampling["seed"])
        if self.sampling["mode"] == "random":
            picked = [
                candidates[ii] for ii in rng.choice(len(candidates), limit, False)
            ]
        else:
            picked = self._sample_uniform(
                candidates,
                outputs,
                masks_by_traj,
                limit,
                max_devi_f,
                rng,
            )
        return self._group_candidates(picked, len(files))

    def _sample_uniform(
        self,
        candidates: List[Tuple[int, int]],
        outputs,
        masks_by_traj,
        limit: int,
        max_devi_f: Optional[List[np.ndarray]],
        rng,
    ) -> List[Tuple[int, int]]:
        field = self.sampling["field"]
        n_bins = self.sampling["n_bins"]
        within_bin = self.sampling["within_bin"]
        if within_bin == "max_deviation" and max_devi_f is None:
            raise FatalError("max_deviation sampling requires force model deviations")

        buckets = [dict() for _ in self.regions]
        for candidate in candidates:
            traj_idx, frame_idx = candidate
            fields, values = outputs[traj_idx]
            value = values[frame_idx, fields.index(field)]
            for region_idx, (region, mask) in enumerate(
                zip(self.regions, masks_by_traj[traj_idx])
            ):
                if not mask[frame_idx]:
                    continue
                lower, upper = region[field]
                bin_idx = min(
                    int((value - lower) / (upper - lower) * n_bins), n_bins - 1
                )
                buckets[region_idx].setdefault(bin_idx, []).append(candidate)

        capacities = [
            len({item for items in region.values() for item in items})
            for region in buckets
        ]
        region_quotas = self._balanced_quotas(capacities, limit)
        picked = []
        picked_set = set()
        for region, quota in zip(buckets, region_quotas):
            available = {
                bin_idx: [item for item in items if item not in picked_set]
                for bin_idx, items in region.items()
            }
            available = {key: value for key, value in available.items() if value}
            bin_quotas = self._bin_quotas(available, quota)
            for bin_idx, count in bin_quotas.items():
                chosen = self._pick_within_bin(
                    available[bin_idx], count, within_bin, max_devi_f, rng
                )
                picked.extend(chosen)
                picked_set.update(chosen)

        if len(picked) < limit:
            remaining = [item for item in candidates if item not in picked_set]
            picked.extend(
                self._pick_within_bin(
                    remaining, limit - len(picked), within_bin, max_devi_f, rng
                )
            )
        return picked[:limit]

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
    def _bin_quotas(cls, buckets, total: int):
        bins = sorted(buckets)
        if not bins or total <= 0:
            return {}
        if total <= len(bins):
            if total == 1:
                selected_bins = [bins[len(bins) // 2]]
            else:
                selected_bins = [
                    bins[round(idx * (len(bins) - 1) / (total - 1))]
                    for idx in range(total)
                ]
            return {bin_idx: 1 for bin_idx in selected_bins}
        quotas = [1] * len(bins)
        extra = cls._balanced_quotas(
            [len(buckets[bin_idx]) - 1 for bin_idx in bins], total - len(bins)
        )
        return {
            bin_idx: quota + increment
            for bin_idx, quota, increment in zip(bins, quotas, extra)
        }

    @staticmethod
    def _pick_within_bin(candidates, count, mode, max_devi_f, rng):
        if count <= 0:
            return []
        if mode == "random":
            order = rng.permutation(len(candidates))[:count]
            return [candidates[ii] for ii in order]
        return sorted(
            candidates,
            key=lambda item: (
                -max_devi_f[item[0]][item[1]],
                item[0],
                item[1],
            ),
        )[:count]

    @staticmethod
    def _group_candidates(candidates, ntraj):
        grouped = [[] for _ in range(ntraj)]
        for traj_idx, frame_idx in sorted(candidates):
            grouped[traj_idx].append(frame_idx)
        return grouped

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
