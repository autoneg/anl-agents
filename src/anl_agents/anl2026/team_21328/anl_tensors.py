"""Utilities for turning ANL scenarios into fixed-shape tensors.

The Dreamer-side code expects static shapes. ANL domains are variable, so this
module pads issue/value tables to a fixed maximum and carries masks alongside
the padded values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


DEFAULT_MAX_ISSUES = 10
DEFAULT_MAX_VALUES = 10


@dataclass(frozen=True)
class UtilityTable:
    """Padded linear-additive utility table."""

    name: str
    weights: np.ndarray
    values: np.ndarray
    reserved_value: np.float32
    issue_mask: np.ndarray
    value_mask: np.ndarray
    discount: np.float32 | None = None

    def utility(self, offer_values: Sequence[int]) -> np.float32:
        return masked_utility(offer_values, self.weights, self.values, self.issue_mask)


@dataclass(frozen=True)
class PaddedScenario:
    """A scenario with padded self/opponent utility tables."""

    name: str
    domain_path: Path
    self_profile_path: Path
    opponent_profile_path: Path
    issue_names: tuple[str, ...]
    value_names: tuple[tuple[str, ...], ...]
    num_issues: np.int32
    num_values: np.ndarray
    issue_mask: np.ndarray
    value_mask: np.ndarray
    self_utility: UtilityTable
    opponent_utility: UtilityTable
    max_issues: int
    max_values: int

    def offer_to_indices(self, offer: Any) -> np.ndarray:
        return offer_to_indices(offer, self.issue_names, self.value_names, self.max_issues)

    def indices_to_offer(self, indices: Sequence[int]) -> tuple[Any, ...]:
        return indices_to_offer(indices, self.value_names, self.issue_mask)

    def self_utility_value(self, offer_or_indices: Any) -> np.float32:
        indices = (
            offer_or_indices
            if _looks_like_indices(offer_or_indices)
            else self.offer_to_indices(offer_or_indices)
        )
        return self.self_utility.utility(indices)

    def opponent_utility_value(self, offer_or_indices: Any) -> np.float32:
        indices = (
            offer_or_indices
            if _looks_like_indices(offer_or_indices)
            else self.offer_to_indices(offer_or_indices)
        )
        return self.opponent_utility.utility(indices)


def load_padded_scenario(
    scenario_dir: str | Path,
    *,
    self_profile: str | Path | None = None,
    opponent_profile: str | Path | None = None,
    self_index: int = 0,
    opponent_index: int = 1,
    max_issues: int = DEFAULT_MAX_ISSUES,
    max_values: int = DEFAULT_MAX_VALUES,
) -> PaddedScenario:
    """Load an ANL scenario folder into padded tensors.

    Args:
        scenario_dir: Directory containing one domain yml and two profile ymls.
        self_profile: Optional profile filename/path for the learner side.
        opponent_profile: Optional profile filename/path for the opponent side.
        self_index: Profile index to use if ``self_profile`` is not provided.
        opponent_index: Profile index to use if ``opponent_profile`` is not provided.
        max_issues: Padded number of issues.
        max_values: Padded number of values per issue.
    """

    scenario_dir = Path(scenario_dir)
    domain_path, profile_paths = _find_domain_and_profiles(scenario_dir)
    if self_profile is not None:
        self_path = _resolve_profile_path(scenario_dir, self_profile)
    else:
        self_path = profile_paths[self_index]
    if opponent_profile is not None:
        opponent_path = _resolve_profile_path(scenario_dir, opponent_profile)
    else:
        opponent_path = profile_paths[opponent_index]
    if self_path == opponent_path:
        raise ValueError(f"self and opponent profiles must differ: {self_path}")

    domain = _load_yaml(domain_path)
    issues = domain.get("issues") or []
    issue_names = tuple(str(issue["name"]) for issue in issues)
    value_names = tuple(
        tuple(value for value in issue.get("values", ())) for issue in issues
    )
    _validate_domain_size(issue_names, value_names, max_issues, max_values)

    issue_mask, value_mask, num_values = _build_masks(
        value_names, max_issues, max_values
    )
    self_table = _load_utility_table(
        self_path, issue_names, value_names, issue_mask, value_mask, max_issues, max_values
    )
    opponent_table = _load_utility_table(
        opponent_path,
        issue_names,
        value_names,
        issue_mask,
        value_mask,
        max_issues,
        max_values,
    )
    return PaddedScenario(
        name=scenario_dir.name,
        domain_path=domain_path,
        self_profile_path=self_path,
        opponent_profile_path=opponent_path,
        issue_names=issue_names,
        value_names=value_names,
        num_issues=np.int32(len(issue_names)),
        num_values=num_values,
        issue_mask=issue_mask,
        value_mask=value_mask,
        self_utility=self_table,
        opponent_utility=opponent_table,
        max_issues=max_issues,
        max_values=max_values,
    )


def load_all_padded_scenarios(
    scenarios_root: str | Path,
    *,
    max_issues: int = DEFAULT_MAX_ISSUES,
    max_values: int = DEFAULT_MAX_VALUES,
) -> list[PaddedScenario]:
    """Load every scenario folder under ``scenarios_root``."""

    root = Path(scenarios_root)
    scenarios = []
    for path in sorted(root.iterdir()):
        if path.is_dir():
            scenarios.append(
                load_padded_scenario(
                    path, max_issues=max_issues, max_values=max_values
                )
            )
    return scenarios


def offer_to_indices(
    offer: Any,
    issue_names: Sequence[str],
    value_names: Sequence[Sequence[Any]],
    max_issues: int = DEFAULT_MAX_ISSUES,
) -> np.ndarray:
    """Convert an ANL outcome tuple/dict into padded value indices."""

    if isinstance(offer, dict):
        raw_values = [offer[name] for name in issue_names]
    else:
        raw_values = list(offer)
    if len(raw_values) != len(issue_names):
        raise ValueError(
            f"offer has {len(raw_values)} values but domain has {len(issue_names)} issues"
        )
    indices = np.zeros((max_issues,), dtype=np.int32)
    for i, value in enumerate(raw_values):
        try:
            indices[i] = list(value_names[i]).index(value)
        except ValueError as exc:
            raise ValueError(
                f"unknown value {value!r} for issue {issue_names[i]!r}"
            ) from exc
    return indices


def indices_to_offer(
    indices: Sequence[int],
    value_names: Sequence[Sequence[Any]],
    issue_mask: Sequence[bool] | np.ndarray,
) -> tuple[Any, ...]:
    """Convert padded value indices back into an ANL outcome tuple."""

    indices = list(indices)
    offer = []
    for i, active in enumerate(np.asarray(issue_mask, dtype=bool)):
        if not active:
            continue
        value_idx = int(indices[i])
        values = value_names[i]
        if value_idx < 0 or value_idx >= len(values):
            raise ValueError(
                f"value index {value_idx} is invalid for issue {i} with {len(values)} values"
            )
        offer.append(values[value_idx])
    return tuple(offer)


def masked_utility(
    offer_values: Sequence[int],
    weights: np.ndarray,
    values: np.ndarray,
    issue_mask: Sequence[bool] | np.ndarray,
) -> np.float32:
    """Compute a masked linear-additive utility for padded offer indices."""

    offer_values = np.asarray(offer_values, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    issue_mask = np.asarray(issue_mask, dtype=bool)
    if offer_values.shape[0] < weights.shape[0]:
        raise ValueError(
            f"offer_values length {offer_values.shape[0]} is smaller than weights length {weights.shape[0]}"
        )
    chosen = values[np.arange(weights.shape[0]), offer_values[: weights.shape[0]]]
    return np.float32(np.sum(weights * chosen * issue_mask.astype(np.float32)))


def _find_domain_and_profiles(scenario_dir: Path) -> tuple[Path, list[Path]]:
    files = sorted(
        path
        for path in scenario_dir.glob("*.yml")
        if not path.name.startswith("_")
    )
    loaded = [(path, _load_yaml(path)) for path in files]
    domains = [
        path
        for path, data in loaded
        if data.get("type") == "DiscreteCartesianOutcomeSpace" and data.get("issues")
    ]
    if len(domains) != 1:
        raise ValueError(
            f"expected exactly one domain yml in {scenario_dir}, found {len(domains)}"
        )
    profiles = [path for path, _ in loaded if path != domains[0]]
    if len(profiles) < 2:
        raise ValueError(f"expected at least two profile yml files in {scenario_dir}")
    return domains[0], profiles


def _resolve_profile_path(scenario_dir: Path, profile: str | Path) -> Path:
    path = Path(profile)
    if not path.is_absolute():
        path = scenario_dir / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_utility_table(
    profile_path: Path,
    issue_names: Sequence[str],
    value_names: Sequence[Sequence[Any]],
    issue_mask: np.ndarray,
    value_mask: np.ndarray,
    max_issues: int,
    max_values: int,
) -> UtilityTable:
    data = _load_yaml(profile_path)
    ufun = data.get("ufun") or data
    if ufun.get("type") != "LinearAdditiveUtilityFunction":
        raise ValueError(
            f"{profile_path} contains unsupported utility type {ufun.get('type')!r}"
        )
    raw_weights = ufun.get("weights") or []
    raw_values = ufun.get("values") or []
    if len(raw_weights) != len(issue_names) or len(raw_values) != len(issue_names):
        raise ValueError(
            f"{profile_path} has weights={len(raw_weights)} values={len(raw_values)} "
            f"for issues={len(issue_names)}"
        )

    weights = np.zeros((max_issues,), dtype=np.float32)
    values = np.zeros((max_issues, max_values), dtype=np.float32)
    weights[: len(raw_weights)] = np.asarray(raw_weights, dtype=np.float32)
    for i, table in enumerate(raw_values):
        mapping = table.get("mapping") or {}
        for j, value in enumerate(value_names[i]):
            if value not in mapping:
                raise ValueError(
                    f"{profile_path} is missing value {value!r} for issue {issue_names[i]!r}"
                )
            values[i, j] = np.float32(mapping[value])

    return UtilityTable(
        name=str(data.get("name") or ufun.get("name") or profile_path.stem),
        weights=weights,
        values=values,
        reserved_value=np.float32(
            ufun.get("reserved_value", data.get("reserved_value", 0.0))
        ),
        issue_mask=issue_mask.copy(),
        value_mask=value_mask.copy(),
        discount=(
            np.float32(data["discount"]) if data.get("discount") is not None else None
        ),
    )


def _build_masks(
    value_names: Sequence[Sequence[Any]],
    max_issues: int,
    max_values: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    issue_mask = np.zeros((max_issues,), dtype=bool)
    value_mask = np.zeros((max_issues, max_values), dtype=bool)
    num_values = np.zeros((max_issues,), dtype=np.int32)
    for i, values in enumerate(value_names):
        issue_mask[i] = True
        num_values[i] = len(values)
        value_mask[i, : len(values)] = True
    return issue_mask, value_mask, num_values


def _validate_domain_size(
    issue_names: Sequence[str],
    value_names: Sequence[Sequence[Any]],
    max_issues: int,
    max_values: int,
) -> None:
    if len(issue_names) > max_issues:
        raise ValueError(
            f"domain has {len(issue_names)} issues, exceeding max_issues={max_issues}"
        )
    too_large = [
        (name, len(values))
        for name, values in zip(issue_names, value_names)
        if len(values) > max_values
    ]
    if too_large:
        raise ValueError(
            f"domain has issues exceeding max_values={max_values}: {too_large}"
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return data


def _looks_like_indices(value: Any) -> bool:
    return isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer)
