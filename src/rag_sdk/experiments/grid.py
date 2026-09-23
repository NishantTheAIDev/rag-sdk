"""Expanding experiment parameter sweeps into concrete configurations."""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from rag_sdk.config import (
    BM25RetrievalConfig,
    CohereRerankerConfig,
    CrossEncoderRerankerConfig,
    DenseRetrievalConfig,
    FixedTokenChunkerConfig,
    HybridRetrievalConfig,
    MMRRetrievalConfig,
    NoRerankerConfig,
    ParentChildChunkerConfig,
    RagConfig,
    RecursiveChunkerConfig,
    SemanticChunkerConfig,
    SentenceWindowChunkerConfig,
    StructureAwareChunkerConfig,
)
from rag_sdk.config.loader import ConfigError
from rag_sdk.config.models import (
    ChunkerConfigBase,
    RerankerConfigBase,
    RetrievalConfigBase,
)

_VARIANT_DEFAULTS = {
    "dense": DenseRetrievalConfig,
    "bm25": BM25RetrievalConfig,
    "hybrid": HybridRetrievalConfig,
    "mmr": MMRRetrievalConfig,
    "recursive": RecursiveChunkerConfig,
    "fixed": FixedTokenChunkerConfig,
    "sentence_window": SentenceWindowChunkerConfig,
    "parent_child": ParentChildChunkerConfig,
    "semantic": SemanticChunkerConfig,
    "structure_aware": StructureAwareChunkerConfig,
    "none": NoRerankerConfig,
    "cross_encoder": CrossEncoderRerankerConfig,
    "cohere": CohereRerankerConfig,
}

# Fields shared by every variant of a section. They survive a strategy switch
# so that, e.g., sweeping ``retrieval.strategy`` keeps the configured
# ``candidate_k`` instead of resetting it to the new strategy's default.
_SHARED_FIELDS: dict[str, type[BaseModel]] = {
    "chunking": ChunkerConfigBase,
    "retrieval": RetrievalConfigBase,
    "reranker": RerankerConfigBase,
}


class ExperimentParameterWarning(UserWarning):
    """Emitted when a sweep parameter does not apply to a strategy."""


@dataclass(frozen=True)
class GridVariant:
    """A concrete configuration plus the sweep parameters skipped for it."""

    config: RagConfig
    skipped_parameters: list[str] = field(default_factory=list)


def apply_override(
    config: RagConfig,
    dot_path: str,
    value: object,
    *,
    skip_missing: bool = False,
    skipped: list[str] | None = None,
) -> RagConfig:
    """Return ``config`` with the value at ``dot_path`` replaced by ``value``.

    When the path targets a strategy discriminator (``chunking.strategy``,
    ``retrieval.strategy`` or ``reranker.strategy``), the selected strategy's
    default configuration is applied so that later overrides can tune it.
    Fields shared by every variant of that section (such as ``top_k``,
    ``candidate_k``, ``chunk_size`` and ``overlap``) keep their current values.
    Selecting a reranker strategy when no reranker is configured creates one.

    With ``skip_missing=True``, overrides whose path does not exist in the
    current configuration are ignored instead of raising (used during sweeps
    where a parameter only applies to some strategies); the skipped paths are
    appended to ``skipped`` when provided.
    """
    data = config.model_dump(mode="json")
    if not _apply_to_data(data, dot_path, value, skip_missing=skip_missing, skipped=skipped):
        return config
    return RagConfig.model_validate(data)


def _apply_to_data(
    data: dict[str, object],
    dot_path: str,
    value: object,
    *,
    skip_missing: bool,
    skipped: list[str] | None,
) -> bool:
    """Apply one override to a dumped config in place; return whether it applied."""
    if not dot_path:
        raise ConfigError("Experiment parameter path must not be empty")
    parts = dot_path.split(".")
    last = parts[-1]
    current: object = data
    for depth, part in enumerate(parts[:-1]):
        is_strategy_parent = depth == len(parts) - 2 and last == "strategy"
        if (
            is_strategy_parent
            and isinstance(current, dict)
            and part in _SHARED_FIELDS
            and current.get(part) is None
        ):
            # e.g. sweeping reranker.strategy with no reranker configured yet.
            current[part] = {}
        if not isinstance(current, dict) or part not in current:
            return _missing(dot_path, skip_missing, skipped)
        current = current[part]
    # A freshly created (empty) section accepts its first ``strategy``.
    if not isinstance(current, dict) or (
        last not in current and not (last == "strategy" and not current)
    ):
        return _missing(dot_path, skip_missing, skipped)
    if last == "strategy":
        section = ".".join(parts[:-1])
        shared = _SHARED_FIELDS.get(section)
        preserved = (
            {key: current[key] for key in shared.model_fields if key in current}
            if shared is not None
            else {}
        )
        current.clear()
        current.update(_variant_default(value))
        current.update(preserved)
    else:
        current[last] = value
    return True


def _missing(dot_path: str, skip_missing: bool, skipped: list[str] | None) -> bool:
    if not skip_missing:
        raise ConfigError(f"Unknown experiment parameter: {dot_path!r}")
    _record_skipped(skipped, dot_path)
    return False


def expand_grid(
    base_config: RagConfig, parameters: dict[str, list[object]]
) -> list[RagConfig]:
    """Build the cartesian product of all parameter combinations.

    Strategy selections are applied first so deeper overrides can tune the
    selected variant; each combination is validated once after every override
    is applied, so declaration order does not matter (for example sweeping
    ``chunking.chunk_size`` below the default ``overlap`` together with a
    smaller ``chunking.overlap``).
    Overrides that do not apply to a strategy (for example sweeping
    ``retrieval.fusion.method`` while another combination selects the dense
    strategy) are skipped for that combination with a warning. An empty sweep
    produces a single baseline configuration.
    """
    return [variant.config for variant in expand_grid_with_detail(base_config, parameters)]


def expand_grid_with_detail(
    base_config: RagConfig, parameters: dict[str, list[object]]
) -> list[GridVariant]:
    """Like :func:`expand_grid`, but each variant reports the parameters skipped.

    Skipped parameters are emitted as :class:`ExperimentParameterWarning` and
    recorded on the returned :class:`GridVariant`.
    """
    if not parameters:
        return [GridVariant(config=base_config)]
    keys = list(parameters)
    ordered_keys = sorted(keys, key=lambda key: not key.endswith(".strategy"))
    value_lists = [parameters[key] for key in ordered_keys]
    combos = itertools.product(*value_lists)
    variants: list[GridVariant] = []
    for combo in combos:
        data = base_config.model_dump(mode="json")
        skipped: list[str] = []
        for key, value in zip(ordered_keys, combo, strict=True):
            _apply_to_data(data, key, value, skip_missing=True, skipped=skipped)
        try:
            variant = RagConfig.model_validate(data)
        except ValidationError as exc:
            combination = ", ".join(
                f"{key}={value!r}" for key, value in zip(ordered_keys, combo, strict=True)
            )
            raise ConfigError(
                f"Invalid experiment combination ({combination}): {exc}"
            ) from exc
        for path in skipped:
            warnings.warn(
                f"Experiment parameter {path!r} does not apply to strategy "
                f"{_strategy_for(variant, path)!r} and was skipped for this "
                f"combination",
                ExperimentParameterWarning,
                stacklevel=2,
            )
        variants.append(GridVariant(config=variant, skipped_parameters=skipped))
    return variants


def _strategy_for(config: RagConfig, dot_path: str) -> str | None:
    if dot_path.startswith("retrieval."):
        return config.retrieval.strategy
    if dot_path.startswith("chunking."):
        return config.chunking.strategy
    if dot_path.startswith("reranker.") and config.reranker is not None:
        return config.reranker.strategy
    return None


def _record_skipped(skipped: list[str] | None, dot_path: str) -> None:
    if skipped is not None:
        skipped.append(dot_path)


def _variant_default(name: object) -> dict[str, object]:
    if not isinstance(name, str) or name not in _VARIANT_DEFAULTS:
        raise ConfigError(f"Unknown strategy for experiment sweep: {name!r}")
    model = _VARIANT_DEFAULTS[name]
    # Some variants declare ``strategy`` without a default, so pass it explicitly.
    return model.model_validate({"strategy": name}).model_dump(mode="json")