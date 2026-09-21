"""Expanding experiment parameter sweeps into concrete configurations."""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

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

    When the path targets a strategy discriminator (``chunking.strategy`` or
    ``retrieval.strategy``), the selected strategy's default configuration is
    applied so that later overrides can tune it. With ``skip_missing=True``,
    overrides whose path does not exist in the current configuration are
    ignored instead of raising (used during sweeps where a parameter only
    applies to some strategies); the skipped paths are appended to ``skipped``
    when provided.
    """
    if not dot_path:
        raise ConfigError("Experiment parameter path must not be empty")
    data = config.model_dump(mode="json")
    parts = dot_path.split(".")
    current = data
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            if skip_missing:
                _record_skipped(skipped, dot_path)
                return config
            raise ConfigError(f"Unknown experiment parameter: {dot_path!r}")
        current = current[part]
    last = parts[-1]
    if not isinstance(current, dict) or last not in current:
        if skip_missing:
            _record_skipped(skipped, dot_path)
            return config
        raise ConfigError(f"Unknown experiment parameter: {dot_path!r}")
    if last == "strategy":
        current.clear()
        current.update(_variant_default(value))
    else:
        current[last] = value
    return RagConfig.model_validate(data)


def expand_grid(
    base_config: RagConfig, parameters: dict[str, list[object]]
) -> list[RagConfig]:
    """Build the cartesian product of all parameter combinations.

    Parameters are applied in declaration order, with strategy selections
    always applied first so deeper overrides can tune the selected variant.
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
        variant = base_config
        skipped: list[str] = []
        for key, value in zip(ordered_keys, combo, strict=True):
            variant = apply_override(
                variant, key, value, skip_missing=True, skipped=skipped
            )
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