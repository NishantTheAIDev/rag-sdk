"""Score fusion for hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from rag_sdk.retrieval.base import RetrievalResult


def rrf_fuse(
    dense: Sequence[RetrievalResult],
    lexical: Sequence[RetrievalResult],
    k: int = 60,
) -> list[RetrievalResult]:
    """Fuse two rankings with Reciprocal Rank Fusion.

    Each ranking contributes ``1 / (k + rank)`` per result and contributions
    accumulate across rankings. Returned results carry the fused score.
    """
    fused: dict[str, list[float, RetrievalResult]] = {}
    dense_rank = {result.chunk.id: rank for rank, result in enumerate(dense, start=1)}
    for rank, result in enumerate(dense, start=1):
        fused[result.chunk.id] = [1.0 / (k + rank), result]
    for rank, result in enumerate(lexical, start=1):
        contribution = 1.0 / (k + rank)
        entry = fused.get(result.chunk.id)
        if entry is None:
            fused[result.chunk.id] = [contribution, result]
        else:
            entry[0] += contribution
    results = [
        result.model_copy(update={"score": contribution})
        for contribution, result in fused.values()
    ]
    results.sort(
        key=lambda result: (
            -result.score,
            dense_rank.get(result.chunk.id, 0),
        )
    )
    return results


def weighted_fuse(
    dense: Sequence[RetrievalResult],
    lexical: Sequence[RetrievalResult],
    dense_weight: float,
    lexical_weight: float,
) -> list[RetrievalResult]:
    """Fuse two rankings by min-max normalized weighted scores.

    Scores are min-max normalized within each source so they are comparable,
    then combined as ``weight_dense * score_dense + weight_lexical * score_lexical``.
    """
    dense_by_id = _normalized(dense)
    lexical_by_id = _normalized(lexical)
    combined: dict[str, list[float, RetrievalResult]] = {}
    for result in dense:
        combined[result.chunk.id] = [
            dense_weight * dense_by_id[result.chunk.id],
            result,
        ]
    for result in lexical:
        score = lexical_weight * lexical_by_id[result.chunk.id]
        entry = combined.get(result.chunk.id)
        if entry is not None:
            entry[0] += score
        else:
            combined[result.chunk.id] = [score, result]
    results = [
        result.model_copy(update={"score": score})
        for score, result in combined.values()
    ]
    results.sort(key=lambda result: result.score, reverse=True)
    return results


def _normalized(results: Sequence[RetrievalResult]) -> dict[str, float]:
    if not results:
        return {}
    lowest = min(result.score for result in results)
    highest = max(result.score for result in results)
    span = highest - lowest
    if span <= 0:
        return {result.chunk.id: 1.0 for result in results}
    return {
        result.chunk.id: (result.score - lowest) / span for result in results
    }