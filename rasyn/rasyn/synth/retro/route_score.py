"""Route-level scoring (RETRO_PLAN R-4 + L6 — fixed weights, no v1 sweeps).

route_score = 0.4 * prod(step_plausibility)
            + 0.3 * forward_pass_rate
            - 0.1 * step_count_norm
            - 0.1 * cost_norm
            - 0.1 * risk_flags_norm

All normalised quantities are in [0, 1]. Each component is computed
deterministically from RouteTree / per-step ForwardValidationResult /
buyability lookups.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable

from rasyn.synth.retro.schemas import (
    BuyabilityRecord,
    ConditionPrediction,
    ForwardValidationResult,
    RetroStep,
    RouteTree,
)


@dataclass(frozen=True)
class RouteScoreWeights:
    """RETRO_PLAN L6 fixed weights. Do NOT sweep in v1."""

    plausibility_product: float = 0.4
    forward_pass_rate: float = 0.3
    step_count: float = 0.1
    cost: float = 0.1
    risk_flags: float = 0.1


@dataclass(frozen=True)
class RouteScoreInputs:
    step_plausibility_product: float
    forward_pass_rate: float
    step_count_norm: float
    cost_norm: float
    risk_flags_norm: float


def _norm_step_count(n: int, max_steps: int) -> float:
    if max_steps <= 0:
        return 0.0
    return min(1.0, n / max_steps)


def _norm_cost(total_cost_usd: float | None, cost_cap_usd: float = 100.0) -> float:
    if total_cost_usd is None:
        return 0.5  # mid-tier penalty when unknown
    return min(1.0, total_cost_usd / cost_cap_usd)


def _norm_risk_flags(risk_flags: list[str], max_flags: int = 5) -> float:
    if max_flags <= 0:
        return 0.0
    return min(1.0, len(risk_flags) / max_flags)


def compute_route_score_inputs(
    steps: list[RetroStep],
    fvr_list: list[ForwardValidationResult],
    buyables_records: Iterable[BuyabilityRecord | None],
    *,
    risk_flags: list[str] | None = None,
    max_steps: int = 8,
) -> RouteScoreInputs:
    """Compute the 5 components from the route's per-step + per-leaf data."""
    risk_flags = risk_flags or []
    plausibility = prod((s.confidence for s in steps), start=1.0) if steps else 0.0
    if fvr_list:
        n_pass = sum(1 for f in fvr_list if f.pass_rule != "fail")
        forward_rate = n_pass / len(fvr_list)
    else:
        forward_rate = 0.0
    cost = 0.0
    n_with_cost = 0
    for rec in buyables_records:
        if rec is None:
            continue
        if rec.cost_per_g_usd is not None:
            cost += rec.cost_per_g_usd
            n_with_cost += 1
    total_cost = cost if n_with_cost else None
    return RouteScoreInputs(
        step_plausibility_product=float(plausibility),
        forward_pass_rate=float(forward_rate),
        step_count_norm=_norm_step_count(len(steps), max_steps),
        cost_norm=_norm_cost(total_cost),
        risk_flags_norm=_norm_risk_flags(risk_flags),
    )


def route_score(inputs: RouteScoreInputs, *, weights: RouteScoreWeights | None = None) -> float:
    w = weights or RouteScoreWeights()
    return (
        w.plausibility_product * inputs.step_plausibility_product
        + w.forward_pass_rate * inputs.forward_pass_rate
        - w.step_count * inputs.step_count_norm
        - w.cost * inputs.cost_norm
        - w.risk_flags * inputs.risk_flags_norm
    )


def attach_score_to_route(
    tree: RouteTree,
    steps: list[RetroStep],
    fvr_list: list[ForwardValidationResult],
    buyables_records: Iterable[BuyabilityRecord | None],
    *,
    risk_flags: list[str] | None = None,
    max_steps: int = 8,
    weights: RouteScoreWeights | None = None,
) -> tuple[float, RouteScoreInputs]:
    """Helper: compute inputs + score in one call. Returns (score, inputs)."""
    inputs = compute_route_score_inputs(
        steps, fvr_list, buyables_records,
        risk_flags=risk_flags, max_steps=max_steps,
    )
    return route_score(inputs, weights=weights), inputs
