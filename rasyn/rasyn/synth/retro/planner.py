"""Retro* / A*-style AND-OR tree search planner (RETRO_PLAN R-5, Lock 8).

The planner maintains an AND-OR tree rooted at the target molecule:
  - OR_molecule node: pick one of several possible expansions.
      cost = min over child AND_step nodes (best disconnection so far).
  - AND_step node: ALL children OR nodes must be solved.
      cost = sum over child OR_molecule nodes + step_cost (-log plausibility).

Search procedure (best-first):
  1. Push the root OR_molecule into a priority queue keyed by f(n) = g(n) + h(n).
     g(n) = realized cost so far (initially 0).
     h(n) = V(n) from R-4 value model.
  2. Pop the node with smallest f. If it is buyable, mark it solved and
     propagate the solution upward.
  3. Otherwise, expand: query proposers, build AND_step children, each
     with its OR_molecule precursors. Push new OR_molecules into the queue.
  4. Repeat until termination:
       - All leaves of the root's best subtree are buyable (success), OR
       - time budget exceeded, OR
       - max_iterations reached.

The result is the top-K candidate RouteTrees ranked by realized cost.
"""
from __future__ import annotations

import heapq
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rasyn.synth.retro.buyability import BuyabilityIndex
from rasyn.synth.retro.conditions import ConditionsPredictor
from rasyn.synth.retro.proposers.base import RetroProposer
from rasyn.synth.retro.route_score import attach_score_to_route, compute_route_score_inputs
from rasyn.synth.retro.schemas import (
    BuyabilityRecord,
    ConditionPrediction,
    ForwardValidationResult,
    ProposerOutput,
    ReactionClass,
    RetroStep,
    RouteRationale,
    RouteTree,
    RouteTreeNode,
)
from rasyn.synth.retro.validator import ForwardValidator
from rasyn.synth.retro.value_model import ValueModel

logger = logging.getLogger("retro.planner")


@dataclass
class PlannerConfig:
    max_steps: int = 8
    max_iterations: int = 200
    time_budget_s: float = 60.0
    top_k_per_proposer: int = 5
    top_k_routes: int = 10
    enable_diffusion: bool = True
    require_forward_validation: bool = True
    tier1_only: bool = False
    seed: int = 42


@dataclass(order=True)
class _PQNode:
    priority: float
    counter: int
    node_id: str = field(compare=False)


@dataclass
class _PlannerState:
    nodes: dict[str, RouteTreeNode] = field(default_factory=dict)
    parents: dict[str, list[str]] = field(default_factory=dict)
    or_open: list[_PQNode] = field(default_factory=list)
    counter: int = 0
    steps_index: dict[str, RetroStep] = field(default_factory=dict)
    fvr_index: dict[str, ForwardValidationResult] = field(default_factory=dict)
    cond_index: dict[str, ConditionPrediction] = field(default_factory=dict)


class RetroPlanner:
    """Retro* AND-OR tree search over the 5-channel proposer ensemble."""

    def __init__(
        self,
        proposers: list[RetroProposer],
        buyability: BuyabilityIndex,
        value_model: ValueModel,
        validator: ForwardValidator,
        conditions: ConditionsPredictor,
        cfg: PlannerConfig | None = None,
    ):
        self.proposers = proposers
        self.buyability = buyability
        self.value_model = value_model
        self.validator = validator
        self.conditions = conditions
        self.cfg = cfg or PlannerConfig()

    # ----- internal helpers -----

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    def _push_or(self, state: _PlannerState, node: RouteTreeNode, g: float) -> None:
        h = self.value_model.value(node)
        state.counter += 1
        heapq.heappush(state.or_open, _PQNode(g + h, state.counter, node.node_id))

    def _make_or_node(self, smi: str, ik: str, depth: int) -> RouteTreeNode:
        is_buyable = self.buyability.is_buyable(ik)
        rec = self.buyability.lookup(ik)
        return RouteTreeNode(
            node_id=self._new_id("OR"),
            node_type="OR_molecule",
            molecule_inchi_key=ik,
            molecule_smiles=smi,
            is_buyable=is_buyable,
            buyability_record_inchi_key=rec.inchi_key if rec else None,
            depth=depth,
            expanded=is_buyable,
        )

    def _expand_or(self, state: _PlannerState, or_node: RouteTreeNode) -> None:
        """Query proposers, build AND_step children, push new OR nodes."""
        outs: list[ProposerOutput] = []
        for prop in self.proposers:
            if prop.channel == "diffusion" and not self.cfg.enable_diffusion:
                continue
            try:
                out = prop.propose(
                    target_smiles=or_node.molecule_smiles or "",
                    target_inchi_key=or_node.molecule_inchi_key or "",
                    top_k=self.cfg.top_k_per_proposer,
                )
                outs.append(out)
            except Exception as e:
                logger.warning("proposer %s failed on %s: %s",
                                prop.channel, or_node.molecule_smiles, e)

        and_children_ids = []
        for out in outs:
            for k, (precursor_ikeys, precursor_smis, conf, rc) in enumerate(zip(
                out.candidates, out.candidate_smiles, out.confidences, out.reaction_class_predictions,
            )):
                step = RetroStep(
                    retro_step_id=self._new_id("S"),
                    product_inchi_key=or_node.molecule_inchi_key or "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
                    precursor_inchi_keys=precursor_ikeys,
                    reaction_class=rc or "unclassified",
                    proposed_by_channel=out.channel,
                    proposed_by_top_k_rank=k,
                    confidence=max(0.0, min(1.0, float(conf))),
                )

                # Conditions prediction
                if self.conditions is not None:
                    cond = self.conditions.predict(
                        reactant_smiles_list=precursor_smis,
                        product_smiles=or_node.molecule_smiles or "",
                        reactant_inchi_keys=precursor_ikeys,
                        product_inchi_key=or_node.molecule_inchi_key or "",
                        reaction_class=step.reaction_class,
                    )
                    state.cond_index[step.retro_step_id] = cond

                # Forward validation
                fvr = self.validator.validate_step(
                    step, precursor_smiles=precursor_smis,
                    target_smiles=or_node.molecule_smiles or "",
                )
                state.fvr_index[step.retro_step_id] = fvr

                if self.cfg.require_forward_validation and fvr.pass_rule == "fail":
                    continue

                # Build the AND_step node and its OR_molecule children
                and_node = RouteTreeNode(
                    node_id=self._new_id("AND"),
                    node_type="AND_step",
                    retro_step=step.model_copy(update={
                        "forward_validation_pass": fvr.pass_rule != "fail",
                        "forward_tanimoto_to_target": fvr.tanimoto_to_target,
                    }),
                    depth=or_node.depth + 1,
                )
                state.steps_index[step.retro_step_id] = and_node.retro_step

                child_or_ids = []
                for precursor_ikey, precursor_smi in zip(precursor_ikeys, precursor_smis):
                    child = self._make_or_node(precursor_smi, precursor_ikey, depth=or_node.depth + 2)
                    state.nodes[child.node_id] = child
                    state.parents.setdefault(child.node_id, []).append(and_node.node_id)
                    child_or_ids.append(child.node_id)
                    if not child.is_buyable and child.depth <= 2 * self.cfg.max_steps:
                        self._push_or(state, child, g=float(and_node.depth))

                and_node = and_node.model_copy(update={"children_node_ids": child_or_ids})
                state.nodes[and_node.node_id] = and_node
                and_children_ids.append(and_node.node_id)

        # Update OR node with children
        new_or = or_node.model_copy(update={
            "children_node_ids": and_children_ids,
            "expanded": True,
        })
        state.nodes[or_node.node_id] = new_or

    # ----- route assembly -----

    def _check_node_solved(self, state: _PlannerState, node_id: str, max_steps: int) -> bool:
        node = state.nodes[node_id]
        if node.node_type == "OR_molecule":
            if node.is_buyable:
                return True
            if not node.expanded:
                return False
            return any(
                self._check_node_solved(state, c, max_steps)
                for c in node.children_node_ids
            )
        # AND_step: all children must be solved AND depth <= max_steps
        if node.depth > 2 * max_steps:
            return False
        return all(self._check_node_solved(state, c, max_steps) for c in node.children_node_ids)

    def _enumerate_routes(self, state: _PlannerState, root_id: str, max_routes: int) -> list[list[str]]:
        """Enumerate min-cost solved AND-OR sub-trees from root.

        Returns up to `max_routes` lists of node_ids. Each list is a tree
        with the root in node 0; AND_step children chosen one per
        OR_molecule.
        """
        routes: list[list[str]] = []

        def _dfs(node_id: str, acc: list[str]) -> bool:
            node = state.nodes[node_id]
            acc.append(node_id)
            if node.node_type == "OR_molecule":
                if node.is_buyable:
                    return True
                # Try each AND_step child; recurse depth-first
                for c in node.children_node_ids:
                    sub = list(acc)
                    if _dfs(c, sub):
                        if len(routes) < max_routes:
                            routes.append(sub)
                        if len(routes) >= max_routes:
                            return True
                return False
            # AND_step: ALL OR_molecule children must succeed (any one assignment)
            sub_acc = list(acc)
            for c in node.children_node_ids:
                ok = False
                # Each child must have at least one solved path
                sub2 = list(sub_acc)
                if _dfs(c, sub2):
                    sub_acc = sub2
                    ok = True
                if not ok:
                    return False
            acc.clear()
            acc.extend(sub_acc)
            return True

        _dfs(root_id, [])
        return routes

    def _build_route_tree(self, state: _PlannerState, node_ids: list[str], root_id: str) -> RouteTree:
        nodes = [state.nodes[nid] for nid in node_ids]
        steps = [n.retro_step for n in nodes if n.node_type == "AND_step" and n.retro_step is not None]
        leaves = [n for n in nodes if n.node_type == "OR_molecule" and n.is_buyable]
        purchasable_fraction = (
            sum(1 for n in nodes if n.node_type == "OR_molecule" and n.is_buyable)
            / max(1, sum(1 for n in nodes if n.node_type == "OR_molecule"))
        )
        longest = max((n.depth for n in nodes), default=0) // 2
        return RouteTree(
            tree_id=self._new_id("TREE"),
            target_inchi_key=state.nodes[root_id].molecule_inchi_key or "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
            target_smiles=state.nodes[root_id].molecule_smiles or "",
            nodes=nodes,
            root_node_id=root_id,
            step_count=len(steps),
            longest_linear_sequence=longest,
            all_leaves_buyable=all(
                n.is_buyable for n in nodes if n.node_type == "OR_molecule" and not n.children_node_ids
            ),
            purchasable_fraction=purchasable_fraction,
            risk_score=0.0,
        )

    # ----- public API -----

    def plan(
        self,
        target_smiles: str,
        target_inchi_key: str,
    ) -> list[RouteTree]:
        state = _PlannerState()
        root = self._make_or_node(target_smiles, target_inchi_key, depth=0)
        state.nodes[root.node_id] = root
        if root.is_buyable:
            return [self._build_route_tree(state, [root.node_id], root.node_id)]
        self._push_or(state, root, g=0.0)

        t0 = time.time()
        iters = 0
        while state.or_open and iters < self.cfg.max_iterations:
            if time.time() - t0 > self.cfg.time_budget_s:
                logger.info("planner: time budget exhausted at iter %d", iters)
                break
            pq_node = heapq.heappop(state.or_open)
            node = state.nodes.get(pq_node.node_id)
            if node is None or node.expanded or node.is_buyable:
                continue
            if node.depth > 2 * self.cfg.max_steps:
                continue
            self._expand_or(state, node)
            iters += 1

        if not self._check_node_solved(state, root.node_id, self.cfg.max_steps):
            logger.info("planner: no buyable route found within budget")
            return []

        routes_node_ids = self._enumerate_routes(state, root.node_id, self.cfg.top_k_routes)
        return [self._build_route_tree(state, node_ids, root.node_id) for node_ids in routes_node_ids]
