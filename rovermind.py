"""RoverMind — IR-01 Energy-Aware Robot Grid Exploration.

Pure software simulation. The simulator owns the hidden grid; the controller
receives only its current position, remaining energy, revealed cells and the
previous action. It never receives the hidden grid or unrevealed occupancy.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

SEED = 20260911
N = 30
BASE = (0, 0)
OBSTACLE_P = 0.20
REACHABLE_MIN = 0.70
INITIAL_ENERGY = 400
REVEAL_RADIUS = 2
SAFETY_RESERVE = 20

DIRECTIONS = ((0, 1, "UP"), (1, 0, "RIGHT"), (0, -1, "DOWN"), (-1, 0, "LEFT"))
Position = Tuple[int, int]


def neighbors(p: Position, size: int = N) -> Iterable[Tuple[Position, str]]:
    x, y = p
    for dx, dy, action in DIRECTIONS:
        q = (x + dx, y + dy)
        if 0 <= q[0] < size and 0 <= q[1] < size:
            yield q, action


def reachable_cells(grid: np.ndarray) -> Set[Position]:
    """True-world helper used only by the simulator/evaluator."""
    q: deque[Position] = deque([BASE])
    seen: Set[Position] = {BASE}
    while q:
        p = q.popleft()
        for qpos, _ in neighbors(p, grid.shape[0]):
            if grid[qpos[1], qpos[0]] and qpos not in seen:
                seen.add(qpos)
                q.append(qpos)
    return seen


def build_grid(seed: int = SEED) -> Tuple[np.ndarray, Set[Position], int]:
    """Generate the required 30x30 map using NumPy PCG64."""
    rng = np.random.Generator(np.random.PCG64(seed))
    attempts = 0
    while True:
        attempts += 1
        free = rng.random((N, N)) >= OBSTACLE_P
        free[BASE[1], BASE[0]] = True
        reachable = reachable_cells(free)
        if len(reachable) >= REACHABLE_MIN * int(free.sum()):
            return free, reachable, attempts


def reveal(grid: np.ndarray, center: Position) -> Dict[Position, bool]:
    """Reveal true occupancy only inside Manhattan distance <= 2."""
    x, y = center
    result: Dict[Position, bool] = {}
    for yy in range(max(0, y - REVEAL_RADIUS), min(N, y + REVEAL_RADIUS + 1)):
        span = REVEAL_RADIUS - abs(yy - y)
        for xx in range(max(0, x - span), min(N, x + span + 1)):
            result[(xx, yy)] = bool(grid[yy, xx])
    return result


def bfs(start: Position, known_free: Set[Position]):
    """BFS over known-free cells only. Safe for controller use."""
    if start not in known_free:
        return {}, {}
    q: deque[Position] = deque([start])
    parent: Dict[Position, Optional[Position]] = {start: None}
    dist: Dict[Position, int] = {start: 0}
    while q:
        p = q.popleft()
        for nxt, _ in neighbors(p):
            if nxt in known_free and nxt not in parent:
                parent[nxt] = p
                dist[nxt] = dist[p] + 1
                q.append(nxt)
    return parent, dist


def first_step(parent: Dict[Position, Optional[Position]], start: Position, target: Position) -> Position:
    if target == start:
        return start
    if target not in parent:
        raise RuntimeError("Target is not connected in the known map.")
    p = target
    while parent[p] != start:
        if parent[p] is None:
            raise RuntimeError("Path reconstruction failed.")
        p = parent[p]
    return p


@dataclass(frozen=True)
class Observation:
    position: Position
    energy: int
    revealed: Dict[Position, bool]
    previous_action: Optional[str]


@dataclass(frozen=True)
class Decision:
    action: str
    target: Optional[Position] = None
    reason: str = ""


class RoverMindController:
    """Greedy frontier controller that is strictly online."""

    ACTIONS = {name: (dx, dy) for dx, dy, name in DIRECTIONS}

    def __init__(self, safety_reserve: int = SAFETY_RESERVE):
        self.known: Dict[Position, bool] = {}
        self.position = BASE
        self.energy = INITIAL_ENERGY
        self.safety_reserve = safety_reserve
        self.return_mode = False

    def update(self, obs: Observation) -> None:
        self.position = obs.position
        self.energy = obs.energy
        self.known.update(obs.revealed)
        self.known[BASE] = True

    def decide(self, obs: Observation) -> Decision:
        self.update(obs)
        if self.energy <= 0:
            raise RuntimeError("Controller cannot act with zero energy.")

        # ONLY explicitly revealed free cells are used for routing.
        known_free = {p for p, is_free in self.known.items() if is_free}
        known_free.add(BASE)

        base_parent, base_dist = bfs(BASE, known_free)
        current_parent, current_dist = bfs(self.position, known_free)
        home = base_dist.get(self.position)
        if home is None:
            raise RuntimeError("Known-safe route to base disappeared.")

        # Exact benchmark termination: BASE + explicit DONE.
        if self.position == BASE:
            frontiers = self._frontiers(current_dist)
            if not frontiers or self.energy <= self.safety_reserve + 1:
                return Decision("DONE", reason="At base; no safe useful exploration remains.")
            self.return_mode = False

        # Turn back once the available energy is too close to the known home cost.
        if self.position != BASE and self.energy <= home + self.safety_reserve:
            self.return_mode = True

        if self.return_mode:
            target = base_parent.get(self.position)
            if target is None:
                raise RuntimeError("Could not reconstruct return step.")
            return Decision("MOVE", target, f"TURN BACK: E={self.energy}, home={home}")

        # Greedy online frontier selection.
        frontiers = self._frontiers(current_dist)
        if frontiers:
            min_d = min(current_dist[p] for p in frontiers)
            nearest = [p for p in frontiers if current_dist[p] == min_d]
            target = min(nearest, key=lambda p: (-self._unknown_count(p), p[0], p[1]))

            if target != self.position:
                nxt = first_step(current_parent, self.position, target)
                post_home = base_dist.get(nxt)
                # Pay 1 now; preserve return distance plus the safety reserve.
                if post_home is not None and self.energy - 1 >= post_home + self.safety_reserve:
                    return Decision("MOVE", nxt, f"GO FRONTIER: post_home={post_home}")

            # If no safe forward step exists, return.
            self.return_mode = True
            target = base_parent.get(self.position)
            if target is None:
                raise RuntimeError("No known-safe return step.")
            return Decision("MOVE", target, f"TURN BACK: frontier no longer safe, home={home}")

        if self.position == BASE:
            return Decision("DONE", reason="At base; finish safely.")

        self.return_mode = True
        target = base_parent.get(self.position)
        if target is None:
            raise RuntimeError("No known-safe return step.")
        return Decision("MOVE", target, f"RETURN: no frontier, home={home}")

    def _unknown_count(self, p: Position) -> int:
        return sum(1 for nxt, _ in neighbors(p) if nxt not in self.known)

    def _frontiers(self, current_dist: Dict[Position, int]) -> List[Position]:
        return [p for p in current_dist if self._unknown_count(p) > 0]


@dataclass
class RunResult:
    seed: int
    success: bool
    coverage: float
    energy_remaining: int
    steps: int
    movement_attempts: int
    obstacle_attempts: int
    reachable_free_cells: int
    observed_reachable_cells: int
    coverage_energy_product: float
    generation_attempts: int
    trajectory: List[Position]
    coverage_over_time: List[float]
    energy_over_time: List[int]
    actions: List[str]
    reasons: List[str]


def run_simulation(seed: int = SEED, max_steps: int = 20_000) -> RunResult:
    """Run one simulation with the exact termination rule."""
    grid, reachable, generation_attempts = build_grid(seed)
    position = BASE
    energy = INITIAL_ENERGY
    previous_action: Optional[str] = None
    controller = RoverMindController()

    current_reveal = reveal(grid, position)
    observed = {p for p, is_free in current_reveal.items() if is_free and p in reachable}
    trajectory = [position]
    coverage_history = [len(observed) / len(reachable)]
    energy_history = [energy]
    actions: List[str] = []
    reasons: List[str] = []
    movement_attempts = 0
    obstacle_attempts = 0

    for _ in range(max_steps):
        if energy == 0:
            break

        # Online boundary: ONLY current reveal + state crosses into controller.
        decision = controller.decide(
            Observation(position, energy, current_reveal, previous_action)
        )
        reasons.append(decision.reason)

        if decision.action == "DONE":
            actions.append("DONE")
            return _result(
                seed, position == BASE, reachable, observed, energy, generation_attempts,
                trajectory, coverage_history, energy_history, actions, reasons,
                movement_attempts, obstacle_attempts,
            )

        if decision.action != "MOVE" or decision.target is None:
            raise RuntimeError(f"Invalid controller decision: {decision}")

        target = decision.target
        if all(nxt != target for nxt, _ in neighbors(position)):
            raise RuntimeError(f"Illegal movement target: {target}")

        movement_attempts += 1
        energy -= 1
        if grid[target[1], target[0]]:
            position = target
            actions.append("MOVE")
        else:
            obstacle_attempts += 1
            actions.append("ATTEMPT_OBSTACLE")

        current_reveal = reveal(grid, position)
        observed.update(p for p, is_free in current_reveal.items() if is_free and p in reachable)
        previous_action = "MOVE"
        trajectory.append(position)
        coverage_history.append(len(observed) / len(reachable))
        energy_history.append(energy)

    return _result(
        seed, False, reachable, observed, energy, generation_attempts,
        trajectory, coverage_history, energy_history, actions, reasons,
        movement_attempts, obstacle_attempts,
    )


def _result(
    seed: int,
    success: bool,
    reachable: Set[Position],
    observed: Set[Position],
    energy: int,
    generation_attempts: int,
    trajectory: List[Position],
    coverage_history: List[float],
    energy_history: List[int],
    actions: List[str],
    reasons: List[str],
    movement_attempts: int,
    obstacle_attempts: int,
) -> RunResult:
    cov = len(observed) / len(reachable) if reachable else 1.0
    ef = energy / INITIAL_ENERGY if success else 0.0
    return RunResult(
        seed=seed, success=success, coverage=cov, energy_remaining=energy,
        steps=len(trajectory) - 1, movement_attempts=movement_attempts,
        obstacle_attempts=obstacle_attempts, reachable_free_cells=len(reachable),
        observed_reachable_cells=len(observed), coverage_energy_product=cov * ef,
        generation_attempts=generation_attempts, trajectory=trajectory,
        coverage_over_time=coverage_history, energy_over_time=energy_history,
        actions=actions, reasons=reasons,
    )


def save_outputs(result: RunResult, out_dir: str = ".") -> None:
    """Write development metrics, trajectory, and a PNG visualization."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = {
        "seed": result.seed,
        "success": result.success,
        "coverage": result.coverage,
        "energy_remaining": result.energy_remaining,
        "steps": result.steps,
        "movement_attempts": result.movement_attempts,
        "obstacle_attempts": result.obstacle_attempts,
        "visited_free_cells": result.observed_reachable_cells,
        "reachable_free_cells": result.reachable_free_cells,
        "coverage_energy_product": result.coverage_energy_product,
        "generation_attempts": result.generation_attempts,
    }
    (out / "rovermind_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (out / "rovermind_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "x", "y", "energy", "coverage_pct", "action", "reason"])
        for i, p in enumerate(result.trajectory):
            action = "START" if i == 0 else result.actions[i - 1]
            reason = "Initial observation" if i == 0 else result.reasons[i - 1]
            w.writerow([i, p[0], p[1], result.energy_over_time[i],
                        f"{100 * result.coverage_over_time[i]:.6f}", action, reason])

    grid, _, _ = build_grid(result.seed)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(~grid, origin="lower", interpolation="nearest")
    xs, ys = zip(*result.trajectory)
    ax.plot(xs, ys, linewidth=1.2, label="Exploration path")
    ax.scatter([BASE[0]], [BASE[1]], s=120, marker="s", label="Base")
    ax.scatter([result.trajectory[-1][0]], [result.trajectory[-1][1]], s=70, marker="o", label="Final")
    ax.set_title(f"RoverMind - IR-01 | Coverage {100 * result.coverage:.2f}% | Return {result.success}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "rovermind_exploration.png", dpi=180)
    plt.close(fig)


def print_result(r: RunResult) -> None:
    print("\nRoverMind development run")
    print("-" * 36)
    print(f"Seed:                    {r.seed}")
    print(f"Success / returned:      {r.success}")
    print(f"Steps:                   {r.steps}")
    print(f"Energy remaining:        {r.energy_remaining}")
    print(f"Reachable free cells:    {r.reachable_free_cells}")
    print(f"Observed reachable:      {r.observed_reachable_cells}")
    print(f"Coverage:                {100 * r.coverage:.2f}%")
    print(f"Movement attempts:       {r.movement_attempts}")
    print(f"Obstacle attempts:       {r.obstacle_attempts}")
    print(f"Coverage x E_fraction:   {r.coverage_energy_product:.6f}")
    print(f"Grid generation attempts:{r.generation_attempts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=int, default=0,
                        help="run N deterministic seeds starting from 20260911")
    args = parser.parse_args()

    result = run_simulation(SEED)
    print_result(result)
    save_outputs(result)
    print("Wrote: rovermind_metrics.json")
    print("Wrote: rovermind_trajectory.csv")
    print("Wrote: rovermind_exploration.png")

    if args.benchmark:
        results = [run_simulation(SEED + i) for i in range(args.benchmark)]
        mean_cov = float(np.mean([r.coverage for r in results]))
        return_rate = float(np.mean([r.success for r in results]))
        mean_ckek = float(np.mean([r.coverage_energy_product for r in results]))
        print("\nLocal benchmark")
        print("-" * 36)
        print(f"Seeds evaluated:         {args.benchmark}")
        print(f"Mean coverage:           {100 * mean_cov:.2f}%")
        print(f"Return success fraction: {return_rate:.2%}")
        print(f"Mean CkEk:               {mean_ckek:.6f}")


if __name__ == "__main__":
    main()
