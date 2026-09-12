"""RoverMind — IR-01 Energy-Aware Robot Grid Exploration.

Pure software / simulation implementation.

The environment owns the hidden true grid. The controller only receives the
currently revealed map, current position, previous actions, and remaining
energy. It never receives the hidden grid or unrevealed occupancy.

Run:
    python rovermind.py
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

SEED = 20260911
GRID_SIZE = 30
OBSTACLE_PROBABILITY = 0.20
REACHABILITY_THRESHOLD = 0.70
INITIAL_ENERGY = 400
REVEAL_RADIUS = 2
BASE: Tuple[int, int] = (0, 0)
FREE = 0
OBSTACLE = 1
UNKNOWN = -1
Position = Tuple[int, int]
DIRECTIONS: Tuple[Position, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def in_bounds(pos: Position, size: int = GRID_SIZE) -> bool:
    r, c = pos
    return 0 <= r < size and 0 <= c < size


def neighbors(pos: Position, size: int = GRID_SIZE) -> Iterable[Position]:
    r, c = pos
    for dr, dc in DIRECTIONS:
        nxt = (r + dr, c + dc)
        if in_bounds(nxt, size):
            yield nxt


def reachable_free_cells(grid: np.ndarray, base: Position = BASE) -> Set[Position]:
    """True-world evaluator helper; never called by the controller."""
    if grid[base] == OBSTACLE:
        return set()
    seen: Set[Position] = {base}
    queue: deque[Position] = deque([base])
    while queue:
        cur = queue.popleft()
        for nxt in neighbors(cur, grid.shape[0]):
            if nxt not in seen and grid[nxt] == FREE:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def generate_grid(
    seed: int = SEED,
    size: int = GRID_SIZE,
    obstacle_probability: float = OBSTACLE_PROBABILITY,
    reachability_threshold: float = REACHABILITY_THRESHOLD,
) -> Tuple[np.ndarray, Set[Position], int]:
    """Generate the required deterministic map using NumPy PCG64."""
    rng = np.random.Generator(np.random.PCG64(seed))
    attempts = 0
    while True:
        attempts += 1
        grid = (rng.random((size, size)) < obstacle_probability).astype(np.int8)
        grid[BASE] = FREE
        reachable = reachable_free_cells(grid, BASE)
        free_count = int(np.count_nonzero(grid == FREE))
        if free_count > 0 and len(reachable) / free_count >= reachability_threshold:
            return grid, reachable, attempts


def reveal_cells(
    true_grid: np.ndarray,
    revealed: np.ndarray,
    center: Position,
    radius: int = REVEAL_RADIUS,
) -> None:
    """Reveal true occupancy within Manhattan distance 2 of the position."""
    cr, cc = center
    size = true_grid.shape[0]
    for r in range(max(0, cr - radius), min(size, cr + radius + 1)):
        row_span = radius - abs(r - cr)
        c0 = max(0, cc - row_span)
        c1 = min(size, cc + row_span + 1)
        revealed[r, c0:c1] = true_grid[r, c0:c1]


def coverage(revealed: np.ndarray, reachable: Set[Position]) -> float:
    """Coverage for evaluation; the controller never receives `reachable`."""
    if not reachable:
        return 1.0
    observed = sum(revealed[p] == FREE for p in reachable)
    return observed / len(reachable)


def shortest_known_free_path(
    revealed: np.ndarray,
    start: Position,
    goal: Position,
) -> Optional[List[Position]]:
    """Shortest path using only cells explicitly known to be free."""
    if revealed[start] != FREE or revealed[goal] != FREE:
        return None
    queue: deque[Position] = deque([start])
    parent: Dict[Position, Optional[Position]] = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == goal:
            path: List[Position] = []
            node: Optional[Position] = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]
        for nxt in neighbors(cur, revealed.shape[0]):
            if nxt in parent or revealed[nxt] != FREE:
                continue
            parent[nxt] = cur
            queue.append(nxt)
    return None


@dataclass
class Decision:
    action: str
    target: Optional[Position] = None
    reason: str = ""


@dataclass
class OnlineExplorer:
    """Greedy, genuinely online controller."""

    base: Position = BASE
    position: Position = BASE
    action_history: List[str] = field(default_factory=list)

    @staticmethod
    def unknown_neighbors(revealed: np.ndarray, pos: Position) -> List[Position]:
        return [n for n in neighbors(pos, revealed.shape[0]) if revealed[n] == UNKNOWN]

    def known_return_path(self, revealed: np.ndarray) -> Optional[List[Position]]:
        return shortest_known_free_path(revealed, self.position, self.base)

    def choose_nearest_frontier_path(self, revealed: np.ndarray) -> Optional[List[Position]]:
        """Find nearest known-free frontier using only the revealed map."""
        size = revealed.shape[0]
        queue: deque[Position] = deque([self.position])
        parent: Dict[Position, Optional[Position]] = {self.position: None}
        distance: Dict[Position, int] = {self.position: 0}
        candidates: List[Position] = []
        best_distance: Optional[int] = None

        while queue:
            cur = queue.popleft()
            dist = distance[cur]
            if best_distance is not None and dist > best_distance:
                break
            if any(revealed[n] == UNKNOWN for n in neighbors(cur, size)):
                if best_distance is None:
                    best_distance = dist
                candidates.append(cur)
                continue
            for nxt in neighbors(cur, size):
                if nxt in parent or revealed[nxt] != FREE:
                    continue
                parent[nxt] = cur
                distance[nxt] = dist + 1
                queue.append(nxt)

        if not candidates:
            return None

        target = min(
            candidates,
            key=lambda p: (-len(self.unknown_neighbors(revealed, p)), p[0], p[1]),
        )
        path: List[Position] = []
        node: Optional[Position] = target
        while node is not None:
            path.append(node)
            node = parent[node]
        return path[::-1]

    def decide(self, revealed: np.ndarray, remaining_energy: int) -> Decision:
        """Choose exactly one action from revealed information and current state.

        Let d be the shortest known-free distance to base.

        Unknown exploration requires E >= d + 3:
          - 1 energy for the exploratory attempt;
          - up to d + 1 energy to return if the new cell is free;
          - 1 energy must still remain after reaching base, otherwise the run
            would end immediately at zero before the controller can output DONE.

        For a known-free move, after paying 1 energy the controller requires
        enough remaining energy to reach base and still have 1 unit left.
        Otherwise it turns back along the known-free route.
        """
        if revealed.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError("Controller expects a 30x30 revealed map.")
        if revealed[self.position] != FREE:
            raise RuntimeError("Current position must be known free.")
        if remaining_energy <= 0:
            raise RuntimeError("Controller cannot act with zero energy.")

        home_path = self.known_return_path(revealed)
        if home_path is None:
            raise RuntimeError("No known-safe route to base; invariant broken.")
        d_home = len(home_path) - 1
        frontier_path = self.choose_nearest_frontier_path(revealed)

        # No known frontier: return to base, then finish.
        if frontier_path is None:
            if self.position == self.base:
                self.action_history.append("DONE")
                return Decision("DONE", reason="No frontier remains; output DONE.")
            if remaining_energy >= d_home + 1:
                target = home_path[1]
                self.action_history.append("MOVE")
                return Decision(
                    "MOVE", target,
                    f"RETURN: d_home={d_home}, E={remaining_energy}",
                )
            raise RuntimeError("Energy insufficient for guaranteed return.")

        # At a frontier, probe an unknown neighbor only with d+3 energy.
        if self.position == frontier_path[-1]:
            unknown = self.unknown_neighbors(revealed, self.position)
            if unknown and remaining_energy >= d_home + 3:
                target = unknown[0]
                self.action_history.append("MOVE")
                return Decision(
                    "MOVE", target,
                    f"EXPLORE: E={remaining_energy}, d_home={d_home}, threshold={d_home + 3}",
                )

            # Safety threshold reached: turn back now.
            if self.position == self.base:
                self.action_history.append("DONE")
                return Decision("DONE", reason="At base; energy too low for another safe probe.")
            if remaining_energy >= d_home + 1:
                target = home_path[1]
                self.action_history.append("MOVE")
                return Decision(
                    "MOVE", target,
                    f"TURN BACK: E={remaining_energy}, d_home={d_home}, reserve=1",
                )
            raise RuntimeError("Energy insufficient for guaranteed return.")

        # Travel toward nearest frontier through cells known to be free.
        if len(frontier_path) >= 2:
            target = frontier_path[1]
            next_home = shortest_known_free_path(revealed, target, self.base)
            if next_home is None:
                raise RuntimeError("Chosen move would break the known route to base.")
            post_move_home = len(next_home) - 1
            if remaining_energy - 1 >= post_move_home + 1:
                self.action_history.append("MOVE")
                return Decision(
                    "MOVE", target,
                    f"GO FRONTIER: post_move_home={post_move_home}",
                )

        # Exploration is no longer safe: return through known-free cells.
        if self.position != self.base and remaining_energy >= d_home + 1:
            target = home_path[1]
            self.action_history.append("MOVE")
            return Decision(
                "MOVE", target,
                f"TURN BACK: safety threshold reached, d_home={d_home}",
            )

        # At base, DONE is allowed even with only 1 energy because DONE costs 0.
        self.action_history.append("DONE")
        return Decision("DONE", reason="At base; finish safely.")


@dataclass
class SimulationResult:
    grid: np.ndarray
    reachable_cells: Set[Position]
    revealed: np.ndarray
    trajectory: List[Position]
    coverage_over_time: List[float]
    energy_over_time: List[int]
    actions: List[str]
    reasons: List[str]
    returned_successfully: bool
    final_coverage: float
    remaining_energy: int
    ckek: float
    generation_attempts: int


def run_simulation(seed: int = SEED, max_steps: int = 20_000) -> SimulationResult:
    """Run the hidden environment and controller step-by-step."""
    true_grid, reachable, attempts = generate_grid(seed=seed)
    revealed = np.full((GRID_SIZE, GRID_SIZE), UNKNOWN, dtype=np.int8)
    controller = OnlineExplorer()
    position = BASE
    energy = INITIAL_ENERGY

    reveal_cells(true_grid, revealed, position)
    trajectory = [position]
    coverage_history = [coverage(revealed, reachable)]
    energy_history = [energy]
    actions: List[str] = []
    reasons: List[str] = []
    returned_successfully = False

    for _ in range(max_steps):
        if energy == 0:
            break
        controller.position = position

        # Critical online boundary: only revealed information is passed in.
        decision = controller.decide(revealed.copy(), energy)
        reasons.append(decision.reason)

        if decision.action == "DONE":
            returned_successfully = position == BASE
            actions.append("DONE")
            break

        if decision.action != "MOVE" or decision.target is None:
            raise RuntimeError(f"Invalid controller decision: {decision}")

        target = decision.target
        if target not in set(neighbors(position)):
            raise RuntimeError(f"Illegal movement target: {target}")

        # Every movement attempt costs exactly 1 unit.
        energy -= 1
        if true_grid[target] == FREE:
            position = target
            actions.append("MOVE")
        else:
            # Obstacle attempt: pay 1, remain at current position.
            actions.append("ATTEMPT_OBSTACLE")

        # Only after the action is the new local information revealed.
        reveal_cells(true_grid, revealed, position)
        trajectory.append(position)
        coverage_history.append(coverage(revealed, reachable))
        energy_history.append(energy)

    final_coverage = coverage_history[-1]
    remaining_fraction = energy / INITIAL_ENERGY if returned_successfully else 0.0
    ckek = final_coverage * remaining_fraction

    return SimulationResult(
        grid=true_grid,
        reachable_cells=reachable,
        revealed=revealed,
        trajectory=trajectory,
        coverage_over_time=coverage_history,
        energy_over_time=energy_history,
        actions=actions,
        reasons=reasons,
        returned_successfully=returned_successfully,
        final_coverage=final_coverage,
        remaining_energy=energy,
        ckek=ckek,
        generation_attempts=attempts,
    )


def save_logs(result: SimulationResult, out_dir: str = "logs") -> None:
    """Save trajectory and final metrics for the development run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "development_trajectory.csv").open("w", encoding="utf-8") as f:
        f.write("step,row,col,energy,coverage_pct,action,reason\n")
        for step, pos in enumerate(result.trajectory):
            action = "START" if step == 0 else result.actions[step - 1]
            reason = "Initial observation" if step == 0 else result.reasons[step - 1]
            f.write(
                f"{step},{pos[0]},{pos[1]},{result.energy_over_time[step]},"
                f"{result.coverage_over_time[step] * 100:.6f},{action},"
                f"{reason.replace(',', ';')}\n"
            )

    (out / "development_metrics.txt").write_text(
        "RoverMind — IR-01 development run\n"
        f"seed={SEED}\n"
        f"grid={GRID_SIZE}x{GRID_SIZE}\n"
        f"generation_attempts={result.generation_attempts}\n"
        f"reachable_free_cells={len(result.reachable_cells)}\n"
        f"steps={len(result.trajectory) - 1}\n"
        f"energy_used={INITIAL_ENERGY - result.remaining_energy}\n"
        f"remaining_energy={result.remaining_energy}\n"
        f"final_coverage_pct={result.final_coverage * 100:.6f}\n"
        f"returned_successfully={result.returned_successfully}\n"
        f"ckek={result.ckek:.6f}\n",
        encoding="utf-8",
    )


def plot_result(result: SimulationResult, save_path: str = "logs/exploration.png") -> None:
    """Save a matplotlib visualization of the grid, path, and base."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(result.grid, cmap="gray_r", origin="upper", interpolation="nearest")
    rows = [p[0] for p in result.trajectory]
    cols = [p[1] for p in result.trajectory]
    ax.plot(cols, rows, linewidth=1.6, label="Exploration path")
    ax.scatter(
        [BASE[1]], [BASE[0]], s=170, marker="s", edgecolors="black",
        linewidths=1.5, label="Base (0,0)", zorder=5,
    )
    ax.set_title(
        f"RoverMind IR-01 | Coverage {result.final_coverage * 100:.2f}% | "
        f"Return {result.returned_successfully}"
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(0, GRID_SIZE, 5))
    ax.set_yticks(range(0, GRID_SIZE, 5))
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def main() -> None:
    result = run_simulation()
    save_logs(result)
    plot_result(result)

    print("=" * 64)
    print("RoverMind — IR-01 Energy-Aware Robot Grid Exploration")
    print("=" * 64)
    print(f"Seed (NumPy PCG64):      {SEED}")
    print(f"Grid:                    {GRID_SIZE}x{GRID_SIZE}")
    print(f"Generation attempts:    {result.generation_attempts}")
    print(f"Reachable free cells:   {len(result.reachable_cells)}")
    print(f"Steps:                   {len(result.trajectory) - 1}")
    print(f"Energy used:             {INITIAL_ENERGY - result.remaining_energy}")
    print(f"Remaining energy:        {result.remaining_energy}")
    print(f"Final coverage:          {result.final_coverage * 100:.2f}%")
    print(f"Returned successfully:   {result.returned_successfully}")
    print(f"CkEk:                    {result.ckek:.6f}")
    print("Logs:                    logs/development_trajectory.csv")
    print("Metrics:                 logs/development_metrics.txt")
    print("Plot:                    logs/exploration.png")
    print("=" * 64)


if __name__ == "__main__":
    main()
