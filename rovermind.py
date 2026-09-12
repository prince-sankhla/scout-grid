"""RoverMind — IR-01 Energy-Aware Robot Grid Exploration.

Pure software simulation.

The environment owns the hidden true grid. The OnlineExplorer controller only
receives the currently revealed map, current position, previous action history,
and remaining energy. It never receives the hidden true grid.

Run:
    python rovermind.py
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


# True-world helpers belong only to the environment/evaluator.
def reachable_free_cells(grid: np.ndarray) -> set[Position]:
    if grid[BASE] == OBSTACLE:
        return set()
    seen: set[Position] = {BASE}
    q: deque[Position] = deque([BASE])
    while q:
        cur = q.popleft()
        for nxt in neighbors(cur):
            if nxt not in seen and grid[nxt] == FREE:
                seen.add(nxt)
                q.append(nxt)
    return seen


def generate_grid(
    seed: int = SEED,
    size: int = GRID_SIZE,
    obstacle_probability: float = OBSTACLE_PROBABILITY,
    reachability_threshold: float = REACHABILITY_THRESHOLD,
) -> Tuple[np.ndarray, set[Position], int]:
    """Generate the deterministic problem map with NumPy PCG64."""
    rng = np.random.Generator(np.random.PCG64(seed))
    attempts = 0
    while True:
        attempts += 1
        grid = (rng.random((size, size)) < obstacle_probability).astype(np.int8)
        grid[BASE] = FREE
        reachable = reachable_free_cells(grid)
        free_count = int(np.count_nonzero(grid == FREE))
        if free_count and len(reachable) / free_count >= reachability_threshold:
            return grid, reachable, attempts


def reveal_cells(
    true_grid: np.ndarray,
    revealed: np.ndarray,
    center: Position,
    radius: int = REVEAL_RADIUS,
) -> None:
    """Environment observation: reveal true occupancy within Manhattan radius."""
    cr, cc = center
    size = true_grid.shape[0]
    for r in range(max(0, cr - radius), min(size, cr + radius + 1)):
        remaining = radius - abs(r - cr)
        c0 = max(0, cc - remaining)
        c1 = min(size, cc + remaining + 1)
        revealed[r, c0:c1] = true_grid[r, c0:c1]


def coverage(revealed: np.ndarray, reachable: set[Position]) -> float:
    if not reachable:
        return 1.0
    observed = sum(revealed[p] == FREE for p in reachable)
    return observed / len(reachable)


# Online-only path utility.
def shortest_known_free_path(
    revealed: np.ndarray,
    start: Position,
    goal: Position,
) -> Optional[List[Position]]:
    """BFS using only cells explicitly known to be free."""
    if revealed[start] != FREE or revealed[goal] != FREE:
        return None
    q: deque[Position] = deque([start])
    parent: Dict[Position, Optional[Position]] = {start: None}
    while q:
        cur = q.popleft()
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
            q.append(nxt)
    return None


@dataclass
class Decision:
    action: str
    target: Optional[Position] = None
    reason: str = ""


@dataclass
class OnlineExplorer:
    """Greedy controller that receives only the revealed map and state."""
    base: Position = BASE
    position: Position = BASE
    action_history: List[str] = field(default_factory=list)
    last_reason: str = ""

    @staticmethod
    def unknown_neighbors(revealed: np.ndarray, pos: Position) -> List[Position]:
        return [n for n in neighbors(pos, revealed.shape[0]) if revealed[n] == UNKNOWN]

    def known_return_path(self, revealed: np.ndarray) -> Optional[List[Position]]:
        return shortest_known_free_path(revealed, self.position, self.base)

    def choose_nearest_frontier_path(self, revealed: np.ndarray) -> Optional[List[Position]]:
        """Nearest frontier found with one BFS over known-free cells only."""
        size = revealed.shape[0]
        q: deque[Position] = deque([self.position])
        parent: Dict[Position, Optional[Position]] = {self.position: None}
        distance: Dict[Position, int] = {self.position: 0}
        candidates: List[Position] = []
        best_distance: Optional[int] = None

        while q:
            cur = q.popleft()
            d = distance[cur]
            if best_distance is not None and d > best_distance:
                break
            if any(revealed[n] == UNKNOWN for n in neighbors(cur, size)):
                if best_distance is None:
                    best_distance = d
                candidates.append(cur)
                continue
            for nxt in neighbors(cur, size):
                if nxt in parent or revealed[nxt] != FREE:
                    continue
                parent[nxt] = cur
                distance[nxt] = d + 1
                q.append(nxt)

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
        """Choose one action from current revealed information only.

        Safety invariant:
        Let d be the shortest known-free distance to base.
        - Unknown exploration is allowed only when E >= d + 2.
        - A known-free move is allowed only when the post-move return route
          still fits in E-1 and leaves one unit at base for the final DONE turn.
        - Otherwise the controller turns back along the known-free route.
        """
        if revealed.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError("Controller expects a 30x30 revealed map.")
        if revealed[self.position] != FREE:
            raise RuntimeError("Current position is not known to be free.")

        home_path = self.known_return_path(revealed)
        if home_path is None:
            raise RuntimeError("No known-safe route to base: invariant broken.")
        return_distance = len(home_path) - 1
        frontier_path = self.choose_nearest_frontier_path(revealed)

        # At base, finish only when there is no frontier. Otherwise probe a
        # nearby unknown cell when the d+2 safety condition allows it.
        if self.position == self.base:
            if frontier_path is None:
                self.last_reason = "No frontier remains; output DONE."
                self.action_history.append("DONE")
                return Decision("DONE", reason=self.last_reason)
            unknown = self.unknown_neighbors(revealed, self.position)
            if unknown and remaining_energy >= return_distance + 2:
                target = unknown[0]
                self.last_reason = "Probe unknown neighbor from base."
                self.action_history.append("MOVE")
                return Decision("MOVE", target, self.last_reason)

        # At a frontier, explore one unknown neighbor if safe.
        if frontier_path is not None and self.position == frontier_path[-1]:
            unknown = self.unknown_neighbors(revealed, self.position)
            if unknown and remaining_energy >= return_distance + 2:
                target = unknown[0]
                self.last_reason = (
                    f"Explore unknown: E={remaining_energy}, return_distance={return_distance}."
                )
                self.action_history.append("MOVE")
                return Decision("MOVE", target, self.last_reason)

        # Move through known-free cells toward the nearest frontier only if the
        # destination still has a known-safe return route with one unit reserved.
        if frontier_path is not None and len(frontier_path) >= 2:
            target = frontier_path[1]
            next_home = shortest_known_free_path(revealed, target, self.base)
            if next_home is None:
                raise RuntimeError("Chosen move breaks the known return route.")
            next_return_distance = len(next_home) - 1
            if remaining_energy - 1 >= next_return_distance + 1:
                self.last_reason = (
                    f"Move toward frontier; post-move return distance={next_return_distance}."
                )
                self.action_history.append("MOVE")
                return Decision("MOVE", target, self.last_reason)

        # Turn back before it becomes impossible to reach base with one unit left.
        if self.position != self.base:
            if remaining_energy >= return_distance + 1:
                target = home_path[1]
                self.last_reason = (
                    f"TURN BACK: E={remaining_energy} < exploration threshold "
                    f"{return_distance + 2}."
                )
                self.action_history.append("MOVE")
                return Decision("MOVE", target, self.last_reason)
            raise RuntimeError("Energy insufficient for guaranteed return with reserve.")

        # At base, no safe exploration remains.
        self.last_reason = "At base; output DONE."
        self.action_history.append("DONE")
        return Decision("DONE", reason=self.last_reason)


@dataclass
class SimulationResult:
    grid: np.ndarray
    reachable_cells: set[Position]
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
    """Local environment loop; only this function owns the hidden grid."""
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
        if energy <= 0:
            break
        controller.position = position

        # IMPORTANT: controller sees only this revealed copy, never true_grid.
        decision = controller.decide(revealed.copy(), energy)
        reasons.append(decision.reason)

        if decision.action == "DONE":
            returned_successfully = position == BASE
            actions.append("DONE")
            break

        if decision.action != "MOVE" or decision.target is None:
            raise RuntimeError(f"Illegal controller decision: {decision}")

        target = decision.target
        if target not in set(neighbors(position)):
            raise RuntimeError(f"Illegal movement target: {target}")

        # Every movement attempt costs exactly one energy unit.
        energy -= 1
        if true_grid[target] == FREE:
            position = target
            actions.append("MOVE")
        else:
            # Obstacle attempt: pay 1, position stays unchanged.
            actions.append("ATTEMPT_OBSTACLE")

        # New local observation becomes available only after the action.
        reveal_cells(true_grid, revealed, position)
        trajectory.append(position)
        coverage_history.append(coverage(revealed, reachable))
        energy_history.append(energy)

        if energy == 0:
            break

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
    """Write development trajectory and final metrics."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "development_trajectory.csv").open("w", encoding="utf-8") as f:
        f.write("step,row,col,energy,coverage_pct,action,reason\n")
        for i, pos in enumerate(result.trajectory):
            action = "START" if i == 0 else result.actions[i - 1]
            reason = "Initial observation" if i == 0 else result.reasons[i - 1]
            f.write(
                f"{i},{pos[0]},{pos[1]},{result.energy_over_time[i]},"
                f"{result.coverage_over_time[i] * 100:.6f},{action},"
                f"{reason.replace(',', ';')}\n"
            )

    (out / "development_metrics.txt").write_text(
        "IR-01 development run\n"
        f"seed={SEED}\n"
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
    """Plot the development grid, path, and base."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(result.grid, cmap="gray_r", origin="upper", interpolation="nearest")
    rows = [p[0] for p in result.trajectory]
    cols = [p[1] for p in result.trajectory]
    ax.plot(cols, rows, linewidth=1.6, label="Exploration path")
    ax.scatter([BASE[1]], [BASE[0]], s=170, marker="s", edgecolors="black",
               linewidths=1.5, label="Base (0,0)", zorder=5)
    ax.set_title(
        f"RoverMind IR-01 | Coverage {result.final_coverage * 100:.2f}% | "
        f"Return {result.returned_successfully}"
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(0, GRID_SIZE, 5))
    ax.set_yticks(range(0, GRID_SIZE, 5))
    ax.grid(False)
    ax.legend(loc="upper right")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()


def print_metrics(result: SimulationResult) -> None:
    print("=" * 68)
    print("RoverMind — IR-01 Energy-Aware Robot Grid Exploration")
    print("=" * 68)
    print(f"Seed (NumPy PCG64):        {SEED}")
    print(f"Grid:                     {GRID_SIZE}x{GRID_SIZE}")
    print(f"Generation attempts:      {result.generation_attempts}")
    print(f"Reachable free cells:     {len(result.reachable_cells)}")
    print(f"Steps / attempts:         {len(result.trajectory) - 1}")
    print(f"Energy used:              {INITIAL_ENERGY - result.remaining_energy}")
    print(f"Remaining energy:         {result.remaining_energy}")
    print(f"Final coverage:           {result.final_coverage * 100:.2f}%")
    print(f"Returned successfully:    {result.returned_successfully}")
    print(f"CkEk:                     {result.ckek:.6f}")
    print("Logs:                     logs/development_trajectory.csv")
    print("Metrics:                  logs/development_metrics.txt")
    print("Plot:                     logs/exploration.png")
    print("=" * 68)


if __name__ == "__main__":
    result = run_simulation()
    print_metrics(result)
    save_logs(result)
    plot_result(result)
