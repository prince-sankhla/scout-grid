"""IR-01: Energy-Aware Robot Grid Exploration (pure software simulation)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

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


def in_bounds(pos: Position) -> bool:
    r, c = pos
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE


def neighbors(pos: Position):
    r, c = pos
    for dr, dc in DIRECTIONS:
        nxt = (r + dr, c + dc)
        if in_bounds(nxt):
            yield nxt


def reachable_free_cells(grid: np.ndarray) -> Set[Position]:
    """True-world helper used only by the simulator/evaluator."""
    if grid[BASE] == OBSTACLE:
        return set()
    seen: Set[Position] = {BASE}
    q = deque([BASE])
    while q:
        cur = q.popleft()
        for nxt in neighbors(cur):
            if nxt not in seen and grid[nxt] == FREE:
                seen.add(nxt)
                q.append(nxt)
    return seen


def generate_grid(
    seed: int = SEED,
    obstacle_probability: float = OBSTACLE_PROBABILITY,
    reachability_threshold: float = REACHABILITY_THRESHOLD,
) -> Tuple[np.ndarray, Set[Position], int]:
    """Generate deterministically with PCG64 and regenerate until >=70% reachable."""
    rng = np.random.Generator(np.random.PCG64(seed))
    attempt = 0
    while True:
        attempt += 1
        grid = (rng.random((GRID_SIZE, GRID_SIZE)) < obstacle_probability).astype(np.int8)
        grid[BASE] = FREE
        reachable = reachable_free_cells(grid)
        free_count = int(np.count_nonzero(grid == FREE))
        if free_count and len(reachable) / free_count >= reachability_threshold:
            return grid, reachable, attempt


def shortest_known_path(
    revealed: np.ndarray, start: Position, goal: Position
) -> Optional[List[Position]]:
    """BFS over KNOWN-FREE cells only. No hidden grid is read here."""
    if not in_bounds(start) or not in_bounds(goal):
        return None
    if revealed[start] != FREE or revealed[goal] != FREE:
        return None

    q = deque([start])
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
        for nxt in neighbors(cur):
            if nxt in parent or revealed[nxt] != FREE:
                continue
            parent[nxt] = cur
            q.append(nxt)
    return None


def reveal(true_grid: np.ndarray, revealed: np.ndarray, center: Position) -> None:
    """Simulator observation: reveal TRUE occupancy within Manhattan distance 2."""
    cr, cc = center
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if abs(r - cr) + abs(c - cc) <= REVEAL_RADIUS:
                revealed[r, c] = true_grid[r, c]


class OnlineController:
    """Online controller. Its decision function sees only the revealed map."""

    def __init__(self, base: Position = BASE) -> None:
        self.base = base
        self.position = base
        self.previous_action: Optional[Position] = None
        self.action_history: List[Position] = []

    @staticmethod
    def frontier_cells(revealed: np.ndarray) -> List[Position]:
        """Known-free cells adjacent to at least one unknown cell."""
        frontier: List[Position] = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                pos = (r, c)
                if revealed[pos] == FREE and any(revealed[n] == UNKNOWN for n in neighbors(pos)):
                    frontier.append(pos)
        return frontier

    @staticmethod
    def unknown_neighbors(revealed: np.ndarray, pos: Position) -> List[Position]:
        return [n for n in neighbors(pos) if revealed[n] == UNKNOWN]

    def known_return_path(self, revealed: np.ndarray) -> Optional[List[Position]]:
        return shortest_known_path(revealed, self.position, self.base)

    def choose_frontier(self, revealed: np.ndarray) -> Optional[List[Position]]:
        """Return a shortest known-free path to the nearest reachable frontier."""
        q = deque([self.position])
        parent: Dict[Position, Optional[Position]] = {self.position: None}
        distance: Dict[Position, int] = {self.position: 0}

        while q:
            cur = q.popleft()
            for nxt in neighbors(cur):
                if nxt in parent or revealed[nxt] != FREE:
                    continue
                parent[nxt] = cur
                distance[nxt] = distance[cur] + 1
                q.append(nxt)

        frontier = [p for p in self.frontier_cells(revealed) if p in distance]
        if not frontier:
            return None

        target = min(frontier, key=lambda p: (distance[p], -len(self.unknown_neighbors(revealed, p)), p))
        path: List[Position] = []
        node: Optional[Position] = target
        while node is not None:
            path.append(node)
            node = parent[node]
        return path[::-1]

    def decide(self, revealed: np.ndarray, remaining_energy: int) -> str:
        """
        Return either MOVE or DONE. The next position for MOVE is stored in
        previous_action. STOP is intentionally not used: the official run-end
        conditions remain 'base + DONE' or energy == 0.
        """
        if revealed.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError("Expected a 30x30 revealed map.")
        if revealed[self.position] != FREE:
            raise RuntimeError("Controller position must be known free.")

        home_path = self.known_return_path(revealed)
        if home_path is None:
            raise RuntimeError("Controller lost its known route to base.")
        return_distance = len(home_path) - 1

        frontier_path = self.choose_frontier(revealed)

        # At base, DONE is safe whenever there is not enough energy to perform
        # one exploratory action and still guarantee a later return + DONE.
        if self.position == self.base and remaining_energy < 3:
            self.previous_action = None
            return "DONE"

        if frontier_path is None:
            # The known connected component is fully revealed. Return to base.
            if self.position == self.base:
                self.previous_action = None
                return "DONE"
            if remaining_energy < return_distance + 1:
                raise RuntimeError("Energy invariant broken before return move.")
            self.previous_action = home_path[1]
            self.action_history.append(self.previous_action)
            return "MOVE"

        # If already on the frontier, try one unknown cell only when we can
        # still get home and have one energy unit left to issue DONE afterwards.
        if self.position == frontier_path[-1]:
            unknown = self.unknown_neighbors(revealed, self.position)
            if unknown:
                # Worst case is an obstacle: position does not change. Best case
                # is free: returning through the old known route costs d+1.
                # To guarantee both return and a final DONE, reserve one extra
                # energy unit: E >= d + 3.
                if remaining_energy >= return_distance + 3:
                    self.previous_action = unknown[0]
                    self.action_history.append(self.previous_action)
                    return "MOVE"

                # Exploration is no longer safe, so turn back immediately.
                if self.position != self.base:
                    if remaining_energy < return_distance + 1:
                        raise RuntimeError("Insufficient energy for guaranteed return.")
                    self.previous_action = home_path[1]
                    self.action_history.append(self.previous_action)
                    return "MOVE"
                self.previous_action = None
                return "DONE"

        # Move one step through known-free cells toward the nearest frontier.
        if len(frontier_path) >= 2:
            nxt = frontier_path[1]
            next_home = shortest_known_path(revealed, nxt, self.base)
            if next_home is None:
                raise RuntimeError("Chosen known-free move would lose the known return path.")
            next_return_distance = len(next_home) - 1

            # After paying this 1-unit move, retain enough energy to return and
            # still have one unit available to output DONE at base.
            if remaining_energy >= next_return_distance + 2:
                self.previous_action = nxt
                self.action_history.append(nxt)
                return "MOVE"

        # Safety fallback: stop exploring and return along the known route.
        if self.position != self.base:
            if remaining_energy < return_distance + 1:
                raise RuntimeError("Insufficient energy for guaranteed return.")
            self.previous_action = home_path[1]
            self.action_history.append(self.previous_action)
            return "MOVE"

        self.previous_action = None
        return "DONE"


@dataclass
class SimulationResult:
    grid: np.ndarray
    reachable_cells: Set[Position]
    revealed: np.ndarray
    trajectory: List[Position]
    coverage_over_time: List[float]
    energy_over_time: List[int]
    actions: List[str]
    returned_successfully: bool
    final_coverage: float
    ckek: float
    generation_attempts: int


def coverage(revealed: np.ndarray, reachable: Set[Position]) -> float:
    if not reachable:
        return 1.0
    observed = sum(revealed[p] == FREE for p in reachable)
    return observed / len(reachable)


def run_simulation(seed: int = SEED, max_steps: int = 20_000) -> SimulationResult:
    """Environment loop. Only this function owns the hidden true grid."""
    grid, reachable, attempts = generate_grid(seed=seed)
    revealed = np.full((GRID_SIZE, GRID_SIZE), UNKNOWN, dtype=np.int8)
    controller = OnlineController()
    position = BASE
    energy = INITIAL_ENERGY

    reveal(grid, revealed, position)
    trajectory = [position]
    energy_over_time = [energy]
    coverage_over_time = [coverage(revealed, reachable)]
    actions: List[str] = []
    returned_successfully = False

    for _ in range(max_steps):
        if energy <= 0:
            break

        controller.position = position
        action = controller.decide(revealed.copy(), energy)
        if action == "DONE":
            returned_successfully = position == BASE
            actions.append("DONE")
            break
        if action != "MOVE" or controller.previous_action is None:
            raise RuntimeError(f"Unexpected controller action: {action}")

        target = controller.previous_action
        if target not in set(neighbors(position)):
            raise RuntimeError(f"Illegal requested move: {target}")

        energy -= 1
        if grid[target] == FREE:
            position = target
            actions.append("MOVE")
        else:
            # Obstacle attempt: spend 1 energy and stay in place.
            actions.append("ATTEMPT_OBSTACLE")

        reveal(grid, revealed, position)
        trajectory.append(position)
        energy_over_time.append(energy)
        coverage_over_time.append(coverage(revealed, reachable))

        if energy == 0:
            break

    final_coverage = coverage_over_time[-1]
    energy_fraction = energy / INITIAL_ENERGY if returned_successfully else 0.0
    ckek = final_coverage * energy_fraction

    return SimulationResult(
        grid=grid,
        reachable_cells=reachable,
        revealed=revealed,
        trajectory=trajectory,
        coverage_over_time=coverage_over_time,
        energy_over_time=energy_over_time,
        actions=actions,
        returned_successfully=returned_successfully,
        final_coverage=final_coverage,
        ckek=ckek,
        generation_attempts=attempts,
    )


def print_metrics(result: SimulationResult) -> None:
    print("=" * 62)
    print("IR-01: Energy-Aware Robot Grid Exploration")
    print("=" * 62)
    print(f"Seed (NumPy PCG64):       {SEED}")
    print(f"Grid:                    {GRID_SIZE}x{GRID_SIZE}")
    print(f"Generation attempts:     {result.generation_attempts}")
    print(f"Reachable free cells:    {len(result.reachable_cells)}")
    print(f"Actions executed:        {len(result.actions)}")
    print(f"Final remaining energy:  {result.energy_over_time[-1]}")
    print(f"Final coverage:          {result.final_coverage * 100:.2f}%")
    print(f"Returned successfully:   {result.returned_successfully}")
    print(f"CkEk:                    {result.ckek:.6f}")
    print("=" * 62)


def plot_result(result: SimulationResult) -> None:
    """Show the hidden evaluation grid, path, and base."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(result.grid, origin="upper", interpolation="nearest", cmap="gray_r")

    rows = [p[0] for p in result.trajectory]
    cols = [p[1] for p in result.trajectory]
    ax.plot(cols, rows, linewidth=1.5, label="Path")
    ax.scatter(cols[0], rows[0], s=130, marker="s", edgecolors="black", label="Base")

    ax.set_title(
        f"IR-01 | Coverage {result.final_coverage * 100:.2f}% | Return {result.returned_successfully}"
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(0, GRID_SIZE, 5))
    ax.set_yticks(range(0, GRID_SIZE, 5))
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    result = run_simulation()
    print_metrics(result)
    plot_result(result)
