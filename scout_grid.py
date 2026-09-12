"""
IR-01: Energy-Aware Robot Grid Exploration

Pure software / simulation implementation.

The controller is online: it only receives the cells revealed by the simulator,
its current position, previous actions, and remaining energy. It never receives
or reads the hidden true grid.

Run:
    python scout_grid.py

Requirements:
    numpy
    matplotlib
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Problem constants
# -----------------------------
SEED = 20260911
GRID_SIZE = 30
OBSTACLE_PROBABILITY = 0.20
REACHABILITY_THRESHOLD = 0.70
INITIAL_ENERGY = 400
REVEAL_MANHATTAN_RADIUS = 2
BASE: Tuple[int, int] = (0, 0)

FREE = 0
OBSTACLE = 1
UNKNOWN = -1

Position = Tuple[int, int]
Action = Tuple[int, int]

# 4-directional movement only.
DIRECTIONS: Tuple[Action, ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


# -----------------------------
# Grid generation and true-world helpers
# -----------------------------
def _in_bounds(position: Position) -> bool:
    r, c = position
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE


def _neighbors(position: Position):
    r, c = position
    for dr, dc in DIRECTIONS:
        nxt = (r + dr, c + dc)
        if _in_bounds(nxt):
            yield nxt


def _reachable_free_cells(grid: np.ndarray, start: Position = BASE) -> set[Position]:
    """Return all free cells reachable from start using 4-directional movement."""
    if grid[start] == OBSTACLE:
        return set()

    visited: set[Position] = {start}
    queue: Deque[Position] = deque([start])

    while queue:
        current = queue.popleft()
        for nxt in _neighbors(current):
            if nxt in visited:
                continue
            if grid[nxt] != FREE:
                continue
            visited.add(nxt)
            queue.append(nxt)

    return visited


def generate_grid(
    seed: int = SEED,
    size: int = GRID_SIZE,
    obstacle_probability: float = OBSTACLE_PROBABILITY,
    reachability_threshold: float = REACHABILITY_THRESHOLD,
) -> Tuple[np.ndarray, set[Position], int]:
    """
    Generate the exact problem grid procedure.

    - NumPy PCG64 RNG with the requested seed.
    - Each non-base cell is independently an obstacle with probability 0.20.
    - Regenerate until at least 70% of free non-obstacle cells are reachable
      from the base.

    The same RNG stream is used across regeneration attempts, which makes the
    full procedure deterministic for the given seed.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    base = BASE

    attempt = 0
    while True:
        attempt += 1

        # False = free, True = obstacle.
        grid = (rng.random((size, size)) < obstacle_probability).astype(np.int8)
        grid[base] = FREE

        reachable = _reachable_free_cells(grid, base)
        free_count = int(np.count_nonzero(grid == FREE))

        if free_count == 0:
            continue

        reachable_fraction = len(reachable) / free_count
        if reachable_fraction >= reachability_threshold:
            return grid, reachable, attempt


def calculate_revealed_coverage(
    revealed: np.ndarray,
    reachable_cells: set[Position],
) -> float:
    """Coverage = reachable free cells observed so far / total reachable free cells."""
    observed_reachable = sum(
        revealed[position] == FREE for position in reachable_cells
    )
    if not reachable_cells:
        return 1.0
    return observed_reachable / len(reachable_cells)


# -----------------------------
# Online controller
# -----------------------------
@dataclass
class OnlineController:
    """
    Genuinely online exploration controller.

    The controller only stores and uses the revealed map. UNKNOWN cells remain
    hidden from its decision logic until the simulator reveals them.
    """

    base: Position = BASE
    energy_capacity: int = INITIAL_ENERGY
    position: Position = BASE
    done: bool = False
    previous_action: Optional[Position] = None
    action_history: List[Position] = field(default_factory=list)
    energy_history: List[int] = field(default_factory=list)

    def reset(self, start: Position = BASE) -> None:
        self.position = start
        self.done = False
        self.previous_action = None
        self.action_history.clear()
        self.energy_history.clear()

    @staticmethod
    def _shortest_known_free_path(
        revealed: np.ndarray,
        start: Position,
        goal: Position,
    ) -> Optional[List[Position]]:
        """
        BFS using ONLY cells currently known to be free.

        The returned path includes start and goal. None means no known-safe
        path exists.
        """
        if not _in_bounds(start) or not _in_bounds(goal):
            return None
        if revealed[start] != FREE or revealed[goal] != FREE:
            return None

        queue: Deque[Position] = deque([start])
        parent: Dict[Position, Optional[Position]] = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal:
                path: List[Position] = []
                node: Optional[Position] = goal
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path

            for nxt in _neighbors(current):
                if nxt in parent:
                    continue
                if revealed[nxt] != FREE:
                    continue
                parent[nxt] = current
                queue.append(nxt)

        return None

    def known_return_distance(self, revealed: np.ndarray) -> Optional[int]:
        """Shortest return distance through cells already known to be free."""
        path = self._shortest_known_free_path(revealed, self.position, self.base)
        if path is None:
            return None
        return len(path) - 1

    @staticmethod
    def _frontier_cells(revealed: np.ndarray) -> List[Position]:
        """
        A frontier is a KNOWN-FREE cell with at least one UNKNOWN neighbor.
        """
        frontier: List[Position] = []
        rows, cols = revealed.shape
        for r in range(rows):
            for c in range(cols):
                if revealed[r, c] != FREE:
                    continue
                pos = (r, c)
                if any(revealed[nxt] == UNKNOWN for nxt in _neighbors(pos)):
                    frontier.append(pos)
        return frontier

    @staticmethod
    def _unknown_neighbors(revealed: np.ndarray, position: Position) -> List[Position]:
        return [nxt for nxt in _neighbors(position) if revealed[nxt] == UNKNOWN]

    def _best_frontier_target(
        self,
        revealed: np.ndarray,
    ) -> Optional[Tuple[Position, List[Position]]]:
        """
        Find the nearest frontier using only the known-free map.

        Tie-breaks prefer the frontier with more unknown adjacent cells,
        because reaching it is likely to reveal more new information.
        """
        current_to_everywhere: Dict[Position, Tuple[int, Optional[Position]]] = {}
        queue: Deque[Position] = deque([self.position])
        current_to_everywhere[self.position] = (0, None)

        while queue:
            current = queue.popleft()
            for nxt in _neighbors(current):
                if nxt in current_to_everywhere:
                    continue
                if revealed[nxt] != FREE:
                    continue
                current_to_everywhere[nxt] = (
                    current_to_everywhere[current][0] + 1,
                    current,
                )
                queue.append(nxt)

        frontier = [
            p for p in self._frontier_cells(revealed) if p in current_to_everywhere
        ]
        if not frontier:
            return None

        target = min(
            frontier,
            key=lambda p: (
                current_to_everywhere[p][0],
                -len(self._unknown_neighbors(revealed, p)),
                p[0],
                p[1],
            ),
        )

        # Reconstruct the known-safe path to the chosen frontier.
        path = [target]
        node = target
        while node != self.position:
            parent = current_to_everywhere[node][1]
            if parent is None:
                return None
            path.append(parent)
            node = parent
        path.reverse()
        return target, path

    def _choose_unknown_move(
        self,
        revealed: np.ndarray,
    ) -> Optional[Position]:
        """
        Choose an adjacent UNKNOWN cell from the current frontier.

        No hidden information is used. The fixed direction order keeps
        behavior deterministic.
        """
        unknowns = self._unknown_neighbors(revealed, self.position)
        return unknowns[0] if unknowns else None

    def decide(self, revealed: np.ndarray, remaining_energy: int) -> str:
        """
        Select the next action from the currently revealed map.

        Returns:
            "MOVE" = take one 4-directional step specified by target in
                     previous_action (the simulator validates the target), or
            "DONE" = exploration is complete and robot is at base.

        Safety rule:
        If the controller is not at base, it computes the shortest path back
        to base through KNOWN-FREE cells. It must have enough energy to survive
        one more action and still retain a known route home.

        In particular, if we try an UNKNOWN cell it might be an obstacle and
        the position would not change. Therefore exploratory moves are allowed
        only when:

            remaining_energy >= known_return_distance + 2

        After one energy unit is spent, the controller still has enough energy
        for the known return route even in the worst case where the attempted
        destination is an obstacle.
        """
        if self.done:
            return "DONE"

        if revealed.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError("Revealed map must be a 30x30 array.")

        if revealed[self.position] != FREE:
            raise RuntimeError("Controller position must be a revealed free cell.")

        # If at base, stop as soon as no reachable known frontier remains.
        if self.position == self.base:
            frontier = self._frontier_cells(revealed)
            if not frontier:
                self.done = True
                return "DONE"

        return_distance = self.known_return_distance(revealed)
        if return_distance is None:
            # This should not occur because the route home is always retained.
            # Treat it as an emergency rather than inventing a hidden route.
            if self.position == self.base:
                self.done = True
                return "DONE"
            raise RuntimeError("No known-safe route back to base.")

        # Already at base: exploration is pending, so pick a nearby frontier.
        # The same safety check below applies to movement.
        target_info = self._best_frontier_target(revealed)

        if target_info is None:
            # No known frontier means there is nothing left to reveal in the
            # known connected component. Return to base before DONE.
            if self.position != self.base:
                if remaining_energy < return_distance:
                    # At this point, no safe action can be invented from the
                    # revealed map; flag failure rather than peeking ahead.
                    return "STOP"
                path_home = self._shortest_known_free_path(
                    revealed, self.position, self.base
                )
                assert path_home is not None
                next_pos = path_home[1]
                self.previous_action = next_pos
                self.action_history.append(next_pos)
                return "MOVE"

            self.done = True
            return "DONE"

        _, path_to_frontier = target_info

        # If we are already on a frontier, first inspect one unknown adjacent
        # cell. Otherwise walk one step along known-safe cells toward frontier.
        if self.position == path_to_frontier[-1]:
            unknown_target = self._choose_unknown_move(revealed)
            if unknown_target is not None:
                # Worst case: the unknown target is an obstacle and we stay
                # where we are. We still need enough energy to get home.
                if remaining_energy >= return_distance + 2:
                    self.previous_action = unknown_target
                    self.action_history.append(unknown_target)
                    return "MOVE"

                # Not enough energy to safely explore: return home now.
                if remaining_energy >= return_distance and self.position != self.base:
                    path_home = self._shortest_known_free_path(
                        revealed, self.position, self.base
                    )
                    assert path_home is not None
                    next_pos = path_home[1]
                    self.previous_action = next_pos
                    self.action_history.append(next_pos)
                    return "MOVE"

                if self.position == self.base:
                    # At base with too little energy, we can safely finish only
                    # if no frontier remains. Otherwise continuing would violate
                    # the return guarantee, so stop with the safe failure state.
                    return "STOP"

                return "STOP"

        # Travel through cells already known to be free.
        if len(path_to_frontier) >= 2:
            next_pos = path_to_frontier[1]
            # A known-free move is also required to preserve a route home.
            if remaining_energy - 1 >= self._known_return_distance_after_known_move(
                revealed, next_pos
            ):
                self.previous_action = next_pos
                self.action_history.append(next_pos)
                return "MOVE"

        # Could not safely advance toward exploration. Return home.
        if self.position != self.base and remaining_energy >= return_distance:
            path_home = self._shortest_known_free_path(
                revealed, self.position, self.base
            )
            assert path_home is not None
            if len(path_home) >= 2:
                next_pos = path_home[1]
                self.previous_action = next_pos
                self.action_history.append(next_pos)
                return "MOVE"

        if self.position == self.base and not self._frontier_cells(revealed):
            self.done = True
            return "DONE"

        return "STOP"

    def _known_return_distance_after_known_move(
        self,
        revealed: np.ndarray,
        next_position: Position,
    ) -> int:
        path = self._shortest_known_free_path(revealed, next_position, self.base)
        if path is None:
            # The caller must not move into a known cell if it loses the known
            # route home.
            return 10**9
        return len(path) - 1


# -----------------------------
# Local simulator
# -----------------------------
@dataclass
class SimulationResult:
    grid: np.ndarray
    reachable_cells: set[Position]
    revealed: np.ndarray
    trajectory: List[Position]
    coverage_over_time: List[float]
    energy_over_time: List[int]
    actions: List[str]
    returned_successfully: bool
    final_coverage: float
    ckek: float
    steps: int
    generation_attempts: int


def reveal_cells(
    true_grid: np.ndarray,
    revealed: np.ndarray,
    center: Position,
    radius: int = REVEAL_MANHATTAN_RADIUS,
) -> None:
    """Reveal TRUE occupancy within Manhattan distance radius of center."""
    cr, cc = center
    for r in range(true_grid.shape[0]):
        for c in range(true_grid.shape[1]):
            if abs(r - cr) + abs(c - cc) <= radius:
                revealed[r, c] = true_grid[r, c]


def run_simulation(
    max_steps: int = 20_000,
    seed: int = SEED,
) -> SimulationResult:
    """Run the online controller against the hidden generated grid."""
    grid, reachable_cells, generation_attempts = generate_grid(seed=seed)

    revealed = np.full_like(grid, UNKNOWN)
    controller = OnlineController()
    controller.reset(BASE)

    position = BASE
    energy = INITIAL_ENERGY

    # Initial observation at base.
    reveal_cells(grid, revealed, position)

    trajectory: List[Position] = [position]
    coverage_over_time: List[float] = [
        calculate_revealed_coverage(revealed, reachable_cells)
    ]
    energy_over_time: List[int] = [energy]
    actions: List[str] = []

    returned_successfully = False

    for _ in range(max_steps):
        if energy <= 0:
            break

        action = controller.decide(revealed.copy(), energy)

        if action == "DONE":
            returned_successfully = position == BASE
            actions.append("DONE")
            break

        if action == "STOP":
            # A controller using only current information could not find a safe
            # continuation; do not invent an action.
            actions.append("STOP")
            break

        if action != "MOVE" or controller.previous_action is None:
            raise RuntimeError(f"Invalid controller action: {action}")

        target = controller.previous_action
        if target not in set(_neighbors(position)):
            raise RuntimeError(f"Controller requested illegal move: {target}")

        # One action always costs exactly one energy unit.
        energy -= 1

        if grid[target] == OBSTACLE:
            # Attempted obstacle: stay put; obstacle is only learned now if it
            # was not already revealed.
            actions.append("ATTEMPT_OBSTACLE")
        else:
            position = target
            actions.append("MOVE")

        # The simulator reveals the observation only after the action occurs.
        reveal_cells(grid, revealed, position)

        # Keep controller position synchronized with the observed result.
        controller.position = position
        controller.energy_history.append(energy)

        trajectory.append(position)
        coverage_over_time.append(
            calculate_revealed_coverage(revealed, reachable_cells)
        )
        energy_over_time.append(energy)

        # The simulator should stop immediately at zero energy.
        if energy == 0:
            break

        # If controller has returned to base it may later output DONE after it
        # verifies there is no frontier left.

    if position == BASE and actions and actions[-1] == "DONE":
        returned_successfully = True

    final_coverage = coverage_over_time[-1]
    remaining_energy_fraction = (energy / INITIAL_ENERGY) if returned_successfully else 0.0
    ckek = final_coverage * remaining_energy_fraction

    return SimulationResult(
        grid=grid,
        reachable_cells=reachable_cells,
        revealed=revealed,
        trajectory=trajectory,
        coverage_over_time=coverage_over_time,
        energy_over_time=energy_over_time,
        actions=actions,
        returned_successfully=returned_successfully,
        final_coverage=final_coverage,
        ckek=ckek,
        steps=len(actions),
        generation_attempts=generation_attempts,
    )


# -----------------------------
# Reporting + visualization
# -----------------------------
def print_metrics(result: SimulationResult) -> None:
    print("=" * 60)
    print("IR-01: Energy-Aware Robot Grid Exploration")
    print("=" * 60)
    print(f"Grid size:                 {GRID_SIZE}x{GRID_SIZE}")
    print(f"Seed (PCG64):              {SEED}")
    print(f"Generation attempts:       {result.generation_attempts}")
    print(f"Reachable free cells:      {len(result.reachable_cells)}")
    print(f"Steps/actions executed:    {result.steps}")
    print(f"Final energy:              {result.energy_over_time[-1]}")
    print(f"Final coverage:            {result.final_coverage * 100:.2f}%")
    print(f"Returned successfully:     {result.returned_successfully}")
    print(f"CkEk:                      {result.ckek:.6f}")
    print("=" * 60)


def plot_result(result: SimulationResult) -> None:
    """Show the true grid, trajectory, and base marker for evaluation."""
    fig, ax = plt.subplots(figsize=(9, 9))

    # Obstacles are shown as 1, free cells as 0.
    ax.imshow(result.grid, origin="upper", interpolation="nearest", cmap="gray_r")

    if result.trajectory:
        rows = [p[0] for p in result.trajectory]
        cols = [p[1] for p in result.trajectory]
        ax.plot(cols, rows, linewidth=1.5, label="Path")
        ax.scatter(
            cols[0], rows[0], s=120, marker="s", edgecolors="black", label="Base"
        )

    ax.set_title(
        f"IR-01 Exploration | Coverage {result.final_coverage * 100:.2f}% | "
        f"Return {result.returned_successfully}"
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(0, GRID_SIZE, 5))
    ax.set_yticks(range(0, GRID_SIZE, 5))
    ax.grid(False)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    result = run_simulation()
    print_metrics(result)
    plot_result(result)
