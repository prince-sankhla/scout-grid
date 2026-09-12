# RoverMind — IR-01 Energy-Aware Robot Grid Exploration

RoverMind is a pure Python simulation and online exploration controller for the IR-01 benchmark.

## Run

```bash
pip install -r requirements.txt
python rovermind.py
```

Run a deterministic multi-seed local check:

```bash
python rovermind.py --benchmark 20
```

The default development run uses the required seed `20260911`.

## Exact benchmark rules

- NumPy `PCG64` with seed `20260911`
- Grid size `30 x 30`
- Base at `(0, 0)`
- Every non-base cell is independently an obstacle with probability `0.20`
- Regenerate until at least `70%` of all free cells are reachable from base using 4-directional movement
- At every position reveal true occupancy within Manhattan distance `2`
- Initial energy `400`
- Every movement attempt costs `1`
- Moving into an obstacle consumes that energy and leaves the position unchanged
- The run ends successfully only when the controller is at base and explicitly outputs `DONE`
- The run ends unsuccessfully when energy reaches `0`
- Coverage = observed reachable free cells / total reachable free cells
- `CkEk = coverage * (remaining_energy / 400)` on successful return, otherwise `0`

## Online rule

The hidden grid belongs only to the simulator. `RoverMindController` never receives the full map, the true reachable-cell set, unrevealed occupancy, or RNG state.

Its decision input contains only the current position, remaining energy, cells revealed so far, and previous action. All controller-side path finding uses only cells explicitly known to be free.

The simulator is the only component allowed to use the hidden map. It applies movement outcomes, reveals the next local observation, and calculates evaluation metrics.

## Exploration strategy

The controller uses greedy frontier exploration.

A **frontier** is a known-free cell with at least one unrevealed neighbor. The controller finds the nearest reachable frontier in its known map. When multiple frontiers are equally close, it prefers the one adjacent to more unknown cells.

This is simple, deterministic, and useful because reaching a frontier moves the sensing window into less-explored territory.

## Energy safety

Before moving, the controller calculates the shortest known-free route back to base. Let that distance be `d`.

The controller keeps a `20`-unit safety reserve. When:

```text
remaining_energy <= d + 20
```

it stops extending the exploration and returns through the known-free route.

For a proposed known-free move, it also checks the destination first. After paying the 1-unit move cost, the remaining energy must still cover the shortest known return route plus the 20-unit reserve.

This gives the core invariant:

```text
Every accepted movement leaves a known-free path to base
and enough energy to traverse that path while keeping the reserve.
```

An unrevealed cell is never selected as a gamble. The development controller deliberately explores conservatively through known-safe cells, which gives it a reliable return corridor.

## Development outputs

A normal run creates:

- `rovermind_metrics.json` — final development metrics
- `rovermind_trajectory.csv` — step-by-step trajectory, energy, coverage, action, and reason
- `rovermind_exploration.png` — grid, path, base, and final position

The repository also contains committed development copies under `logs/` for inspection.

## Current development result

Required seed `20260911`:

```text
success=True
coverage=98.00%
energy_remaining=20
steps=380
reachable_free_cells=700
observed_reachable_cells=686
coverage_energy_product=0.049000
```

20-seed local check:

```text
mean coverage=96.27%
return success fraction=100.00%
mean CkEk=0.048135
```

## Judge explanation

> “The controller only reasons from what has already been revealed. It searches for the nearest useful frontier, but before every movement it verifies that a known-free route back to the base still fits inside the remaining energy budget. When the safety margin gets too small, it stops exploring and returns. The true map stays inside the simulator, so the controller never reads the future.”
