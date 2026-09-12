# RoverMind — IR-01 Energy-Aware Robot Grid Exploration

RoverMind is a pure Python simulation and online exploration controller for the IR-01 benchmark.

## Run

```bash
pip install -r requirements.txt
python rovermind.py
```

For a local multi-seed check:

```bash
python rovermind.py --benchmark 20
```

The default run uses the required seed `20260911`.

## Exact rules implemented

- NumPy `PCG64` with seed `20260911`
- `30 x 30` grid
- Base at `(0, 0)`
- Each non-base cell has obstacle probability `0.20`
- The grid is regenerated until at least `70%` of free cells are reachable from the base using 4-directional movement
- At every position, the simulator reveals true occupancy inside Manhattan distance `2`
- Initial energy is `400`
- Every movement attempt costs `1` energy
- An obstacle attempt does not change the position
- Success requires the controller to be at the base and explicitly output `DONE`
- Energy reaching `0` ends the run unsuccessfully
- Coverage is observed reachable free cells divided by total reachable free cells
- `CkEk = coverage * (remaining_energy / 400)` on a successful return; otherwise it is `0`

## How the online rule is protected

The simulator owns the true map. `RoverMindController` never receives that map.

The controller receives only an `Observation` containing:

- current position
- remaining energy
- the cells revealed so far
- previous action

All path finding inside the controller runs on cells that are explicitly known to be free. Unknown cells are never inspected for hidden occupancy.

The true map is used only by the simulator to apply the requested move, reveal the next local observation, and calculate evaluation metrics.

## Exploration strategy

The controller uses a **frontier strategy**.

A frontier is a cell that is already known to be free and has at least one unrevealed neighbor. The controller finds the nearest reachable frontier using only its known map. If several frontiers are equally close, it prefers the one touching more unknown cells.

This is simple, deterministic, and useful because moving toward a frontier brings the sensing area into less-explored parts of the grid.

## Energy safety: when does it turn back?

Before going farther, the controller calculates the shortest route from its current position to the base using only known-free cells. Call that distance `d`.

The controller keeps a safety reserve of `20` energy units. When:

```text
remaining_energy <= d + 20
```

it stops trying to extend the exploration and switches to return mode.

For a normal known-free step toward a frontier, it also checks the destination first. If the proposed destination would leave too little energy to reach the base with the reserve still available, that step is rejected and the controller turns back instead.

This creates a simple safety guarantee:

```text
After every accepted movement,
there is still a known-free route to base,
and enough energy to traverse that route while keeping the reserve.
```

The controller normally moves only through cells already known to be free, so its current path remains a reliable fallback corridor. An unrevealed cell is never entered just to gamble on its occupancy.

## Development outputs

Running the program creates:

- `rovermind_metrics.json` — final development metrics
- `rovermind_trajectory.csv` — step-by-step position, energy, coverage, action, and decision reason
- `rovermind_exploration.png` — development visualization of the grid, path, base, and final position

The visualization shows the hidden development map so the run can be inspected. That map is not passed into the controller.

## Judge explanation

> “The controller only reasons from what has already been revealed. It moves toward the nearest useful frontier, but before every move it checks the known shortest route back to the base. When the remaining energy gets too close to that return cost, it stops exploring and comes back. The simulator keeps the true map separate, so the controller never reads the future.”
