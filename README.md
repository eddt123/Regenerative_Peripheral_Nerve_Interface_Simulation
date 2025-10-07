# Regenerative Peripheral Nerve Interface (RPNI) Simulation

This repository hosts tooling to evaluate optimisation strategies for a COMSOL-based model of a regenerative peripheral nerve interface. The core entry point is `RPNI_simulation.py`, which orchestrates multiple optimisation algorithms and records their performance when driving a 12-channel electrode array wrapped around the nerve interface model stored in `12-ch 3D RPNI sim.mph`.

## Prerequisites

### Software
- **Python 3.9+** (the scripts rely on `pathlib`, `numpy`, and the `mph` client library).
- **COMSOL Multiphysics® with LiveLink™ for MATLAB®/Java API** capable of being launched through the `mph` Python package. The simulation has been tested with COMSOL Multiphysics 6.x running in server mode.
- **COMSOL Server access credentials** with permission to load and solve the `12-ch 3D RPNI sim.mph` model file.

### Python dependencies
Install dependencies into your environment:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` is not provided, ensure at minimum that `numpy`, `scipy`, `pandas`, and `mph` are installed:

```bash
pip install numpy scipy pandas mph
```

## Repository structure

- `RPNI_simulation.py` – Main orchestration script that launches multiple optimisers against the COMSOL model.
- `12-ch 3D RPNI sim.mph` – COMSOL model file referenced by the script.
- `models/` – Implementations of the various optimisation algorithms (Bayesian optimisation, CMA-ES, DE, PSO, etc.).
- `utils/` – Helper functions including objective function definitions.
- `data/` – Created automatically on first run; houses CSV logs of optimiser performance.

## Running the simulation

1. **Start the COMSOL server**: Launch COMSOL Multiphysics with Java API access enabled (e.g., `comsol server`) and note the hostname/port if different from the default `localhost:2036`.
2. **Configure the COMSOL client (optional)**: The script relies on the default connection settings of the `mph` package. If your server uses a non-default address, set the `COMSOL_SERVER` environment variable before running:

   ```bash
   export COMSOL_SERVER="hostname:port"
   ```

3. **Execute the script** from the repository root:

   ```bash
   python RPNI_simulation.py
   ```

   The script will iterate through configured optimiser hyperparameter sweeps. Depending on the number of combinations and COMSOL solve times, this process can take several hours.

### Output location

All optimiser runs log their results as CSV files in `data/RPNI_sim_PSO/`. Each file name encodes the optimiser type and hyperparameters (e.g., `bo_i20_xi0.01_t20000.csv`). The CSVs typically contain:

- Optimiser iteration metadata (e.g., iteration index, sampled currents).
- Objective values (selectivity scores) returned by the COMSOL evaluation.

You can post-process these results with your preferred analysis tooling (Python, MATLAB, etc.).

## Customisation tips

- **Target fibre indices**: Adjust `target_indices` in `RPNI_simulation.py` to evaluate different nerve fibre locations.
- **Current bounds**: Modify the `current_ranges` dictionary to reflect alternative stimulation current limits.
- **Optimiser settings**: Each optimiser block exposes hyperparameters such as population size, learning rate, or exploration constants. Tune these to match your experimentation budget.

## Troubleshooting

- **COMSOL out-of-memory errors**: The script resets the COMSOL client every third simulation run (`RESET_INTERVAL = 3`). Increase or decrease this value if you experience instability.
- **Connection issues**: Ensure the COMSOL server is reachable from the machine running the script and that your license allows external connections.
- **Missing CSV outputs**: Confirm write permissions for the `data/` directory and monitor terminal output for exceptions during optimisation.

## Citation

If you use this simulation suite in academic work, please cite the associated publications or acknowledge the tooling accordingly.
