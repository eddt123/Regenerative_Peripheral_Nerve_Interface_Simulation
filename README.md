RPNI_sim_benchmark.py — Benchmarking Optimisers for Selective Peripheral Nerve Stimulation

This repository contains a fair, dimension-scaled, statistically powered benchmarking framework for optimisation algorithms applied to current steering in Regenerative Peripheral Nerve Interfaces (RPNIs) and similar peripheral nerve stimulation systems.

The framework evaluates how well different black-box optimisers can maximise neural selectivity when controlling multi-electrode stimulation patterns under a grounded boundary model.

The aim is to identify the most reliable optimisation approaches for:

Closed-loop sensory mapping


🔬 Scientific Motivation

Peripheral nerve and RPNI stimulation produce complex electric fields in biological tissue.
Different combinations of electrode currents lead to different fascicular activation patterns, and the goal is to maximise selectivity:

selectivity
=
𝐴
target
mean
(
𝐴
off-target
)
selectivity=
mean(A
off-target
	​

)
A
target
	​

	​


Optimising these currents is a non-convex, multi-modal, highly non-linear black-box problem.

Conventional approaches (parameter sweeps, manual tuning) do not scale to systems with:

8–25 electrodes

12D+ continuous current control

Patient-specific target fields

Tight experimental budgets (100–1000 stim evaluations)

This script provides a rigorous comparison of advanced optimisation algorithms to determine which methods achieve the highest selectivity under fixed evaluation budgets.

🚀 What This Benchmark Does
✔ Dimension-scaled and equal-budget

Each optimiser receives exactly:

200 × N evaluations

where N = number of electrodes, ensuring fair comparisons.

✔ Grounded boundary (no zero-sum current constraint)

Currents are restricted to:

[-1 mA, +1 mA]


and a grounded tissue boundary handles return paths automatically.

✔ Full multi-optimiser comparison

Included algorithms:

Category	Algorithms
Evolutionary	CMA-ES, Differential Evolution, CEM
Swarm	Particle Swarm Optimisation (PSO)
Model-based	Sequential Bayesian Optimisation (GP-based)
Local search	Simulated Annealing
Baseline	Random Search
Novel methods	Multiple Multi-Stage CMA-ES variants

✔ Multi-Stage Optimisation (Novel)

We include several new multi-stage methods, designed specifically for PNS and RPNI stimulation:

Pair-Sweep CMA-ES
Sweep all electrode pairs → average best → warm-start CMA-ES.

Pairs-Basis CMA-ES → Full CMA-ES
Optimise in low-dimensional pair basis → expand to full 12D.

Pair-Sweep Multi-Start CMA-ES
Sweep → choose K best → run short CMA-ES refinements → take best.

These reflect the biological intuition that stimulation direction (dipoles) dominates first-order effects, and fine-control should only be explored after coarse spatial focus is found.

🧠 RPNI & Nerve Simulation Model

run_selectivity_simulation() computes:

Electric fields in a cylindrical RPNI / nerve model

Activating function or field-based activation

Target fascicle activation

Off-target activation sampling within tissue

Final selectivity metric

Tissue parameters include:

Radius = 1 cm

Height = 4 cm

Conductivity = 0.25 S/m

≥1200 off-target sample points

The model is easily replaced or extended with COMSOL, FEM, or ASCENT-generated fields.

📊 Outputs & Interpretation
1. Per-run CSV logs

Store every evaluated stimulation pattern:

eval index

selectivity

best so far

all electrode currents (I₀…IN₋₁)

optimiser metadata

stage index for multi-stage models

2. Progress plots

*_progress.png → evaluation vs selectivity.

3. Summary tables

optimizer_summary.csv

statistics_summary.csv

pairwise_tests.csv

landscape_characterization.csv

4. Performance visualisations

Boxplots by dimension

Convergence curves (mean ± std)

Pairwise significance heatmaps

Landscape difficulty metrics

🏆 Most Important Metric

For comparing optimisers, the key number is:

⭐ Final best selectivity under equal evaluation budgets

This represents:

quality of the best found stimulation pattern

ability to escape local minima

speed and reliability

practical performance in patient trials

Secondary metrics:

eval index where best was found (speed)

variability (std)

significance tests (Mann–Whitney + effect size)

🧪 How to Run the Benchmark
python RPNI_sim_benchmark.py


All results are saved under:

data/benchmark_extra/


To enable/disable algorithms, edit the main loop:

# summaries.append(run_cma(...))
# summaries.append(run_bo(...))
# summaries.append(run_pso(...))
# summaries.append(run_de(...))
# summaries.append(run_cem(...))
# summaries.append(run_sa(...))

summaries.append(run_ms_pairs_sweep_then_cma(...))
summaries.append(run_ms_pairs_cma_then_full_cma(...))
summaries.append(run_ms_pairs_sweep_multi_cma(...))

🔧 Extending the Framework

You can easily add:

new basis sets (tripoles, focal tripoles, anatomical priors)

additional multi-stage strategies

Bayesian optimisation variants (qEI, TPE, RF-BO)

Gradient-estimated variants (finite-difference CMA-ES)

Hardware-in-the-loop optimisation for real RPNI implants

The design is modular: each optimiser is one function returning a standardised result dictionary.

📝 Recommended Use Cases

Research into selective peripheral nerve stimulation

RPNI sensory mapping algorithms

Current steering optimisation

Benchmarking black-box optimisers on biophysical problems

Designing next-generation closed-loop prosthetic interfaces

🤝 Citation

If you use this benchmark in research, please cite:

"Closed-loop optimisation for selective stimulation in Regenerative Peripheral Nerve Interfaces."
Ed Turner et al., 2025.