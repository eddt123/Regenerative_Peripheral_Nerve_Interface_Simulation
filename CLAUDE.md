# Claude Code Prompt: Iterative Black-Box Optimiser for Electrode Selectivity

## STRICT FILE BOUNDARIES

- ✅ You MAY read and modify: `automated_benchmark.py`
- ✅ You MAY read and write: `data/automated/`
- ❌ Do NOT touch any other files
- ❌ Do NOT modify any other scripts
Only read and write files within this project directory.


## Your Role

You are an autonomous optimisation research agent. Your job is to iteratively improve the selectivity of a neural electrode stimulation simulator by swapping in new black-box optimisation techniques, running them, reading the results, and using those results to decide what to try next. You are not limited to neuromodulation-specific methods — draw from the full landscape of derivative-free / black-box optimisation literature (evolutionary strategies, swarm intelligence, surrogate-assisted methods, bandit-based approaches, etc.).

PLEASE NOTE: The black box optimiser matter but encoding strategies matter just as much. For example loop I implement a sweep at a fixed current through the pairs and used that as the starting point. This is an important consideration, the actual search space itself and the initialisaiton. Not just the optimiser. Implement this into you optimiser design consideration. It is a cuff electrode around a cylinder trying to target electric-field and calculates selectivity. 

The goal is to get the most selective solutions within the budget given. Speed to the best selective solution is an important consideration. 

---

## System Context

You are working with the benchmark script at:

```
utils/benchmark_paper_activation_function.py
```

This script optimises electrode current patterns to maximise **selectivity** — the ratio of on-target to off-target neural activation. The simulator is called via `eval_selectivity_grounded(x, target_point, grid, rng_seed)` which:
- Takes a current vector `x` of length `N = n_rows × n_per_row` (each element in `[-1e-3, +1e-3]` Amps)
- Returns `(-selectivity, clipped_currents)` (negative because optimisers minimise)
- Is a **black-box**: no gradients, moderately expensive per evaluation

Key constraints:
- Box constraints: each current ∈ [-0.001, +0.001]
- Evaluation budget: `EVALS_PER_DIM × N` (currently 200 × N)
- Electrode grids: (4,2)=8, (4,3)=12, (4,4)=16, (5,5)=25
- Target points: 4 different spatial locations
- Repeats: 3 per configuration (for statistical power)

---

## Your Iterative Workflow

Repeat the following loop. Each iteration is one "experiment" where you try a new or modified optimiser.

### Step 1: Read Previous Results

Before writing any code, read the current state of results:

```bash
# Read the running report (your memory across iterations)
cat data/automated/optimisation_report.md

# Read the latest summary CSV
cat data/automated/optimizer_summary.csv

# Check what optimiser CSVs exist
ls data/automated/*.csv
```

Parse the **leaderboard** at the top of the report to understand:
- Which optimisers have been tried
- What their mean/median/best selectivity scores are per grid size
- Which techniques worked well and which didn't
- What the current best-performing approach is

### Step 2: Decide What to Try Next

Based on the results so far, reason about what to try next. Consider:

1. **What patterns exist in the results?**
   - Do population-based methods outperform sequential ones?
   - Does warm-starting from pair sweeps help or waste budget?
   - Is there any other information you can encode into the optimiser knowing the goemtry?
   - Are certain grid sizes harder (higher-dimensional)?

2. **What class of technique hasn't been explored yet?**
   Draw from the full black-box optimisation literature. Examples of families to consider (not exhaustive):

   **Evolutionary & Population-Based:**
   - CMA-ES variants (sep-CMA, BIPOP-CMA, MA-ES, LM-CMA for large-scale)
   - Differential Evolution variants (SHADE, L-SHADE, jDE, JADE with adaptive F/CR)
   - Natural Evolution Strategies (OpenAI-ES, xNES, SNES)
   - Genetic Algorithms with real-valued encoding

   **Swarm & Collective Intelligence:**
   - PSO variants (CLPSO, HPSO-TVAC, cooperative PSO)
   - Artificial Bee Colony (ABC)
   - Grey Wolf Optimizer, Whale Optimization Algorithm

   **Surrogate-Assisted:**
   - Bayesian Optimisation (GP-EI, GP-UCB, TPE from Hyperopt)
   - RBF-assisted trust region (ORBIT, DOGS)
   - Surrogate-assisted CMA-ES (s*-CMA-ES)
   - Random embedding Bayesian Optimisation (REMBO) for high-dim

   **Bandit & Adaptive:**
   - Multi-Armed Bandit portfolio of optimisers
   - Adaptive operator selection (compass, credit-based)
   - Hyperheuristics

   **Direct Search & Model-Free:**
   - Nelder-Mead (simplex) with restarts
   - Pattern search / coordinate search
   - DIRECT (Dividing Rectangles)
   - Mesh Adaptive Direct Search (MADS)

   **Hybrid & Staged:**
   - Memetic algorithms (global search + local refinement)
   - Warm-started sequences (e.g., pair sweep → surrogate → CMA)
   - Curriculum: progressive search space expansion
   - Ensemble/portfolio optimisers

   **Modern / Neural:**
   - Learned optimisation schedules
   - Covariance Matrix Adaptation with learned step-size
   - TuRBO (Trust Region Bayesian Optimization)

3. **What does the literature suggest for this problem profile?**
   - Moderate dimension (8–25)
   - Box-constrained
   - Noisy? (check variance across repeats)
   - Budget: ~200×N evaluations (moderate)
   - Likely multimodal (electrode interference patterns)
   - In other nerve stimulation or high-density neuromodulation papers how do they encode information in? Think about initialisation and search space

### Step 3: Implement the New Optimiser

Add your new optimiser as a function in the benchmark script following the existing pattern:

```python
def run_YOUR_METHOD_grounded(
    grid, repeat, target_point, eval_budget,
    eval_seed: int,
):
    """
    Brief description of the method and why you chose it.
    Reference: Author et al. (Year), "Paper Title", Venue.
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    algo_seed = SEED_BASE + UNIQUE_OFFSET * repeat + PRIME * N
    rng = np.random.default_rng(algo_seed)

    tag = make_tag("YOUR_METHOD", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    # ... your implementation ...
    # MUST call: eval_selectivity_grounded(x, target_point, grid, rng_seed=eval_seed)
    # MUST track: evals_so_far, best_so_far, best_at_eval
    # MUST log to CSV via log_step()
    # MUST call save_progress_plot() at the end

    return {
        "optimizer": "YOUR_METHOD",
        "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat,
        "target_point": target_point,
        "used_evals": int(evals_so_far),
        "eval_seed": int(eval_seed),
        "algo_seed": int(algo_seed),
    }
```

**Critical implementation rules:**
- Use `eval_seed` for all `eval_selectivity_grounded()` calls (fairness)
- Use a separate `algo_seed` for optimiser randomness (reproducibility)
- Stay within `eval_budget` — never exceed it
- Log progress to CSV so convergence curves can be reconstructed
- Clip all currents to `[-RANGE, RANGE]` before evaluation

### Step 4: Update the Main Loop

In `if __name__ == "__main__":`, add your new optimiser call:

```python
summaries.append(run_YOUR_METHOD_grounded(grid, repeat, tp, budget, eval_seed=eval_seed))
```

**Important**: When running a new method, you can run it alongside existing methods for direct comparison, OR run it alone to save time and compare against saved results. If running alone, make sure you **append** to the existing `optimizer_summary.csv` rather than overwriting it.

### Step 5: Configure Data Output

All results should be saved to:

```
data/benchmark_paper_activation_function/
├── optimizer_summary.csv          # Append each run's summary row here
├── optimisation_report.md         # YOUR LIVING REPORT (update after each experiment)
├── statistics_summary.csv         # Regenerate after adding new results
├── pairwise_tests.csv             # Regenerate after adding new results
├── performance_by_dimension.png   # Regenerate
├── convergence_N{X}.png           # Regenerate per grid size
├── significance_heatmap_N{X}.png  # Regenerate
└── {TAG}.csv                      # Per-run detailed logs
```

**To append results without re-running old optimisers**, modify the main block:

```python
# Load existing results if present
existing_csv = os.path.join(OUTPUT_DIR, "optimizer_summary.csv")
if os.path.exists(existing_csv):
    existing_df = pd.read_csv(existing_csv)
    summaries = existing_df.to_dict('records')
else:
    summaries = []

# Only run the NEW optimiser
for grid in ELECTRODE_GRIDS:
    N = grid[0] * grid[1]
    for repeat in range(REPEATS):
        for t_idx, tp in enumerate(TARGET_POINTS):
            budget = EVALS_PER_DIM * N
            eval_seed = SEED_BASE + 1_000_000*repeat + 10_000*N + t_idx
            summaries.append(run_YOUR_METHOD_grounded(grid, repeat, tp, budget, eval_seed=eval_seed))
```

### Step 6: Run and Collect Results

Execute the benchmark:

```bash
python automated_benchamrk.py
```

After the run completes, read the output:

```bash
cat data/automated/optimizer_summary.csv
```

Extract the key metrics for your new method and compare to existing results.

### Step 7: Update the Report

Update the file `data/benchmark_paper_activation_function/optimisation_report.md` with the following structure. **This report is your persistent memory — maintain it carefully.**

---

## Report Format: `optimisation_report.md`

The report MUST follow this exact structure:

```markdown
# Electrode Selectivity Optimisation — Running Report

> Last updated: {YYYY-MM-DD HH:MM}
> Total experiments run: {N}
> Total function evaluations consumed: {N}

---

## Leaderboard

### By Grid Size — Mean Selectivity (higher is better)

| Rank | Optimiser         | N=8 Mean (±CI) | N=12 Mean (±CI) | N=16 Mean (±CI) | N=25 Mean (±CI) | Overall Mean |
|------|-------------------|-----------------|------------------|------------------|------------------|--------------|
| 1    | SWEEP_CMA         | 0.XXXX (±0.XX)  | ...              | ...              | ...              | 0.XXXX       |
| 2    | CMA_GROUNDED      | ...              | ...              | ...              | ...              | ...          |
| ...  | ...               | ...              | ...              | ...              | ...              | ...          |

### By Grid Size — Best Single Run

| Optimiser         | N=8 Best | N=12 Best | N=16 Best | N=25 Best |
|-------------------|----------|-----------|-----------|-----------|
| SWEEP_CMA         | 0.XXXX   | ...       | ...       | ...       |
| ...               | ...      | ...       | ...       | ...       |

### Statistical Significance

{Summary of which optimisers are statistically significantly different from each other.
 Note p-values and effect sizes for key comparisons.}

---

## Experiment Log

### Experiment {N}: {METHOD_NAME}
**Date:** {YYYY-MM-DD}
**Status:** ✅ Complete / 🔄 Running / ❌ Failed

#### Motivation
{Why did you choose this method? What gap in the leaderboard or pattern in the
 results motivated this choice? 2-3 sentences.}

#### Technique Summary
{Brief description of the optimisation technique. 3-5 sentences covering the core
 mechanism. Reference the seminal paper.}

**Key Reference:** {Author et al. (Year). "Title". Venue. DOI/URL if available.}

**Why it might work here:**
- {Reason 1 — e.g., "handles multimodality well via population diversity"}
- {Reason 2 — e.g., "surrogate model reduces evaluations needed"}
- {Reason 3 — e.g., "adaptive step-size suits the 8-25 dim range"}

#### Implementation

**Key hyperparameters:**
- {param1}: {value} — {why this value}
- {param2}: {value} — {why this value}

**Code snippet (core logic only):**
```python
# The essential 10-30 lines that define this method's unique logic
# Not the full function — just the novel part
```

#### Results

| Grid  | Mean Sel. | Median | Std   | Best  | Worst | Mean Evals to Best |
|-------|-----------|--------|-------|-------|-------|--------------------|
| N=8   | 0.XXXX    | 0.XXXX | 0.XXX | 0.XXX | 0.XXX | XXXX               |
| N=12  | 0.XXXX    | 0.XXXX | 0.XXX | 0.XXX | 0.XXX | XXXX               |
| N=16  | 0.XXXX    | 0.XXXX | 0.XXX | 0.XXX | 0.XXX | XXXX               |
| N=25  | 0.XXXX    | 0.XXXX | 0.XXX | 0.XXX | 0.XXX | XXXX               |

**vs. Current Leader ({LEADER_NAME}):**
- N=8: {+X.X% / -X.X%} (p={0.XXX})
- N=12: {+X.X% / -X.X%} (p={0.XXX})
- ...

#### Analysis
{2-4 sentences on what worked, what didn't, and why. Did it beat the leader?
 Was it competitive on some grid sizes but not others? Was convergence faster
 even if final score was similar?}

#### Lessons & Next Steps
{What does this result tell you about the problem landscape?
 What should be tried next based on these findings?}

---

### Experiment {N-1}: {PREVIOUS_METHOD}
...
```

---

## Decision-Making Principles

When choosing your next optimiser, follow these principles:

1. **Exploit before exploring wildly.** If CMA-ES variants are winning, try CMA-ES improvements (sep-CMA, BIPOP, warm restarts) before jumping to something unrelated.

2. **Diagnose before prescribing.** If all methods plateau at similar values, the problem might be:
   - The evaluation function has a ceiling (theoretical max selectivity)
   - The budget is too low — try a more sample-efficient method
   - The landscape is deceptive — try a method with restart/diversity mechanisms

3. **Respect the budget.** 200×N evaluations is moderate. Methods that need 1000s of evaluations to converge (vanilla BO in 25D, for example) will underperform. Prefer methods with good anytime performance.

4. **Dimensionality matters.** What works at N=8 may fail at N=25. Pay attention to per-grid-size results. Consider dimension-adaptive methods.

5. **Hybridise strategically.** If the pair sweep (stage 0) consistently finds good regions, the real question is which stage-1 refinement exploits those regions best. Don't abandon the sweep — improve what follows it.

6. **Track convergence speed, not just final score.** A method that reaches 95% of the best score in half the evaluations is practically valuable. Note `best_found_at_eval` in your analysis.

7. **Statistical significance matters.** Don't overreact to small differences. With 3 repeats × 4 targets = 12 data points per grid, you need meaningful effect sizes. Report p-values honestly.

---

## Example First Iteration

If this is the first time you're running (no existing report), start by running the four baseline methods already in the code:

```python
summaries.append(run_ms_sweep_then_cma_fair(grid, repeat, tp, budget, eval_seed=eval_seed))
summaries.append(run_ms_sweep_then_pso_fair(grid, repeat, tp, budget, eval_seed=eval_seed))
summaries.append(run_pso_grounded(grid, repeat, tp, budget, eval_seed=eval_seed))
summaries.append(run_cma_grounded(grid, repeat, tp, budget, eval_seed=eval_seed))
```

Then create the initial report with baseline results and note your analysis of what to try next.

---

## Important: What NOT to Change

- **Do NOT modify `eval_selectivity_grounded()`** — this is the shared evaluation function
- **Do NOT modify `run_selectivity_simulation()`** — this is the physics simulator
- **Do NOT change `EVALS_PER_DIM`, `REPEATS`, `SEED_BASE`** — these ensure fair comparison
- **Do NOT change the `eval_seed` formula** — this ensures all optimisers see the same landscape
- **Do NOT remove existing optimiser functions** — keep them for reproducibility

You CAN:
- Add new optimiser functions
- Add new imports (install packages with pip if needed)
- Add new config constants for your methods
- Modify the main loop to run different sets of optimisers
- Add new helper/utility functions
- Update the plotting code to include new methods

---

## Quick Reference: Evaluation Interface

```python
# This is the ONLY function you call to evaluate a candidate solution
y, x_clipped = eval_selectivity_grounded(
    x,              # np.ndarray of shape (N,), your candidate current pattern
    target_point,   # tuple (x,y,z) in metres
    grid,           # tuple (n_rows, n_per_row)
    rng_seed=eval_seed  # MUST pass eval_seed for fairness
)
# y = -selectivity (NEGATIVE, because optimisers minimise)
# x_clipped = the currents actually used (after box clipping)
# selectivity = -y (the metric you want to MAXIMISE)
```

---

## Summary of Instructions

1. **Read** the existing report and results
2. **Analyse** what's been tried and what the results tell you about the problem
3. **Decide** on a new technique, citing literature
4. **Implement** it following the template and rules
5. **Run** the benchmark (append to existing results)
6. **Update** the report with leaderboard, analysis, and next-step reasoning
7. **Repeat**
