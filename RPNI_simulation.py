
from pathlib import Path
import mph
import numpy as np

from models.greedy_random_sim      import GreedyRandomOptimizer
from models.bo_model_sim          import BOSimulation
from models.cmaes_sim             import CMAESOptimizer
from models.gradient_descent_sim  import GradientDescentSimulator
from models.de_sim                import DESimulation
from models.additional_optimisers_sim import (
    TPESimulation,
    RandomForestBOSimulation,
    CEMOptimizer,
    PowellLocalOptimizer,
    BanditOptimizer,
    PSOSimulation,
    #TuRBOBOSimulation
)

from utils.objective_functions_sim import compute_selectivity

# ─── COMSOL Session Management ─────────────────────────────────────────
_client     = None
_model      = None
_call_count = 0
RESET_INTERVAL = 3  # restart COMSOL every x calls to avoid OOM

def get_client_and_model():
    global _client, _model
    if _client is None:
        _client = mph.start()
        model_path = Path(__file__).resolve().parent / "12-ch 3D RPNI sim.mph"
        _model  = _client.load(str(model_path))
        _model.build()                   # mesh ONCE
    return _client, _model

def run_comsol(model, param_dict):
    # clear old solution & results to stop heap growth
    try:
        model.sol("sol1").clearSolution()
        model.result().clear()           # drop result datasets/plots
    except Exception:
        pass                             # first call, objects may not exist
    # update parameters
    for k, v in param_dict.items():
        model.parameter(k, str(v))
    model.solve("Study 1")               # solve only, no new mesh 


def reset_client():
    global _client, _model
    if _client is not None:
        try: _client.terminate()
        except: pass
    _client = None
    _model  = None

# ─── Helpers ───────────────────────────────────────────────────────────

def extract_e_potential(model) -> np.ndarray:
    return np.array(model.evaluate("V2"))

def simulate_selectivity(param_dict, target_idx):
    global _call_count
    _call_count += 1
    if _call_count % RESET_INTERVAL == 0:   # reset every Nth call
        reset_client()

    _, model = get_client_and_model()
    run_comsol(model, param_dict)
    vdata = np.array(model.evaluate("V2"))
    return compute_selectivity(vdata, target_idx)

# ─── Main Script ─────────────────────────────────────────────────────
def main():
    out_dir = Path("data") / "RPNI_sim_PSO"
    out_dir.mkdir(parents=True, exist_ok=True)

    current_ranges = {f"I_d{i}": (-1.0, 1.0) for i in range(1, 13)}
    target_indices = [20000]  #do 16786, 20000 and 25000

    # 1) Greedy-Random
    gr_iters      = [5, 10, 20, 40]    # number of random iterations
    gr_candidates = [40, 20, 10, 5]    # samples per iteration (gr_iters * gr_candidates ≈ 200)

    # 2) BO (GP) hyperparams
    bo_iters      = [180, 190, 200]    # total BO iterations (bo_initial + bo_iters ≈ 200)
    bo_initial    = [10, 20, 40]       # initial random points
    xi_values     = [0.001, 0.01, 0.1]  # exploration vs exploitation

    # 3) CMA-ES hyperparams
    sigma_values     = [0.4, 0.8, 1.2]  # initial search radius
    popsize_values   = [5, 10, 20]      # population size
    cma_iters_values = [40, 20, 10]     # generations (popsize * gens ≈ 200)

    # 4) DE hyperparams
    de_popsize_vals  = [5, 10, 20]                # popsize multiplier (popsize * gens * D ≈ 200)
    de_mutation_vals = [(0.3, 0.9), (0.5, 1.0)]   # F range for mutation
    de_rec_vals      = [0.5, 0.7, 0.9]            # crossover probability CR
    de_maxiter_vals  = [10, 20, 40]               # generations

    # 5) GD hyperparams
    gd_iters   = [200]             # total gradient steps
    lr_values  = [0.01, 0.1, 0.5]   # learning rates
    eps_values = [1e-1, 1e-2, 1e-3] # finite-difference step sizes

    # 6) TPE hyperparams
    tpe_trials_vals = [200]        # Optuna TPE trials

    # 7) RF-BO hyperparams
    rf_calls_vals = [200]          # scikit-optimize surrogate calls

    # 8) CEM hyperparams
    cem_popsize_vals    = [20, 50, 100] # CEM population sizes
    cem_elite_frac_vals = [0.1, 0.2, 0.3] # elite fractions
    cem_iters_vals      = [4, 10, 20]   # iterations (popsize * iters ≈ 200)
    cem_smooth_vals     = [0.7, 0.9]    # smoothing factors

    # 9) Powell hyperparams
    powell_restarts_vals = [5, 10, 20] # random restarts

    # 10) Bandit hyperparams
    bandit_res_vals   = [3, 5, 7]        # discretisation resolution per axis
    bandit_pulls_vals = [200]            # total pulls
    bandit_c_vals     = [0.5, 1.0, 2.0]  # UCB exploration constant

    # 11) Particle Swarm Optimization (PSO) hyperparams
    pso_popsize_vals = [20]    # swarm size
    pso_w_vals       = [0.5]  # inertia weights
    pso_c1_vals      = [1.5]       # cognitive coefficients
    pso_c2_vals      = [1.5]       # social coefficients
    pso_iters_vals   = [10]    # iterations (popsize * iters ≈ 200)

    # 12) TuRBO hyperparams
    turbo_init_vals     = [20, 40]   # number of Sobol (random) initial points
    turbo_maxeval_vals  = [200]      # total evaluations (incl. init)
    turbo_batch_vals    = [1, 4]     # proposals per BO step



    for tidx in target_indices:

        #GP-BO
        for bo_initial in bo_initial:
            for xi in xi_values:
                bo_csv = out_dir / f"bo_i{bo_initial}_xi{xi}_t{tidx}.csv"
                bo_opt = BOSimulation(
                    current_ranges     = current_ranges,
                    target_idx         = tidx,
                    simulate_fn        = simulate_selectivity,
                    out_csv            = bo_csv,
                    n_iters            = 200 - bo_initial,   # to keep total ~200
                    candidates_per_iter= gr_candidates,
                    n_initial_points   = bo_initial,
                    acq_func           = "EI",
                    random_state       = 42,
                    xi                 = xi,
                )
                bo_opt.optimize()


        # CMA-ES
        for sigma in sigma_values:
            for popsize in popsize_values:
                for cma_iters in cma_iters_values:
                    cma_csv = out_dir / f"cmaes_t{tidx}_sig{sigma}_pop{popsize}_gen{cma_iters}.csv"
                    cma_opt = CMAESOptimizer(
                        current_ranges = current_ranges,
                        target_idx     = tidx,
                        simulate_fn    = simulate_selectivity,
                        out_csv        = cma_csv,
                        sigma          = sigma,
                        popsize        = popsize,
                        n_iters        = cma_iters,
                    )
                    cma_opt.optimize()

        #DE
        for pop in de_popsize_vals:
            for mut in de_mutation_vals:
                for rec in de_rec_vals:
                    for gens in de_maxiter_vals:
                        de_csv = out_dir / (
                            f"de_t{tidx}_pop{pop}_mut{mut[0]}-{mut[1]}"
                            f"_rec{rec}_gen{gens}.csv"
                        )
                        de_opt = DESimulation(
                            current_ranges  = current_ranges,
                            target_idx      = tidx,
                            simulate_fn     = simulate_selectivity,
                            out_csv         = de_csv,
                            popsize         = pop,
                            mutation        = mut,
                            recombination   = rec,
                            maxiter         = gens,
                        )
                        de_opt.optimize()

        # Gradient Descent
        for lr in lr_values:
            for eps in eps_values:
                gd_csv = out_dir / f"gd_t{tidx}_lr_{lr}_eps_{eps}.csv"
                gd_opt = GradientDescentSimulator(
                    current_ranges = current_ranges,
                    target_idx     = tidx,
                    simulate_fn    = simulate_selectivity,
                    out_csv        = gd_csv,
                    learning_rate  = lr,
                    n_iters        = gd_iters,
                    eps            = eps,
                )
                gd_opt.optimize()

        # TPE (Optuna)
        for trials in tpe_trials_vals:
            tpe_csv = out_dir / f"tpe_{trials}_target_{tidx}.csv"
            tpe_opt = TPESimulation(
                current_ranges = current_ranges,
                target_idx     = tidx,
                simulate_fn    = simulate_selectivity,
                out_csv        = tpe_csv,
                n_trials       = trials,
                random_seed    = 42,
            )
            tpe_opt.optimize()

        # Random-Forest BO (skopt)
        for calls in rf_calls_vals:
            rf_csv = out_dir / f"rfbo_{calls}_target_{tidx}.csv"
            rf_opt = RandomForestBOSimulation(
                current_ranges = current_ranges,
                target_idx     = tidx,
                simulate_fn    = simulate_selectivity,
                out_csv        = rf_csv,
                n_calls        = calls,
                random_seed    = 42,
            )
            rf_opt.optimize()

        # CEM
        for pop in cem_popsize_vals:
            for elite in cem_elite_frac_vals:
                for iters in cem_iters_vals:
                    for smooth in cem_smooth_vals:
                        cem_csv = out_dir / f"cem_pop{pop}_elite{elite}_it{iters}_s{smooth}_t{tidx}.csv"
                        cem_opt = CEMOptimizer(
                            current_ranges = current_ranges,
                            target_idx     = tidx,
                            simulate_fn    = simulate_selectivity,
                            out_csv        = cem_csv,
                            popsize        = pop,
                            elite_frac     = elite,
                            n_iters        = iters,
                            smoothing      = smooth,
                            random_seed    = 42,
                        )
                        cem_opt.optimize()

        # Powell Local
        for rest in powell_restarts_vals:
            pow_csv = out_dir / f"powell_{rest}restarts_t{tidx}.csv"
            pow_opt = PowellLocalOptimizer(
                current_ranges = current_ranges,
                target_idx     = tidx,
                simulate_fn    = simulate_selectivity,
                out_csv        = pow_csv,
                n_restarts     = rest,
                random_seed    = 42,
            )
            pow_opt.optimize()

        # Bandit (UCB)
        for res in bandit_res_vals:
            for pulls in bandit_pulls_vals:
                for c in bandit_c_vals:
                    ban_csv = out_dir / f"bandit_res{res}_pulls{pulls}_c{c}_t{tidx}.csv"
                    ban_opt = BanditOptimizer(
                        current_ranges = current_ranges,
                        target_idx     = tidx,
                        simulate_fn    = simulate_selectivity,
                        out_csv        = ban_csv,
                        resolution     = res,
                        total_pulls    = pulls,
                        c              = c,
                    )
                    ban_opt.optimize()

        # # particle swarm optimisation
        for popsize in pso_popsize_vals:
            for w in pso_w_vals:
                for c1 in pso_c1_vals:
                    for c2 in pso_c2_vals:
                        for nit in pso_iters_vals:
                            pso_csv = out_dir / f"pso_pop{popsize}_w{w}_c1{c1}_c2{c2}_it{nit}_t{tidx}.csv"
                            pso_opt = PSOSimulation(
                                current_ranges = current_ranges,
                                target_idx     = tidx,
                                simulate_fn    = simulate_selectivity,
                                out_csv        = pso_csv,
                                popsize        = popsize,
                                w              = w,
                                c1             = c1,
                                c2             = c2,
                                n_iters        = nit,
                                random_seed    = 42,
                            )
                            pso_opt.optimize()

        # TuRBO (Trust-Region BO)
        # for n_init in turbo_init_vals:
        #     for max_evals in turbo_maxeval_vals:
        #         for batch in turbo_batch_vals:
        #             turbo_csv = out_dir / (
        #                 f"turbo_init{n_init}_eval{max_evals}_batch{batch}_t{tidx}.csv"
        #             )
        #             turbo_opt = TuRBOBOSimulation(
        #                 current_ranges = current_ranges,
        #                 target_idx     = tidx,
        #                 simulate_fn    = simulate_selectivity,
        #                 out_csv        = turbo_csv,
        #                 n_init         = n_init,
        #                 max_evals      = max_evals,
        #                 batch_size     = batch,
        #                 random_seed    = 42,
        #             )
        #             turbo_opt.optimize()


    reset_client()


if __name__ == "__main__":
    main()
