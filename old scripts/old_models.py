def run_random_search(grid, repeat, target_point, eval_budget):
    """Pure random search baseline."""
    N = grid[0] * grid[1]
    seed = SEED_BASE + 10000*repeat + N
    rng = np.random.default_rng(seed)
    tag = make_tag("Random", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")
    
    best_so_far = -np.inf
    best_at_eval = 0
    xs_axis, step_vals, best_vals = [], [], []
    
    header_written = False
    for i in tqdm(range(eval_budget), desc=tag, leave=False):
        x = rng.uniform(-RANGE, RANGE, N)
        y, x_orig, x_proj = eval_selectivity(x, target_point, grid, rng_seed=seed)
        
        sel = -y
        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = i + 1
        
        xs_axis.append(i + 1)
        step_vals.append(sel)
        best_vals.append(best_so_far)
        
        # Log every 10th evaluation to reduce file size
        if (i + 1) % 10 == 0 or i == 0 or i == eval_budget - 1:
            log_step(csv_path, {
                "optimizer": "Random",
                "n_rows": grid[0], "n_per_row": grid[1], "N": N,
                "repeat": repeat,
                "eval_index": i + 1,
                "evals_so_far": i + 1,
                "current_selectivity": sel,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
            }, header_written)
            header_written = True
    
    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    
    return {
        "optimizer": "Random", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": eval_budget
    }


def run_cma(grid, repeat, target_point, eval_budget):
    """
    CMA-ES (dimension-scaled) with correct projection handling.
    Evaluates at projected feasible points and tells the optimizer
    those same evaluated points to maintain model consistency.
    """
    N = grid[0] * grid[1]
    popsize = cma_popsize(N)
    seed = SEED_BASE + 1000*repeat + N
    tag = make_tag("CMAES", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    # Budget accounting
    iters = max(1, eval_budget // popsize)
    used_budget = iters * popsize

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-RANGE, RANGE, N)
    #x0 = project_currents(x0, -RANGE, RANGE, ZERO_SUM)

    es = CMAEvolutionStrategy(
        x0, CMA_SIGMA0,
        {"popsize": popsize, "verb_disp": 0, "seed": seed, "bounds": [-RANGE, RANGE]}
    )

    best_so_far = -np.inf
    best_at_eval = 0
    xs_axis, iter_vals, best_vals = [], [], []
    evals_so_far = 0
    header_written = False

    for it in tqdm(range(iters), desc=tag, leave=False):
        X_ask = es.ask()

        # Evaluate at projected feasible points
        X_eval, Y = [], []
        for x_ask in X_ask:
            y, x_proj = eval_selectivity_grounded(x_ask, target_point, grid, rng_seed=seed)
            X_eval.append(x_proj)
            Y.append(y)

        # Tell CMA the *evaluated* (projected) points
        es.tell(X_eval, Y)

        evals_so_far += len(Y)
        k = int(np.argmin(Y))
        step_best = -Y[k]

        if step_best > best_so_far:
            best_so_far = step_best
            best_at_eval = evals_so_far

        xs_axis.append(evals_so_far)
        iter_vals.append(step_best)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "CMAES",
            "n_rows": grid[0], "n_per_row": grid[1], "N": N,
            "sigma0": CMA_SIGMA0, "popsize": popsize,
            "repeat": repeat,
            "step_index": it + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": step_best,
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
            'currents': X_ask
        }, header_written)
        header_written = True

    save_progress_plot(xs_axis, iter_vals, best_vals, tag, target_point)

    return {
        "optimizer": "CMAES", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": used_budget, "popsize": popsize
    }


def run_bo(grid, repeat, target_point, eval_budget):
    """
    Sequential Bayesian Optimization (skopt) with correct projection.
    Evaluates at projected feasible points and tells the optimizer
    the same evaluated points.
    """
    N = grid[0] * grid[1]
    n_init = bo_n_initial(N)
    seed = SEED_BASE + 2000*repeat + N
    tag = make_tag(f"BO_{BO_ACQ}", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    opt = SkOptimizer(
        [(-RANGE, RANGE)] * N,
        base_estimator="GP",
        acq_func=BO_ACQ,
        acq_func_kwargs={"kappa": BO_KAPPA, "xi": BO_XI},
        n_initial_points=n_init,
        random_state=seed
    )

    best_so_far = -np.inf
    best_at_eval = 0
    xs_axis, step_vals, best_vals = [], [], []
    header_written = False

    for i in tqdm(range(eval_budget), desc=tag, leave=False):
        # Ask one point at a time (sequential BO)
        x_ask = np.array(opt.ask(), dtype=float)

        # Evaluate at projected feasible point
        y, x_orig, x_proj = eval_selectivity(x_ask, target_point, grid, rng_seed=seed)

        # Tell the optimizer the *evaluated* point
        opt.tell(x_proj.tolist(), float(y))

        sel = -y
        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = i + 1

        xs_axis.append(i + 1)
        step_vals.append(sel)
        best_vals.append(best_so_far)

        # Log every 10th evaluation
        if (i + 1) % 10 == 0 or i == 0 or i == eval_budget - 1:
            log_step(csv_path, {
                "optimizer": "BO",
                "n_rows": grid[0], "n_per_row": grid[1], "N": N,
                "acq_func": BO_ACQ, "kappa": BO_KAPPA, "xi": BO_XI,
                "n_initial": n_init,
                "repeat": repeat,
                "eval_index": i + 1,
                "evals_so_far": i + 1,
                "current_selectivity": sel,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
            }, header_written)
            header_written = True

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "BO", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": eval_budget, "n_initial": n_init
    }


def run_pso(grid, repeat, target_point, eval_budget):
    N = grid[0] * grid[1]
    popsize = pso_popsize(N)
    seed = SEED_BASE + 3000*repeat + N
    rng = np.random.default_rng(seed)
    tag = make_tag("PSO", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")
    
    # Budget accounting
    iters = max(1, eval_budget // popsize)
    used_budget = iters * popsize
    
    # Init swarm in feasible space
    X = np.vstack([
        project_currents(rng.uniform(-RANGE, RANGE, N), -RANGE, RANGE, ZERO_SUM)
        for _ in range(popsize)
    ])
    V = np.zeros_like(X)
    
    # Evaluate initial population
    pbest_pos = X.copy()
    pbest_val = np.empty(popsize, dtype=float)
    for i in range(popsize):
        y, _, _ = eval_selectivity(pbest_pos[i], target_point, grid, rng_seed=seed)
        pbest_val[i] = y
    
    g_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_val = pbest_val[g_idx]
    
    best_so_far = -gbest_val
    best_at_eval = popsize
    xs_axis = [popsize]
    step_vals = [best_so_far]
    best_vals = [best_so_far]
    evals_so_far = popsize
    
    header_written = False
    log_step(csv_path, {
        "optimizer": "PSO",
        "n_rows": grid[0], "n_per_row": grid[1], "N": N,
        "popsize": popsize, "w": PSO_W, "c1": PSO_C1, "c2": PSO_C2,
        "repeat": repeat,
        "step_index": 1,
        "evals_so_far": evals_so_far,
        "step_best_selectivity": best_so_far,
        "best_so_far": best_so_far,
        "best_found_at_eval": best_at_eval,
        "target_x": target_point[0],
        "target_y": target_point[1],
        "target_z": target_point[2],
    }, header_written)
    header_written = True
    
    # Main loop
    for it in tqdm(range(1, iters), desc=tag, leave=False):
        r1 = rng.random(size=(popsize, N))
        r2 = rng.random(size=(popsize, N))
        
        V = PSO_W*V + PSO_C1*r1*(pbest_pos - X) + PSO_C2*r2*(gbest_pos - X)
        if PSO_VCLAMP is not None:
            V = np.clip(V, -PSO_VCLAMP, PSO_VCLAMP)
        
        X_new = X + V
        # Project to feasible space
        X = np.vstack([
            project_currents(x, -RANGE, RANGE, ZERO_SUM) for x in X_new
        ])
        
        it_best_sel = -np.inf
        for i in range(popsize):
            y, _, x_proj = eval_selectivity(X[i], target_point, grid, rng_seed=seed)
            X[i] = x_proj  # Ensure consistency
            
            if y < pbest_val[i]:
                pbest_val[i] = y
                pbest_pos[i] = x_proj
            
            sel = -y
            if sel > it_best_sel:
                it_best_sel = sel
        
        g_idx = int(np.argmin(pbest_val))
        if pbest_val[g_idx] < gbest_val:
            gbest_val = pbest_val[g_idx]
            gbest_pos = pbest_pos[g_idx].copy()
        
        evals_so_far += popsize
        if it_best_sel > best_so_far:
            best_so_far = it_best_sel
            best_at_eval = evals_so_far
        
        xs_axis.append(evals_so_far)
        step_vals.append(it_best_sel)
        best_vals.append(best_so_far)
        
        log_step(csv_path, {
            "optimizer": "PSO",
            "n_rows": grid[0], "n_per_row": grid[1], "N": N,
            "popsize": popsize, "w": PSO_W, "c1": PSO_C1, "c2": PSO_C2,
            "repeat": repeat,
            "step_index": it + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": it_best_sel,
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
        }, header_written)
    
    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    
    return {
        "optimizer": "PSO", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": used_budget, "popsize": popsize
    }

def run_de(grid, repeat, target_point, eval_budget):
    N = grid[0] * grid[1]
    seed = SEED_BASE + 4000*repeat + N
    rng = np.random.default_rng(seed)
    tag = make_tag("DE", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    NP = de_population_size(N)          # number of candidate vectors in the population
    iters = max(1, eval_budget // NP)   # each generation evaluates ~NP candidates
    used_budget = iters * NP

    # Init feasible population
    X = np.vstack([
        project_currents(rng.uniform(-RANGE, RANGE, N), -RANGE, RANGE, ZERO_SUM)
        for _ in range(NP)
    ])
    # Evaluate
    F_vals = np.empty(NP, dtype=float)
    for i in range(NP):
        y, _, _ = eval_selectivity(X[i], target_point, grid, rng_seed=seed)
        F_vals[i] = y  # minimize y

    best_idx = int(np.argmin(F_vals))
    best_val = F_vals[best_idx]
    best_vec = X[best_idx].copy()

    best_so_far = -best_val
    best_at_eval = NP

    xs_axis  = [NP]
    step_vals = [best_so_far]
    best_vals = [best_so_far]
    header_written = False

    log_step(csv_path, {
        "optimizer": "DE",
        "n_rows": grid[0], "n_per_row": grid[1], "N": N,
        "popsize": NP, "F": DE_F, "CR": DE_CR,
        "repeat": repeat,
        "step_index": 1,
        "evals_so_far": NP,
        "step_best_selectivity": best_so_far,
        "best_so_far": best_so_far,
        "best_found_at_eval": best_at_eval,
        "target_x": target_point[0],
        "target_y": target_point[1],
        "target_z": target_point[2],
    }, header_written)
    header_written = True

    # Main generations
    for gen in tqdm(range(1, iters), desc=tag, leave=False):
        X_new = X.copy()
        F_new = F_vals.copy()

        for i in range(NP):
            # choose 3 distinct indices != i
            idxs = rng.choice([j for j in range(NP) if j != i], size=3, replace=False)
            r1, r2, r3 = idxs
            # mutation
            V = X[r1] + DE_F * (X[r2] - X[r3])
            # binomial crossover
            cross = rng.random(N) < DE_CR
            if not np.any(cross):
                cross[rng.integers(0, N)] = True
            U = np.where(cross, V, X[i])
            # project & evaluate
            U = project_currents(U, -RANGE, RANGE, ZERO_SUM)
            y, _, Uproj = eval_selectivity(U, target_point, grid, rng_seed=seed)
            # selection
            if y < F_vals[i]:
                X_new[i] = Uproj
                F_new[i] = y

        X, F_vals = X_new, F_new

        # Best of this generation
        b_idx = int(np.argmin(F_vals))
        if F_vals[b_idx] < best_val:
            best_val = F_vals[b_idx]
            best_vec = X[b_idx].copy()

        evals_so_far = (gen+1) * NP
        step_best = -float(np.min(F_vals))
        if step_best > best_so_far:
            best_so_far = step_best
            best_at_eval = evals_so_far

        xs_axis.append(evals_so_far)
        step_vals.append(step_best)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "DE",
            "n_rows": grid[0], "n_per_row": grid[1], "N": N,
            "popsize": NP, "F": DE_F, "CR": DE_CR,
            "repeat": repeat,
            "step_index": gen + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": step_best,
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
        }, header_written)

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    return {
        "optimizer": "DE", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": used_budget, "popsize": NP
    }


def run_cem(grid, repeat, target_point, eval_budget):
    N = grid[0] * grid[1]
    seed = SEED_BASE + 5000*repeat + N
    rng = np.random.default_rng(seed)
    tag = make_tag("CEM", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    K = cem_popsize(N)
    iters = max(1, eval_budget // K)
    used_budget = iters * K

    m = np.zeros(N, dtype=float)           # start at 0 mA
    s = np.full(N, CEM_SIGMA0, dtype=float)

    best_so_far = -np.inf
    best_at_eval = 0
    xs_axis, step_vals, best_vals = [], [], []
    header_written = False
    evals_so_far = 0

    for it in tqdm(range(iters), desc=tag, leave=False):
        # Sample & project
        X = m + s * rng.standard_normal(size=(K, N))
        X = np.vstack([project_currents(x, -RANGE, RANGE, ZERO_SUM) for x in X])

        Y = np.empty(K, dtype=float)
        for i in range(K):
            y, _, _ = eval_selectivity(X[i], target_point, grid, rng_seed=seed)
            Y[i] = y
        evals_so_far += K

        # Rank by MIN(y) -> MAX(selectivity)
        idx = np.argsort(Y)
        elite_k = max(1, int(np.ceil(CEM_ELITE_FRAC * K)))
        elites = X[idx[:elite_k]]

        # Update parameters with smoothing
        new_m = elites.mean(axis=0)
        new_s = elites.std(axis=0)
        m = (1 - CEM_ALPHA) * m + CEM_ALPHA * new_m
        s = (1 - CEM_ALPHA) * s + CEM_ALPHA * np.maximum(new_s, CEM_SIGMA_MIN)

        # Clip to bounds (center & spread)
        m = np.clip(m, -RANGE, RANGE)
        s = np.clip(s, CEM_SIGMA_MIN, RANGE)

        step_best = -float(np.min(Y))
        if step_best > best_so_far:
            best_so_far = step_best
            best_at_eval = evals_so_far

        xs_axis.append(evals_so_far)
        step_vals.append(step_best)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "CEM",
            "n_rows": grid[0], "n_per_row": grid[1], "N": N,
            "popsize": K, "elite_frac": CEM_ELITE_FRAC, "alpha": CEM_ALPHA,
            "repeat": repeat,
            "step_index": it + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": step_best,
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
        }, header_written)
        header_written = True

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    return {
        "optimizer": "CEM", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": used_budget, "popsize": K
    }



def run_sa(grid, repeat, target_point, eval_budget):
    N = grid[0] * grid[1]
    seed = SEED_BASE + 6000*repeat + N
    rng = np.random.default_rng(seed)
    tag = make_tag("SA", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    # Warm-up to set T0 by objective spread (deduct from budget)
    warm = min(SA_WARMUP_SAMPLES, max(0, eval_budget // 10))
    temps = []
    if warm > 0:
        vals = []
        for _ in range(warm):
            x = rng.uniform(-RANGE, RANGE, N)
            x = project_currents(x, -RANGE, RANGE, ZERO_SUM)
            y, _, _ = eval_selectivity(x, target_point, grid, rng_seed=seed)
            vals.append(float(y))
        vals = np.array(vals)
        T0 = np.std(vals) if np.std(vals) > 1e-12 else 1.0
    else:
        T0 = 1.0

    remaining = max(1, eval_budget - warm)

    # Start from best warm-up or random feasible
    if warm > 0:
        best_idx = int(np.argmin(vals))
        x = project_currents(rng.uniform(-RANGE, RANGE, N), -RANGE, RANGE, ZERO_SUM) if best_idx < 0 else None
        x = project_currents(rng.uniform(-RANGE, RANGE, N), -RANGE, RANGE, ZERO_SUM) if x is None else x
    else:
        x = project_currents(rng.uniform(-RANGE, RANGE, N), -RANGE, RANGE, ZERO_SUM)

    y, _, _ = eval_selectivity(x, target_point, grid, rng_seed=seed)

    x_best = x.copy()
    y_best = y

    best_so_far = -float(y_best)
    best_at_eval = warm + 1

    xs_axis  = [warm + 1]
    step_vals = [best_so_far]
    best_vals = [best_so_far]
    header_written = False

    T = T0
    for k in tqdm(range(1, remaining), desc=tag, leave=False):
        # Gaussian proposal + projection
        prop = x + SA_STEP_SIGMA * rng.standard_normal(N)
        prop = project_currents(prop, -RANGE, RANGE, ZERO_SUM)
        y_prop, _, _ = eval_selectivity(prop, target_point, grid, rng_seed=seed)

        # Metropolis acceptance (minimize y)
        dy = y_prop - y
        if dy < 0 or rng.random() < np.exp(-dy / max(T, 1e-12)):
            x, y = prop, y_prop

        # Track global best
        if y < y_best:
            x_best, y_best = x.copy(), y

        # Update temp
        T *= SA_ALPHA

        evals_so_far = warm + 1 + k
        step_best = -float(y_best)
        if step_best > best_so_far:
            best_so_far = step_best
            best_at_eval = evals_so_far

        # Log sparsely to keep files small
        if k == 1 or k == remaining-1 or k % max(10, N) == 0:
            xs_axis.append(evals_so_far)
            step_vals.append(-float(y))
            best_vals.append(best_so_far)
            log_step(csv_path, {
                "optimizer": "SA",
                "n_rows": grid[0], "n_per_row": grid[1], "N": N,
                "alpha": SA_ALPHA, "step_sigma": SA_STEP_SIGMA, "T0": T0,
                "repeat": repeat,
                "step_index": k + 1,
                "evals_so_far": evals_so_far,
                "step_best_selectivity": -float(y),
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
            }, header_written)
            header_written = True

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    return {
        "optimizer": "SA", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat, "target_point": target_point,
        "used_evals": warm + remaining
    }


def run_ms_pairs_sweep_then_cma(grid, repeat, target_point, eval_budget,
                                I0=0.8e-3, top_k_pairs=10):
    """
    Stage 0: exhaustive sweep over all electrode pairs at fixed current +/-I0.
             Each pattern is: +I0 at i, -I0 at j, others 0.
             Uses at most N*(N-1)/2 evaluations (or fewer if budget smaller).

    Stage 1: 12-D CMA-ES warm-started at the weighted mean of the top-K pairs
             (by selectivity). Remaining budget is used for CMA-ES.

    No zero-sum projection, grounded boundary only.
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row
    seed = SEED_BASE + 7000*repeat + N
    rng = np.random.default_rng(seed)

    tag = make_tag("MS_SWEEP_CMA", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    # ---------------- Stage 0: Pair sweep ----------------
    pairs = [(i, j) for i in range(N) for j in range(N) if j > i]
    max_pairs = len(pairs)

    # If budget too small, subsample pairs
    if max_pairs > eval_budget:
        rng.shuffle(pairs)
        pairs = pairs[:eval_budget]
        max_pairs = len(pairs)

    pair_results = []
    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    xs_axis, step_vals, best_vals = [], [], []
    header_written = False

    for idx, (i, j) in enumerate(tqdm(pairs,
                                      desc=f"{tag}_stage0_pairsweep",
                                      leave=False)):
        currents = np.zeros(N, dtype=float)
        currents[i] = +I0
        currents[j] = -I0

        y, x_used = eval_selectivity_grounded(currents, target_point, grid, rng_seed=seed)
        sel = -y
        evals_so_far += 1

        pair_results.append((sel, x_used.copy()))

        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = evals_so_far

        xs_axis.append(evals_so_far)
        step_vals.append(sel)
        best_vals.append(best_so_far)

        # Log sparsely to limit file size
        if (idx == 0) or (idx == max_pairs - 1) or ((idx + 1) % 10 == 0):
            log_step(csv_path, {
                "optimizer": "MS_SWEEP_CMA",
                "stage": 0,
                "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
                "repeat": repeat,
                "evals_so_far": evals_so_far,
                "step_best_selectivity": sel,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
                'currents': currents
            }, header_written)
            header_written = True

        if evals_so_far >= eval_budget:
            break

    # If we used all budget in sweep, just return best sweep result
    if evals_so_far >= eval_budget or len(pair_results) == 0:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "MS_SWEEP_CMA",
            "tag": tag,
            "best": float(best_so_far),
            "best_found_at_eval": int(best_at_eval),
            "N": N, "grid": grid, "repeat": repeat,
            "target_point": target_point,
            "used_evals": evals_so_far,
        }

    # ---------------- Build warm-start mean from top-K pairs ----------------
    pair_results.sort(key=lambda t: t[0], reverse=True)
    k = min(top_k_pairs, len(pair_results))
    top = pair_results[:k]
    # Simple average of current vectors from top-K
    x0 = np.mean([c for (_, c) in top], axis=0)
    x0 = np.clip(x0, -RANGE, RANGE)

    # Rough sigma based on variation among top patterns
    if k > 1:
        stacked = np.stack([c for (_, c) in top], axis=0)
        sigma_est = np.std(stacked, axis=0).mean()
        sigma0 = max(0.1 * RANGE, min(0.5 * RANGE, sigma_est))
    else:
        sigma0 = 0.3 * RANGE

    # ---------------- Stage 1: 12-D CMA-ES ----------------
    remaining = max(0, eval_budget - evals_so_far)
    if remaining <= 0:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "MS_SWEEP_CMA",
            "tag": tag,
            "best": float(best_so_far),
            "best_found_at_eval": int(best_at_eval),
            "N": N, "grid": grid, "repeat": repeat,
            "target_point": target_point,
            "used_evals": evals_so_far,
        }

    popsize = cma_popsize(N)
    iters = max(1, remaining // popsize)

    es = CMAEvolutionStrategy(
        x0, sigma0,
        {"popsize": popsize,
         "verb_disp": 0,
         "seed": seed + 1,
         "bounds": [-RANGE, RANGE]}
    )

    for it in tqdm(range(iters), desc=f"{tag}_stage1_cma12d", leave=False):
        X_ask = es.ask()
        Y = []
        gen_best = -np.inf

        for x in X_ask:
            x_c = np.clip(x, -RANGE, RANGE)
            y, _ = eval_selectivity_grounded(x_c, target_point, grid, rng_seed=seed)
            sel = -y
            Y.append(y)

            evals_so_far += 1
            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
            if sel > gen_best:
                gen_best = sel

        es.tell(X_ask, Y)

        xs_axis.append(evals_so_far)
        step_vals.append(gen_best)
        best_vals.append(best_so_far)

        log_step(csv_path, {
            "optimizer": "MS_SWEEP_CMA",
            "stage": 1,
            "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
            "repeat": repeat,
            "step_index": it + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": gen_best,
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
        }, header_written)

        if evals_so_far >= eval_budget:
            break

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "MS_SWEEP_CMA",
        "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat,
        "target_point": target_point,
        "used_evals": evals_so_far,
    }


def run_ms_hierarchical_cma(
    grid,
    repeat,
    target_point,
    eval_budget,
    frac_sweep=0.25,
    frac_subspace=0.35,
    I0=0.8e-3,
    top_k_pairs=20,
    max_subspace_dim=8,
    plateau_eps=5e-3,
    stall_gens=5,
    min_gens=3,
):
    """
    Multi-Stage Hierarchical CMA-ES (MS_HIER_CMA)
    ============================================

    Stage 0 (coarse, structured):
        - Sweep a subset of all +/-I0 dipolar pairs (i<j), grounded only.
        - Collect best pair patterns.

    Stage 1 (data-driven low-dim subspace CMA):
        - Take top-K pair patterns, compute PCA/SVD basis in R^N.
        - Optimise coefficients 'a' in a d-dimensional subspace:
              currents = m + B @ a
        - Plateau-based early stopping; unused evals go to Stage 2.

    Stage 2 (full 12-D CMA-ES):
        - CMA-ES in full current space, starting from best currents so far,
          using remaining budget.

    No zero-sum projection: all evaluation via eval_selectivity_grounded
    (simple clipping to [-RANGE, RANGE]).
    """
    from cma import CMAEvolutionStrategy

    n_rows, n_per_row = grid
    N = n_rows * n_per_row
    seed = SEED_BASE + 7300 * repeat + N
    rng = np.random.default_rng(seed)

    tag = make_tag("MS_HIER_CMA", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)

    xs_axis, step_vals, best_vals = [], [], []
    header_written = False

    # --------------------------------------------------------------
    # Stage 0: Pair sweep (coarse structured exploration)
    # --------------------------------------------------------------
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    max_pairs_possible = len(all_pairs)
    max_pairs_budget = int(frac_sweep * eval_budget)

    n_pairs_eval = min(max_pairs_possible, max_pairs_budget)
    pair_results = []

    if n_pairs_eval > 0:
        rng.shuffle(all_pairs)
        pairs = all_pairs[:n_pairs_eval]

        for idx, (i, j) in enumerate(
            tqdm(pairs, desc=f"{tag}_stage0_pairsweep", leave=False)
        ):
            currents = np.zeros(N, dtype=float)
            currents[i] = +I0
            currents[j] = -I0

            y, x_used = eval_selectivity_grounded(
                currents, target_point, grid, rng_seed=seed
            )
            sel = -y
            evals_so_far += 1

            pair_results.append((sel, x_used.copy()))

            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
                best_currents = x_used.copy()

            xs_axis.append(evals_so_far)
            step_vals.append(sel)
            best_vals.append(best_so_far)

            # Log sparsely: row currents = *literal* currents for this eval
            if (
                idx == 0
                or idx == n_pairs_eval - 1
                or ((idx + 1) % 10 == 0)
            ):
                row = {
                    "optimizer": "MS_HIER_CMA",
                    "stage": 0,
                    "n_rows": n_rows,
                    "n_per_row": n_per_row,
                    "N": N,
                    "repeat": repeat,
                    "evals_so_far": evals_so_far,
                    "step_best_selectivity": sel,
                    "best_so_far": best_so_far,
                    "best_found_at_eval": best_at_eval,
                    "target_x": target_point[0],
                    "target_y": target_point[1],
                    "target_z": target_point[2],
                    "Currents": x_used.tolist(),  # literal for this eval
                }
                log_step(csv_path, row, header_written)
                header_written = True

            if evals_so_far >= eval_budget:
                break

    # If we already exhausted the budget, stop here
    if evals_so_far >= eval_budget:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "MS_HIER_CMA",
            "tag": tag,
            "best": float(best_so_far),
            "best_found_at_eval": int(best_at_eval),
            "N": N,
            "grid": grid,
            "repeat": repeat,
            "target_point": target_point,
            "used_evals": evals_so_far,
        }

    # --------------------------------------------------------------
    # Stage 1: Data-driven low-dim subspace CMA (PCA on best pairs)
    # --------------------------------------------------------------
    remaining = eval_budget - evals_so_far

    if remaining > N * 4 and len(pair_results) >= 2:
        pair_results.sort(key=lambda t: t[0], reverse=True)
        K = min(top_k_pairs, len(pair_results))
        top = pair_results[:K]

        M = np.stack([c for (sel, c) in top], axis=0)  # (K, N)
        m_vec = np.mean(M, axis=0)
        M_centered = M - m_vec[None, :]

        U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        d_sub = min(max_subspace_dim, Vt.shape[0], Vt.shape[1])

        if d_sub >= 1:
            B_sub = Vt[:d_sub, :].T  # (N, d_sub)

            budget_stage1 = min(int(frac_subspace * eval_budget), remaining)
            popsize1 = cma_popsize(d_sub)
            iters1 = max(1, budget_stage1 // popsize1)

            a0 = np.zeros(d_sub, dtype=float)
            sigma0_sub = 1.0

            es1 = CMAEvolutionStrategy(
                a0,
                sigma0_sub,
                {
                    "popsize": popsize1,
                    "verb_disp": 0,
                    "seed": seed + 1,
                },
            )

            stage1_last_improve = best_so_far if np.isfinite(best_so_far) else -1e9
            gens_since_improve = 0

            for gen in tqdm(
                range(iters1), desc=f"{tag}_stage1_subspace", leave=False
            ):
                A_ask = es1.ask()
                Y = []
                gen_best = -np.inf
                gen_best_currents = None  # <-- literal for this generation

                for a in A_ask:
                    currents = m_vec + B_sub @ a
                    currents = np.clip(currents, -RANGE, RANGE)
                    y, x_used = eval_selectivity_grounded(
                        currents, target_point, grid, rng_seed=seed
                    )
                    sel = -y
                    Y.append(y)

                    evals_so_far += 1

                    # global best
                    if sel > best_so_far:
                        best_so_far = sel
                        best_at_eval = evals_so_far
                        best_currents = x_used.copy()

                    # per-generation best (for logging)
                    if sel > gen_best:
                        gen_best = sel
                        gen_best_currents = x_used.copy()

                    if evals_so_far >= eval_budget:
                        break

                es1.tell(A_ask, Y)

                xs_axis.append(evals_so_far)
                step_vals.append(gen_best)
                best_vals.append(best_so_far)

                # log: currents = literal gen-best currents this generation
                row = {
                    "optimizer": "MS_HIER_CMA",
                    "stage": 1,
                    "subspace_dim": d_sub,
                    "n_rows": n_rows,
                    "n_per_row": n_per_row,
                    "N": N,
                    "repeat": repeat,
                    "step_index": gen + 1,
                    "evals_so_far": evals_so_far,
                    "step_best_selectivity": gen_best,
                    "best_so_far": best_so_far,
                    "best_found_at_eval": best_at_eval,
                    "target_x": target_point[0],
                    "target_y": target_point[1],
                    "target_z": target_point[2],
                    "Currents": (
                        gen_best_currents.tolist()
                        if gen_best_currents is not None
                        else None
                    ),
                }
                log_step(csv_path, row, header_written)
                header_written = True

                # plateau check on global best
                if best_so_far > stage1_last_improve + plateau_eps:
                    stage1_last_improve = best_so_far
                    gens_since_improve = 0
                else:
                    gens_since_improve += 1

                if (
                    gen + 1 >= min_gens
                    and gens_since_improve >= stall_gens
                ):
                    break

                if evals_so_far >= eval_budget:
                    break

    # --------------------------------------------------------------
    # Stage 2: Full 12-D CMA-ES, warm-started
    # --------------------------------------------------------------
    remaining = eval_budget - evals_so_far
    if remaining > 0:
        popsize2 = cma_popsize(N)
        iters2 = max(1, remaining // popsize2)

        if not np.isfinite(best_so_far):
            x0 = rng.uniform(-RANGE, RANGE, N)
        else:
            x0 = best_currents.copy()

        sigma0_full = 0.6 * RANGE

        es2 = CMAEvolutionStrategy(
            x0,
            sigma0_full,
            {
                "popsize": popsize2,
                "verb_disp": 0,
                "seed": seed + 2,
                "bounds": [-RANGE, RANGE],
            },
        )

        for gen in tqdm(
            range(iters2), desc=f"{tag}_stage2_full", leave=False
        ):
            X_ask = es2.ask()
            Y = []
            gen_best = -np.inf
            gen_best_currents = None

            for x in X_ask:
                x_c = np.clip(x, -RANGE, RANGE)
                y, x_used = eval_selectivity_grounded(
                    x_c, target_point, grid, rng_seed=seed
                )
                sel = -y
                Y.append(y)

                evals_so_far += 1

                # global best
                if sel > best_so_far:
                    best_so_far = sel
                    best_at_eval = evals_so_far
                    best_currents = x_used.copy()

                # per-generation best
                if sel > gen_best:
                    gen_best = sel
                    gen_best_currents = x_used.copy()

                if evals_so_far >= eval_budget:
                    break

            es2.tell(X_ask, Y)

            xs_axis.append(evals_so_far)
            step_vals.append(gen_best)
            best_vals.append(best_so_far)

            row = {
                "optimizer": "MS_HIER_CMA",
                "stage": 2,
                "n_rows": n_rows,
                "n_per_row": n_per_row,
                "N": N,
                "repeat": repeat,
                "step_index": gen + 1,
                "evals_so_far": evals_so_far,
                "step_best_selectivity": gen_best,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
                "Currents": (
                    gen_best_currents.tolist()
                    if gen_best_currents is not None
                    else None
                ),
            }
            log_step(csv_path, row, header_written)
            header_written = True

            if evals_so_far >= eval_budget:
                break

    # One final plot over the *whole* run (all stages)
    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "MS_HIER_CMA",
        "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N,
        "grid": grid,
        "repeat": repeat,
        "target_point": target_point,
        "used_evals": evals_so_far,
    }

def run_ms_adaptive_cma(
     grid,
    repeat,
    target_point,
    eval_budget,
    frac_sweep=0.15,
    eta=3,
    max_trials=8,
    min_cma_evals=80,
    I0=0.8e-3,
):
    """
    Multi-Stage Hyperband-style CMA-ES (MS_HYPERBAND_CMA)
    =====================================================

    Literature inspiration:
      - Successive Halving / Hyperband from hyperparameter optimisation
        (e.g. Li et al., 2017): allocate small budgets to many
        configurations, then repeatedly keep the best fraction and
        increase their budget.

    Here:
      Stage 0: Pair sweep
        - Use fixed dipolar +/-I0 patterns over all electrode pairs.
        - Evaluate up to frac_sweep * eval_budget pairs.
        - Keep best max_trials pairs as structured full-space seeds.

      Stage 1: Hyperband / Successive-Halving CMA-ES
        - Create T <= max_trials CMA-ES runs in full N-d space,
          each initialised from one of the best pair patterns.
        - Rung 0: run all T for a small budget b0 (in function evaluations).
        - Rung 1..s-1: keep top ~1/eta runs, give them larger per-run
          budgets b1 = b0*eta, b2 = b0*eta^2, etc.
        - All CMA runs are *full-space* and re-use their internal state:
          we continue optimisation between rungs.

      Stage 2: Final exploitation
        - Pick the single best trial found so far.
        - Spend any remaining evaluations continuing that CMA-ES instance.

    This combines:
        - Structured warm starts from the sweep,
        - Multi-start CMA-ES,
        - Hyperband-style budget allocation,
      which (to my knowledge) has not been applied to current steering
      in peripheral nerve / RPNI neuromodulation.

    All evaluations use eval_selectivity_grounded (box-clipped currents).
    """

    from cma import CMAEvolutionStrategy

    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    seed = SEED_BASE + 8200 * repeat + N
    rng = np.random.default_rng(seed)

    tag = make_tag("MS_HYPERBAND_CMA", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)

    xs_axis, step_vals, best_vals = [], [], []
    header_written = False

    # --------------------------------------------------------------
    # Stage 0: Pair sweep (coarse exploration)
    # --------------------------------------------------------------
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    n_pairs_total = len(all_pairs)
    sweep_budget = int(frac_sweep * eval_budget)
    n_pairs_eval = min(n_pairs_total, sweep_budget)

    pair_records = []  # dicts: {"sel", "currents", "i", "j"}

    if n_pairs_eval > 0:
        rng.shuffle(all_pairs)
        sweep_pairs = all_pairs[:n_pairs_eval]

        for idx, (i, j) in enumerate(
            tqdm(sweep_pairs, desc=f"{tag}_stage0_pairsweep", leave=False)
        ):
            currents = np.zeros(N, dtype=float)
            currents[i] = +I0
            currents[j] = -I0

            y, x_used = eval_selectivity_grounded(
                currents, target_point, grid, rng_seed=seed
            )
            sel = -y
            evals_so_far += 1

            pair_records.append(
                {"sel": sel, "currents": x_used.copy(), "i": i, "j": j}
            )

            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
                best_currents = x_used.copy()

            xs_axis.append(evals_so_far)
            step_vals.append(sel)
            best_vals.append(best_so_far)

            # sparse logging
            if (
                idx == 0
                or idx == n_pairs_eval - 1
                or ((idx + 1) % 10 == 0)
            ):
                row = {
                    "optimizer": "MS_HYPERBAND_CMA",
                    "stage": 0,
                    "n_rows": n_rows,
                    "n_per_row": n_per_row,
                    "N": N,
                    "repeat": repeat,
                    "evals_so_far": evals_so_far,
                    "step_best_selectivity": sel,
                    "best_so_far": best_so_far,
                    "best_found_at_eval": best_at_eval,
                    "target_x": target_point[0],
                    "target_y": target_point[1],
                    "target_z": target_point[2],
                    "currents": x_used.tolist(),
                    "pair_i": i,
                    "pair_j": j,
                }
                log_step(csv_path, row, header_written)
                header_written = True

            if evals_so_far >= eval_budget:
                break

    # Early exit if sweep used all budget or nothing was evaluated
    if evals_so_far >= eval_budget or len(pair_records) == 0:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "MS_HYPERBAND_CMA",
            "tag": tag,
            "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
            "best_found_at_eval": int(best_at_eval),
            "N": N,
            "grid": grid,
            "repeat": repeat,
            "target_point": target_point,
            "used_evals": evals_so_far,
        }

    # --------------------------------------------------------------
    # Stage 1: Hyperband / Successive-Halving CMA-ES (full N-d)
    # --------------------------------------------------------------
    # Sort sweep patterns by selectivity
    pair_records_sorted = sorted(pair_records, key=lambda r: r["sel"], reverse=True)

    T = min(max_trials, len(pair_records_sorted))  # number of CMA starts
    remaining_for_cma = eval_budget - evals_so_far

    # If almost no budget left, fall back to one CMA from best pair
    if remaining_for_cma <= min_cma_evals:
        x0 = pair_records_sorted[0]["currents"].copy()
        popsize = cma_popsize(N)
        iters = max(1, remaining_for_cma // popsize)

        es = CMAEvolutionStrategy(
            x0,
            CMA_SIGMA0,
            {"popsize": popsize, "verb_disp": 0,
             "seed": seed + 1, "bounds": [-RANGE, RANGE]},
        )

        for gen in tqdm(
            range(iters), desc=f"{tag}_fallback_cma", leave=False
        ):
            X_ask = es.ask()
            Y = []
            gen_best = -np.inf
            gen_best_curr = None

            for x in X_ask:
                if evals_so_far >= eval_budget:
                    break
                x_c = np.clip(x, -RANGE, RANGE)
                y, x_used = eval_selectivity_grounded(
                    x_c, target_point, grid, rng_seed=seed
                )
                sel = -y
                Y.append(y)
                evals_so_far += 1

                if sel > best_so_far:
                    best_so_far = sel
                    best_at_eval = evals_so_far
                    best_currents = x_used.copy()

                if sel > gen_best:
                    gen_best = sel
                    gen_best_curr = x_used.copy()

            if len(Y) == 0:
                break

            es.tell(X_ask[: len(Y)], Y)

            xs_axis.append(evals_so_far)
            step_vals.append(gen_best)
            best_vals.append(best_so_far)

            row = {
                "optimizer": "MS_HYPERBAND_CMA",
                "stage": 1,
                "rung": -1,
                "trial_id": 0,
                "n_rows": n_rows,
                "n_per_row": n_per_row,
                "N": N,
                "repeat": repeat,
                "step_index": gen + 1,
                "evals_so_far": evals_so_far,
                "step_best_selectivity": gen_best,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
                "currents": (
                    gen_best_curr.tolist() if gen_best_curr is not None else None
                ),
            }
            log_step(csv_path, row, header_written)
            header_written = True

            if evals_so_far >= eval_budget:
                break

        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "MS_HYPERBAND_CMA",
            "tag": tag,
            "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
            "best_found_at_eval": int(best_at_eval),
            "N": N,
            "grid": grid,
            "repeat": repeat,
            "target_point": target_point,
            "used_evals": evals_so_far,
        }

    # Decide number of rungs s and base budget b0, Hyperband-style
    # We aim for s in {3,2,1} such that b0 >= min_cma_evals.
    s = 3
    while s > 1:
        denom = s * T
        b0_candidate = remaining_for_cma // max(denom, 1)
        if b0_candidate >= min_cma_evals:
            break
        s -= 1
    if s <= 1:
        s = 1
    b0 = max(min_cma_evals, remaining_for_cma // max(s * T, 1))

    # Precompute budgets per rung (per trial) and target trial counts
    rung_budgets = []
    rung_trial_counts = []
    n_current = T
    for r in range(s):
        rung_budgets.append(b0 * (eta**r))
        rung_trial_counts.append(max(1, int(np.floor(n_current))))
        n_current = max(1, int(np.floor(n_current / eta)))

    # Create CMA-ES trials
    trials = []
    for t_idx in range(T):
        x0_t = pair_records_sorted[t_idx]["currents"].copy()
        popsize_t = cma_popsize(N)
        es_t = CMAEvolutionStrategy(
            x0_t,
            CMA_SIGMA0,
            {"popsize": popsize_t, "verb_disp": 0,
             "seed": seed + 10 + t_idx, "bounds": [-RANGE, RANGE]},
        )
        trials.append(
            {
                "id": t_idx,
                "es": es_t,
                "popsize": popsize_t,
                "best_sel": pair_records_sorted[t_idx]["sel"],
                "best_currents": x0_t.copy(),
            }
        )

    # Helper to get trial by id
    def get_trial(tr_id):
        for tr in trials:
            if tr["id"] == tr_id:
                return tr
        return None

    active_ids = [tr["id"] for tr in trials]

    # Run rungs
    for rung_idx in range(s):
        if evals_so_far >= eval_budget or len(active_ids) == 0:
            break

        per_trial_budget = rung_budgets[rung_idx]
        # We may only keep top n_keep for the *next* rung
        if rung_idx < s - 1:
            n_keep_next = max(1, int(np.floor(len(active_ids) / eta)))
        else:
            n_keep_next = len(active_ids)

        rung_results = []

        for tr_id in list(active_ids):
            if evals_so_far >= eval_budget:
                break

            tr = get_trial(tr_id)
            if tr is None:
                continue

            es = tr["es"]
            popsize_t = tr["popsize"]

            # Translate budget to generation count
            local_budget = min(per_trial_budget, eval_budget - evals_so_far)
            if local_budget <= 0:
                continue

            max_gens = max(1, int(local_budget // popsize_t))

            for gen in tqdm(
                range(max_gens),
                desc=f"{tag}_stage1_rung{rung_idx}_trial{tr_id}",
                leave=False,
            ):
                if evals_so_far >= eval_budget:
                    break

                X_ask = es.ask()
                Y = []
                gen_best = -np.inf
                gen_best_curr = None

                for x in X_ask:
                    if evals_so_far >= eval_budget:
                        break

                    x_c = np.clip(x, -RANGE, RANGE)
                    y, x_used = eval_selectivity_grounded(
                        x_c, target_point, grid, rng_seed=seed
                    )
                    sel = -y
                    Y.append(y)
                    evals_so_far += 1

                    # per-trial best
                    if sel > tr["best_sel"]:
                        tr["best_sel"] = sel
                        tr["best_currents"] = x_used.copy()

                    # global best
                    if sel > best_so_far:
                        best_so_far = sel
                        best_at_eval = evals_so_far
                        best_currents = x_used.copy()

                    if sel > gen_best:
                        gen_best = sel
                        gen_best_curr = x_used.copy()

                if len(Y) == 0:
                    break

                es.tell(X_ask[: len(Y)], Y)

                xs_axis.append(evals_so_far)
                step_vals.append(gen_best)
                best_vals.append(best_so_far)

                row = {
                    "optimizer": "MS_HYPERBAND_CMA",
                    "stage": 1,
                    "rung": rung_idx,
                    "trial_id": tr_id,
                    "n_rows": n_rows,
                    "n_per_row": n_per_row,
                    "N": N,
                    "repeat": repeat,
                    "step_index": gen + 1,
                    "evals_so_far": evals_so_far,
                    "step_best_selectivity": gen_best,
                    "best_so_far": best_so_far,
                    "best_found_at_eval": best_at_eval,
                    "target_x": target_point[0],
                    "target_y": target_point[1],
                    "target_z": target_point[2],
                    "currents": (
                        gen_best_curr.tolist()
                        if gen_best_curr is not None
                        else None
                    ),
                }
                log_step(csv_path, row, header_written)
                header_written = True

            rung_results.append(
                {"trial_id": tr_id, "best_sel": tr["best_sel"]}
            )

        # Select survivors for next rung
        if rung_idx < s - 1 and len(rung_results) > 0:
            rung_results_sorted = sorted(
                rung_results, key=lambda r: r["best_sel"], reverse=True
            )
            active_ids = [
                r["trial_id"] for r in rung_results_sorted[:n_keep_next]
            ]

    # --------------------------------------------------------------
    # Stage 2: Final exploitation of best trial
    # --------------------------------------------------------------
    remaining_final = eval_budget - evals_so_far
    if remaining_final > 0 and len(trials) > 0:
        best_trial = max(trials, key=lambda tr: tr["best_sel"])
        es2 = best_trial["es"]
        popsize2 = best_trial["popsize"]

        max_gens2 = max(1, int(remaining_final // popsize2))

        for gen in tqdm(
            range(max_gens2), desc=f"{tag}_stage2_exploit", leave=False
        ):
            if evals_so_far >= eval_budget:
                break

            X_ask = es2.ask()
            Y = []
            gen_best = -np.inf
            gen_best_curr = None

            for x in X_ask:
                if evals_so_far >= eval_budget:
                    break

                x_c = np.clip(x, -RANGE, RANGE)
                y, x_used = eval_selectivity_grounded(
                    x_c, target_point, grid, rng_seed=seed
                )
                sel = -y
                Y.append(y)
                evals_so_far += 1

                if sel > best_trial["best_sel"]:
                    best_trial["best_sel"] = sel
                    best_trial["best_currents"] = x_used.copy()

                if sel > best_so_far:
                    best_so_far = sel
                    best_at_eval = evals_so_far
                    best_currents = x_used.copy()

                if sel > gen_best:
                    gen_best = sel
                    gen_best_curr = x_used.copy()

            if len(Y) == 0:
                break

            es2.tell(X_ask[: len(Y)], Y)

            xs_axis.append(evals_so_far)
            step_vals.append(gen_best)
            best_vals.append(best_so_far)

            row = {
                "optimizer": "MS_HYPERBAND_CMA",
                "stage": 2,
                "trial_id": best_trial["id"],
                "n_rows": n_rows,
                "n_per_row": n_per_row,
                "N": N,
                "repeat": repeat,
                "step_index": gen + 1,
                "evals_so_far": evals_so_far,
                "step_best_selectivity": gen_best,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
                "currents": (
                    gen_best_curr.tolist()
                    if gen_best_curr is not None
                    else None
                ),
            }
            log_step(csv_path, row, header_written)
            header_written = True

            if evals_so_far >= eval_budget:
                break

    # --------------------------------------------------------------
    # Final plot and return
    # --------------------------------------------------------------
    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "MS_HYPERBAND_CMA",
        "tag": tag,
        "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
        "best_found_at_eval": int(best_at_eval),
        "N": N,
        "grid": grid,
        "repeat": repeat,
        "target_point": target_point,
        "used_evals": evals_so_far,
    }

def run_ms_sweep_then_bo(
    grid,
    repeat,
    target_point,
    eval_budget,
    frac_sweep=0.2,
    I0=0.8e-3,
    top_k_init=12,
    acq_func="EI",
    kappa=1.96,
    xi=0.01,
):
    """
    Multi-Stage Sweep + Bayesian Optimization (MS_SWEEP_BO)
    ======================================================

    Stage 0 (pair sweep, structured exploration):
        - Sweep +/-I0 dipolar pairs over electrodes (i<j), grounded boundary only.
        - Evaluate up to frac_sweep * eval_budget pairs (capped at all pairs).
        - Collect selectivity and full current vectors for each pair; track best.

    Stage 1 (full-space Bayesian optimization):
        - Use the top-K pair patterns as *pseudo-observations* to initialise a
          Gaussian Process model (no extra simulator calls).
        - Run sequential BO in the full N-dimensional current space with EI:
              - Bounds: [-RANGE, RANGE]^N
              - Acquisition: EI with kappa, xi tuned for moderate exploration.
        - All new evaluations are real simulator calls and count toward the
          evaluation budget.
        - Optimisation is directly in current space; clipping to [-RANGE, RANGE]
          is applied before simulation (no zero-sum constraint, grounded only).

    Notes / intended behaviour for N=12:
        - With eval_budget = 200*N = 2400 and frac_sweep=0.2, the full
          pair space (66 pairs) is swept; the rest (~2330 evals) goes to BO.
        - top_k_init ~ 12 gives a good trade-off between using informative
          pair seeds and keeping the GP model numerically stable.

    Returns:
        dict with keys:
            - optimizer: "MS_SWEEP_BO"
            - tag: run identifier
            - best: best selectivity found
            - best_found_at_eval: evaluation index where best was found
            - N, grid, repeat, target_point, used_evals
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    seed = SEED_BASE + 8400 * repeat + N
    rng = np.random.default_rng(seed)

    tag = make_tag("MS_SWEEP_BO", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)

    xs_axis, step_vals, best_vals = [], [], []
    header_written = False

    # --------------------------------------------------------------
    # Stage 0: Pair sweep (fixed dipolar +/-I0)
    # --------------------------------------------------------------
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    max_pairs_possible = len(all_pairs)

    max_pairs_budget = int(frac_sweep * eval_budget)
    n_pairs_eval = min(max_pairs_possible, max_pairs_budget)

    pair_results = []  # list of (sel, currents) for later BO warm-start

    if n_pairs_eval > 0:
        rng.shuffle(all_pairs)
        pairs = all_pairs[:n_pairs_eval]

        for idx, (i, j) in enumerate(
            tqdm(pairs, desc=f"{tag}_stage0_pairsweep", leave=False)
        ):
            currents = np.zeros(N, dtype=float)
            currents[i] = +I0
            currents[j] = -I0

            # Grounded boundary, simple clipping only
            y, x_used = eval_selectivity_grounded(
                currents, target_point, grid, rng_seed=seed
            )
            sel = -y
            evals_so_far += 1

            pair_results.append((sel, x_used.copy()))

            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
                best_currents = x_used.copy()

            xs_axis.append(evals_so_far)
            step_vals.append(sel)
            best_vals.append(best_so_far)

            # Sparse logging to keep CSV manageable
            if (
                idx == 0
                or idx == n_pairs_eval - 1
                or ((idx + 1) % 10 == 0)
            ):
                row = {
                    "optimizer": "MS_SWEEP_BO",
                    "stage": 0,
                    "n_rows": n_rows,
                    "n_per_row": n_per_row,
                    "N": N,
                    "repeat": repeat,
                    "evals_so_far": evals_so_far,
                    "step_best_selectivity": sel,
                    "best_so_far": best_so_far,
                    "best_found_at_eval": best_at_eval,
                    "target_x": target_point[0],
                    "target_y": target_point[1],
                    "target_z": target_point[2],
                    "currents": x_used.tolist(),
                }
                log_step(csv_path, row, header_written)
                header_written = True

            if evals_so_far >= eval_budget:
                break

    # If no budget left or no sweep data, bail early
    if evals_so_far >= eval_budget or len(pair_results) == 0:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "MS_SWEEP_BO",
            "tag": tag,
            "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
            "best_found_at_eval": int(best_at_eval),
            "N": N,
            "grid": grid,
            "repeat": repeat,
            "target_point": target_point,
            "used_evals": evals_so_far,
        }

    # --------------------------------------------------------------
    # Stage 1: Full-space BO warm-started from sweep
    # --------------------------------------------------------------
    remaining_budget = eval_budget - evals_so_far
    if remaining_budget <= 0:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "MS_SWEEP_BO",
            "tag": tag,
            "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
            "best_found_at_eval": int(best_at_eval),
            "N": N,
            "grid": grid,
            "repeat": repeat,
            "target_point": target_point,
            "used_evals": evals_so_far,
        }

    # Sort pair patterns by selectivity (best first)
    pair_results.sort(key=lambda t: t[0], reverse=True)
    k_init = min(top_k_init, len(pair_results))

    # Create BO optimiser over [-RANGE, RANGE]^N
    opt = SkOptimizer(
        [(-RANGE, RANGE)] * N,
        base_estimator="GP",
        acq_func=acq_func,
        acq_func_kwargs={"kappa": kappa, "xi": xi},
        n_initial_points=0,   # we will seed manually from the sweep
        random_state=seed,
    )

    # Seed the GP with top-K swept patterns as pseudo-observations
    # IMPORTANT: we do NOT re-evaluate the simulator; we reuse 'sel' from sweep.
    for sel, currents in pair_results[:k_init]:
        y_val = -float(sel)           # objective = -selectivity
        x_list = np.clip(currents, -RANGE, RANGE).tolist()
        opt.tell(x_list, y_val)

    # Now run sequential BO for the remaining budget
    for i in tqdm(range(remaining_budget), desc=f"{tag}_stage1_bo", leave=False):
        # Ask BO for a candidate point
        x_ask = np.array(opt.ask(), dtype=float)
        x_eval = np.clip(x_ask, -RANGE, RANGE)

        # Evaluate with grounded boundary (no zero-sum)
        y, x_used = eval_selectivity_grounded(
            x_eval, target_point, grid, rng_seed=seed
        )
        sel = -y
        evals_so_far += 1

        # Tell BO about the evaluated point
        opt.tell(x_used.tolist(), float(y))

        # Track global best
        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = evals_so_far
            best_currents = x_used.copy()

        xs_axis.append(evals_so_far)
        step_vals.append(sel)
        best_vals.append(best_so_far)

        # Log sparsely during BO
        if (
            (i == 0)
            or (i == remaining_budget - 1)
            or ((i + 1) % 10 == 0)
        ):
            row = {
                "optimizer": "MS_SWEEP_BO",
                "stage": 1,
                "n_rows": n_rows,
                "n_per_row": n_per_row,
                "N": N,
                "repeat": repeat,
                "eval_index": evals_so_far,
                "evals_so_far": evals_so_far,
                "current_selectivity": sel,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
                "currents": x_used.tolist(),
            }
            log_step(csv_path, row, header_written)
            header_written = True

        if evals_so_far >= eval_budget:
            break

    # Final plot over full run (sweep + BO)
    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "MS_SWEEP_BO",
        "tag": tag,
        "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
        "best_found_at_eval": int(best_at_eval),
        "N": N,
        "grid": grid,
        "repeat": repeat,
        "target_point": target_point,
        "used_evals": evals_so_far,
    }

def run_ms_sweep_then_pso(
    grid,
    repeat,
    target_point,
    eval_budget,
    frac_sweep=0.25,
    I0=0.8e-3,
    top_k_pairs=20,
):
    """
    Multi-Stage Sweep + PSO (grounded boundary, no zero-sum)
    =======================================================

    Stage 0: Fixed-current pair sweep (structured exploration)
        - For each unordered electrode pair (i < j), apply dipolar pattern:
              +I0 at i, -I0 at j, others 0.
        - Evaluate via eval_selectivity_grounded (box clipping only).
        - Use at most frac_sweep * eval_budget evaluations (or all pairs if cheaper).
        - Collect all evaluated pair patterns and their selectivities.

    Stage 1: Full-space PSO warm-started from best pairs
        - Use top-K pair patterns as structured initial particles in full N-D space.
        - Remaining particles initialised randomly in [-RANGE, RANGE]^N.
        - PSO hyperparameters identical to run_pso (grounded version):
              popsize = pso_popsize(N)
              w       = PSO_W
              c1, c2  = PSO_C1, PSO_C2
              velocity clamp = PSO_VCLAMP
        - All evaluations via eval_selectivity_grounded (grounded, no zero-sum).
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    seed = SEED_BASE + 9100 * repeat + N
    rng = np.random.default_rng(seed)

    tag = make_tag("MS_SWEEP_PSO_GROUNDED", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)

    xs_axis, step_vals, best_vals = [], [], []
    header_written = False

    # --------------------------------------------------------------
    # Stage 0: Pair sweep (dipolar +/-I0, grounded boundary)
    # --------------------------------------------------------------
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    n_pairs_total = len(all_pairs)

    max_pairs_budget = int(frac_sweep * eval_budget)
    n_pairs_eval = min(n_pairs_total, max_pairs_budget)

    pair_results = []  # list of (sel, currents) for warm-starting PSO

    if n_pairs_eval > 0:
        rng.shuffle(all_pairs)
        pairs = all_pairs[:n_pairs_eval]

        for idx, (i, j) in enumerate(
            tqdm(pairs, desc=f"{tag}_stage0_pairsweep", leave=False)
        ):
            currents = np.zeros(N, dtype=float)
            currents[i] = +I0
            currents[j] = -I0

            # eval_selectivity_grounded: NO zero-sum, just [-RANGE, RANGE] clipping
            y, x_c = eval_selectivity_grounded(
                currents, target_point, grid, rng_seed=seed
            )
            sel = -y
            evals_so_far += 1

            pair_results.append((sel, x_c.copy()))

            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
                best_currents = x_c.copy()

            xs_axis.append(evals_so_far)
            step_vals.append(sel)
            best_vals.append(best_so_far)

            # Log sparsely
            if (
                idx == 0
                or idx == n_pairs_eval - 1
                or ((idx + 1) % 10 == 0)
            ):
                row = {
                    "optimizer": "MS_SWEEP_PSO_GROUNDED",
                    "stage": 0,
                    "n_rows": n_rows,
                    "n_per_row": n_per_row,
                    "N": N,
                    "repeat": repeat,
                    "evals_so_far": evals_so_far,
                    "step_best_selectivity": sel,
                    "best_so_far": best_so_far,
                    "best_found_at_eval": best_at_eval,
                    "target_x": target_point[0],
                    "target_y": target_point[1],
                    "target_z": target_point[2],
                    "currents": x_c.tolist(),
                    "pair_i": i,
                    "pair_j": j,
                }
                log_step(csv_path, row, header_written)
                header_written = True

            if evals_so_far >= eval_budget:
                break

    # If we exhausted budget or never evaluated anything, stop here
    if evals_so_far >= eval_budget or len(pair_results) == 0:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "MS_SWEEP_PSO_GROUNDED",
            "tag": tag,
            "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
            "best_found_at_eval": int(best_at_eval),
            "N": N,
            "grid": grid,
            "repeat": repeat,
            "target_point": target_point,
            "used_evals": evals_so_far,
        }

    # --------------------------------------------------------------
    # Stage 1: PSO in full N-D, warm-started from best pairs
    # --------------------------------------------------------------
    remaining = eval_budget - evals_so_far
    popsize = pso_popsize(N)

    if remaining < popsize:
        # Not enough budget for even one PSO generation
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "MS_SWEEP_PSO_GROUNDED",
            "tag": tag,
            "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
            "best_found_at_eval": int(best_at_eval),
            "N": N,
            "grid": grid,
            "repeat": repeat,
            "target_point": target_point,
            "used_evals": evals_so_far,
        }

    iters = max(1, remaining // popsize)

    # Sort pair patterns by selectivity (best first)
    pair_results.sort(key=lambda t: t[0], reverse=True)
    K = min(top_k_pairs, popsize, len(pair_results))

    rng_pso = np.random.default_rng(seed + 1)

    # Initialise swarm positions: top-K from sweep, rest random
    X = np.zeros((popsize, N), dtype=float)
    for k in range(K):
        X[k] = np.clip(pair_results[k][1], -RANGE, RANGE)
    for i in range(K, popsize):
        X[i] = np.clip(
            rng_pso.uniform(-RANGE, RANGE, N), -RANGE, RANGE
        )

    # Initial velocities
    V = np.zeros_like(X)

    # Evaluate initial swarm
    pbest_pos = X.copy()
    pbest_val = np.empty(popsize, dtype=float)
    for i in range(popsize):
        y, x_c = eval_selectivity_grounded(
            pbest_pos[i], target_point, grid, rng_seed=seed
        )
        pbest_val[i] = y
        X[i] = x_c  # ensure we store clipped position

    g_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_val = pbest_val[g_idx]

    swarm_best_sel = -gbest_val
    evals_so_far += popsize
    if swarm_best_sel > best_so_far:
        best_so_far = swarm_best_sel
        best_at_eval = evals_so_far
        best_currents = gbest_pos.copy()

    xs_axis.append(evals_so_far)
    step_vals.append(swarm_best_sel)
    best_vals.append(best_so_far)

    row = {
        "optimizer": "MS_SWEEP_PSO_GROUNDED",
        "stage": 1,
        "n_rows": n_rows,
        "n_per_row": n_per_row,
        "N": N,
        "popsize": popsize,
        "w": PSO_W,
        "c1": PSO_C1,
        "c2": PSO_C2,
        "repeat": repeat,
        "step_index": 1,
        "evals_so_far": evals_so_far,
        "step_best_selectivity": swarm_best_sel,
        "best_so_far": best_so_far,
        "best_found_at_eval": best_at_eval,
        "target_x": target_point[0],
        "target_y": target_point[1],
        "target_z": target_point[2],
        "currents": gbest_pos.tolist(),
    }
    log_step(csv_path, row, header_written)
    header_written = True

    # Main PSO loop
    for it in tqdm(range(1, iters), desc=f"{tag}_stage1_pso", leave=False):
        if evals_so_far >= eval_budget:
            break

        r1 = rng_pso.random(size=(popsize, N))
        r2 = rng_pso.random(size=(popsize, N))

        V = PSO_W * V + PSO_C1 * r1 * (pbest_pos - X) + PSO_C2 * r2 * (gbest_pos - X)
        if PSO_VCLAMP is not None:
            V = np.clip(V, -PSO_VCLAMP, PSO_VCLAMP)

        X_new = X + V
        X = np.clip(X_new, -RANGE, RANGE)

        it_best_sel = -np.inf
        it_best_currents = None

        for i in range(popsize):
            if evals_so_far >= eval_budget:
                break

            y, x_c = eval_selectivity_grounded(
                X[i], target_point, grid, rng_seed=seed
            )
            X[i] = x_c
            sel = -y
            evals_so_far += 1

            # Update personal best
            if y < pbest_val[i]:
                pbest_val[i] = y
                pbest_pos[i] = x_c

            # Best of this generation
            if sel > it_best_sel:
                it_best_sel = sel
                it_best_currents = x_c.copy()

            # Global best
            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
                best_currents = x_c.copy()

        # Update global best index from pbest_val
        g_idx = int(np.argmin(pbest_val))
        gbest_pos = pbest_pos[g_idx].copy()
        gbest_val = pbest_val[g_idx]

        xs_axis.append(evals_so_far)
        step_vals.append(it_best_sel)
        best_vals.append(best_so_far)

        row = {
            "optimizer": "MS_SWEEP_PSO_GROUNDED",
            "stage": 1,
            "n_rows": n_rows,
            "n_per_row": n_per_row,
            "N": N,
            "popsize": popsize,
            "w": PSO_W,
            "c1": PSO_C1,
            "c2": PSO_C2,
            "repeat": repeat,
            "step_index": it + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": it_best_sel,
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
            "currents": (
                it_best_currents.tolist() if it_best_currents is not None else None
            ),
        }
        log_step(csv_path, row, header_written)

        if evals_so_far >= eval_budget:
            break

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "MS_SWEEP_PSO_GROUNDED",
        "tag": tag,
        "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
        "best_found_at_eval": int(best_at_eval),
        "N": N,
        "grid": grid,
        "repeat": repeat,
        "target_point": target_point,
        "used_evals": evals_so_far,
    }

def run_ms_sweep_then_tr_pso(
    grid,
    repeat,
    target_point,
    eval_budget,
    frac_sweep=0.25,
    I0=0.8e-3,
    popscale=1.0,
    tr_init_radius=None,
    tr_min_radius=0.05,
    tr_expand=1.5,
    tr_shrink=0.5,
    improve_tol=1e-3,
):
    """
    Multi-Stage Sweep + Trust-Region PSO (grounded)
    ==============================================

    Stage 0: Fixed-current pair sweep (same as MS_SWEEP_PSO_GROUNDED)
        - +/-I0 dipolar patterns over all unordered pairs.
        - Evaluated via eval_selectivity_grounded.
        - Best pattern defines initial trust-region centre.

    Stage 1: Trust-Region PSO in full N-D space
        - Centre c starts at best currents from sweep.
        - Radius r starts at tr_init_radius * RANGE (default 0.5*RANGE).
        - Particles' steps are restricted to a ball of radius r around c.
        - If generation best improves global best by > improve_tol ⇒ expand r.
        - Else shrink r.
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row

    if tr_init_radius is None:
        tr_init_radius = 0.5  # fraction of RANGE

    seed = SEED_BASE + 9200 * repeat + N
    rng = np.random.default_rng(seed)

    tag = make_tag("MS_SWEEP_TR_PSO_GROUNDED", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)

    xs_axis, step_vals, best_vals = [], [], []
    header_written = False

    # --------------------------------------------------------------
    # Stage 0: Pair sweep (same as above, grounded only)
    # --------------------------------------------------------------
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    n_pairs_total = len(all_pairs)

    max_pairs_budget = int(frac_sweep * eval_budget)
    n_pairs_eval = min(n_pairs_total, max_pairs_budget)

    sweep_best_sel = -np.inf
    sweep_best_currents = np.zeros(N, dtype=float)

    if n_pairs_eval > 0:
        rng.shuffle(all_pairs)
        pairs = all_pairs[:n_pairs_eval]

        for idx, (i, j) in enumerate(
            tqdm(pairs, desc=f"{tag}_stage0_pairsweep", leave=False)
        ):
            currents = np.zeros(N, dtype=float)
            currents[i] = +I0
            currents[j] = -I0

            y, x_c = eval_selectivity_grounded(
                currents, target_point, grid, rng_seed=seed
            )
            sel = -y
            evals_so_far += 1

            if sel > sweep_best_sel:
                sweep_best_sel = sel
                sweep_best_currents = x_c.copy()

            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
                best_currents = x_c.copy()

            xs_axis.append(evals_so_far)
            step_vals.append(sel)
            best_vals.append(best_so_far)

            # Sparse logging
            if (
                idx == 0
                or idx == n_pairs_eval - 1
                or ((idx + 1) % 10 == 0)
            ):
                row = {
                    "optimizer": "MS_SWEEP_TR_PSO_GROUNDED",
                    "stage": 0,
                    "n_rows": n_rows,
                    "n_per_row": n_per_row,
                    "N": N,
                    "repeat": repeat,
                    "evals_so_far": evals_so_far,
                    "step_best_selectivity": sel,
                    "best_so_far": best_so_far,
                    "best_found_at_eval": best_at_eval,
                    "target_x": target_point[0],
                    "target_y": target_point[1],
                    "target_z": target_point[2],
                    "currents": x_c.tolist(),
                    "pair_i": i,
                    "pair_j": j,
                }
                log_step(csv_path, row, header_written)
                header_written = True

            if evals_so_far >= eval_budget:
                break

    # If no sweep evaluations, or budget exhausted, stop here
    if evals_so_far >= eval_budget or sweep_best_sel == -np.inf:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "MS_SWEEP_TR_PSO_GROUNDED",
            "tag": tag,
            "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
            "best_found_at_eval": int(best_at_eval),
            "N": N,
            "grid": grid,
            "repeat": repeat,
            "target_point": target_point,
            "used_evals": evals_so_far,
        }

    # --------------------------------------------------------------
    # Stage 1: Trust-Region PSO (grounded)
    # --------------------------------------------------------------
    remaining = eval_budget - evals_so_far
    popsize = int(popscale * pso_popsize(N))

    if remaining < popsize:
        save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
        return {
            "optimizer": "MS_SWEEP_TR_PSO_GROUNDED",
            "tag": tag,
            "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
            "best_found_at_eval": int(best_at_eval),
            "N": N,
            "grid": grid,
            "repeat": repeat,
            "target_point": target_point,
            "used_evals": evals_so_far,
        }

    iters = max(1, remaining // popsize)

    rng_tr = np.random.default_rng(seed + 1)

    # Trust-region centre and radius
    centre = sweep_best_currents.copy()
    radius = tr_init_radius * RANGE
    min_radius = tr_min_radius * RANGE

    # Initialise swarm around centre inside trust region
    X = centre[None, :] + radius * rng_tr.standard_normal(size=(popsize, N))
    X = np.clip(X, -RANGE, RANGE)
    V = np.zeros_like(X)

    # Evaluate initial swarm
    pbest_pos = X.copy()
    pbest_val = np.empty(popsize, dtype=float)
    for i in range(popsize):
        y, x_c = eval_selectivity_grounded(
            pbest_pos[i], target_point, grid, rng_seed=seed
        )
        pbest_val[i] = y
        X[i] = x_c

    g_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_val = pbest_val[g_idx]

    gen_best_sel = -gbest_val
    evals_so_far += popsize
    if gen_best_sel > best_so_far:
        best_so_far = gen_best_sel
        best_at_eval = evals_so_far
        best_currents = gbest_pos.copy()

    xs_axis.append(evals_so_far)
    step_vals.append(gen_best_sel)
    best_vals.append(best_so_far)

    row = {
        "optimizer": "MS_SWEEP_TR_PSO_GROUNDED",
        "stage": 1,
        "n_rows": n_rows,
        "n_per_row": n_per_row,
        "N": N,
        "popsize": popsize,
        "w": PSO_W,
        "c1": PSO_C1,
        "c2": PSO_C2,
        "repeat": repeat,
        "step_index": 1,
        "evals_so_far": evals_so_far,
        "step_best_selectivity": gen_best_sel,
        "best_so_far": best_so_far,
        "best_found_at_eval": best_at_eval,
        "target_x": target_point[0],
        "target_y": target_point[1],
        "target_z": target_point[2],
        "currents": gbest_pos.tolist(),
        "tr_radius": radius,
    }
    log_step(csv_path, row, header_written)
    header_written = True

    # Main TR-PSO loop
    for it in tqdm(range(1, iters), desc=f"{tag}_stage1_tr_pso", leave=False):
        if evals_so_far >= eval_budget:
            break

        prev_best = best_so_far

        r1 = rng_tr.random(size=(popsize, N))
        r2 = rng_tr.random(size=(popsize, N))

        V = PSO_W * V + PSO_C1 * r1 * (pbest_pos - X) + PSO_C2 * r2 * (gbest_pos - X)
        if PSO_VCLAMP is not None:
            V = np.clip(V, -PSO_VCLAMP, PSO_VCLAMP)

        # Proposed move
        X_new = X + V

        # Enforce trust region: pull back towards centre if outside radius
        disp = X_new - centre[None, :]
        dist = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, radius / dist)
        X_tr = centre[None, :] + disp * scale

        # Clip to box
        X = np.clip(X_tr, -RANGE, RANGE)

        it_best_sel = -np.inf
        it_best_curr = None

        for i in range(popsize):
            if evals_so_far >= eval_budget:
                break

            y, x_c = eval_selectivity_grounded(
                X[i], target_point, grid, rng_seed=seed
            )
            X[i] = x_c
            sel = -y
            evals_so_far += 1

            # Personal best
            if y < pbest_val[i]:
                pbest_val[i] = y
                pbest_pos[i] = x_c

            # Generation best
            if sel > it_best_sel:
                it_best_sel = sel
                it_best_curr = x_c.copy()

            # Global best
            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
                best_currents = x_c.copy()
                centre = x_c.copy()  # recenter TR on new best

        # Update global best index
        g_idx = int(np.argmin(pbest_val))
        gbest_pos = pbest_pos[g_idx].copy()
        gbest_val = pbest_val[g_idx]

        # Adapt trust-region radius based on improvement
        if best_so_far > prev_best + improve_tol:
            radius = min(tr_expand * radius, RANGE)
        else:
            radius = max(tr_shrink * radius, min_radius)

        xs_axis.append(evals_so_far)
        step_vals.append(it_best_sel)
        best_vals.append(best_so_far)

        row = {
            "optimizer": "MS_SWEEP_TR_PSO_GROUNDED",
            "stage": 1,
            "n_rows": n_rows,
            "n_per_row": n_per_row,
            "N": N,
            "popsize": popsize,
            "w": PSO_W,
            "c1": PSO_C1,
            "c2": PSO_C2,
            "repeat": repeat,
            "step_index": it + 1,
            "evals_so_far": evals_so_far,
            "step_best_selectivity": it_best_sel,
            "best_so_far": best_so_far,
            "best_found_at_eval": best_at_eval,
            "target_x": target_point[0],
            "target_y": target_point[1],
            "target_z": target_point[2],
            "currents": (
                it_best_curr.tolist() if it_best_curr is not None else None
            ),
            "tr_radius": radius,
        }
        log_step(csv_path, row, header_written)

        if evals_so_far >= eval_budget:
            break

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "MS_SWEEP_TR_PSO_GROUNDED",
        "tag": tag,
        "best": float(best_so_far if np.isfinite(best_so_far) else 0.0),
        "best_found_at_eval": int(best_at_eval),
        "N": N,
        "grid": grid,
        "repeat": repeat,
        "target_point": target_point,
        "used_evals": evals_so_far,
    }

def run_ms_shaped_cma(
    grid,
    repeat,
    target_point,
    eval_budget,
    frac_sweep=0.20,
    I0=0.8e-3,
    top_k_stat=15,  # Number of top sweep points to estimate covariance shape
):
    """
    Multi-Stage Covariance-Shaped CMA-ES (MS_SHAPED_CMA)
    ====================================================
    Beats standard warm-started CMA-ES by pre-shaping the covariance matrix.
    
    Stage 0: Pair sweep (same as MS_SWEEP_CMA).
    Stage 1: Shaped CMA-ES.
             - Starts at the single best point found in sweep (not the mean).
             - Calculates the standard deviation of the Top-K sweep vectors.
             - Uses these stds to scale the initial CMA-ES search axes.
             
    Why it wins: It tells CMA-ES immediately which dimensions (electrodes) 
    are sensitive and which are irrelevant, skipping the 'burn-in' phase.
    """
    from cma import CMAEvolutionStrategy

    n_rows, n_per_row = grid
    N = n_rows * n_per_row
    seed = SEED_BASE + 9500 * repeat + N
    rng = np.random.default_rng(seed)

    tag = make_tag("MS_SHAPED_CMA", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)

    xs_axis, step_vals, best_vals = [], [], []
    header_written = False

    # --------------------------------------------------------------
    # Stage 0: Pair sweep
    # --------------------------------------------------------------
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    n_pairs_total = len(all_pairs)
    max_pairs_budget = int(frac_sweep * eval_budget)
    n_pairs_eval = min(n_pairs_total, max_pairs_budget)

    pair_results = []  # Store (selectivity, vector)

    if n_pairs_eval > 0:
        rng.shuffle(all_pairs)
        pairs = all_pairs[:n_pairs_eval]

        for idx, (i, j) in enumerate(
            tqdm(pairs, desc=f"{tag}_stage0_pairsweep", leave=False)
        ):
            currents = np.zeros(N, dtype=float)
            currents[i] = +I0
            currents[j] = -I0

            # Grounded boundary, simple clipping
            y, x_c = eval_selectivity_grounded(
                currents, target_point, grid, rng_seed=seed
            )
            sel = -y
            evals_so_far += 1
            
            # Store for covariance estimation
            pair_results.append((sel, x_c.copy()))

            if sel > best_so_far:
                best_so_far = sel
                best_at_eval = evals_so_far
                best_currents = x_c.copy()

            xs_axis.append(evals_so_far)
            step_vals.append(sel)
            best_vals.append(best_so_far)

            # Sparse logging
            if (idx == 0) or (idx == n_pairs_eval - 1) or ((idx + 1) % 10 == 0):
                log_step(csv_path, {
                    "optimizer": "MS_SHAPED_CMA",
                    "stage": 0,
                    "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
                    "repeat": repeat,
                    "evals_so_far": evals_so_far,
                    "step_best_selectivity": sel,
                    "best_so_far": best_so_far,
                    "best_found_at_eval": best_at_eval,
                    "target_x": target_point[0],
                    "target_y": target_point[1],
                    "target_z": target_point[2],
                    "currents": x_c.tolist()
                }, header_written)
                header_written = True

            if evals_so_far >= eval_budget:
                break

    # --------------------------------------------------------------
    # Stage 1: Covariance-Shaped CMA-ES
    # --------------------------------------------------------------
    remaining = eval_budget - evals_so_far
    
    # If we have enough data to shape the covariance
    if remaining > 0 and len(pair_results) > 0:
        
        # 1. Analyze Top-K results to shape the search
        pair_results.sort(key=lambda x: x[0], reverse=True)
        k_stat = min(top_k_stat, len(pair_results))
        top_vectors = np.array([p[1] for p in pair_results[:k_stat]])
        
        # Calculate empirical standard deviation per dimension
        # Add a small floor (eps) so we don't completely freeze dimensions
        eps_sig = 0.05 * RANGE
        shaped_stds = np.std(top_vectors, axis=0) + eps_sig
        
        # Normalize stds so the max scaling factor is 1.0 (handled by sigma0)
        scale_factors = shaped_stds / (np.max(shaped_stds) + 1e-9)
        
        # Start EXACTLY at the best point found (not the mean)
        x0 = best_currents.copy()
        
        # Initial global step size
        sigma0 = 0.4 * RANGE 

        popsize = cma_popsize(N)
        iters = max(1, remaining // popsize)

        # Pass scaling factors to CMA via 'CMA_stds'
        es = CMAEvolutionStrategy(
            x0,
            sigma0,
            {
                "popsize": popsize,
                "verb_disp": 0,
                "seed": seed + 1,
                "bounds": [-RANGE, RANGE],
                "CMA_stds": scale_factors,  # <--- THE KEY INNOVATION
            },
        )

        for gen in tqdm(range(iters), desc=f"{tag}_stage1_shaped_cma", leave=False):
            X_ask = es.ask()
            Y = []
            gen_best = -np.inf
            gen_best_curr = None

            for x in X_ask:
                if evals_so_far >= eval_budget:
                    break
                
                # Evaluate
                x_c = np.clip(x, -RANGE, RANGE)
                y, x_used = eval_selectivity_grounded(
                    x_c, target_point, grid, rng_seed=seed
                )
                sel = -y
                Y.append(y)
                evals_so_far += 1

                if sel > best_so_far:
                    best_so_far = sel
                    best_at_eval = evals_so_far
                    best_currents = x_used.copy()
                
                if sel > gen_best:
                    gen_best = sel
                    gen_best_curr = x_used.copy()

            if len(Y) == 0:
                break
                
            es.tell(X_ask[:len(Y)], Y)

            xs_axis.append(evals_so_far)
            step_vals.append(gen_best)
            best_vals.append(best_so_far)

            log_step(csv_path, {
                "optimizer": "MS_SHAPED_CMA",
                "stage": 1,
                "n_rows": n_rows, "n_per_row": n_per_row, "N": N,
                "repeat": repeat,
                "step_index": gen + 1,
                "evals_so_far": evals_so_far,
                "step_best_selectivity": gen_best,
                "best_so_far": best_so_far,
                "best_found_at_eval": best_at_eval,
                "target_x": target_point[0],
                "target_y": target_point[1],
                "target_z": target_point[2],
                "currents": gen_best_curr.tolist() if gen_best_curr is not None else None
            }, header_written)

            if evals_so_far >= eval_budget:
                break

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)

    return {
        "optimizer": "MS_SHAPED_CMA",
        "tag": tag,
        "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval),
        "N": N, "grid": grid, "repeat": repeat,
        "target_point": target_point,
        "used_evals": evals_so_far,
    }

def run_ms_sweep_sep_cma(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    Multi-Stage Sweep + Separable CMA-ES (Fixed for partial generation crash)
    =========================================================================
    Stage 0: Pair sweep to find the best coarse solution.
    Stage 1: Sep-CMA-ES initialized at that best solution.
    """
    from cma import CMAEvolutionStrategy
    
    n_rows, n_per_row = grid
    N = n_rows * n_per_row
    seed = SEED_BASE + 15000 * repeat + N
    rng = np.random.default_rng(seed)
    
    tag = make_tag("MS_SWEEP_SEP_CMA", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)
    xs_axis, step_vals, best_vals = [], [], []
    header_written = False

    # --- STAGE 0: Pair Sweep ---
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    sweep_limit = int(0.2 * eval_budget) # Use max 20% of budget for sweep
    
    if len(all_pairs) > sweep_limit:
        rng.shuffle(all_pairs)
        pairs = all_pairs[:sweep_limit]
    else:
        pairs = all_pairs

    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        sel = -y
        evals_so_far += 1
        
        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = evals_so_far
            best_currents = x_used.copy()
            
        xs_axis.append(evals_so_far)
        step_vals.append(sel)
        best_vals.append(best_so_far)
        
        if evals_so_far >= eval_budget:
            break

    # --- STAGE 1: Sep-CMA-ES ---
    remaining = eval_budget - evals_so_far
    if remaining > 0:
        # Start exactly at the sweep winner
        x0 = best_currents.copy()
        
        # Smaller step size because we are arguably close to the solution
        sigma0 = 0.2 * RANGE 
        
        popsize = cma_popsize(N)
        
        es = CMAEvolutionStrategy(
            x0, sigma0,
            {
                "popsize": popsize, 
                "verb_disp": 0, 
                "seed": seed + 1, 
                "bounds": [-RANGE, RANGE],
                "CMA_diagonal": True, 
            }
        )

        while evals_so_far < eval_budget:
            X_ask = es.ask()
            Y = []
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
                Y.append(y)
                evals_so_far += 1
                
                if -y > best_so_far:
                    best_so_far = -y
                    best_at_eval = evals_so_far
            
            # FIX: Do not tell partial population if budget ran out
            if len(Y) < popsize:
                break
            
            es.tell(X_ask, Y)
            
            step_best = -np.min(Y)
            xs_axis.append(evals_so_far)
            step_vals.append(step_best)
            best_vals.append(best_so_far)
            
            log_step(csv_path, {
                "optimizer": "MS_SWEEP_SEP_CMA", "N": N, "repeat": repeat,
                "evals_so_far": evals_so_far, "best_so_far": best_so_far,
                "step_best": step_best, "target_x": target_point[0]
            }, header_written)
            header_written = True

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    return {
        "optimizer": "MS_SWEEP_SEP_CMA", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, 
        "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far
    }

def run_ms_sweep_memetic(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    Multi-Stage Sweep + Memetic CMA-ES (Fixed for partial generation crash)
    =======================================================================
    Stage 0: Pair Sweep.
    Stage 1: CMA-ES initialized at best pair.
    Stage 2: Periodic Nelder-Mead local search ('polish') to snap to optima.
    """
    from cma import CMAEvolutionStrategy
    from scipy.optimize import minimize

    n_rows, n_per_row = grid
    N = n_rows * n_per_row
    seed = SEED_BASE + 16000 * repeat + N
    rng = np.random.default_rng(seed)
    
    tag = make_tag("MS_SWEEP_MEMETIC", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)
    xs_axis, step_vals, best_vals = [], [], []
    header_written = False

    # --- STAGE 0: Pair Sweep ---
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    sweep_limit = int(0.2 * eval_budget)
    if len(all_pairs) > sweep_limit:
        rng.shuffle(all_pairs)
        pairs = all_pairs[:sweep_limit]
    else:
        pairs = all_pairs

    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        sel = -y
        evals_so_far += 1
        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = evals_so_far
            best_currents = x_used.copy()
        xs_axis.append(evals_so_far)
        step_vals.append(sel)
        best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    # --- STAGE 1: Memetic CMA ---
    remaining = eval_budget - evals_so_far
    if remaining > 0:
        x0 = best_currents.copy()
        sigma0 = 0.3 * RANGE
        popsize = cma_popsize(N)
        
        es = CMAEvolutionStrategy(x0, sigma0, {
            "popsize": popsize, "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE, RANGE]
        })

        NM_FREQ = 10   # Polish every 10 generations
        NM_BUDGET = 50 # Max evals for polish
        gen_count = 0

        while evals_so_far < eval_budget:
            gen_count += 1
            X_ask = es.ask()
            Y = []
            gen_best_x = None
            gen_best_y = np.inf
            
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
                Y.append(y)
                evals_so_far += 1
                
                if y < gen_best_y:
                    gen_best_y = y
                    gen_best_x = x_used.copy()

                if -y > best_so_far:
                    best_so_far = -y
                    best_at_eval = evals_so_far
                    best_currents = x_used.copy()

            # FIX: Check for partial population
            if len(Y) < popsize:
                break

            es.tell(X_ask, Y)

            # --- MEMETIC POLISH STEP ---
            if gen_count % NM_FREQ == 0 and evals_so_far < eval_budget - NM_BUDGET:
                
                def obj(x):
                    nonlocal evals_so_far, best_so_far, best_at_eval, best_currents
                    if evals_so_far >= eval_budget: return 0.0
                    if np.any(np.abs(x) > RANGE): return 1e9 # Soft constraint
                    y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
                    evals_so_far += 1
                    if -y > best_so_far:
                        best_so_far = -y
                        best_at_eval = evals_so_far
                        best_currents = x_used.copy()
                    return y
                
                # Run local search starting from current generation best
                res = minimize(obj, gen_best_x, method='Nelder-Mead', 
                               options={'maxfev': NM_BUDGET, 'xatol': 1e-5})
                
                # If local search found something better, inject it back into CMA
                if res.success:
                    es.inject_best_solution(res.x)

            step_best = -np.min(Y)
            xs_axis.append(evals_so_far)
            step_vals.append(step_best)
            best_vals.append(best_so_far)
            
            log_step(csv_path, {
                "optimizer": "MS_SWEEP_MEMETIC", "N": N, "repeat": repeat,
                "evals_so_far": evals_so_far, "best_so_far": best_so_far
            }, header_written)
            header_written = True

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    return {
        "optimizer": "MS_SWEEP_MEMETIC", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, 
        "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far
    }

def run_ms_sweep_lshade(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    Multi-Stage Sweep + L-SHADE (Fixed for partial generation crash)
    ================================================================
    Stage 0: Pair Sweep.
    Stage 1: L-SHADE, seeded with sweep winner.
    """
    n_rows, n_per_row = grid
    N = n_rows * n_per_row
    seed = SEED_BASE + 17000 * repeat + N
    rng = np.random.default_rng(seed)
    
    tag = make_tag("MS_SWEEP_LSHADE", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)
    xs_axis, best_vals = [], []
    header_written = False

    # --- STAGE 0: Pair Sweep ---
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    sweep_limit = int(0.2 * eval_budget)
    if len(all_pairs) > sweep_limit:
        rng.shuffle(all_pairs)
        pairs = all_pairs[:sweep_limit]
    else:
        pairs = all_pairs

    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        sel = -y
        evals_so_far += 1
        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = evals_so_far
            best_currents = x_used.copy()
        xs_axis.append(evals_so_far)
        best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    # --- STAGE 1: L-SHADE ---
    remaining = eval_budget - evals_so_far
    if remaining > 0:
        # L-SHADE Constants
        pop_size_init = 18 * N
        pop_size_min = 4
        H = 6
        M_CR = np.full(H, 0.5); M_F = np.full(H, 0.5); k_mem = 0
        archive = []
        
        # Initialize Population (Random)
        pop = rng.uniform(-RANGE, RANGE, (pop_size_init, N))
        
        # *** INJECT SWEEP WINNER ***
        pop[0] = best_currents.copy()
        
        fitness = np.zeros(pop_size_init)
        
        # Evaluate Initial Population
        for i in range(pop_size_init):
            if evals_so_far >= eval_budget: break
            y, x_used = eval_selectivity_grounded(pop[i], target_point, grid, rng_seed=seed)
            fitness[i] = y
            evals_so_far += 1
            if -y > best_so_far:
                best_so_far = -y
                best_at_eval = evals_so_far
        
        curr_pop_size = pop_size_init
        
        while evals_so_far < eval_budget:
            # Linear Population Reduction
            progress = (evals_so_far - sweep_limit) / remaining # Approx progress
            progress = np.clip(progress, 0, 1)
            next_pop_size = int(round(pop_size_init - (pop_size_init - pop_size_min) * progress))
            next_pop_size = max(pop_size_min, next_pop_size)
            
            if curr_pop_size > next_pop_size:
                idx = np.argsort(fitness)
                pop = pop[idx[:next_pop_size]]
                fitness = fitness[idx[:next_pop_size]]
                curr_pop_size = next_pop_size
            
            # Generate Parameters
            CR = np.clip(np.random.normal(M_CR[np.random.randint(0, H, curr_pop_size)], 0.1), 0, 1)
            F = np.clip(np.random.cauchy(M_F[np.random.randint(0, H, curr_pop_size)], 0.1), 0, 1)

            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros(curr_pop_size)
            success_F, success_CR, diff_fitness = [], [], []
            
            pop_archive = np.vstack((pop, np.array(archive))) if len(archive) > 0 else pop
            
            # Evolution
            p_best_rate = 0.11
            n_p_best = max(2, int(curr_pop_size * p_best_rate))
            sorted_idx = np.argsort(fitness)
            
            for i in range(curr_pop_size):
                if evals_so_far >= eval_budget: break
                
                p_best = pop[rng.choice(sorted_idx[:n_p_best])]
                r1 = pop[rng.choice([x for x in range(curr_pop_size) if x != i])]
                
                r2_candidates = list(range(len(pop_archive)))
                if i < len(pop_archive): r2_candidates.remove(i)
                r2 = pop_archive[rng.choice(r2_candidates)]
                
                v = pop[i] + F[i] * (p_best - pop[i]) + F[i] * (r1 - r2)
                j_rand = rng.integers(0, N)
                u = np.where(rng.random(N) < CR[i], v, pop[i])
                u[j_rand] = v[j_rand]
                
                y, u_used = eval_selectivity_grounded(u, target_point, grid, rng_seed=seed)
                evals_so_far += 1
                
                if y < fitness[i]:
                    new_pop[i] = u_used
                    new_fitness[i] = y
                    success_F.append(F[i])
                    success_CR.append(CR[i])
                    diff_fitness.append(fitness[i] - y)
                    archive.append(pop[i].copy())
                    if -y > best_so_far:
                        best_so_far = -y
                        best_at_eval = evals_so_far
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
            
            # FIX: If we broke the loop early, do not update population with partial/empty data
            if evals_so_far >= eval_budget:
                break

            pop = new_pop
            fitness = new_fitness
            
            # Update Memory
            if len(success_F) > 0:
                w = np.array(diff_fitness) / np.sum(diff_fitness)
                M_CR[k_mem] = np.sum(w * np.array(success_CR))
                M_F[k_mem] = np.sum(w * np.array(success_F)**2) / np.sum(w * np.array(success_F))
                k_mem = (k_mem + 1) % H
            
            while len(archive) > pop_size_init: archive.pop(rng.integers(0, len(archive)))
            
            xs_axis.append(evals_so_far)
            best_vals.append(best_so_far)
            
            if evals_so_far % 50 == 0:
                log_step(csv_path, {
                    "optimizer": "MS_SWEEP_LSHADE", "N": N, "repeat": repeat,
                    "evals_so_far": evals_so_far, "best_so_far": best_so_far
                }, header_written)
                header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {
        "optimizer": "MS_SWEEP_LSHADE", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, 
        "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far
    }

def run_ms_sweep_progressive_cma(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    Multi-Stage Sweep + Progressive CMA-ES (Fixed Logging)
    ======================================================
    Stage 0: Pair sweep to find the basin of attraction.
    Stage 1: CMA-ES where the max current limit starts LOW and 
             progressively expands to the full RANGE.
    """
    from cma import CMAEvolutionStrategy
    
    n_rows, n_per_row = grid
    N = n_rows * n_per_row
    seed = SEED_BASE + 18000 * repeat + N
    rng = np.random.default_rng(seed)
    
    tag = make_tag("MS_SWEEP_PROG_CMA", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0
    best_so_far = -np.inf
    best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)
    xs_axis, step_vals, best_vals = [], [], []
    header_written = False

    # --- STAGE 0: Pair Sweep ---
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    sweep_limit = int(0.2 * eval_budget)
    
    if len(all_pairs) > sweep_limit:
        rng.shuffle(all_pairs)
        pairs = all_pairs[:sweep_limit]
    else:
        pairs = all_pairs

    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        sel = -y
        evals_so_far += 1
        
        if sel > best_so_far:
            best_so_far = sel
            best_at_eval = evals_so_far
            best_currents = x_used.copy()
            
        xs_axis.append(evals_so_far)
        step_vals.append(sel)
        best_vals.append(best_so_far)
        
        if evals_so_far >= eval_budget: break

    # --- STAGE 1: Progressive CMA ---
    remaining = eval_budget - evals_so_far
    if remaining > 0:
        x0 = best_currents.copy()
        
        # 1. Determine Starting Cap
        start_cap = max(np.max(np.abs(x0)), 0.2 * RANGE)
        
        # 2. Initialize CMA
        sigma0 = 0.2 * start_cap 
        
        popsize = cma_popsize(N)
        
        es = CMAEvolutionStrategy(
            x0, sigma0,
            {
                "popsize": popsize, 
                "verb_disp": 0, 
                "seed": seed + 1, 
                "bounds": [-RANGE, RANGE], 
            }
        )
        
        growth_evals = 0.5 * remaining

        while evals_so_far < eval_budget:
            X_ask = es.ask()
            Y = []
            
            # --- DYNAMIC BOUND CALCULATION ---
            stage1_evals = evals_so_far - (eval_budget - remaining)
            
            if stage1_evals < growth_evals:
                ratio = stage1_evals / growth_evals
                current_cap = start_cap + (RANGE - start_cap) * ratio
            else:
                current_cap = RANGE
            
            current_cap = min(current_cap, RANGE)

            for x in X_ask:
                if evals_so_far >= eval_budget: break
                
                # Apply the Progressive Cap
                x_c = np.clip(x, -current_cap, current_cap)
                
                y, x_used = eval_selectivity_grounded(x_c, target_point, grid, rng_seed=seed)
                Y.append(y)
                evals_so_far += 1
                
                if -y > best_so_far:
                    best_so_far = -y
                    best_at_eval = evals_so_far
                    best_currents = x_used.copy()
            
            if len(Y) < popsize:
                break
            
            es.tell(X_ask[:len(Y)], Y)
            
            step_best = -np.min(Y)
            xs_axis.append(evals_so_far)
            step_vals.append(step_best)
            best_vals.append(best_so_far)
            
            # Sparse logging
            if evals_so_far % 20 == 0:
                log_step(csv_path, {
                    "optimizer": "MS_SWEEP_PROG_CMA", "N": N, "repeat": repeat,
                    "evals_so_far": evals_so_far, "best_so_far": best_so_far,
                    "current_cap": current_cap,
                    "currents": best_currents.tolist() 
                }, header_written)
                header_written = True

    save_progress_plot(xs_axis, step_vals, best_vals, tag, target_point)
    return {
        "optimizer": "MS_SWEEP_PROG_CMA", "tag": tag, "best": float(best_so_far),
        "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, 
        "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far
    }

# ======================================================================
# PROGRESSIVE SEARCH SPACE VARIANTS (10 IMPLEMENTATIONS)
# ======================================================================

def run_prog_cma_linear_50(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    1. Linear Growth (50% duration).
    Standard baseline. Expands search space linearly over first half of run.
    """
    from cma import CMAEvolutionStrategy
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 20000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("PROG_LIN_50", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)
    header_written = False; xs_axis, step_vals, best_vals = [], [], []

    # Stage 0: Sweep
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    sweep_limit = int(0.2 * eval_budget)
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far:
            best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    # Stage 1: Linear Progressive
    remaining = eval_budget - evals_so_far
    if remaining > 0:
        x0 = best_currents.copy()
        start_cap = max(np.max(np.abs(x0)), 0.15 * RANGE)
        sigma0 = 0.2 * start_cap
        es = CMAEvolutionStrategy(x0, sigma0, {"popsize": cma_popsize(N), "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE, RANGE]})
        
        growth_evals = 0.5 * remaining 

        while evals_so_far < eval_budget:
            X_ask = es.ask(); Y = []
            stage1_evals = evals_so_far - (eval_budget - remaining)
            
            ratio = min(1.0, stage1_evals / growth_evals)
            current_cap = start_cap + (RANGE - start_cap) * ratio
            
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(np.clip(x, -current_cap, current_cap), target_point, grid, rng_seed=seed)
                Y.append(y); evals_so_far += 1
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()

            if len(Y) < es.popsize: break
            es.tell(X_ask[:len(Y)], Y)
            
            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0:
                log_step(csv_path, {"optimizer": "PROG_LIN_50", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "cap": current_cap, "currents": best_currents.tolist()}, header_written); header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "PROG_LIN_50", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}


def run_prog_cma_exp_slow(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    2. Exponential/Quadratic Growth.
    Stays low for longer, accelerating limit expansion only near the end.
    """
    from cma import CMAEvolutionStrategy
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 21000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("PROG_EXP_SLOW", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0
    best_currents = np.zeros(N, dtype=float)
    header_written = False; xs_axis, best_vals = [], []

    sweep_limit = int(0.2 * eval_budget)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    remaining = eval_budget - evals_so_far
    if remaining > 0:
        x0 = best_currents.copy()
        start_cap = max(np.max(np.abs(x0)), 0.15 * RANGE)
        es = CMAEvolutionStrategy(x0, 0.2 * start_cap, {"popsize": cma_popsize(N), "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE, RANGE]})
        growth_evals = 0.6 * remaining 

        while evals_so_far < eval_budget:
            X_ask = es.ask(); Y = []
            progress = min(1.0, (evals_so_far - (eval_budget - remaining)) / growth_evals)
            
            ratio = progress ** 2  
            current_cap = start_cap + (RANGE - start_cap) * ratio
            
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(np.clip(x, -current_cap, current_cap), target_point, grid, rng_seed=seed)
                Y.append(y); evals_so_far += 1
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()

            if len(Y) < es.popsize: break
            es.tell(X_ask[:len(Y)], Y)
            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0: log_step(csv_path, {"optimizer": "PROG_EXP_SLOW", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "cap": current_cap, "currents": best_currents.tolist()}, header_written); header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "PROG_EXP_SLOW", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}


def run_prog_cma_step_curriculum(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    3. Step/Curriculum Learning.
    Increases bounds in 3 discrete stages (Low -> Medium -> High).
    """
    from cma import CMAEvolutionStrategy
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 22000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("PROG_STEP", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0; best_currents = np.zeros(N)
    header_written = False; xs_axis, best_vals = [], []

    sweep_limit = int(0.2 * eval_budget)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    remaining = eval_budget - evals_so_far
    if remaining > 0:
        x0 = best_currents.copy()
        start_cap = max(np.max(np.abs(x0)), 0.15 * RANGE)
        es = CMAEvolutionStrategy(x0, 0.2 * start_cap, {"popsize": cma_popsize(N), "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE, RANGE]})
        
        level1_end = evals_so_far + 0.33 * remaining
        level2_end = evals_so_far + 0.66 * remaining
        cap_level1 = start_cap
        cap_level2 = start_cap + 0.5 * (RANGE - start_cap)
        cap_level3 = RANGE

        while evals_so_far < eval_budget:
            X_ask = es.ask(); Y = []
            
            if evals_so_far < level1_end: current_cap = cap_level1
            elif evals_so_far < level2_end: current_cap = cap_level2
            else: current_cap = cap_level3
            
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(np.clip(x, -current_cap, current_cap), target_point, grid, rng_seed=seed)
                Y.append(y); evals_so_far += 1
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()

            if len(Y) < es.popsize: break
            es.tell(X_ask[:len(Y)], Y)
            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0: log_step(csv_path, {"optimizer": "PROG_STEP", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "cap": current_cap, "currents": best_currents.tolist()}, header_written); header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "PROG_STEP", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}


def run_prog_cma_sigmoid(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    4. Sigmoidal Growth.
    S-curve expansion. Slow start -> Fast expansion -> Slow finish.
    """
    from cma import CMAEvolutionStrategy
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 23000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("PROG_SIGMOID", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0; best_currents = np.zeros(N)
    header_written = False; xs_axis, best_vals = [], []

    sweep_limit = int(0.2 * eval_budget)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    remaining = eval_budget - evals_so_far
    if remaining > 0:
        x0 = best_currents.copy()
        start_cap = max(np.max(np.abs(x0)), 0.15 * RANGE)
        es = CMAEvolutionStrategy(x0, 0.2 * start_cap, {"popsize": cma_popsize(N), "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE, RANGE]})
        growth_evals = 0.6 * remaining 

        while evals_so_far < eval_budget:
            X_ask = es.ask(); Y = []
            
            t = (evals_so_far - (eval_budget - remaining)) / growth_evals
            if t >= 1.0: ratio = 1.0
            else:
                sig = 1 / (1 + np.exp(-10 * (t - 0.5)))
                sig_0 = 1 / (1 + np.exp(5)); sig_1 = 1 / (1 + np.exp(-5))
                ratio = (sig - sig_0) / (sig_1 - sig_0)
                ratio = np.clip(ratio, 0, 1)

            current_cap = start_cap + (RANGE - start_cap) * ratio
            
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(np.clip(x, -current_cap, current_cap), target_point, grid, rng_seed=seed)
                Y.append(y); evals_so_far += 1
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
            
            if len(Y) < es.popsize: break
            es.tell(X_ask[:len(Y)], Y)
            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0: log_step(csv_path, {"optimizer": "PROG_SIGMOID", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "cap": current_cap, "currents": best_currents.tolist()}, header_written); header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "PROG_SIGMOID", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}


def run_prog_cma_delayed(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    5. Delayed Expansion.
    Locks the current limit at the low 'Sweep' level for 30% of the run.
    """
    from cma import CMAEvolutionStrategy
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 24000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("PROG_DELAY", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0; best_currents = np.zeros(N)
    header_written = False; xs_axis, best_vals = [], []

    sweep_limit = int(0.2 * eval_budget)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    remaining = eval_budget - evals_so_far
    if remaining > 0:
        x0 = best_currents.copy()
        start_cap = max(np.max(np.abs(x0)), 0.15 * RANGE)
        es = CMAEvolutionStrategy(x0, 0.2 * start_cap, {"popsize": cma_popsize(N), "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE, RANGE]})
        
        delay_end = evals_so_far + 0.3 * remaining
        growth_end = evals_so_far + 0.8 * remaining

        while evals_so_far < eval_budget:
            X_ask = es.ask(); Y = []
            
            if evals_so_far < delay_end:
                current_cap = start_cap 
            elif evals_so_far < growth_end:
                ratio = (evals_so_far - delay_end) / (growth_end - delay_end)
                current_cap = start_cap + (RANGE - start_cap) * ratio
            else:
                current_cap = RANGE
            
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(np.clip(x, -current_cap, current_cap), target_point, grid, rng_seed=seed)
                Y.append(y); evals_so_far += 1
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()

            if len(Y) < es.popsize: break
            es.tell(X_ask[:len(Y)], Y)
            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0: log_step(csv_path, {"optimizer": "PROG_DELAY", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "cap": current_cap, "currents": best_currents.tolist()}, header_written); header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "PROG_DELAY", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}


def run_prog_cma_trust_funnel(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    6. Trust Funnel.
    Bounds are [BestSweep - R, BestSweep + R]. R grows from small to full.
    """
    from cma import CMAEvolutionStrategy
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 25000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("PROG_FUNNEL", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0; best_currents = np.zeros(N)
    header_written = False; xs_axis, best_vals = [], []

    sweep_limit = int(0.2 * eval_budget)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    remaining = eval_budget - evals_so_far
    if remaining > 0:
        center_vec = best_currents.copy() 
        start_radius = 0.15 * RANGE
        es = CMAEvolutionStrategy(center_vec, 0.2 * start_radius, {"popsize": cma_popsize(N), "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE, RANGE]})
        growth_evals = 0.6 * remaining 

        while evals_so_far < eval_budget:
            X_ask = es.ask(); Y = []
            
            t = (evals_so_far - (eval_budget - remaining)) / growth_evals
            ratio = min(1.0, t)
            current_radius = start_radius + (RANGE * 2) * ratio 
            
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                
                lower = np.maximum(-RANGE, center_vec - current_radius)
                upper = np.minimum(RANGE,  center_vec + current_radius)
                x_c = np.clip(x, lower, upper)

                y, x_used = eval_selectivity_grounded(x_c, target_point, grid, rng_seed=seed)
                Y.append(y); evals_so_far += 1
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()

            if len(Y) < es.popsize: break
            es.tell(X_ask[:len(Y)], Y)
            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0: log_step(csv_path, {"optimizer": "PROG_FUNNEL", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "radius": current_radius, "currents": best_currents.tolist()}, header_written); header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "PROG_FUNNEL", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}


def run_prog_cma_soft_penalty(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    7. Soft Penalty.
    Instead of hard clipping, adds a penalty to the objective if current > cap.
    """
    from cma import CMAEvolutionStrategy
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 26000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("PROG_PENALTY", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0; best_currents = np.zeros(N)
    header_written = False; xs_axis, best_vals = [], []

    sweep_limit = int(0.2 * eval_budget)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    remaining = eval_budget - evals_so_far
    if remaining > 0:
        x0 = best_currents.copy()
        start_cap = max(np.max(np.abs(x0)), 0.15 * RANGE)
        es = CMAEvolutionStrategy(x0, 0.2 * start_cap, {"popsize": cma_popsize(N), "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE*1.5, RANGE*1.5]})
        growth_evals = 0.6 * remaining 

        while evals_so_far < eval_budget:
            X_ask = es.ask(); Y = []
            t = (evals_so_far - (eval_budget - remaining)) / growth_evals
            current_cap = start_cap + (RANGE - start_cap) * min(1.0, t)
            
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                
                x_eval = np.clip(x, -RANGE, RANGE) 
                y, x_used = eval_selectivity_grounded(x_eval, target_point, grid, rng_seed=seed)
                
                violation = np.sum(np.maximum(0, np.abs(x) - current_cap))
                penalty = 100.0 * violation 
                
                Y.append(y + penalty)
                evals_so_far += 1
                
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()

            if len(Y) < es.popsize: break
            es.tell(X_ask[:len(Y)], Y)
            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0: log_step(csv_path, {"optimizer": "PROG_PENALTY", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "cap": current_cap, "currents": best_currents.tolist()}, header_written); header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "PROG_PENALTY", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}


def run_prog_cma_aggressive_start(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    8. Aggressive Start.
    Starts the cap at 50% of RANGE immediately. Skips the 'low energy' fine tuning.
    """
    from cma import CMAEvolutionStrategy
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 27000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("PROG_AGGRESSIVE", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0; best_currents = np.zeros(N)
    header_written = False; xs_axis, best_vals = [], []

    sweep_limit = int(0.2 * eval_budget)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    remaining = eval_budget - evals_so_far
    if remaining > 0:
        x0 = best_currents.copy()
        start_cap = 0.5 * RANGE 
        es = CMAEvolutionStrategy(x0, 0.2 * start_cap, {"popsize": cma_popsize(N), "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE, RANGE]})
        growth_evals = 0.5 * remaining 

        while evals_so_far < eval_budget:
            X_ask = es.ask(); Y = []
            t = (evals_so_far - (eval_budget - remaining)) / growth_evals
            current_cap = start_cap + (RANGE - start_cap) * min(1.0, t)
            
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(np.clip(x, -current_cap, current_cap), target_point, grid, rng_seed=seed)
                Y.append(y); evals_so_far += 1
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()

            if len(Y) < es.popsize: break
            es.tell(X_ask[:len(Y)], Y)
            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0: log_step(csv_path, {"optimizer": "PROG_AGGRESSIVE", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "cap": current_cap, "currents": best_currents.tolist()}, header_written); header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "PROG_AGGRESSIVE", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}


def run_prog_cma_stagnation(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    9. Stagnation-Based Growth.
    Increases the current cap only when the best score fails to improve.
    """
    from cma import CMAEvolutionStrategy
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 28000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("PROG_STAG", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0; best_currents = np.zeros(N)
    header_written = False; xs_axis, best_vals = [], []

    sweep_limit = int(0.2 * eval_budget)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    remaining = eval_budget - evals_so_far
    if remaining > 0:
        x0 = best_currents.copy()
        current_cap = max(np.max(np.abs(x0)), 0.15 * RANGE)
        es = CMAEvolutionStrategy(x0, 0.2 * current_cap, {"popsize": cma_popsize(N), "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE, RANGE]})
        
        gens_no_improve = 0
        last_best = best_so_far

        while evals_so_far < eval_budget:
            X_ask = es.ask(); Y = []
            
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(np.clip(x, -current_cap, current_cap), target_point, grid, rng_seed=seed)
                Y.append(y); evals_so_far += 1
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()

            if len(Y) < es.popsize: break
            es.tell(X_ask[:len(Y)], Y)
            
            if best_so_far > last_best + 1e-4:
                last_best = best_so_far
                gens_no_improve = 0
            else:
                gens_no_improve += 1
            
            if gens_no_improve >= 3:
                current_cap = min(RANGE, current_cap + 0.1 * RANGE)
                gens_no_improve = 0

            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0: log_step(csv_path, {"optimizer": "PROG_STAG", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "cap": current_cap, "currents": best_currents.tolist()}, header_written); header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "PROG_STAG", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}


def run_prog_cma_late_bloomer(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    10. Late Bloomer.
    Uses 100% of the available budget to grow the limit.
    The cap only reaches MAX at the very last generation.
    """
    from cma import CMAEvolutionStrategy
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 29000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("PROG_LATE", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0; best_currents = np.zeros(N)
    header_written = False; xs_axis, best_vals = [], []

    sweep_limit = int(0.2 * eval_budget)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    remaining = eval_budget - evals_so_far
    if remaining > 0:
        x0 = best_currents.copy()
        start_cap = max(np.max(np.abs(x0)), 0.15 * RANGE)
        es = CMAEvolutionStrategy(x0, 0.2 * start_cap, {"popsize": cma_popsize(N), "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE, RANGE]})
        growth_evals = remaining 

        while evals_so_far < eval_budget:
            X_ask = es.ask(); Y = []
            t = (evals_so_far - (eval_budget - remaining)) / growth_evals
            current_cap = start_cap + (RANGE - start_cap) * min(1.0, t)
            
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(np.clip(x, -current_cap, current_cap), target_point, grid, rng_seed=seed)
                Y.append(y); evals_so_far += 1
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()

            if len(Y) < es.popsize: break
            es.tell(X_ask[:len(Y)], Y)
            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0: log_step(csv_path, {"optimizer": "PROG_LATE", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "cap": current_cap, "currents": best_currents.tolist()}, header_written); header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "PROG_LATE", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}

def run_ms_sweep_surrogate_cma(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    Multi-Stage Sweep + Surrogate-Assisted CMA-ES
    =============================================
    Stage 0: Pair sweep.
    Stage 1: CMA-ES with a local RBF surrogate model.
             - Collects (X, y) pairs.
             - Trains a cheap RBF interpolator.
             - Asks for λ * 10 points from CMA.
             - Ranks them using the surrogate.
             - Evaluates only the top λ on the real simulator.
    """
    from cma import CMAEvolutionStrategy
    from scipy.interpolate import Rbf
    
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 30000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("MS_SURROGATE_CMA", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0; best_currents = np.zeros(N)
    xs_axis, best_vals = [], []
    header_written = False
    
    # Archive for surrogate training
    archive_X = []
    archive_y = []

    # Stage 0: Sweep
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    sweep_limit = int(0.2 * eval_budget)
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        
        archive_X.append(x_used)
        archive_y.append(y) # Minimize y (negative selectivity)
        
        if -y > best_so_far:
            best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    # Stage 1: Surrogate CMA
    remaining = eval_budget - evals_so_far
    if remaining > 0:
        x0 = best_currents.copy()
        sigma0 = 0.3 * RANGE
        popsize = cma_popsize(N)
        
        es = CMAEvolutionStrategy(x0, sigma0, {"popsize": popsize, "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE, RANGE]})
        
        # Pre-screening factor (how many points to simulate on surrogate per real eval)
        n_prescreen = 10 
        
        while evals_so_far < eval_budget:
            # 1. Train Surrogate (if enough data)
            # Use only recent/local history to keep it fast and accurate locally
            n_train = min(len(archive_X), 200) 
            X_train = np.array(archive_X[-n_train:])
            y_train = np.array(archive_y[-n_train:])
            
            try:
                # Simple linear RBF is robust and fast
                rbf = Rbf(*X_train.T, y_train, function='linear')
                has_model = True
            except:
                has_model = False

            # 2. Ask & Prescreen
            if has_model:
                # Ask for many more points than we need
                X_candidates = es.ask(popsize * n_prescreen)
                
                # Predict fitness
                y_pred = []
                valid_candidates = []
                for x in X_candidates:
                    # Clip first
                    xc = np.clip(x, -RANGE, RANGE)
                    # RBF prediction
                    pred = rbf(*xc)
                    y_pred.append(pred)
                    valid_candidates.append(xc)
                
                # Pick the best 'popsize' candidates according to surrogate
                # We want to MINIMIZE y (which is -Selectivity)
                sort_idx = np.argsort(y_pred)
                X_ask = [X_candidates[i] for i in sort_idx[:popsize]]
            else:
                X_ask = es.ask()

            # 3. Evaluate Real
            Y = []
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(np.clip(x, -RANGE, RANGE), target_point, grid, rng_seed=seed)
                Y.append(y)
                evals_so_far += 1
                
                archive_X.append(x_used)
                archive_y.append(y)
                
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()

            if len(Y) < popsize: break
            es.tell(X_ask[:len(Y)], Y)
            
            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0:
                log_step(csv_path, {"optimizer": "MS_SURROGATE_CMA", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "currents": best_currents.tolist()}, header_written); header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "MS_SURROGATE_CMA", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}

def run_ms_sweep_restart_cma(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    Multi-Stage Sweep + Restart CMA-ES (IPOP)
    =========================================
    Stage 0: Pair sweep.
    Stage 1: Run CMA-ES. If it converges (stagnates), RESTART it with 
             DOUBLE the population size.
    
    Why it works: Small populations are fast but get stuck. Large populations
    are slow but robust. This gives you the best of both worlds.
    """
    from cma import CMAEvolutionStrategy
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 31000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("MS_RESTART_CMA", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0; best_currents = np.zeros(N)
    xs_axis, best_vals = [], []
    header_written = False

    # Stage 0: Sweep
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    sweep_limit = int(0.2 * eval_budget)
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    # Stage 1: IPOP Restarts
    popsize = cma_popsize(N)
    restart_count = 0
    
    while evals_so_far < eval_budget:
        # Start a new run
        # First run starts at sweep best. Subsequent runs start random (or near best)
        if restart_count == 0:
            x0 = best_currents.copy()
            sigma0 = 0.3 * RANGE
        else:
            # Perturb slightly from best known to escape local trap
            x0 = best_currents + rng.normal(0, 0.1 * RANGE, N)
            sigma0 = 0.5 * RANGE # Larger sigma for restart
        
        es = CMAEvolutionStrategy(x0, sigma0, {
            "popsize": popsize, 
            "verb_disp": 0, 
            "seed": seed + restart_count, 
            "bounds": [-RANGE, RANGE],
            "tolflatfitness": 10,  # Stop if stuck
            "tolfun": 1e-4         # Stop if converged
        })
        
        while not es.stop() and evals_so_far < eval_budget:
            X_ask = es.ask()
            Y = []
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
                Y.append(y); evals_so_far += 1
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
            
            if len(Y) < popsize: break
            es.tell(X_ask[:len(Y)], Y)
            
            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0:
                log_step(csv_path, {"optimizer": "MS_RESTART_CMA", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "restart": restart_count, "popsize": popsize, "currents": best_currents.tolist()}, header_written); header_written = True

        restart_count += 1
        popsize *= 2 # Double population for next restart

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "MS_RESTART_CMA", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}

def run_ms_sweep_hybrid_cma_de(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    Multi-Stage Sweep + Hybrid CMA-DE
    =================================
    Stage 0: Pair sweep.
    Stage 1: Interleaved CMA-ES and Differential Evolution.
             - Run CMA-ES for k generations.
             - Run DE for k generations (using CMA population as input).
             - Swap information by injecting best DE solutions into CMA.
    """
    from cma import CMAEvolutionStrategy
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 32000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("MS_HYBRID_CMA_DE", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0; best_currents = np.zeros(N)
    xs_axis, best_vals = [], []
    header_written = False

    # Stage 0: Sweep
    sweep_limit = int(0.2 * eval_budget)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    # Stage 1: Hybrid Loop
    popsize = cma_popsize(N)
    es = CMAEvolutionStrategy(best_currents.copy(), 0.3 * RANGE, {"popsize": popsize, "verb_disp": 0, "seed": seed+1, "bounds": [-RANGE, RANGE]})
    
    # DE Population
    de_pop = rng.uniform(-RANGE, RANGE, (popsize, N))
    de_fitness = np.zeros(popsize)
    # Init DE fitness
    for i in range(popsize):
        if evals_so_far >= eval_budget: break
        y, x_u = eval_selectivity_grounded(de_pop[i], target_point, grid, rng_seed=seed)
        de_fitness[i] = y; evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_u.copy()

    while evals_so_far < eval_budget:
        # --- CMA Step ---
        X_ask = es.ask()
        Y = []
        for x in X_ask:
            if evals_so_far >= eval_budget: break
            y, x_u = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
            Y.append(y); evals_so_far += 1
            if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_u.copy()
        
        if len(Y) < popsize: break
        es.tell(X_ask[:len(Y)], Y)
        
        # --- DE Step ---
        for i in range(popsize):
            if evals_so_far >= eval_budget: break
            # DE/best/1/bin strategy
            idxs = [idx for idx in range(popsize) if idx != i]
            r1, r2 = de_pop[rng.choice(idxs, 2, replace=False)]
            # Use GLOBAL best from CMA/Sweep history
            v = best_currents + 0.8 * (r1 - r2) 
            
            # Crossover
            cross_points = rng.random(N) < 0.9
            u = np.where(cross_points, v, de_pop[i])
            u = np.clip(u, -RANGE, RANGE)
            
            y, u_used = eval_selectivity_grounded(u, target_point, grid, rng_seed=seed)
            evals_so_far += 1
            
            if y < de_fitness[i]:
                de_pop[i] = u_used
                de_fitness[i] = y
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = u_used.copy()
                
                # INJECTION: If DE finds a winner, tell CMA about it!
                try: es.inject_best_solution(u_used)
                except: pass

        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far % 50 == 0:
            log_step(csv_path, {"optimizer": "MS_HYBRID_CMA_DE", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "currents": best_currents.tolist()}, header_written); header_written = True

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "MS_HYBRID_CMA_DE", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}


def run_ms_sweep_cma_lbfgs(grid, repeat, target_point, eval_budget, I0=0.8e-3):
    """
    Multi-Stage Sweep + Hybrid CMA-ES / L-BFGS-B
    ==============================================
    Stage 0: Pair Sweep (Global coarse search).
    Stage 1: Interleaved CMA-ES and L-BFGS-B.
             - CMA-ES handles the covariance/exploration (finding the valley).
             - L-BFGS-B performs aggressive local descent (sliding down the valley).
             - The result of L-BFGS-B is re-injected as the new mean for CMA-ES.
    
    Best for: Physics simulations (current steering) where the landscape is 
    continuous and likely has smooth gradients, but is highly correlated.
    """
    from cma import CMAEvolutionStrategy
    from scipy.optimize import minimize
    
    n_rows, n_per_row = grid; N = n_rows * n_per_row
    seed = SEED_BASE + 33000 * repeat + N; rng = np.random.default_rng(seed)
    tag = make_tag("MS_CMA_LBFGS", grid, repeat, target_point)
    csv_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")

    evals_so_far = 0; best_so_far = -np.inf; best_at_eval = 0; best_currents = np.zeros(N)
    xs_axis, best_vals = [], []
    header_written = False

    # --- Stage 0: Pair Sweep ---
    # Identifies the correct 'basin' of attraction
    sweep_limit = int(0.2 * eval_budget)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs); pairs = all_pairs[:sweep_limit]
    
    for i, j in tqdm(pairs, desc=f"{tag}_stage0", leave=False):
        x = np.zeros(N); x[i] = I0; x[j] = -I0
        y, x_used = eval_selectivity_grounded(x, target_point, grid, rng_seed=seed)
        evals_so_far += 1
        if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
        xs_axis.append(evals_so_far); best_vals.append(best_so_far)
        if evals_so_far >= eval_budget: break

    # --- Stage 1: CMA-ES <-> L-BFGS-B Loop ---
    remaining = eval_budget - evals_so_far
    if remaining > 0:
        # Initialize CMA at the sweep's best point
        x0 = best_currents.copy()
        sigma0 = 0.3 * RANGE
        popsize = cma_popsize(N)
        
        es = CMAEvolutionStrategy(x0, sigma0, {
            "popsize": popsize, 
            "verb_disp": 0, 
            "seed": seed+1, 
            "bounds": [-RANGE, RANGE]
        })

        # Cycle parameters
        cma_generations_per_cycle = 20  # Run CMA for a bit to learn geometry
        lbfgs_max_evals = 30            # Then do a quick gradient slide
        
        gen_counter = 0

        while evals_so_far < eval_budget:
            # A. Run CMA-ES Block
            # -------------------
            X_ask = es.ask()
            Y = []
            for x in X_ask:
                if evals_so_far >= eval_budget: break
                y, x_used = eval_selectivity_grounded(np.clip(x, -RANGE, RANGE), target_point, grid, rng_seed=seed)
                Y.append(y); evals_so_far += 1
                if -y > best_so_far: best_so_far = -y; best_at_eval = evals_so_far; best_currents = x_used.copy()
            
            if len(Y) < popsize: break
            es.tell(X_ask[:len(Y)], Y)
            gen_counter += 1

            # Logging
            xs_axis.append(evals_so_far); best_vals.append(best_so_far)
            if evals_so_far % 50 == 0:
                 log_step(csv_path, {"optimizer": "MS_CMA_LBFGS", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "phase": "CMA", "currents": best_currents.tolist()}, header_written); header_written = True

            # B. Trigger L-BFGS-B Jump?
            # -------------------------
            if gen_counter >= cma_generations_per_cycle and evals_so_far < eval_budget:
                gen_counter = 0 # Reset counter
                
                # Define local objective wrapper for SciPy
                # Note: SciPy minimizes, and 'y' from eval is already negative selectivity (so we minimize y)
                def local_objective(x_loc):
                    nonlocal evals_so_far, best_so_far, best_at_eval, best_currents
                    if evals_so_far >= eval_budget: raise StopIteration
                    
                    val, x_u = eval_selectivity_grounded(np.clip(x_loc, -RANGE, RANGE), target_point, grid, rng_seed=seed)
                    evals_so_far += 1
                    
                    if -val > best_so_far: 
                        best_so_far = -val
                        best_at_eval = evals_so_far
                        best_currents = x_u.copy()
                    return val

                try:
                    # Start L-BFGS-B from current CMA mean (center of distribution)
                    # We limit maxfun to ensure we don't drain the whole budget in one local search
                    res = minimize(local_objective, es.mean, method='L-BFGS-B', 
                                   bounds=[(-RANGE, RANGE)] * N, 
                                   options={'maxfun': lbfgs_max_evals, 'ftol': 1e-9})
                    
                    # C. Re-Injection
                    # ---------------
                    # Move the CMA-ES Mean to the new, optimized location found by L-BFGS-B
                    es.mean = res.x
                    
                    # Optional: Slightly re-inflate sigma if it became too small, 
                    # to encourage exploration around this new deep valley
                    if es.sigma < 0.01 * RANGE:
                         es.sigma = 0.05 * RANGE

                    # Log the jump
                    xs_axis.append(evals_so_far); best_vals.append(best_so_far)
                    log_step(csv_path, {"optimizer": "MS_CMA_LBFGS", "N": N, "repeat": repeat, "evals_so_far": evals_so_far, "best_so_far": best_so_far, "phase": "LBFGS_JUMP", "currents": best_currents.tolist()}, header_written)

                except StopIteration:
                    break # Budget exhausted inside L-BFGS-B

    save_progress_plot(xs_axis, best_vals, best_vals, tag, target_point)
    return {"optimizer": "MS_CMA_LBFGS", "tag": tag, "best": float(best_so_far), "best_found_at_eval": int(best_at_eval), "N": N, "grid": grid, "repeat": repeat, "target_point": target_point, "used_evals": evals_so_far}