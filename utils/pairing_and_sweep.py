

# ==============================================================
# INITIAL ELECTRODE SWEEP
# ==============================================================
def electrode_sweep_initialisation(n_electrodes, target_point, sweep_amplitude=0.5e-3):
    """
    Stimulates each electrode one at a time with ±amplitude while others are zero.
    Returns a list of (currents, selectivity) pairs to use as warm-start data.
    """
    results = []
    for i in range(n_electrodes):
        for sign in [+1, -1]:
            currents = np.zeros(n_electrodes)
            currents[i] = sign * sweep_amplitude
            val = -eval_selectivity(currents, target_point)  # negative (minimiser)
            results.append((currents, val))
    return results
