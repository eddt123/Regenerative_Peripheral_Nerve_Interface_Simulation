import numpy as np

def compute_selectivity(vdata: np.ndarray,
                        target_idx: int,
                        mode: str = "average",
                        penalty_weight: float = 1.0,
                        eps: float = 1e-12
                       ) -> float:
    """
    Compute selectivity for a target point, with two penalty modes:
      - "sum":    use sum of all other amplitudes   [original]
      - "average": use average of all other amplitudes
    You can also scale the penalty by `penalty_weight`.
    """
    # 1) amplitudes per position
    amps = np.mean(np.abs(vdata), axis=0)  # shape (npos,)
    a_t = amps[target_idx]

    # 2) off-target measure
    if mode == "sum":
        off = np.sum(amps) - a_t
    elif mode == "average":
        npos = amps.size
        off = (np.sum(amps) - a_t) / (npos - 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 3) apply weight & compute ratio
    denom = penalty_weight * off + eps
    return a_t / denom
