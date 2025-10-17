import numpy as np

# ---------- Geometry & layout ----------
def layout_electrodes(n_rows=4, n_per_row=3, radius=0.01, height=0.04,
                      z0=-0.02, z1=0.02, rotation_offset=0.0):
    z_vals = np.linspace(z0, z1, n_rows) if height is None else np.linspace(-height/2, height/2, n_rows)
    thetas = np.linspace(0, 2*np.pi, n_per_row, endpoint=False) + rotation_offset
    positions = []
    for z in z_vals:
        for th in thetas:
            x = radius * np.cos(th)
            y = radius * np.sin(th)
            positions.append([x, y, z])
    return np.array(positions)  # (N,3)

# ---------- Utilities (optional) ----------
def pair_balanced_currents(N, i1, i2, a1, a2):
    """Create a vector with exactly two active electrodes that is internally zero-sum."""
    cur = np.zeros(N, dtype=float)
    m = 0.5 * (a1 + a2)
    cur[i1] = a1 - m
    cur[i2] = a2 - m
    return cur

def target_between_pair(sources_pos, i1, i2, inward_ratio=0.80, keep_z="i1"):
    """Place target at the midpoint between i1 and i2, then pull radially inward."""
    p1, p2 = sources_pos[i1].copy(), sources_pos[i2].copy()
    if keep_z == "i1":   p2[2] = p1[2]
    elif keep_z == "i2": p1[2] = p2[2]
    mid = 0.5 * (p1 + p2)
    mid[:2] *= inward_ratio
    return mid

# ---------- Field / potential (finite electrode) ----------
def electric_field_at_point(r, sources_pos, currents, sigma, electrode_radius=0.001, eps=1e-9):
    r = np.asarray(r)
    R = r[None,:] - sources_pos
    d2 = np.sum(R*R, axis=1)
    a2 = electrode_radius**2
    d32 = (d2 + a2)**1.5
    coeff = currents / (4*np.pi*sigma)
    return (coeff[:,None] * R / d32[:,None]).sum(axis=0)

def potential_at_point(r, sources_pos, currents, sigma, electrode_radius=0.001, eps=1e-9):
    r = np.asarray(r)
    R = r[None,:] - sources_pos
    d = np.sqrt((R*R).sum(axis=1) + electrode_radius**2)
    return np.sum(currents / (4*np.pi*sigma*d))

# ---------- Diagnostics for grounded mode ----------
def boundary_potential_stats(sources_pos, currents, sigma, R_outer=0.1, z=0.0, n_samples=72, electrode_radius=0.001):
    """
    Sample |V| on a circle of radius R_outer (at z) to check that the potential is ~0 at the grounded boundary.
    Returns (mean_abs_V, max_abs_V).
    """
    th = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    vals = []
    for t in th:
        p = np.array([R_outer*np.cos(t), R_outer*np.sin(t), z])
        vals.append(abs(potential_at_point(p, sources_pos, currents, sigma, electrode_radius)))
    vals = np.asarray(vals)
    return float(vals.mean()), float(vals.max())

# ---------- Cylinder sampling ----------
def sample_points_in_cylinder(n_points, radius, z0, z1, exclude_center=None, exclude_radius=0.003, rng=None):
    rng = np.random.default_rng(rng)
    pts = []
    while len(pts) < n_points:
        u = rng.random()
        r = radius * np.sqrt(u)
        th = rng.random() * 2*np.pi
        x, y = r*np.cos(th), r*np.sin(th)
        z = rng.random()*(z1 - z0) + z0
        p = np.array([x, y, z])
        if exclude_center is not None and np.linalg.norm(p - exclude_center) < exclude_radius:
            continue
        pts.append(p)
    return np.array(pts)

# ---------- Activation model ----------
def compute_activating_function(r, sources_pos, currents, sigma, electrode_radius=0.001, delta=1e-5):
    r_plus = np.array(r); r_minus = np.array(r)
    r_plus[2] += delta;   r_minus[2] -= delta
    Vp = potential_at_point(r_plus,  sources_pos, currents, sigma, electrode_radius)
    V0 = potential_at_point(r,       sources_pos, currents, sigma, electrode_radius)
    Vm = potential_at_point(r_minus, sources_pos, currents, sigma, electrode_radius)
    return abs((Vp - 2*V0 + Vm) / (delta**2))  # V/m^2

def activation_from_field(E_vec, E_th=0.5, k=0.1, model="sigmoid", use_activating_function=False,
                          sources_pos=None, currents=None, sigma=None, points=None, electrode_radius=0.001):
    if use_activating_function and all(x is not None for x in [sources_pos, currents, sigma, points]):
        if points.ndim == 1:
            points = points[None, :]
        E_mag = np.array([compute_activating_function(p, sources_pos, currents, sigma, electrode_radius) for p in points])
    else:
        E_mag = np.linalg.norm(E_vec, axis=-1) if getattr(E_vec, "ndim", 1) > 1 else np.array([np.linalg.norm(E_vec)])
    if model == "sigmoid":
        out = 1.0 / (1.0 + np.exp(-(E_mag - E_th) / k))
    elif model == "hill":
        n = 4; out = (E_mag**n) / (E_mag**n + E_th**n)
    elif model == "binary":
        out = (E_mag > E_th).astype(float)
    else:
        raise ValueError("Unknown activation model.")
    return out if len(out) > 1 else out[0]

# ---------- Selectivity metrics ----------
def selectivity_rms(E_target, E_off):
    Et = np.linalg.norm(E_target)
    Erms = np.sqrt(np.mean(np.linalg.norm(E_off, axis=1)**2) + 1e-12)
    return Et / (Erms + 1e-12)

def selectivity_max(E_target, E_off):
    Et = np.linalg.norm(E_target)
    Emax = np.max(np.linalg.norm(E_off, axis=1)) + 1e-12
    return Et / Emax

def selectivity_activation(E_target, E_off, E_th=0.5, k=0.1, model="sigmoid",
                           use_activating_function=False, sources_pos=None,
                           currents=None, sigma=None, target_point=None, off_pts=None,
                           electrode_radius=0.001):
    if use_activating_function and all(x is not None for x in [sources_pos, currents, sigma, target_point, off_pts]):
        A_t   = activation_from_field(E_target[None,:], E_th, k, model, True, sources_pos, currents, sigma,
                                      np.array(target_point)[None,:], electrode_radius)
        A_off = activation_from_field(E_off,              E_th, k, model, True, sources_pos, currents, sigma,
                                      off_pts, electrode_radius)
    else:
        A_t   = activation_from_field(E_target[None,:], E_th, k, model)
        A_off = activation_from_field(E_off,            E_th, k, model)
    A_rms = np.sqrt(np.mean(A_off**2) + 1e-12)
    return A_t / (A_rms + 1e-12)

# ---------- Main simulation ----------
def run_selectivity_simulation(
    n_rows=4, n_per_row=3,
    currents=None,            # length = n_rows*n_per_row
    target_point=(0.0, 0.0, 0.0),
    radius=0.01, height=0.04, sigma=0.25,
    n_off_samples=1000, z0=None, z1=None, rotation_offset=0.0,
    metric="activation", rng=1234,
    E_th=0.4, k=0.1, activation_model="sigmoid",
    use_activating_function=False, electrode_radius=0.001,
    min_target_activation=None,
    return_electrode_idx=None,
    grounded_boundary=True,          # allow non-zero-sum currents (monopolar to distant ground)
    R_outer=0.10,                     # "ground" radius (m), should be >> radius
    record_boundary_stats=False       # compute mean/max |V| at R_outer for diagnostics
):
    """
    If grounded_boundary=False (default): currents must sum to zero (floating array).
    If grounded_boundary=True: no zero-sum required; the outer boundary at R_outer acts as ground (Dirichlet ~0).
    """
    N = n_rows * n_per_row
    if currents is None:
        raise ValueError("Provide 'currents' of length n_rows*n_per_row.")
    currents = np.asarray(currents, dtype=float).copy()
    if len(currents) != N:
        raise ValueError(f"currents length {len(currents)} != expected {N}")

    # Enforce neutrality unless grounded mode
    if grounded_boundary:
        if return_electrode_idx is not None:
            raise ValueError("Do not set return_electrode_idx when grounded_boundary=True.")
        # allow non-zero-sum currents; I_net is handled by distant ground
        I_net = float(np.sum(currents))
    else:
        if return_electrode_idx is not None:
            s_others = np.sum(currents) - currents[return_electrode_idx]
            currents[return_electrode_idx] = -s_others
        else:
            if abs(np.sum(currents)) > 1e-12:
                raise ValueError("Currents must sum to 0 A or specify return_electrode_idx.")
        I_net = 0.0

    if z0 is None or z1 is None:
        z0, z1 = -height/2, height/2

    sources_pos = layout_electrodes(n_rows, n_per_row, radius, height, z0, z1, rotation_offset)

    # Field at target
    E_t = electric_field_at_point(np.array(target_point), sources_pos, currents, sigma, electrode_radius)

    # Optional soft target-activation penalty
    if min_target_activation is not None:
        if use_activating_function:
            A_t = activation_from_field(E_t[None,:], E_th, k, activation_model, True,
                                        sources_pos, currents, sigma, np.array(target_point)[None,:], electrode_radius)
        else:
            A_t = activation_from_field(E_t[None,:], E_th, k, activation_model)
        soft_penalty = np.clip((min_target_activation - float(A_t))/min_target_activation, 0.0, 1.0)
    else:
        soft_penalty = 0.0

    # Off-target sampling
    exclude_rad = max(2*electrode_radius, 0.002)
    off_pts = sample_points_in_cylinder(n_off_samples, radius, z0, z1,
                                        exclude_center=np.array(target_point),
                                        exclude_radius=exclude_rad, rng=rng)
    E_off = np.vstack([electric_field_at_point(p, sources_pos, currents, sigma, electrode_radius) for p in off_pts])

    # Selectivity metric
    if metric == "rms":
        S = selectivity_rms(E_t, E_off)
    elif metric == "max":
        S = selectivity_max(E_t, E_off)
    elif metric == "activation":
        S = selectivity_activation(E_t, E_off, E_th, k, activation_model,
                                   use_activating_function, sources_pos, currents, sigma,
                                   np.array(target_point), off_pts, electrode_radius)
    else:
        raise ValueError("Unknown metric; use 'rms', 'max', or 'activation'.")

    S = max(0.0, S * (1.0 - soft_penalty))

    # Boundary diagnostics (optional)
    b_mean, b_max = (None, None)
    if record_boundary_stats and grounded_boundary:
        b_mean, b_max = boundary_potential_stats(sources_pos, currents, sigma, R_outer=R_outer, z=0.0,
                                                 n_samples=72, electrode_radius=electrode_radius)

    return {
        "selectivity": float(S),
        "E_target": E_t,
        "target_point": np.array(target_point),
        "sources_pos": sources_pos,
        "currents": currents,
        "sigma": sigma,
        "radius": radius,
        "height": (z1 - z0),
        "metric": metric,
        "grounded_boundary": grounded_boundary,
        "R_outer": R_outer,
        "I_net": I_net,
        "boundary_mean_abs_V": b_mean,
        "boundary_max_abs_V": b_max,
        "activation_params": {
            "E_th": E_th, "k": k, "model": activation_model,
            "use_activating_function": use_activating_function,
            "electrode_radius": electrode_radius,
            "min_target_activation": min_target_activation
        }
    }
