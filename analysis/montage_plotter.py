import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize

# ------------------------------------------------------------
# Electrode ring positions
# ------------------------------------------------------------
def electrode_positions(n=12, radius=1.0):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    return np.stack([x, y], axis=1)

# ------------------------------------------------------------
# Build montage connectivity
# ------------------------------------------------------------
def build_connections(n_electrodes=12, adjacent_only=True):
    connections = []
    for i in range(n_electrodes):
        if adjacent_only:
            pair = [(i, (i+1)%n_electrodes)]
        else:
            step = (i*3 + 1) % n_electrodes
            pair = [(i, step)]
        connections.append(pair)
        tri = [(i, (i+1)%n_electrodes), ((i+1)%n_electrodes, (i+2)%n_electrodes)]
        connections.append(tri)
    return connections

# ------------------------------------------------------------
# Curved gradient field (blue→red)
# ------------------------------------------------------------
def draw_gradient_field(ax, p1, p2, strength=0.25, steps=40):
    # Control point offset for curvature
    mid = 0.5*(p1 + p2)
    perp = np.array([-(p2[1]-p1[1]), p2[0]-p1[0]])
    perp /= np.linalg.norm(perp)
    mid = mid + strength * perp

    # Parameterize curve points properly in 2D
    t = np.linspace(0, 1, steps)[:, None]
    curve = (1-t)**2 * p1[None, :] + 2*(1-t)*t * mid[None, :] + t**2 * p2[None, :]

    # Create gradient line
    cmap = ListedColormap(["#4F9DFF", "#FF6C6C"])
    points = curve.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=Normalize(0, 1),
                        linewidth=2.4, alpha=0.85)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)

# ------------------------------------------------------------
# Plot montage-space growth
# ------------------------------------------------------------
def plot_connectivity_with_field(dims=[1,2,4,8,12], n_electrodes=12, adjacent_only=True):
    pos = electrode_positions(n_electrodes)
    connections = build_connections(n_electrodes, adjacent_only)

    fig, axes = plt.subplots(1, len(dims), figsize=(2.8*len(dims), 3.0))
    if len(dims)==1: axes=[axes]

    for ax, dim in zip(axes, dims):
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f"{dim}-D montage space", fontsize=11, pad=22)
        # electrodes
        ax.scatter(pos[:,0], pos[:,1], s=420, c="white", edgecolor="k", lw=1.1, zorder=3)
        # connections
        for j in range(dim):
            for (a,b) in connections[j]:
                xa, ya = pos[a]; xb, yb = pos[b]
                ax.plot([xa, xb], [ya, yb], color='tab:red', lw=1.6, alpha=0.9, zorder=1)
                draw_gradient_field(ax, pos[a], pos[b], strength=0.3)
        for i,(x,y) in enumerate(pos):
            ax.text(x*1.22, y*1.22, str(i+1), ha="center", va="center", fontsize=7)

    fig.suptitle("Montage-space expansion and electric field spread", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig("montage_field_gradient_poster.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_connectivity_with_field(dims=[1,2,4,8,12], adjacent_only=True)
