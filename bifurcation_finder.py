"""
Automated Bifurcation Analysis of Two-Parameter Families of
Two-Dimensional Dynamical Systems.

Given a system

    dx/dt = f(x, y, alpha, beta)
    dy/dt = g(x, y, alpha, beta)

this tool sweeps a grid of (alpha, beta) parameter values, numerically
locates all equilibria at each grid point, classifies each equilibrium's
stability via the eigenvalues of the Jacobian, and flags points where the
classification indicates a Hopf or saddle-node bifurcation.

Validated against two known systems (see README / report):
  - Hopf normal form: correctly recovers the bifurcation line alpha = 0.
  - Bogdanov-Takens system: correctly recovers the Hopf bifurcation curve
    beta = sqrt(-alpha).

Known limitation: saddle-node detection was NOT reliably demonstrated.
The classification tolerance below is derived from grid spacing
(tol = max(a_step, b_step) / 2), which means the threshold for declaring
a bifurcation changes with grid resolution. Saddle-node points require an
eigenvalue to be almost exactly zero, which a discrete grid will rarely
land on -- this is the direct cause of the saddle-node misses documented
in the accompanying report, not a separate/unrelated bug.

Usage: edit the "Configuration" section at the bottom of this file to
define a new system and parameter ranges, then run the script directly.

Dependencies: numpy, scipy, sympy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sympy as sp


def find_bifurcations_2d(f_expr, g_expr, x_sym, y_sym, a_sym, b_sym,
                          a_range, b_range, xy_search_range,
                          n_a=60, n_b=60, n_ic=6):
    """
    Sweep a 2D parameter grid, locate equilibria, and classify stability.

    Parameters
    ----------
    f_expr, g_expr : sympy expressions
        The right-hand sides dx/dt = f, dy/dt = g, written in terms of
        x_sym, y_sym, a_sym, b_sym.
    x_sym, y_sym : sympy symbols
        State variables.
    a_sym, b_sym : sympy symbols
        The two parameters being swept.
    a_range, b_range : tuple(float, float)
        (min, max) sweep range for each parameter.
    xy_search_range : tuple(float, float)
        (min, max) square window used to seed initial guesses for
        equilibrium root-finding.
    n_a, n_b : int
        Grid resolution along each parameter axis.
    n_ic : int
        Number of initial-condition values per axis (n_ic^2 total guesses
        per parameter point) used to find multiple equilibria via
        scipy.optimize.fsolve, since fsolve is a local root finder.

    Returns
    -------
    list[dict]
        One entry per (equilibrium, parameter point) found, with keys
        'alpha', 'beta', 'x', 'y', 'label'.
    """
    # Build fast numeric functions from the symbolic system and its Jacobian.
    f_num = sp.lambdify((x_sym, y_sym, a_sym, b_sym), f_expr, "numpy")
    g_num = sp.lambdify((x_sym, y_sym, a_sym, b_sym), g_expr, "numpy")

    J_exprs = [
        sp.diff(f_expr, x_sym), sp.diff(f_expr, y_sym),
        sp.diff(g_expr, x_sym), sp.diff(g_expr, y_sym),
    ]
    J_num = [sp.lambdify((x_sym, y_sym, a_sym, b_sym), e, "numpy")
             for e in J_exprs]

    def system_vec(xy, a, b):
        return [f_num(xy[0], xy[1], a, b),
                g_num(xy[0], xy[1], a, b)]

    def jacobian_matrix(x, y, a, b):
        return np.array([[J_num[0](x, y, a, b), J_num[1](x, y, a, b)],
                          [J_num[2](x, y, a, b), J_num[3](x, y, a, b)]],
                         dtype=float)

    def classify(x, y, a, b, tol=1e-3):
        """
        Classify an equilibrium's stability from its Jacobian eigenvalues.

        Order of checks matters: saddle-node is checked before Hopf
        because a saddle-node point can have a near-zero trace too, and
        we want the more specific real-eigenvalue condition to win.
        """
        J = jacobian_matrix(x, y, a, b)
        eigs = np.linalg.eigvals(J)
        re, im = np.real(eigs), np.imag(eigs)

        # Saddle-node: exactly one eigenvalue with both Re and Im ~ 0,
        # while the other eigenvalue is clearly away from zero.
        zero_eigs = (np.abs(re) < tol) & (np.abs(im) < tol)
        if np.sum(zero_eigs) == 1 and np.all(np.abs(re[~zero_eigs]) > tol):
            return 'saddle_node_bifurcation'

        # Hopf: trace ~ 0 (real parts crossing zero together) with
        # nonzero imaginary parts (complex conjugate pair).
        trace = np.trace(J)
        if np.abs(trace) < tol and np.any(np.abs(im) > tol):
            return 'hopf_bifurcation'

        # Saddle: real parts of opposite sign.
        if np.min(re) < -tol and np.max(re) > tol:
            return 'saddle'

        # Stable sink: both real parts negative.
        if np.max(re) < -tol:
            return 'stable_sink'

        # Unstable source: both real parts positive.
        if np.min(re) > tol:
            return 'unstable_source'

        # Fallback: real parts near zero but doesn't match the Hopf
        # condition above (e.g. imaginary parts also ~ 0) -- treated as
        # a center for visualization purposes.
        return 'center'

    # Force alpha = 0 into the sweep so exact-boundary bifurcations (like
    # the Hopf normal form's alpha = 0 line) aren't skipped by the grid.
    # NOTE: this fix is only applied to the alpha axis, not beta -- a
    # bifurcation that falls at a non-grid beta value can still be missed.
    a_sweep = np.unique(np.sort(np.append(np.linspace(a_range[0], a_range[1], n_a), 0.0)))
    b_sweep = np.linspace(b_range[0], b_range[1], n_b)
    a_step = a_sweep[1] - a_sweep[0]
    b_step = b_sweep[1] - b_sweep[0]

    # Classification tolerance is tied to grid spacing. This is a deliberate
    # but imperfect choice: it scales detection sensitivity with resolution,
    # but also means saddle-node points (which require an eigenvalue to be
    # almost exactly zero) are only caught if the grid happens to land close
    # enough -- see module docstring / README for the practical consequence.
    tol = max(a_step, b_step) / 2
    ic_vals = np.linspace(xy_search_range[0], xy_search_range[1], n_ic)

    results = []
    total = len(a_sweep) * n_b
    print(f"Sweeping {total} parameter combinations")

    for i, a_val in enumerate(a_sweep):
        if i % 10 == 0:
            print(f"  alpha = {a_val:.2f}  ({i * n_b}/{total})")
        for b_val in b_sweep:
            found_eqs = []

            # Try multiple initial conditions since fsolve only finds the
            # equilibrium nearest to its starting point.
            for x0 in ic_vals:
                for y0 in ic_vals:
                    try:
                        sol, info, ier, _ = fsolve(
                            system_vec, [x0, y0], args=(a_val, b_val),
                            full_output=True
                        )
                        if ier != 1:
                            continue
                        residual = np.max(np.abs(system_vec(sol, a_val, b_val)))
                        if residual > 1e-10:
                            continue
                        is_new = all(
                            np.linalg.norm(sol - eq) > 1e-4
                            for eq in found_eqs
                        )
                        if is_new:
                            found_eqs.append(sol)
                    except Exception:
                        continue

            for eq in found_eqs:
                label = classify(eq[0], eq[1], a_val, b_val, tol=tol)
                results.append({
                    'alpha': a_val,
                    'beta': b_val,
                    'x': eq[0],
                    'y': eq[1],
                    'label': label,
                })

    print(f"Done. Found {len(results)} equilibrium points total.")
    return results


def plot_bifurcation_2d(results, a_range, b_range, param_names=("alpha", "beta"),
                         title="Bifurcation Diagram"):
    """
    Render a two-panel bifurcation diagram: a full stability map on the
    left, and the isolated Hopf / saddle-node curves on the right.
    Saves the figure to bifurcation_diagram.png in the working directory.

    Parameters
    ----------
    results : list[dict]
        Output of find_bifurcations_2d.
    a_range, b_range : tuple(float, float)
        Axis limits for the right-hand (bifurcation-only) panel.
    param_names : tuple(str, str)
        Axis labels for the two parameters.
    title : str
        Figure title.
    """
    if not results:
        print("No equilibria found -- try widening xy_search_range or parameter ranges.")
        return

    alphas = np.array([r['alpha'] for r in results])
    betas = np.array([r['beta'] for r in results])
    labels = np.array([r['label'] for r in results])

    color_map = {
        'stable_sink': '#2196F3',
        'unstable_source': '#F44336',
        'saddle': '#E0E0E0',
        'center': '#9C27B0',
        'hopf_bifurcation': '#00FF00',
        'saddle_node_bifurcation': '#FF00FF',
    }
    label_names = {
        'stable_sink': 'Stable Sink',
        'unstable_source': 'Unstable Source',
        'saddle': 'Saddle',
        'center': 'Center',
        'hopf_bifurcation': 'Hopf bifurcation',
        'saddle_node_bifurcation': 'Saddle-node bifurcation',
    }
    zorder_map = {
        'hopf_bifurcation': 6,
        'saddle_node_bifurcation': 5,
        'stable_sink': 4,
        'unstable_source': 4,
        'center': 3,
        'saddle': 2,
    }

    def style_ax(ax):
        ax.axvline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.4)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.4)
        ax.set_xlabel(f"Parameter  {param_names[0]}", fontsize=11)
        ax.set_ylabel(f"Parameter  {param_names[1]}", fontsize=11)
        ax.grid(True, alpha=0.15)
        ax.spines[['top', 'right']].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='normal', y=1.01)

    # Left panel: full stability map.
    ax = axes[0]
    for lbl, color in color_map.items():
        mask = labels == lbl
        if mask.any():
            ax.scatter(alphas[mask], betas[mask],
                       s=6, color=color, alpha=1.0,
                       label=label_names[lbl], linewidths=0,
                       zorder=zorder_map.get(lbl, 3))
    style_ax(ax)
    ax.set_title("Stability map in parameter space", fontsize=11)
    ax.legend(markerscale=3, framealpha=0.9, fontsize=8, loc='lower left')

    # Right panel: isolated bifurcation curves only.
    ax2 = axes[1]
    bif_labels = ['hopf_bifurcation', 'saddle_node_bifurcation']
    plotted = False
    for lbl in bif_labels:
        mask = labels == lbl
        if mask.any():
            ax2.scatter(alphas[mask], betas[mask],
                        s=10, color=color_map[lbl], alpha=0.9,
                        label=label_names[lbl], linewidths=0, zorder=4)
            plotted = True
    if not plotted:
        ax2.text(0.5, 0.5, 'No bifurcations detected\nin this parameter range',
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=11, color='gray')
    style_ax(ax2)
    ax2.set_title("Bifurcation curves", fontsize=11)
    ax2.legend(markerscale=3, framealpha=0.9, fontsize=9)
    ax2.set_xlim(a_range[0], a_range[1])
    ax2.set_ylim(b_range[0], b_range[1])

    plt.tight_layout()
    plt.savefig("bifurcation_diagram.png", dpi=150, bbox_inches="tight")
    print("Saved -> bifurcation_diagram.png")
    plt.show()


if __name__ == "__main__":
    # ---- Configuration: edit this section to analyze a new system ----
    x, y, a, b = sp.symbols("x y alpha beta")

    f_equation = y                        # dx/dt
    g_equation = a + b * y + x**2 + x * y  # dy/dt  (Bogdanov-Takens system)

    a_min, a_max = -1.5, 1.5   # sweep range for alpha
    b_min, b_max = -1.5, 1.5   # sweep range for beta
    xy_min, xy_max = -2.0, 2.0  # search window for equilibria

    param_name_a = "alpha"
    param_name_b = "beta"

    print("System:")
    print(f"  dx/dt = {f_equation}")
    print(f"  dy/dt = {g_equation}")
    print()

    results = find_bifurcations_2d(
        f_equation, g_equation,
        x, y, a, b,
        a_range=(a_min, a_max),
        b_range=(b_min, b_max),
        xy_search_range=(xy_min, xy_max),
        n_a=60,
        n_b=60,
    )

    plot_bifurcation_2d(
        results,
        a_range=(a_min, a_max),
        b_range=(b_min, b_max),
        param_names=(param_name_a, param_name_b),
        title=f"Bifurcation diagram  |  dx/dt = {f_equation},  dy/dt = {g_equation}",
    )
