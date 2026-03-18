"""
subplot_registry.py — Subplot definitions for the experiment visualizations.

Each SubplotSpec holds metadata (key, label, group) and a render function
``fn(ax, scenario_data)`` that draws one subplot.  The scenario_data dict is
produced by ExperimentRunner.compute() and extended by build_figure() with
``marker_alpha`` and ``fig``.

The Fig object is retrieved via ``ax.get_figure()`` so render functions work
with both plt.subplots() and matplotlib.figure.Figure() backends.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# SubplotSpec dataclass
# ---------------------------------------------------------------------------

@dataclass
class SubplotSpec:
    key: str
    label: str
    group: str      # 'heatmap' | 'chart' | 'diagnostic'
    render: Callable  # fn(ax, scenario_data) -> None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_overlays(ax, sd, show_legend=False):
    """
    Draw the ellipse, Fixed Anchor, Original Mutable, and strategy markers.
    Strategy markers are semi-transparent (controlled by sd['marker_alpha']).
    Fixed Anchor and Original Mutable stay fully opaque.
    """
    marker_alpha = sd.get('marker_alpha', 0.55)
    ellipse_pts = sd['ellipse_pts']
    g_fixed = sd['g_fixed']
    g_mutable = sd['g_mutable']
    plot_styles = sd['plot_styles']
    details_per_strategy = sd['details_per_strategy']

    # Ellipse
    ax.plot(ellipse_pts[:, 0], ellipse_pts[:, 1], 'k--', label='Breeding Ellipse')

    # Fixed anchor (fully opaque)
    ax.scatter(g_fixed[0], g_fixed[1], c='gold', marker='*', s=300,
               edgecolors='k', label='Fixed Anchor', zorder=10)

    # Original mutable (fully opaque)
    ax.scatter(g_mutable[0], g_mutable[1], c='gray', marker='o', s=100,
               label='Original Mutable', zorder=9)

    # Strategy markers (semi-transparent)
    for s_name, (p_mk, p_col) in plot_styles.items():
        res = details_per_strategy[s_name]
        genotypes = res['out_dict']['genotypes']
        for i, g_cand_plot in enumerate(genotypes):
            is_diff_mut = (g_cand_plot[0] != g_mutable[0] or g_cand_plot[1] != g_mutable[1])
            is_diff_fix = (g_cand_plot[0] != g_fixed[0] or g_cand_plot[1] != g_fixed[1])
            if i > 1 or (is_diff_mut and is_diff_fix):
                ax.scatter(g_cand_plot[0], g_cand_plot[1], marker=p_mk, color=p_col,
                           s=120, label=s_name, zorder=11, alpha=marker_alpha)

    if show_legend:
        loc = 'left' if g_fixed[0] else 'right'
        loc = ('lower ' if g_mutable[1] else 'upper ') + loc
        ax.legend(loc=loc, fontsize='small', framealpha=0.8)


def _get_overlay_data(sd):
    """Lazily compute and cache the analytical overlay data in scenario_data."""
    if 'overlay_data' not in sd:
        sd['overlay_data'] = _compute_analytical_overlay(sd['grid_dict'])
    return sd['overlay_data']


def _compute_analytical_overlay(grid_dict):
    """
    Compute analytical MVP, BB, BBS points and condition regions
    on the (g_new_1, g_new_2) grid.
    """
    g_f = grid_dict['g_fixed']
    g_m = grid_dict['g_mutable']
    gamma = grid_dict['gamma']
    p = grid_dict['p']
    c = grid_dict['c']
    R = grid_dict['R']
    r_g = grid_dict['r_g']
    X = grid_dict['X']
    Y = grid_dict['Y']

    mu = (1 + c) / 2
    S = p * (1 - p) * (1 - c) ** 2
    A = gamma * S

    s_f = g_f[0] + g_f[1]
    d_f = g_f[0] - g_f[1]
    s_m = g_m[0] + g_m[1]
    d_m = g_m[0] - g_m[1]

    r_s = R * np.sqrt(2 * (1 + r_g))
    r_d = R * np.sqrt(2 * (1 - r_g))

    alpha = s_m - s_f
    delta = d_m - d_f
    U_f = mu * s_f - (A / 2) * d_f ** 2

    S_grid = X + Y
    D_grid = X - Y
    P_grid = mu * (S_grid - s_f) - A * d_f * (D_grid - d_f)
    Q_grid = D_grid - d_f
    Q2_grid = Q_grid ** 2
    Q2_safe = np.where(Q2_grid > 1e-20, Q2_grid, 1e-20)

    w_grid = P_grid / (A * Q2_safe)
    U_grid = mu * S_grid - (A / 2) * D_grid ** 2

    V_interior = U_f + P_grid ** 2 / (2 * A * Q2_safe)
    V_grid = np.where(w_grid <= 0, U_f,
             np.where(w_grid >= 1, U_grid, V_interior))

    # BB solution
    s_BB = s_m + r_s
    d_BB = d_m
    g_BB = np.array([(s_BB + d_BB) / 2, (s_BB - d_BB) / 2])
    U_BB = mu * s_BB - (A / 2) * d_BB ** 2
    P_BB = mu * (s_BB - s_f) - A * d_f * (d_BB - d_f)
    Q_BB = d_BB - d_f
    w_BB = P_BB / (A * Q_BB ** 2) if abs(Q_BB) > 1e-12 else np.inf
    if w_BB <= 0:
        V_BB = U_f
    elif w_BB >= 1:
        V_BB = U_BB
    else:
        V_BB = U_f + P_BB ** 2 / (2 * A * Q_BB ** 2)

    # BBS solution
    theta_fine = np.linspace(0, 2 * np.pi, 10000)
    ds_fine = r_s * np.cos(theta_fine)
    dd_fine = r_d * np.sin(theta_fine)
    s_fine = s_m + ds_fine
    d_fine = d_m + dd_fine
    U_fine = mu * s_fine - (A / 2) * d_fine ** 2
    idx_BBS = np.argmax(U_fine)
    theta_BBS = theta_fine[idx_BBS]
    s_BBS = s_fine[idx_BBS]
    d_BBS = d_fine[idx_BBS]
    g_BBS = np.array([(s_BBS + d_BBS) / 2, (s_BBS - d_BBS) / 2])
    U_BBS = U_fine[idx_BBS]
    P_BBS = mu * (s_BBS - s_f) - A * d_f * (d_BBS - d_f)
    Q_BBS = d_BBS - d_f
    w_BBS_port = P_BBS / (A * Q_BBS ** 2) if abs(Q_BBS) > 1e-12 else np.inf
    if w_BBS_port <= 0:
        V_BBS = U_f
    elif w_BBS_port >= 1:
        V_BBS = U_BBS
    else:
        V_BBS = U_f + P_BBS ** 2 / (2 * A * Q_BBS ** 2)

    # MVP solution
    L_tilde = np.sqrt(alpha ** 2 * r_d ** 2 + delta ** 2 * r_s ** 2)
    feasible = L_tilde >= r_s * r_d

    if feasible and abs(L_tilde) > 1e-12:
        base_angle = np.arctan2(delta * r_s, alpha * r_d)
        cos_arg = np.clip(r_s * r_d / L_tilde, -1, 1)
        delta_angle = np.arccos(cos_arg)
        theta1 = np.pi + base_angle + delta_angle
        theta2 = np.pi + base_angle - delta_angle

        def eval_V(theta):
            sn = s_m + r_s * np.cos(theta)
            dn = d_m + r_d * np.sin(theta)
            U_new = mu * sn - (A / 2) * dn ** 2
            P_ = mu * (sn - s_f) - A * d_f * (dn - d_f)
            Q_ = dn - d_f
            if abs(Q_) < 1e-12:
                return U_new, 1.0
            w_ = P_ / (A * Q_ ** 2)
            if w_ <= 0:
                return U_f, w_
            elif w_ >= 1:
                return U_new, w_
            else:
                return U_f + P_ ** 2 / (2 * A * Q_ ** 2), w_

        V1, w1 = eval_V(theta1)
        V2, w2 = eval_V(theta2)
        if V1 >= V2:
            theta_MVP, V_MVP_val, w_MVP_val = theta1, V1, w1
        else:
            theta_MVP, V_MVP_val, w_MVP_val = theta2, V2, w2

        s_MVP = s_m + r_s * np.cos(theta_MVP)
        d_MVP = d_m + r_d * np.sin(theta_MVP)
        g_MVP = np.array([(s_MVP + d_MVP) / 2, (s_MVP - d_MVP) / 2])
    else:
        theta_MVP = theta_BBS
        V_MVP_val = V_BBS
        w_MVP_val = w_BBS_port
        g_MVP = g_BBS

    beats_BB = V_grid > V_BB + 1e-10
    beats_BBS = V_grid > V_BBS + 1e-10
    feasible_mask = grid_dict.get('feasible_mask', np.ones_like(X, dtype=bool))

    return {
        'V_grid': V_grid, 'w_grid': w_grid, 'U_grid': U_grid,
        'beats_BB': beats_BB, 'beats_BBS': beats_BBS,
        'feasible_mask': feasible_mask,
        'V_BB': V_BB, 'V_BBS': V_BBS, 'V_MVP': V_MVP_val,
        'U_BB': U_BB, 'U_BBS': U_BBS,
        'w_BB': w_BB, 'w_MVP': w_MVP_val,
        'g_BB': g_BB, 'g_BBS': g_BBS, 'g_MVP': g_MVP,
        'g_f': g_f, 'g_m': g_m,
        'theta_BB': 0.0, 'theta_BBS': theta_BBS, 'theta_MVP': theta_MVP,
        'r_s': r_s, 'r_d': r_d, 's_m': s_m, 'd_m': d_m,
    }


# ---------------------------------------------------------------------------
# Individual render functions
# ---------------------------------------------------------------------------

def _render_value_added(ax, sd):
    """Portfolio Value Added heatmap (plot 0)."""
    X, Y = sd['X'], sd['Y']
    mv_grid = sd['grid_dict']['stats']['v_port']
    levels = np.linspace(np.min(mv_grid), np.max(mv_grid), 50)
    cf = ax.contourf(X, Y, mv_grid, levels=levels, cmap='RdBu', alpha=0.8)
    cb = ax.get_figure().colorbar(cf, ax=ax)
    cb.set_label('ΔV (Gain over Baseline)')
    cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.contour(X, Y, mv_grid, levels=[0], colors='k', linewidths=1, linestyles='--')
    _add_overlays(ax, sd)
    ax.set_title("Portfolio Value Added")
    ax.set_ylabel("Genotype dim 2")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)


def _render_performance_bar(ax, sd):
    """Dual-axis bar chart: economic gain vs adoption rate (plot 1)."""
    strat_names = sd['strat_names']
    gains = sd['gains']
    adaptions1 = sd['adaptions1']
    adaptions2 = sd['adaptions2']
    plot_styles = sd['plot_styles']
    bar_colors = [plot_styles[n][1] for n in strat_names]

    ax.bar(strat_names, gains, edgecolor='k', alpha=0.6, color=bar_colors, label='Economic Gain')
    ax.set_ylabel("Gain % (Normalized)")
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_ylim(0, 1.05 * 100)

    ax2 = ax.twinx()
    ax2.plot(strat_names, adaptions2, color='darkblue', marker='D', linestyle='None',
             markersize=8, label='Adaptation Rate (Env2)')
    ax2.plot(strat_names, adaptions1, color='darkred', marker='D', linestyle='None',
             markersize=8, label='Adaptation Rate (Env1)')
    ax2.set_ylabel("Adaptation Rate (Share)", color='darkred')
    ax2.set_ylim(0, 1.05)

    ax.set_title("Strategy Performance: Gain vs. Adoption")
    ax.set_xticks(range(len(strat_names)), labels=strat_names, rotation=45, ha='right')


def _render_portfolio_mean(ax, sd):
    """Portfolio Mean heatmap (plot 2)."""
    X, Y = sd['X'], sd['Y']
    cf = ax.contourf(X, Y, sd['grid_dict']['stats']['mean_port'], 50, cmap='viridis', alpha=0.8)
    ax.get_figure().colorbar(cf, ax=ax, label='Mean Yield')
    _add_overlays(ax, sd)
    ax.set_title("Portfolio Mean")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)


def _render_portfolio_var(ax, sd):
    """Portfolio Variance heatmap (plot 3)."""
    X, Y = sd['X'], sd['Y']
    cf = ax.contourf(X, Y, sd['grid_dict']['stats']['var_port'], 50, cmap='magma_r', alpha=0.8)
    ax.get_figure().colorbar(cf, ax=ax, label='Variance')
    _add_overlays(ax, sd)
    ax.set_title("Portfolio Variance")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)


def _render_variety_mean(ax, sd):
    """Single Variety Mean heatmap (plot 4)."""
    X, Y = sd['X'], sd['Y']
    cf = ax.contourf(X, Y, sd['U_grid'], 50, cmap='viridis', alpha=0.8)
    ax.get_figure().colorbar(cf, ax=ax, label='Mean Yield')
    _add_overlays(ax, sd)
    ax.set_title("Single Variety Mean")
    ax.set_ylabel("Genotype dim 2")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)


def _render_variety_var(ax, sd):
    """Single Variety Variance heatmap (plot 5)."""
    X, Y = sd['X'], sd['Y']
    cf = ax.contourf(X, Y, sd['Var_grid'], 50, cmap='magma_r', alpha=0.8)
    ax.get_figure().colorbar(cf, ax=ax, label='Variance')
    _add_overlays(ax, sd)
    ax.set_title("Single Variety Variance")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)


def _render_yield_ev1(ax, sd):
    """Yield in Environment 1, weighted by p (plot 6)."""
    X, Y = sd['X'], sd['Y']
    beta1 = sd['beta1']
    p = sd['p']
    z_ev1 = p * (beta1[0] * X + beta1[1] * Y)
    cf = ax.contourf(X, Y, z_ev1, 50, cmap='viridis', alpha=0.8)
    ax.get_figure().colorbar(cf, ax=ax, label='Yield EV1')
    _add_overlays(ax, sd)
    ax.set_title("Single Variety Yield EV1")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)


def _render_yield_ev2(ax, sd):
    """Yield in Environment 2, weighted by (1-p) (plot 7)."""
    X, Y = sd['X'], sd['Y']
    beta2 = sd['beta2']
    p = sd['p']
    z_ev2 = (1 - p) * (beta2[0] * X + beta2[1] * Y)
    cf = ax.contourf(X, Y, z_ev2, 50, cmap='viridis', alpha=0.8)
    ax.get_figure().colorbar(cf, ax=ax, label='Yield EV2')
    _add_overlays(ax, sd)
    ax.set_title("Single Variety Yield EV2")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)


def _render_adoption_share(ax, sd):
    """Candidate adoption share (w_c) heatmap (plot 8)."""
    X, Y = sd['X'], sd['Y']
    cf = ax.contourf(X, Y, sd['grid_dict']['stats']['w_c'], 50, cmap='RdYlGn', alpha=0.8)
    ax.get_figure().colorbar(cf, ax=ax, label='Adoption Rate')
    _add_overlays(ax, sd)
    ax.set_title("Possible adoption shares")
    ax.set_ylabel("Genotype dim 2")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)


def _render_weight_mutable(ax, sd):
    """Mutable asset weight (w_new_grid) heatmap (plot 9)."""
    X, Y = sd['X'], sd['Y']
    cf = ax.contourf(X, Y, sd['w_new_grid'], 50, cmap='RdYlGn', alpha=0.8)
    ax.get_figure().colorbar(cf, ax=ax, label='Weight mutable')
    _add_overlays(ax, sd)
    ax.set_title("Possible adoption shares g_mut")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)


def _render_covariance(ax, sd):
    """Covariance between candidate and fixed asset (plot 10)."""
    X, Y = sd['X'], sd['Y']
    Cov_grid = sd['grid_dict']['Cov_grid_fix']
    cov_min, cov_max = np.min(Cov_grid), np.max(Cov_grid)

    if cov_min < 0 < cov_max:
        divnorm = TwoSlopeNorm(vmin=cov_min, vcenter=0., vmax=cov_max)
    else:
        divnorm = Normalize(vmin=cov_min, vmax=cov_max)

    cf = ax.contourf(X, Y, Cov_grid, 50, cmap='RdBu_r', norm=divnorm, alpha=0.8)
    ax.get_figure().colorbar(cf, ax=ax, label='Covariance')
    ax.contour(X, Y, Cov_grid, levels=[0], colors='k', linewidths=1, linestyles='--')
    _add_overlays(ax, sd)
    ax.set_title("Covariance (Cand vs Fixed)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)


def _render_mvp_vs_bb(ax, sd):
    """MVP vs BeatBest condition heatmap (plot 12)."""
    X, Y = sd['X'], sd['Y']
    od = _get_overlay_data(sd)
    diff_BB = od['V_grid'] - od['V_BB']
    vmax_bb = np.max(diff_BB)
    vmin_bb = np.min(diff_BB)

    cf_bb = ax.contourf(X, Y, diff_BB, levels=np.linspace(vmin_bb, vmax_bb, 50),
                        cmap='RdBu', alpha=0.8)
    ax.contour(X, Y, diff_BB, levels=[0], colors='black', linewidths=1.5, linestyles=':')
    ax.contour(X, Y, od['w_grid'], levels=[0], colors='magenta', linewidths=2, linestyles=':')
    ax.contour(X, Y, od['w_grid'], levels=[1], colors='orange', linewidths=2, linestyles=':')

    handles = [
        Line2D([0], [0], color='black', linestyle=':', linewidth=1.5),
        Line2D([0], [0], color='magenta', linestyle=':', linewidth=2),
        Line2D([0], [0], color='orange', linestyle=':', linewidth=2),
    ]
    labels = ['MVP = BB', r'$w(\mathbf{g}_{\mathrm{new}})=0$',
              r'$w(\mathbf{g}_{\mathrm{new}})=1$']

    ax.set_ylabel("Genotype dim 2")
    ax.set_xlabel("Genotype dim 1")
    _add_overlays(ax, sd, show_legend=False)
    ax.legend(handles, labels, loc='best', fontsize='small', framealpha=0.8)
    ax.get_figure().colorbar(cf_bb, ax=ax).set_label(
        r'$V_{fn}(\mathbf{g}_{\mathrm{new}}) - V_{fn}(\mathbf{g}^{BB})$')
    ax.set_title('MVP vs BB')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])


def _render_mvp_vs_bbs(ax, sd):
    """MVP vs BBS condition heatmap (plot 13)."""
    X, Y = sd['X'], sd['Y']
    od = _get_overlay_data(sd)
    diff_BBS = od['V_grid'] - od['V_BBS']
    vmax_bbs = np.max(diff_BBS)
    vmin_bbs = np.min(diff_BBS)

    cf_bbs = ax.contourf(X, Y, diff_BBS, levels=np.linspace(vmin_bbs, vmax_bbs, 50),
                          cmap='RdBu', alpha=0.8)
    ax.contour(X, Y, diff_BBS, levels=[0], colors='black', linewidths=1.5, linestyles=':')
    ax.contour(X, Y, od['w_grid'], levels=[0], colors='magenta', linewidths=1, linestyles=':')
    ax.contour(X, Y, od['w_grid'], levels=[1], colors='orange', linewidths=1, linestyles=':')

    handles = [
        Line2D([0], [0], color='black', linestyle=':', linewidth=1.5),
        Line2D([0], [0], color='magenta', linestyle=':', linewidth=1),
        Line2D([0], [0], color='orange', linestyle=':', linewidth=1),
    ]
    labels = ['MVP = BBS', r'$w(\mathbf{g}_{\mathrm{new}})=0$',
              r'$w(\mathbf{g}_{\mathrm{new}})=1$']

    ax.set_xlabel("Genotype dim 1")
    _add_overlays(ax, sd, show_legend=False)
    ax.legend(handles, labels, loc='best', fontsize='small', framealpha=0.8)
    ax.get_figure().colorbar(cf_bbs, ax=ax).set_label(
        r'$V_{fn}(\mathbf{g}_{\mathrm{new}}) - V_{fn}(\mathbf{g}^{BBS})$')
    ax.set_title('MVP vs BBS')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])


def _render_legend(ax, sd):
    """Legend panel showing all strategy markers."""
    marker_alpha = sd.get('marker_alpha', 0.55)
    plot_styles = sd['plot_styles']
    ax.axis('off')

    legend_elements = [
        Line2D([0], [0], color='k', linestyle='--', label='Breeding Ellipse'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=18,
               markeredgecolor='k', label='Fixed Anchor'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12,
               label='Original Mutable'),
    ]
    for s_name, (p_mk, p_col) in plot_styles.items():
        legend_elements.append(
            Line2D([0], [0], marker=p_mk, color='w', markerfacecolor=p_col,
                   markersize=12, label=s_name, alpha=marker_alpha)
        )

    ax.legend(handles=legend_elements, loc='center', fontsize=9,
              frameon=True, borderpad=0.7)
    ax.set_title("Legend", fontweight='bold')


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SUBPLOT_REGISTRY: dict = {
    'value_added':     SubplotSpec('value_added',     'Portfolio Value Added',      'heatmap',     _render_value_added),
    'performance_bar': SubplotSpec('performance_bar', 'Strategy Performance',       'chart',       _render_performance_bar),
    'portfolio_mean':  SubplotSpec('portfolio_mean',  'Portfolio Mean',             'heatmap',     _render_portfolio_mean),
    'portfolio_var':   SubplotSpec('portfolio_var',   'Portfolio Variance',         'heatmap',     _render_portfolio_var),
    'variety_mean':    SubplotSpec('variety_mean',    'Single Variety Mean',        'heatmap',     _render_variety_mean),
    'variety_var':     SubplotSpec('variety_var',     'Single Variety Variance',    'heatmap',     _render_variety_var),
    'yield_ev1':       SubplotSpec('yield_ev1',       'Yield EV1',                  'heatmap',     _render_yield_ev1),
    'yield_ev2':       SubplotSpec('yield_ev2',       'Yield EV2',                  'heatmap',     _render_yield_ev2),
    'adoption_share':  SubplotSpec('adoption_share',  'Adoption Shares (w_c)',      'heatmap',     _render_adoption_share),
    'weight_mutable':  SubplotSpec('weight_mutable',  'Weight Mutable',             'heatmap',     _render_weight_mutable),
    'covariance':      SubplotSpec('covariance',      'Covariance (Cand vs Fixed)', 'heatmap',     _render_covariance),
    'mvp_vs_bb':       SubplotSpec('mvp_vs_bb',       'MVP vs BeatBest',            'diagnostic',  _render_mvp_vs_bb),
    'mvp_vs_bbs':      SubplotSpec('mvp_vs_bbs',      'MVP vs BBS',                 'diagnostic',  _render_mvp_vs_bbs),
    'legend':          SubplotSpec('legend',          'Legend',                     'chart',       _render_legend),
}

DEFAULT_SUBPLOT_IDS = [
    'value_added', 'performance_bar',  'adoption_share','portfolio_mean', 'portfolio_var', 'covariance','mvp_vs_bb', 'mvp_vs_bbs',  'legend',
]
