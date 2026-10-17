"""Publication-quality plotting functions.

Every compared algorithm/variant is distinguished by a *triple* encoding —
color + line style/marker (line plots) or color + hatch (bar plots) — so the
figures stay readable in grayscale print.  Text labels are placed with
offsets away from the data centroid and carry a white background box, and
legends are placed outside the axes whenever they could cover data, so that
text never overlaps plotted elements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from base import savefig
from config import FIGURE_STYLE, USE_LATEX


def _set_style():
    configure_plotting(use_latex=USE_LATEX)


def configure_plotting(style=None, palette='viridis', use_latex=False):
    from base import configure_plotting as base_configure
    base_configure(style=style or 'seaborn-v0_8-whitegrid', palette=palette,
                   use_latex=use_latex)
    if FIGURE_STYLE == 'paper':
        rcParams['axes.grid'] = False
        rcParams['axes.edgecolor'] = '#333333'
        rcParams['axes.linewidth'] = 0.8
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.direction'] = 'in'


# ---------------------------------------------------------------------------
# Per-policy visual encoding: color + marker + line style + hatch.
# The proposed method (OnlineMatching) is visually emphasized.
# ---------------------------------------------------------------------------
POLICY_STYLES = {
    'OnlineMatching': dict(color='#C00000', marker='*', ls='-', hatch='',
                           lw=2.4, ms=11),
    'DedicatedOnly':  dict(color='#4C72B0', marker='o', ls='-', hatch=''),
    'CrowdFirst':     dict(color='#DD8452', marker='s', ls='--', hatch='//'),
    'Hybrid':         dict(color='#55A868', marker='^', ls='-.', hatch='..'),
    'Nearest':        dict(color='#8172B3', marker='D', ls=(0, (3, 1, 1, 1)),
                           hatch='\\\\'),
    'Random':         dict(color='#937860', marker='P', ls=(0, (5, 2)),
                           hatch='++'),
    'FIFOBatch':      dict(color='#DA8BC3', marker='X', ls='-', hatch='oo'),
    'Queueing':       dict(color='#8C8C8C', marker='h', ls='--', hatch='**'),
    'ORToolsVRP':     dict(color='#64B5CD', marker='d', ls='-.', hatch='xx'),
    'Rollout':        dict(color='#B6B6B6', marker='<', ls=':', hatch='..'),
    # Ablation variants of the proposed method
    'Full':           dict(color='#C00000', marker='*', ls='-', hatch='',
                           lw=2.4, ms=11),
    'NoDualPrice':    dict(color='#4C72B0', marker='o', ls='--', hatch='//'),
    'NoDeadlineRisk': dict(color='#55A868', marker='^', ls='-.', hatch='..'),
    'NoQueueLoad':    dict(color='#DD8452', marker='s', ls=':', hatch='xx'),
    'NoDistance':     dict(color='#8172B3', marker='D', ls=(0, (3, 1, 1, 1)),
                           hatch='\\\\'),
}

# Deterministic fallback cycles for names not in the registry.
_FALLBACK_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
                    '#937860', '#DA8BC3', '#8C8C8C', '#64B5CD', '#CCB974']
_FALLBACK_MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', 'h', 'd', '<']
_FALLBACK_LS = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
_FALLBACK_HATCH = ['', '//', '..', 'xx', '\\\\', '++', 'oo', '**']


def style_for(name):
    """Visual style (color/marker/linestyle/hatch) for a policy or variant."""
    if name in POLICY_STYLES:
        return POLICY_STYLES[name]
    i = abs(hash(str(name))) % len(_FALLBACK_COLORS)
    return dict(color=_FALLBACK_COLORS[i], marker=_FALLBACK_MARKERS[i],
                ls=_FALLBACK_LS[i % len(_FALLBACK_LS)],
                hatch=_FALLBACK_HATCH[i % len(_FALLBACK_HATCH)])


def _line_kwargs(name):
    st = style_for(name)
    return dict(color=st['color'], marker=st['marker'], ls=st['ls'],
                linewidth=st.get('lw', 1.6), markersize=st.get('ms', 6),
                markeredgecolor='white', markeredgewidth=0.4)


def _bar_kwargs(name):
    st = style_for(name)
    return dict(color=st['color'], hatch=st['hatch'], edgecolor='black',
                linewidth=0.4)


def _annotate_point(ax, x, y, text, centroid=None, fontsize=9):
    """Annotate a point with the text offset *away* from the data centroid and
    backed by a white box, so labels never collide with markers/lines."""
    if centroid is None:
        dx, dy = 10, 8
    else:
        cx, cy = centroid
        dx, dy = 12, 10
        if x < cx:
            dx = -12
        if y < cy:
            dy = -10
    ha = 'left' if dx >= 0 else 'right'
    ax.annotate(text, (x, y), textcoords='offset points', xytext=(dx, dy),
                fontsize=fontsize, ha=ha,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='none', alpha=0.75), zorder=6)


def plot_backlog(backlogs, labels, duration):
    _set_style()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for b, l in zip(backlogs, labels):
        t = np.arange(len(b)) * (duration / max(1, len(b)))
        kw = _line_kwargs(l)
        ax.plot(t, b, label=l, alpha=0.9, markevery=max(1, len(b) // 12), **kw)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Backlog (orders)')
    ax.set_title('Real-time backlog under crowdsourced delivery policies')
    # Legend outside the axes: the backlog traces occupy the upper area.
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    if FIGURE_STYLE != 'paper':
        ax.grid(True, alpha=0.3)
    return savefig(fig, 'G_backlog_timeseries.png')


def plot_metrics_bar(summary_df, metrics=None, ylabels=None):
    """Bar plot with error bars from replication summary.

    Each policy bar uses its own color + hatch pair.  summary_df is expected
    to contain columns: policy, metric, mean, std, ci_lower, ci_upper.
    """
    _set_style()
    if metrics is None:
        metrics = ['avg_delay_min', 'on_time_rate', 'max_backlog', 'crowd_served']
    if ylabels is None:
        # Terminology standardized across figures (reviewer minor comment 2):
        # "fulfillment time" = elapsed time from order creation to completion.
        ylabels = ['Avg. fulfillment time (min)', 'On-time rate',
                   'Max backlog (orders)', 'Crowd-served orders']

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, col, ylab in zip(axes, metrics, ylabels):
        sub = summary_df[summary_df['metric'] == col]
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.sort_values('mean')
        x = np.arange(len(sub))
        err = (sub['mean'] - sub['ci_lower'], sub['ci_upper'] - sub['mean'])
        for xi, (_, row) in zip(x, sub.iterrows()):
            ax.bar(xi, row['mean'], yerr=[[row['mean'] - row['ci_lower']],
                                          [row['ci_upper'] - row['mean']]],
                   capsize=3, alpha=0.9, **_bar_kwargs(row['policy']))
        ax.set_xticks(x)
        ax.set_xticklabels(sub['policy'], rotation=45, ha='right')
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
    plt.tight_layout()
    return savefig(fig, 'G_metrics_bar.png')


def plot_utilization(results):
    """Dedicated vs crowd utilization bar plot (color + hatch per fleet type)."""
    _set_style()
    rows = []
    for name, m in results.items():
        rows.append({'policy': name, 'type': 'dedicated',
                     'utilization': m['dedicated_utilization']})
        rows.append({'policy': name, 'type': 'crowd',
                     'utilization': m['crowd_utilization']})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    width = 0.35
    policies = df['policy'].unique()
    x = np.arange(len(policies))
    type_style = {'dedicated': dict(color='#4C72B0', hatch=''),
                  'crowd': dict(color='#DD8452', hatch='//')}
    for i, t in enumerate(['dedicated', 'crowd']):
        vals = [df[(df['policy'] == p) & (df['type'] == t)]['utilization'].values[0]
                for p in policies]
        st = type_style[t]
        ax.bar(x + (i - 0.5) * width, vals, width, label=t, alpha=0.9,
               color=st['color'], hatch=st['hatch'], edgecolor='black',
               linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right')
    ax.set_ylabel('Utilization')
    ax.set_title('Dedicated vs crowd utilization')
    ax.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    return savefig(fig, 'G_utilization.png')


def plot_crowd_availability(params):
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(24), params['crowd_avail'], marker='o', color='teal',
            linewidth=2, markersize=5, ls='-')
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Availability probability')
    ax.set_title('Crowd courier availability curve')
    ax.set_ylim(0, 1)
    if FIGURE_STYLE != 'paper':
        ax.grid(True, alpha=0.3)
    return savefig(fig, 'G_crowd_availability.png')


def plot_pareto_frontier(raw_df, x='avg_delay_min', y='total_cost',
                         color='crowd_served'):
    """Plot mean fulfillment time vs. cost per policy and draw the Pareto frontier.

    Each policy is a distinct color+marker; labels are placed away from the
    data centroid with a white background box so they never overlap points,
    dominance arrows or the frontier line.
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    means = []
    for policy, sub in raw_df.groupby('policy'):
        mean = sub.mean(numeric_only=True)
        means.append((policy, float(mean[x]), float(mean[y])))
    xs = [m[1] for m in means]
    ys = [m[2] for m in means]
    centroid = (float(np.mean(xs)), float(np.mean(ys)))
    for policy, mx, my in means:
        st = style_for(policy)
        ax.scatter(mx, my, s=140, alpha=0.9, color=st['color'],
                   marker=st['marker'], edgecolors='black', linewidths=0.6,
                   zorder=5)
        _annotate_point(ax, mx, my, policy, centroid=centroid)

    # Compute Pareto-optimal points for two minimization objectives.
    points = [(p[1], p[2]) for p in means]
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
    frontier = []
    best_y = float('inf')
    for px, py in sorted_points:
        if py < best_y:
            frontier.append((px, py))
            best_y = py
    if len(frontier) >= 2:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, 'k--', linewidth=1.5, label='Pareto frontier')
        # Draw dominance arrows from dominated to frontier points.
        frontier_set = set(frontier)
        for px, py in points:
            if (px, py) not in frontier_set:
                # Nearest frontier point that dominates or is closest.
                target = min(frontier,
                             key=lambda q: (q[0] - px) ** 2 + (q[1] - py) ** 2)
                ax.annotate('', xy=target, xytext=(px, py),
                            arrowprops=dict(arrowstyle='->', color='gray',
                                            lw=0.8, alpha=0.5))
        ax.legend(frameon=False, loc='upper right')

    ax.set_xlabel('Avg. fulfillment time (min)')
    ax.set_ylabel('Total cost (yuan)')
    ax.set_title('Fulfillment-time vs. cost trade-off (Pareto frontier)')
    # Leave headroom so offset labels stay inside the axes.
    ax.margins(x=0.15, y=0.15)
    plt.tight_layout()
    return savefig(fig, 'G_pareto_frontier.png')


def plot_cdf_delays(raw_df):
    """Empirical CDF of per-replication average fulfillment time per policy."""
    _set_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for policy, sub in raw_df.groupby('policy'):
        delays = sub['avg_delay_min'].dropna().sort_values()
        if len(delays) == 0:
            continue
        p = np.arange(1, len(delays) + 1) / len(delays)
        kw = _line_kwargs(policy)
        ax.plot(delays, p, label=policy, markevery=max(1, len(delays) // 8),
                **kw)
    ax.set_xlabel('Average fulfillment time per replication (min)')
    ax.set_ylabel('CDF')
    ax.set_title('Empirical CDF of average fulfillment time')
    ax.legend(frameon=False, loc='lower right')
    plt.tight_layout()
    return savefig(fig, 'G_delay_cdf.png')


def plot_sensitivity_heatmap(sens_df, param_name, metric='avg_delay_min_mean'):
    """Heatmap of sensitivity results (policy x parameter value)."""
    _set_style()
    pivot = sens_df.pivot(index='policy', columns=param_name, values=metric)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Policy')
    ax.set_title(f'Sensitivity of {metric} to {param_name}')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return savefig(fig, f'G_sensitivity_{param_name}.png')


def plot_sensitivity_lines(sens_df, param_name='N_CROWD_MAX',
                           metric='avg_delay_min_mean',
                           operating_point=None):
    """Line version of the sensitivity sweep: one color+marker+linestyle per
    policy, with the calibrated operating point marked by a vertical line."""
    _set_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for policy, sub in sens_df.groupby('policy'):
        sub = sub.sort_values(param_name)
        kw = _line_kwargs(policy)
        ax.plot(sub[param_name], sub[metric], label=policy, **kw)
    if operating_point is not None:
        ax.axvline(operating_point, color='black', ls=':', linewidth=1.2)
        ax.annotate('calibrated\noperating point', (operating_point, ax.get_ylim()[0]),
                    textcoords='offset points', xytext=(6, 10), fontsize=9,
                    ha='left',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='none', alpha=0.8))
    ax.set_xlabel(param_name)
    ax.set_ylabel('Avg. fulfillment time (min)')
    ax.set_title(f'Sensitivity of fulfillment time to {param_name}')
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    if FIGURE_STYLE != 'paper':
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return savefig(fig, f'G_sensitivity_lines_{param_name}.png')


# ---------------------------------------------------------------------------
# Revision experiments: ablation, baseline tuning, circuity, capacity mix
# ---------------------------------------------------------------------------

def plot_ablation(summary_df, metric='avg_delay_min'):
    """Bar chart of the ablation variants with 95% CI, one row per scenario.

    summary_df carries a ``scenario`` column (base / stress); each variant is
    encoded by color + hatch and the full model is visually emphasized.
    """
    _set_style()
    scenarios = (list(summary_df['scenario'].unique())
                 if 'scenario' in summary_df.columns else ['all'])
    metrics = [metric, 'total_cost']
    ylabels = ['Avg. fulfillment time (min)', 'Total cost (yuan)']
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(11, 4.2 * len(scenarios)),
                             squeeze=False)
    for row, scenario in enumerate(scenarios):
        sc_df = (summary_df[summary_df['scenario'] == scenario]
                 if 'scenario' in summary_df.columns else summary_df)
        for ax, col, ylab in zip(axes[row], metrics, ylabels):
            sub = sc_df[sc_df['metric'] == col].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            sub['__order'] = sub['policy'].map(lambda p: 0 if p == 'Full' else 1)
            sub = sub.sort_values(['__order', 'mean'])
            x = np.arange(len(sub))
            for xi, (_, r) in zip(x, sub.iterrows()):
                ax.bar(xi, r['mean'],
                       yerr=[[r['mean'] - r['ci_lower']],
                             [r['ci_upper'] - r['mean']]],
                       capsize=3, alpha=0.9, **_bar_kwargs(r['policy']))
            ax.set_xticks(x)
            ax.set_xticklabels(sub['policy'], rotation=30, ha='right')
            ax.set_ylabel(ylab)
            ax.set_title(f'{ylab} [{scenario} scenario]')
    plt.tight_layout()
    return savefig(fig, 'G_ablation.png')


def plot_baseline_tuning(tuning_df, metric='avg_delay_min'):
    """Tuning curves for the Hybrid and Queueing baseline thresholds."""
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, baseline in zip(axes, ['Hybrid', 'Queueing']):
        sub = tuning_df[tuning_df['baseline'] == baseline]
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.sort_values('value')
        param = sub['parameter'].iloc[0]
        kw = _line_kwargs(baseline)
        ax.errorbar(sub['value'], sub[f'{metric}_mean'],
                    yerr=sub[f'{metric}_std'], capsize=3, **kw)
        best_row = sub.loc[sub[f'{metric}_mean'].idxmin()]
        ax.axvline(best_row['value'], color=kw['color'], ls=':', linewidth=1.0)
        ax.annotate(f"selected = {best_row['value']:g}",
                    (best_row['value'], best_row[f'{metric}_mean']),
                    textcoords='offset points', xytext=(8, -16), fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='none', alpha=0.8))
        ax.set_xlabel(param)
        ax.set_ylabel('Avg. fulfillment time (min)')
        ax.set_title(f'{baseline} baseline tuning')
        if FIGURE_STYLE != 'paper':
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return savefig(fig, 'G_baseline_tuning.png')


def plot_circuity(circ_df, metric='avg_delay_min'):
    """Strategy metrics vs. road-circuity multiplier (ranking stability)."""
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, col, ylab in zip(axes,
                             [f'{metric}_mean', 'on_time_rate_mean'],
                             ['Avg. fulfillment time (min)', 'On-time rate']):
        for policy, sub in circ_df.groupby('policy'):
            sub = sub.sort_values('CIRCUITY_FACTOR')
            ax.plot(sub['CIRCUITY_FACTOR'], sub[col], label=policy,
                    **_line_kwargs(policy))
        ax.set_xlabel('Road-circuity multiplier')
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
        if FIGURE_STYLE != 'paper':
            ax.grid(True, alpha=0.3)
    axes[1].legend(frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    return savefig(fig, 'G_circuity.png')


def plot_capacity_mix(summary_df, metric='avg_delay_min'):
    """Metrics vs. dedicated share of a fixed total fleet."""
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, col, ylab in zip(axes,
                             [f'{metric}_mean', 'total_cost_mean'],
                             ['Avg. fulfillment time (min)', 'Total cost (yuan)']):
        for policy, sub in summary_df.groupby('policy'):
            sub = sub.sort_values('ded_share')
            ax.plot(sub['ded_share'], sub[col], label=policy,
                    **_line_kwargs(policy))
        ax.set_xlabel('Dedicated share of total fleet')
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
        if FIGURE_STYLE != 'paper':
            ax.grid(True, alpha=0.3)
    axes[1].legend(frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    return savefig(fig, 'G_capacity_mix.png')


def plot_regimes(summary_df, metric='avg_delay_min'):
    """Core-policy comparison across congestion regimes (grouped bars).

    summary_df carries a ``scenario`` column (base / stress / extreme); one
    subplot per metric, regimes on the x-axis, policies encoded by
    color + hatch, error bars show the 95% CI over seeds.
    """
    _set_style()
    metrics = [metric, 'on_time_rate']
    ylabels = ['Avg. fulfillment time (min)', 'On-time rate']
    scenarios = list(summary_df['scenario'].unique())
    policies = list(summary_df['policy'].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.5),
                             squeeze=False)
    for ax, col, ylab in zip(axes[0], metrics, ylabels):
        x = np.arange(len(scenarios))
        n_p = len(policies)
        width = 0.8 / max(1, n_p)
        for j, pol in enumerate(policies):
            means, lo, hi = [], [], []
            for sc in scenarios:
                sub = summary_df[(summary_df['scenario'] == sc)
                                 & (summary_df['policy'] == pol)
                                 & (summary_df['metric'] == col)]
                means.append(float(sub['mean'].iloc[0]))
                lo.append(float(sub['mean'].iloc[0] - sub['ci_lower'].iloc[0]))
                hi.append(float(sub['ci_upper'].iloc[0] - sub['mean'].iloc[0]))
            ax.bar(x + (j - (n_p - 1) / 2) * width, means, width,
                   yerr=[lo, hi], capsize=2, alpha=0.9, label=pol,
                   **_bar_kwargs(pol))
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in scenarios])
        ax.set_xlabel('Congestion regime')
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
        if FIGURE_STYLE != 'paper':
            ax.grid(True, axis='y', alpha=0.3)
    axes[0][1].legend(frameon=False, loc='center left',
                      bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    return savefig(fig, 'G_regimes.png')


def plot_cross_city(df, metric='avg_delay_min'):
    """Cross-city comparison: grouped bars (cities on x, color+hatch per policy)."""
    _set_style()
    metrics = [metric, 'on_time_rate', 'total_cost']
    ylabels = ['Avg. fulfillment time (min)', 'On-time rate', 'Total cost (yuan)']
    cities = list(df['city'].unique())
    policies = list(df['policy'].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.5))
    for ax, col, ylab in zip(axes, metrics, ylabels):
        x = np.arange(len(cities))
        n_p = len(policies)
        width = 0.8 / n_p
        for j, pol in enumerate(policies):
            sub = df[df['policy'] == pol].set_index('city').reindex(cities)
            vals = sub[f'{col}_mean'].values
            err = sub[f'{col}_std'].values if f'{col}_std' in sub.columns else None
            ax.bar(x + (j - (n_p - 1) / 2) * width, vals, width, yerr=err,
                   capsize=2, label=pol, alpha=0.9, **_bar_kwargs(pol))
        ax.set_xticks(x)
        ax.set_xticklabels(cities)
        ax.set_xlabel('City')
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
    axes[-1].legend(frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    return savefig(fig, 'G_cross_city.png')


# ---------------------------------------------------------------------------
# Restored summary figures from the original submission (kept under their
# original new_G_* filenames so the manuscript references stay valid).
# ---------------------------------------------------------------------------

def plot_delay_vs_crowd_utilization(metrics_df, policies=None):
    """Scatter of crowd utilization vs. average fulfillment time per policy.

    Each policy has its own color + marker; labels are offset away from the
    data centroid with a white backing box so they never overlap markers.
    ``metrics_df`` is the per-policy mean table (one row per policy) with
    columns ``policy``, ``avg_delay_min`` and ``crowd_utilization``.
    """
    _set_style()
    df = metrics_df.copy()
    if policies is not None:
        keep = [p for p in policies if p in set(df['policy'])]
        df = df[df['policy'].isin(keep)]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pts = [(float(r['crowd_utilization']) * 100.0, float(r['avg_delay_min']),
            r['policy']) for _, r in df.iterrows()]
    centroid = (float(np.mean([p[0] for p in pts])),
                float(np.mean([p[1] for p in pts]))) if pts else None
    for x, y, name in pts:
        st = style_for(name)
        ax.scatter(x, y, s=280, alpha=0.9, color=st['color'], marker=st['marker'],
                   edgecolors='black', linewidths=1.0, zorder=5)
        _annotate_point(ax, x, y, name, centroid=centroid, fontsize=10)
    ax.set_xlabel('Crowd utilization (%)')
    ax.set_ylabel('Avg. fulfillment time (min)')
    ax.set_title('Crowd utilization vs average fulfillment time')
    ax.margins(x=0.2, y=0.2)
    if FIGURE_STYLE != 'paper':
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return savefig(fig, 'new_G_delay_vs_crowd_utilization.png')


def plot_hourly_crowd_availability(params, threshold=0.8):
    """Hourly crowd availability curve with the high-availability window.

    The shaded window spans the hours whose availability probability is at
    least ``threshold`` (contiguous span from the first to the last such hour).
    """
    _set_style()
    avail = np.asarray(params['crowd_avail'], dtype=float) * 100.0
    hours = np.arange(len(avail))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    high = hours[avail >= threshold * 100.0]
    if len(high) > 0:
        ax.axvspan(high.min() - 0.5, high.max() + 0.5, color='green',
                   alpha=0.08, label='High availability window')
    ax.plot(hours, avail, marker='o', color='#2c3e50', linewidth=2,
            markersize=5)
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Crowd availability (%)')
    ax.set_title('Hourly crowd availability calibrated from LaDe Shanghai data')
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0, max(105.0, float(avail.max()) + 5.0))
    # Legend below the curve's early-morning values to avoid the shaded window.
    ax.legend(frameon=False, loc='lower right')
    if FIGURE_STYLE != 'paper':
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return savefig(fig, 'new_G_hourly_crowd_availability.png')


def plot_relative_improvement_bar(metrics_df, proposed='OnlineMatching',
                                  baselines=('DedicatedOnly', 'Hybrid',
                                             'Queueing')):
    """Grouped bars of the proposed policy's improvement over each baseline.

    Metric groups are encoded by color + hatch.  The legend sits below the
    axes so it never covers bars.  Improvements are expressed in percent of
    the baseline value (on-time rate in percentage points).
    """
    _set_style()
    df = metrics_df.set_index('policy')
    if proposed not in df.index:
        return None
    completed_col = 'completed_total' if 'completed_total' in df.columns else 'completed'
    p = df.loc[proposed]

    def _reduction(col):
        return lambda b: (df.loc[b, col] - p[col]) / df.loc[b, col] * 100.0

    specs = [
        ('Avg. fulfillment time reduction', _reduction('avg_delay_min')),
        ('On-time rate increase (pp)', lambda b: (p['on_time_rate'] - df.loc[b, 'on_time_rate']) * 100.0),
        ('Max backlog reduction', _reduction('max_backlog')),
        ('Total cost reduction', _reduction('total_cost')),
        ('Completed orders increase', lambda b: (p[completed_col] - df.loc[b, completed_col]) / df.loc[b, completed_col] * 100.0),
    ]
    rows = []
    for b in baselines:
        if b not in df.index:
            continue
        for label, fn in specs:
            rows.append({'baseline': b, 'metric': label, 'value': float(fn(b))})
    if not rows:
        return None
    imp = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    metric_labels = [s[0] for s in specs]
    base_labels = list(dict.fromkeys(imp['baseline']))
    n_b, n_m = len(base_labels), len(metric_labels)
    width = 0.8 / n_m
    x = np.arange(n_b)
    metric_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    metric_hatches = ['', '//', '..', 'xx', '\\\\']
    for j, mlabel in enumerate(metric_labels):
        vals = [float(imp[(imp['baseline'] == b) & (imp['metric'] == mlabel)]['value'].iloc[0])
                for b in base_labels]
        ax.bar(x + (j - (n_m - 1) / 2) * width, vals, width, label=mlabel,
               alpha=0.9, color=metric_colors[j % len(metric_colors)],
               hatch=metric_hatches[j % len(metric_hatches)],
               edgecolor='black', linewidth=0.4)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(base_labels)
    ax.set_xlabel('Baseline policy')
    ax.set_ylabel('Improvement (%)')
    ax.set_title(f'{proposed} improvement over baseline policies')
    # Legend below the axes: grouped bars can reach the top of the panel.
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=9)
    if FIGURE_STYLE != 'paper':
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return savefig(fig, 'new_G_relative_improvement_bar.png')
