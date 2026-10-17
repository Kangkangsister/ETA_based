"""Standalone paper figures regenerated from output/results.xlsx.

All compared policies are encoded by color + marker (line/scatter) or
color + hatch (bars) via the shared style registry in plots.py, so the
figures remain distinguishable in grayscale.  Labels are offset away from
the data centroid with a white backing box to avoid text/figure overlap.
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from plots import style_for, _line_kwargs, _bar_kwargs, _annotate_point  # noqa: E402

xls = pd.ExcelFile('output/results.xlsx')
metrics = pd.read_excel(xls, sheet_name='metrics')
sensitivity = pd.read_excel(xls, sheet_name='sensitivity')

# ---------- Figure 1: Cost decomposition ----------
fig, ax = plt.subplots(figsize=(10.5, 6))
policies = metrics['policy'].values
cost_components = ['ded_cost', 'crowd_cost', 'delay_cost', 'deadline_cost', 'travel_cost']
labels = ['Dedicated labor', 'Crowd labor', 'Delay cost', 'Deadline violation', 'Travel cost']
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#70AD47']
hatches = ['', '//', '..', 'xx', '\\\\']

bottom = np.zeros(len(policies))
for comp, label, color, hatch in zip(cost_components, labels, colors, hatches):
    vals = metrics[comp].values
    ax.bar(policies, vals, bottom=bottom, label=label, color=color, hatch=hatch,
           edgecolor='black', linewidth=0.4, width=0.6)
    bottom += vals

ax.set_ylabel('Cost (CNY)', fontsize=12)
ax.set_title('Platform Cost Decomposition by Policy', fontsize=14)
# Legend outside the axes: stacked bars fill the panel.
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
ax.set_ylim(0, bottom.max() * 1.15)
for i, total in enumerate(metrics['total_cost'].values):
    ax.text(i, total + 500, f'{total:.0f}', ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('output/figures/G_cost_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------- Figure 2: Sensitivity line plot ----------
fig, ax = plt.subplots(figsize=(10.5, 6))
policies = ['DedicatedOnly', 'Hybrid', 'Queueing', 'OnlineMatching']

for pol in policies:
    sub = sensitivity[sensitivity['policy'] == pol].sort_values('N_CROWD_MAX')
    xs = sub['N_CROWD_MAX'].values
    ys = sub['avg_delay_min_mean'].values
    stds = sub['avg_delay_min_std'].values
    kw = _line_kwargs(pol)
    ax.plot(xs, ys, label=pol, **kw)
    ax.fill_between(xs, ys - stds, ys + stds, color=kw['color'], alpha=0.12)

ax.set_xlabel('Maximum Crowd Fleet Size (N_CROWD_MAX)', fontsize=12)
ax.set_ylabel('Average Delay (min)', fontsize=12)
ax.set_title('Sensitivity of Average Delay to Crowd Fleet Size', fontsize=14)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('output/figures/G_sensitivity_line.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------- Figure 3: Delay vs Cost Pareto (re-draw with annotations) ----------
fig, ax = plt.subplots(figsize=(9, 6.5))
pts = [(row['avg_delay_min'], row['total_cost'], row['policy'])
       for _, row in metrics.iterrows()]
centroid = (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
for x, y, name in pts:
    st = style_for(name)
    ax.scatter(x, y, s=220, color=st['color'], marker=st['marker'],
               edgecolors='black', linewidths=0.8, zorder=5)
    _annotate_point(ax, x, y, name, centroid=centroid, fontsize=9)

ax.set_xlabel('Average Delay (min)', fontsize=12)
ax.set_ylabel('Total Cost (CNY)', fontsize=12)
ax.set_title('Delay-Cost Trade-off Across Policies', fontsize=14)
ax.margins(x=0.18, y=0.18)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('output/figures/G_delay_cost_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()

print('Figures saved.')
