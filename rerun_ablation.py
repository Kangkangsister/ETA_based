"""Standalone driver: re-run the ablation study (revision round 2) only.

Reuses the exact calibration path of main.py (data loading, arrival-rate
calibration, fleet sizing) and then runs ``experiments.ablation.run_ablation``
under the five regimes (base / stress / extreme / contention /
hetero_deadline) with common random numbers and the targeted metrics.

Usage:
    python rerun_ablation.py            # full ABLATION_SEEDS replications
    python rerun_ablation.py 10         # override the number of seeds
"""

import logging
import sys
import time

import pandas as pd

from base import setup_logging, set_seed, empirical_kpis, calibrate_arrival_scale, \
    estimate_fleet_sizes
from config import (SEED, CITY, SIMULATION_MINUTES, N_DEDICATED, N_CROWD_MAX,
                    FLEET_TARGET_UTIL, SURGE_HOURS, SURGE_MULTIPLIER,
                    APPLY_SURGE_MULTIPLIER, ABLATION_SEEDS)
from data.loader import load_delivery, load_pickup, load_trajectory, derive_params
from simulator.core import SimulatorG
from experiments.ablation import run_ablation

log = logging.getLogger(__name__.split('.')[0])


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else ABLATION_SEEDS
    t0 = time.time()
    setup_logging()
    set_seed(SEED)

    log.info('Loading data for city=%s', CITY)
    delivery = load_delivery()
    pickup = load_pickup()
    traj = load_trajectory()
    params = derive_params(delivery, pickup, traj)
    emp = empirical_kpis(city=CITY)
    calibrate_arrival_scale(params, emp, SIMULATION_MINUTES)
    n_dedicated, n_crowd_max = estimate_fleet_sizes(
        params, N_DEDICATED, N_CROWD_MAX,
        simulation_minutes=SIMULATION_MINUTES, target_util=FLEET_TARGET_UTIL)
    params['n_dedicated'] = n_dedicated
    params['n_crowd_max'] = n_crowd_max
    log.info('Fleet: dedicated=%d, crowd_max=%d; arrival scale=%.3f',
             n_dedicated, n_crowd_max, params['arrival_rate_scale'])

    sim_common = dict(duration=SIMULATION_MINUTES, n_dedicated=n_dedicated,
                      n_crowd_max=n_crowd_max, surge_hours=SURGE_HOURS,
                      surge_multiplier=SURGE_MULTIPLIER,
                      apply_surge_multiplier=APPLY_SURGE_MULTIPLIER,
                      arrival_rate_scale=params['arrival_rate_scale'],
                      road_network=None)

    abl_raw, abl_summary, abl_pairwise, abl_effects = run_ablation(
        params, SimulatorG, n_seeds=n_seeds, base_seed=6000, **sim_common)

    out = 'output/ablation_r2.xlsx'
    with pd.ExcelWriter(out) as xw:
        abl_raw.to_excel(xw, sheet_name='ablation_raw', index=False)
        abl_summary.to_excel(xw, sheet_name='ablation', index=False)
        abl_pairwise.to_excel(xw, sheet_name='ablation_pairwise', index=False)
        abl_effects.to_excel(xw, sheet_name='ablation_effects', index=False)
    log.info('Saved %s (%.1fs total)', out, time.time() - t0)

    # Console digest: mean metrics per variant/scenario + significance vs Full.
    key = abl_summary[abl_summary['metric'].isin(
        ['avg_delay_min', 'on_time_rate_promised', 'on_time_rate_tight',
         'deadline_violations_promised', 'p95_delay_min', 'workload_gini',
         'total_cost'])]
    piv = key.pivot_table(index='policy', columns=['scenario', 'metric'],
                          values='mean')
    pd.set_option('display.width', 250)
    print('\n=== Ablation means (revision round 2) ===')
    print(piv.round(3).to_string())
    print('\n=== Wilcoxon vs Full (Holm-corrected) ===')
    cols = ['scenario', 'metric', 'policy_a', 'policy_b', 'mean_a', 'mean_b',
            'p_value', 'significance']
    print(abl_pairwise[cols].to_string(index=False))
    print('\n=== Component contributions (variant - Full, paired) ===')
    ecols = ['scenario', 'variant', 'metric', 'mean_diff', 'rel_diff_pct',
             'wilcoxon_p_holm', 'rank_biserial', 'significance']
    key_eff = abl_effects[abl_effects['metric'].isin(
        ['avg_delay_min', 'on_time_rate_promised', 'workload_gini'])]
    print(key_eff[ecols].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
