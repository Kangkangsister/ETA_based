"""Cross-city validation experiment (revision).

For each LaDe city (sh, hz, cq, jl, yt) the full calibration protocol is
re-applied independently: parameter estimation, arrival-rate calibration,
data-driven fleet sizing, and fair baseline tuning (on a separate seed
range).  The core policies are then compared over multiple seeds, validated
against the city's empirical KPIs, and tested pairwise vs. the proposed
OnlineMatching policy.  This demonstrates that the findings generalize
beyond the default Shanghai setting.

Revision round 3 (external-validity fixes):

- The number of replications per city is raised from 10 to
  ``config.CROSS_CITY_SEEDS`` (30, the same protocol as the main
  comparison): with n=10 the smallest attainable Wilcoxon p-value is
  ~0.00195, so the earlier cross-city tests carried almost no statistical
  information.
- The physical parameters are no longer silently shared from Shanghai.
  LaDe ships detailed GPS trajectories for Shanghai only, so for the other
  cities (a) the doorstep service time is estimated from the city's OWN
  delivery data (scaled empirical fallback in ``derive_params``), (b) the
  shared speed estimate is adjusted by the documented per-city factor
  ``config.CITY_SPEED_FACTORS`` (marked as *assumed* in the per-city
  parameter table), and (c) the pre-delivery overhead is recalibrated per
  city from the city's empirical accept-to-delivery time (as before).  If a
  city-specific trajectory file is present it is used directly
  (``data.loader.load_trajectory_for_city``).
- Every city's calibrated parameters are exported (sheet
  ``cross_city_params``) with their source category, so the paper can state
  exactly which values are estimated, calibrated, or assumed per city.
"""

import time

import numpy as np
import pandas as pd

from base import (setup_logging, set_seed, save_excel, save_latex_table,
                  format_excel_best, empirical_kpis, calibrate_arrival_scale,
                  estimate_fleet_sizes)
from config import (SIMULATION_MINUTES, N_DEDICATED, N_CROWD_MAX, SURGE_HOURS,
                    SURGE_MULTIPLIER, APPLY_SURGE_MULTIPLIER, SEED,
                    FLEET_TARGET_UTIL, BACKLOG_THRESHOLD_GRID,
                    QUEUEING_WAIT_GRID, PICKUP_OVERHEAD_MEAN,
                    PICKUP_OVERHEAD_STD, CITY_SPEED_FACTORS,
                    CROSS_CITY_SEEDS, CROSS_CITY_TUNE_SEEDS)
from data.loader import (load_delivery, load_pickup,
                         load_trajectory_for_city, derive_params)
from simulator.core import SimulatorG
from policies import (DedicatedOnlyPolicy, HybridThresholdPolicy,
                      QueueingThresholdPolicy, OnlineMatchingPolicy)
from experiments.baseline_tuning import tune_baselines, best_thresholds
from experiments.replication import run_replications, pairwise_test_table
from experiments.validation import validate_against_empirical
from plots import plot_cross_city

CITIES = ['sh', 'hz', 'cq', 'jl', 'yt']
_PAIRWISE_METRICS = ['avg_delay_min', 'on_time_rate', 'total_cost']


def _make_policies(hybrid_threshold, queueing_target):
    return {
        'DedicatedOnly': DedicatedOnlyPolicy(),
        'Hybrid': HybridThresholdPolicy(backlog_threshold=int(hybrid_threshold)),
        'Queueing': QueueingThresholdPolicy(target_wait_min=float(queueing_target)),
        'OnlineMatching': OnlineMatchingPolicy(),
    }


def main():
    t0 = time.time()
    log = setup_logging()
    set_seed(SEED)
    log.info('Cross-city validation over cities: %s (%d seeds, %d tuning '
             'seeds per candidate)', CITIES, CROSS_CITY_SEEDS,
             CROSS_CITY_TUNE_SEEDS)

    summary_frames, validation_frames, tuning_rows, pairwise_frames = [], [], [], []
    param_rows = []
    for i, city in enumerate(CITIES):
        log.info('=== City %s: calibration ===', city)
        try:
            delivery = load_delivery(city=city)
            pickup = load_pickup(city=city)
            # City-specific trajectory if present; otherwise the shared
            # Shanghai sample, tagged so derive_params marks the physical
            # parameters as assumed (and uses the city's own data for the
            # doorstep service time).
            traj, traj_city = load_trajectory_for_city(city)
            params = derive_params(
                delivery, pickup, traj, traj_city=traj_city,
                speed_factor=CITY_SPEED_FACTORS.get(city, 1.0))
        except Exception as e:
            log.warning('Skipping %s: %s', city, e)
            continue

        emp = empirical_kpis(city=city)
        calibrate_arrival_scale(params, emp, SIMULATION_MINUTES)

        # Per-city pickup/acceptance overhead calibration: the pre-delivery
        # overhead is recalibrated per city from the city's empirical
        # accept-to-delivery time (same decomposition as the holdout-overhead
        # evidence: overhead = accept-to-delivery - travel - doorstep
        # service), using the city-specific speed.  Without this, cities with
        # longer empirical delivery times (hz/cq/jl/yt) are systematically
        # underestimated.
        emp_avg = emp.get('avg_delivery_time_min')
        trip_km = params.get('mean_trip_km')
        if emp_avg and trip_km is not None and np.isfinite(trip_km):
            trip_min = trip_km / max(1.0, params['speed_mean']) * 60.0
            implied = float(np.clip(emp_avg - trip_min - params['service_mean'],
                                    5.0, 150.0))
            cv = PICKUP_OVERHEAD_STD / PICKUP_OVERHEAD_MEAN  # keep the coefficient of variation
            params['pickup_overhead_mean'] = implied
            params['pickup_overhead_std'] = implied * cv
            params['pickup_overhead_log_mean'] = float(
                np.log(implied ** 2 / np.sqrt(implied ** 2 + (implied * cv) ** 2)))
            params['pickup_overhead_log_std'] = float(np.sqrt(np.log(1 + cv ** 2)))
            log.info('%s: calibrated pickup overhead %.1f min '
                     '(empirical accept-to-delivery %.1f, travel %.1f, service %.1f)',
                     city, implied, emp_avg, trip_min, params['service_mean'])

        n_dedicated, n_crowd_max = estimate_fleet_sizes(
            params, N_DEDICATED, N_CROWD_MAX,
            simulation_minutes=SIMULATION_MINUTES, target_util=FLEET_TARGET_UTIL)
        params['n_dedicated'] = n_dedicated
        params['n_crowd_max'] = n_crowd_max
        log.info('%s: fleet ded=%d crowd_max=%d, arrival scale=%.3f',
                 city, n_dedicated, n_crowd_max,
                 params.get('arrival_rate_scale', 1.0))

        # Per-city parameter provenance (estimated / calibrated / assumed).
        param_rows += [
            {'city': city, 'param': 'speed_mean',
             'value': params['speed_mean'],
             'source': params.get('speed_source', 'estimated')},
            {'city': city, 'param': 'service_mean',
             'value': params['service_mean'],
             'source': params.get('service_source', 'estimated')},
            {'city': city, 'param': 'pickup_overhead_mean',
             'value': params['pickup_overhead_mean'],
             'source': 'calibrated (city accept-to-delivery decomposition)'},
            {'city': city, 'param': 'arrival_rate_scale',
             'value': params['arrival_rate_scale'],
             'source': 'calibrated (city daily order volume)'},
            {'city': city, 'param': 'n_dedicated', 'value': n_dedicated,
             'source': 'calibrated (peak-hour target utilization)'},
            {'city': city, 'param': 'n_crowd_max', 'value': n_crowd_max,
             'source': 'calibrated (peak-hour target utilization)'},
            {'city': city, 'param': 'mean_trip_km',
             'value': params.get('mean_trip_km'),
             'source': 'estimated (city station->dropoff Haversine)'},
        ]

        sim_common = dict(duration=SIMULATION_MINUTES, n_dedicated=n_dedicated,
                          n_crowd_max=n_crowd_max, surge_hours=SURGE_HOURS,
                          surge_multiplier=SURGE_MULTIPLIER,
                          apply_surge_multiplier=APPLY_SURGE_MULTIPLIER,
                          arrival_rate_scale=params['arrival_rate_scale'])

        # Fair per-city baseline tuning on a separate seed range.  Each city
        # is tuned independently, so per-city thresholds may legitimately
        # differ from the Shanghai main-experiment values; both are reported
        # (sheet cross_city_tuning) so the tuning protocol is uniform and
        # transparent across cities.
        _, tune_summary, tuned = tune_baselines(
            params, SimulatorG, BACKLOG_THRESHOLD_GRID, QUEUEING_WAIT_GRID,
            n_seeds=CROSS_CITY_TUNE_SEEDS, base_seed=15000 + i * 100,
            **sim_common)
        tuned_cost = best_thresholds(tune_summary, criterion='cost')
        tuning_rows.append({
            'city': city,
            'hybrid_backlog_threshold': tuned.get('Hybrid'),
            'queueing_target_wait_min': tuned.get('Queueing'),
            'hybrid_backlog_threshold_cost_optimal': tuned_cost.get('Hybrid'),
            'queueing_target_wait_min_cost_optimal': tuned_cost.get('Queueing'),
            'pickup_overhead_mean': params.get('pickup_overhead_mean'),
            'speed_mean': params.get('speed_mean'),
            'service_mean': params.get('service_mean'),
            'n_dedicated': n_dedicated,
            'n_crowd_max': n_crowd_max,
        })
        policies = _make_policies(tuned.get('Hybrid', 15),
                                  tuned.get('Queueing', 10.0))

        raw, summary = run_replications(
            policies, params, SimulatorG, n=CROSS_CITY_SEEDS,
            base_seed=11000 + i * 100, **sim_common)
        summary['city'] = city
        summary_frames.append(summary)

        val = validate_against_empirical(
            summary, city=city, duration_min=SIMULATION_MINUTES,
            baseline_policy='OnlineMatching')
        val['city'] = city
        validation_frames.append(val)

        pw = pairwise_test_table(raw, _PAIRWISE_METRICS)
        if not pw.empty:
            pw = pw[(pw['policy_a'] == 'OnlineMatching')
                    | (pw['policy_b'] == 'OnlineMatching')]
            pw['city'] = city
            pairwise_frames.append(pw)

    if not summary_frames:
        log.warning('No city could be processed; aborting cross-city export')
        return
    summary_all = pd.concat(summary_frames, ignore_index=True)

    # Compact per city x policy table (mean +/- std for key metrics).
    key_metrics = ['avg_delay_min', 'on_time_rate', 'max_backlog', 'total_cost']
    sub = summary_all[summary_all['metric'].isin(key_metrics)]
    compact = sub.pivot_table(index=['city', 'policy'], columns='metric',
                              values='mean')
    compact_std = sub.pivot_table(index=['city', 'policy'], columns='metric',
                                  values='std')
    compact.columns = [f'{c}_mean' for c in compact.columns]
    compact_std.columns = [f'{c}_std' for c in compact_std.columns]
    compact = pd.concat([compact, compact_std], axis=1).reset_index()

    sheets = {
        'cross_city': compact,
        'cross_city_full': summary_all,
        'cross_city_validation': pd.concat(validation_frames, ignore_index=True),
        'cross_city_tuning': pd.DataFrame(tuning_rows),
        'cross_city_params': pd.DataFrame(param_rows),
    }
    if pairwise_frames:
        sheets['cross_city_pairwise'] = pd.concat(pairwise_frames,
                                                  ignore_index=True)
    path = save_excel(sheets, filename='cross_city.xlsx')
    format_excel_best(path)
    save_latex_table(compact, 'cross_city.tex',
                     caption=f'Cross-city validation (mean over '
                             f'{CROSS_CITY_SEEDS} seeds per city)',
                     label='tab:cross_city')
    save_latex_table(pd.DataFrame(param_rows), 'cross_city_params.tex',
                     caption='Per-city calibrated parameters and their source '
                             'categories',
                     label='tab:cross_city_params')
    plot_cross_city(compact)
    log.info('Cross-city results saved to %s (%.1fs)', path, time.time() - t0)


if __name__ == '__main__':
    main()
