"""Main experiment script for the crowdsourced delivery simulation.

Experimental protocol (revised version):

1. Load LaDe data and calibrate parameters (arrival rates, speed, service
   times, crowd availability, fleet sizes, arrival-rate scale).
2. Tune the baseline thresholds (Hybrid backlog B, Queueing target wait) on
   an independent seed range, so the baselines are fairly calibrated
   (reviewer comment 4).
3. Single-seed comparison of all strategies (diagnostics + backlog traces).
4. N_REPLICATIONS (=30) independent replications per strategy with paired
   Wilcoxon tests and Holm-Bonferroni correction (reviewer comment 1).
5. Sensitivity analysis over the maximum crowd fleet size.
6. Extended revision experiments: ablation of the OnlineMatching marginal-cost
   components under the base / stress / extreme / contention /
   hetero-deadline regimes (comment 3), a
   comparison of the core policies across the same regimes, road-circuity
   robustness (comment 5), dedicated / crowd capacity-mix scenarios
   (comment 8), and holdout-period validation with a data-based check of the
   pre-delivery overhead (comments 6 and 7).
7. Validation against empirical KPIs, fluid lower bound, and export of Excel
   sheets, LaTeX tables, figures and reproducibility metadata.
"""

import argparse
import time

# NOTE: on Windows the OR-Tools native DLLs fail to load (WinError 127) once
# the numpy/pandas DLLs are already loaded.  Importing ortools first makes the
# VRP offline baseline available on Windows as well; on Linux (the server
# environment) ortools imports cleanly regardless.  If it is unavailable the
# policy disables itself gracefully and the exclusion is recorded in
# metadata.json.
try:
    import ortools  # noqa: F401
    _ORTOOLS_AVAILABLE = True
except Exception:
    _ORTOOLS_AVAILABLE = False

import numpy as np
import pandas as pd

from base import (setup_logging, set_seed, configure_plotting, save_excel,
                  save_latex_table, save_metadata, format_excel_best,
                  empirical_kpis, calibrate_arrival_scale,
                  estimate_fleet_sizes, statistical_procedure)
from config import (SIMULATION_MINUTES, WARMUP_MINUTES, N_DEDICATED, N_CROWD_MAX,
                    DEDICATED_CAPACITY, CROWD_CAPACITY, FLEET_TARGET_UTIL,
                    SURGE_HOURS, SURGE_MULTIPLIER, APPLY_SURGE_MULTIPLIER,
                    ARRIVAL_RATE_SCALE, SEED, CITY,
                    N_REPLICATIONS, USE_LATEX, USE_ROAD_NETWORK,
                    ROAD_NETWORK_MAX_ROWS,
                    DEADLINE_MIN, DEADLINE_STD, DEADLINE_MIN_CLIP,
                    ON_TIME_THRESHOLD_MIN, DEADLINE_BUFFER_MIN,
                    PICKUP_OVERHEAD_MEAN, PICKUP_OVERHEAD_STD,
                    COST_DELAY_PER_MIN, COST_DEADLINE_VIOLATION, COST_KM,
                    COST_DEDICATED_PER_HOUR, COST_CROWD_PER_HOUR,
                    DUAL_LR, DUAL_PRICE_STEP, DUAL_PRICE_DECAY,
                    DUAL_PRICE_STEP_DECAY, DUAL_PRICE_INACTIVE_DECAY,
                    BACKLOG_THRESHOLD, CIRCUITY_FACTOR,
                    CORE_POLICIES, RUN_EXTENDED_EXPERIMENTS, RUN_CROSS_CITY,
                    SENSITIVITY_SEEDS, BACKLOG_THRESHOLD_GRID,
                    QUEUEING_WAIT_GRID, BASELINE_TUNING_SEEDS,
                    ABLATION_SEEDS, CIRCUITY_GRID, CIRCUITY_SEEDS,
                    CAPACITY_MIX_DED_SHARES, CAPACITY_MIX_SEEDS,
                    CAPACITY_MIX_MODE,
                    HOLDOUT_TRAIN_FRACTION, HOLDOUT_SEEDS, HOLDOUT_POLICIES,
                    REGIME_SEEDS, RUN_EXPENSIVE_BASELINES,
                    EXPENSIVE_BASELINE_SEEDS, EXPENSIVE_BASELINE_BASE_SEED,
                    ROLLOUT_HORIZON, ROLLOUT_N_SAMPLES,
                    ROLLOUT_ENABLE_THRESHOLD,
                    RUN_COST_TUNED_CHECK, COST_TUNED_SEEDS,
                    COST_TUNED_BASE_SEED)
from data.loader import (load_delivery, load_pickup,
                         load_trajectory_for_city, derive_params)
from data.network import RoadNetwork
from simulator.core import SimulatorG
from policies import (DedicatedOnlyPolicy, CrowdFirstPolicy,
                      HybridThresholdPolicy,
                      NearestPolicy, RandomPolicy, FIFOBatchPolicy,
                      QueueingThresholdPolicy, OnlineMatchingPolicy,
                      ORToolsVRPPolicy, RolloutPolicy)
from experiments.replication import run_replications, pairwise_test_table
from experiments.sensitivity import sensitivity_grid
from experiments.validation import validate_against_empirical, theoretical_bounds
from experiments.baseline_tuning import tune_baselines, best_thresholds
from experiments.ablation import run_ablation
from experiments.regimes import regime_comparison
from experiments.circuity import circuity_robustness
from experiments.capacity_mix import capacity_mix_experiment
from experiments.holdout import holdout_validation
from plots import (plot_backlog, plot_metrics_bar, plot_utilization,
                   plot_crowd_availability, plot_pareto_frontier,
                   plot_cdf_delays, plot_sensitivity_heatmap,
                   plot_sensitivity_lines,
                   plot_ablation, plot_baseline_tuning, plot_circuity,
                   plot_capacity_mix, plot_regimes,
                   plot_delay_vs_crowd_utilization,
                   plot_hourly_crowd_availability,
                   plot_relative_improvement_bar,
                   configure_plotting)


def _make_policies(hybrid_threshold=BACKLOG_THRESHOLD, queueing_target=10.0):
    """Construct the policy registry, optionally with tuned baseline thresholds."""
    return {
        'DedicatedOnly': DedicatedOnlyPolicy(),
        'CrowdFirst': CrowdFirstPolicy(),
        'Hybrid': HybridThresholdPolicy(backlog_threshold=int(hybrid_threshold)),
        'Nearest': NearestPolicy(),
        'Random': RandomPolicy(),
        # FIFOBatch is kept importable but excluded by default because, with
        # immediate assignment each minute, it collapses to Nearest.
        'FIFOBatch': FIFOBatchPolicy(batch_size=5),
        'Queueing': QueueingThresholdPolicy(target_wait_min=float(queueing_target)),
        'OnlineMatching': OnlineMatchingPolicy(),
        # OR-Tools is disabled on Windows by default (DLL issues).  Include it
        # only when the solver is actually available to avoid a silent Nearest
        # fallback in the comparison.
        'ORToolsVRP': ORToolsVRPPolicy(),
        'Rollout': RolloutPolicy(),
    }


def main(fast=False, selected_policies=None, n_replications=None):
    t0 = time.time()
    log = setup_logging()
    set_seed(SEED)
    configure_plotting(use_latex=USE_LATEX)
    log.info('Project G: Crowdsourced delivery platform simulation')
    log.info('Selected plan: A (full conference/journal-grade upgrade)')
    if fast:
        log.info('FAST MODE: reduced policies, seeds and grids')

    # ------------------------------------------------------------------
    # Data loading and calibration
    # ------------------------------------------------------------------
    log.info('Loading data for city=%s', CITY)
    delivery = load_delivery()
    pickup = load_pickup()
    traj, traj_city = load_trajectory_for_city(CITY)
    params = derive_params(delivery, pickup, traj, traj_city=traj_city)
    # Calibrate the global arrival-rate scale so that the expected number of
    # orders generated during the simulation horizon matches the empirical
    # daily order count scaled to that horizon.  We use the full delivery file
    # count (via empirical_kpis) rather than the sampled data so that validation
    # is internally consistent.
    emp = empirical_kpis(city=CITY)
    calibrate_arrival_scale(params, emp, SIMULATION_MINUTES)
    log.info('Arrival-rate scale: %.3f (empirical daily orders: %.1f)',
             params['arrival_rate_scale'],
             emp.get('total_orders', 0) / max(1, emp.get('n_days', 1)))
    log.info('Parameters derived')
    log.info('Stations: %s', params['stations'])
    log.info('Speed: %.2f +/- %.2f km/h', params['speed_mean'], params['speed_std'])
    log.info('Service: %.2f +/- %.2f min', params['service_mean'], params['service_std'])
    n_dedicated, n_crowd_max = estimate_fleet_sizes(
        params, N_DEDICATED, N_CROWD_MAX,
        simulation_minutes=SIMULATION_MINUTES, target_util=FLEET_TARGET_UTIL)
    # Make actual fleet sizes available to policies that need them (e.g.
    # QueueingThresholdPolicy uses the true dedicated fleet size).
    params['n_dedicated'] = n_dedicated
    params['n_crowd_max'] = n_crowd_max
    log.info('Fleet size estimates: dedicated=%d, crowd_max=%d (observed=%d couriers)',
             n_dedicated, n_crowd_max, params.get('n_couriers_observed', 0))

    # Optional road network (lightweight Shanghai subset)
    road_network = None
    if USE_ROAD_NETWORK:
        try:
            road_network = RoadNetwork(
                city_bounds=(13370000, 13400000, 3520000, 3560000),
                max_rows=10000 if fast else ROAD_NETWORK_MAX_ROWS).build()
            log.info('Road network built: %d nodes', len(road_network.nodes))
        except Exception as e:
            log.warning('Road network disabled: %s', e)

    # Common simulator keyword arguments shared by all experiments.
    sim_common = dict(duration=SIMULATION_MINUTES, n_dedicated=n_dedicated,
                      n_crowd_max=n_crowd_max, surge_hours=SURGE_HOURS,
                      surge_multiplier=SURGE_MULTIPLIER,
                      apply_surge_multiplier=APPLY_SURGE_MULTIPLIER,
                      arrival_rate_scale=params['arrival_rate_scale'],
                      road_network=road_network)

    # ------------------------------------------------------------------
    # Baseline tuning (reviewer comment 4): select the Hybrid backlog
    # threshold and the Queueing target waiting time on an independent seed
    # range before the main comparison.  The tuning summary marks both the
    # delay-optimal and the cost-optimal values plus the Pareto-optimal
    # candidates, so the selection criterion is transparent (revision
    # analysis: a delay-only criterion could be read as favouring the
    # proposed method).
    # ------------------------------------------------------------------
    tuning_raw = tuning_summary = None
    tuned = {}
    tuned_cost = {}
    if RUN_EXTENDED_EXPERIMENTS:
        tune_seeds = 3 if fast else BASELINE_TUNING_SEEDS
        log.info('Tuning baseline thresholds (%d seeds per candidate)', tune_seeds)
        tuning_raw, tuning_summary, tuned = tune_baselines(
            params, SimulatorG, BACKLOG_THRESHOLD_GRID, QUEUEING_WAIT_GRID,
            n_seeds=tune_seeds, base_seed=5000, **sim_common)
        tuned_cost = best_thresholds(tuning_summary, criterion='cost')
    hybrid_threshold = int(tuned.get('Hybrid', BACKLOG_THRESHOLD))
    queueing_target = float(tuned.get('Queueing', 10.0))
    log.info('Baseline thresholds used: Hybrid B=%d, Queueing target=%.1f min '
             '(delay-optimal; cost-optimal: B=%s, w=%s)',
             hybrid_threshold, queueing_target,
             tuned_cost.get('Hybrid'), tuned_cost.get('Queueing'))

    # ------------------------------------------------------------------
    # Policies
    # ------------------------------------------------------------------
    all_policies = _make_policies(hybrid_threshold, queueing_target)
    # Filter out policies that are unavailable or redundant on this platform.
    ortools_ok = getattr(all_policies['ORToolsVRP'], '_has_ortools', False)
    log.info('OR-Tools available: %s (import preflight: %s)',
             ortools_ok, _ORTOOLS_AVAILABLE)
    def _policy_usable(name, pol):
        if name == 'ORToolsVRP':
            return getattr(pol, '_has_ortools', False)
        return True
    all_policies = {k: v for k, v in all_policies.items() if _policy_usable(k, v)}
    # Policies excluded from the default comparison because they duplicate
    # another policy or are too expensive for the main batch.  Rollout runs in
    # the dedicated expensive-baselines block below (own seed range, paired
    # against the core policies) instead of being silently dropped.
    default_exclude = {'FIFOBatch', 'Rollout'}
    if fast and selected_policies:
        policies = {k: all_policies[k] for k in selected_policies if k in all_policies}
    elif fast:
        policies = {k: all_policies[k] for k in CORE_POLICIES if k in all_policies}
    elif selected_policies:
        policies = {k: all_policies[k] for k in selected_policies if k in all_policies}
    else:
        policies = {k: v for k, v in all_policies.items() if k not in default_exclude}
    # The four strategies reported in the paper; extended experiments run on
    # this core set so the protocol matches the main comparison.
    core_policies = {k: policies[k] for k in CORE_POLICIES if k in policies}

    # ------------------------------------------------------------------
    # Single-seed run for quick comparison and backlog traces
    # ------------------------------------------------------------------
    log.info('Running single-seed comparison (%d policies)', len(policies))
    results = {}
    backlogs = []
    backlog_labels = []
    for name, pol in policies.items():
        sim = SimulatorG(params, pol, seed=SEED, **sim_common)
        metrics = sim.run()
        results[name] = metrics
        backlogs.append(sim.metrics.backlog_log)
        backlog_labels.append(name)
        log.info('%s: avg_delay=%.2f min, on_time=%.2f%%, max_backlog=%d, '
                 'crowd_served=%d, total_cost=%.1f',
                 name, metrics['avg_delay_min'], metrics['on_time_rate'] * 100,
                 metrics['max_backlog'], metrics['crowd_served'],
                 metrics['total_cost'])

    df_metrics_single_seed = pd.DataFrame(results).T.reset_index().rename(
        columns={'index': 'policy'})

    # ------------------------------------------------------------------
    # Replications for statistical robustness (reviewer comment 1: N >= 30)
    # ------------------------------------------------------------------
    reps = n_replications if n_replications is not None else N_REPLICATIONS
    if fast and n_replications is None:
        reps = 3
    log.info('Running %d replications per policy', reps)
    raw_df, summary_df = run_replications(
        policies, params, SimulatorG, n=reps, base_seed=SEED, **sim_common)

    pairwise = pairwise_test_table(raw_df, ['avg_delay_min', 'on_time_rate',
                                            'max_backlog', 'total_cost'])

    # ------------------------------------------------------------------
    # Sensitivity analysis: N_CROWD_MAX
    # ------------------------------------------------------------------
    log.info('Running sensitivity analysis on N_CROWD_MAX')
    # The sweep must cover the calibrated operating point (n_crowd_max) so the
    # reader can verify where the system actually operates, and extend beyond
    # it to show the saturation plateau.
    sens_values = [0, 20, 40, 60, 80] if fast else sorted(
        {0, 20, 40, 60, 80, 100, n_crowd_max, int(round(n_crowd_max * 1.3))})
    sens_n_seeds = 3 if fast else SENSITIVITY_SEEDS
    sens_df = sensitivity_grid(
        core_policies, params, SimulatorG, 'N_CROWD_MAX', sens_values,
        n_seeds=sens_n_seeds, base_seed=2000, **sim_common)

    # ------------------------------------------------------------------
    # Extended revision experiments
    # ------------------------------------------------------------------
    abl_raw = abl_summary = abl_pairwise = abl_effects = None
    circ_df = mix_raw = mix_summary = None
    reg_raw = reg_summary = reg_pairwise = None
    holdout = None
    cost_tuned_raw = cost_tuned_summary = cost_tuned_pairwise = None
    expensive_raw = expensive_summary = expensive_pairwise = None
    if RUN_EXTENDED_EXPERIMENTS:
        # Comment 3: ablation of the OnlineMatching marginal-cost components
        # under the base, stress and extreme regimes.
        abl_seeds = 3 if fast else ABLATION_SEEDS
        abl_raw, abl_summary, abl_pairwise, abl_effects = run_ablation(
            params, SimulatorG, n_seeds=abl_seeds, base_seed=6000, **sim_common)

        # Regime comparison: the core policies re-run under the same
        # base / stress / extreme regimes, showing how the relative ranking
        # and the OnlineMatching advantage evolve with congestion.
        reg_seeds = 3 if fast else REGIME_SEEDS
        reg_raw, reg_summary, reg_pairwise = regime_comparison(
            core_policies, params, SimulatorG, n_seeds=reg_seeds,
            base_seed=10000, **sim_common)

        # Comment 5: robustness to the Haversine distance approximation via
        # road-circuity multipliers.
        cir_seeds = 3 if fast else CIRCUITY_SEEDS
        circ_df = circuity_robustness(
            core_policies, params, SimulatorG, CIRCUITY_GRID,
            n_seeds=cir_seeds, base_seed=7000, **sim_common)

        # Comment 8: dedicated / crowd capacity composition at constant total
        # CARRYING CAPACITY (iso-capacity; see CAPACITY_MIX_MODE).  Anchoring
        # on the calibrated fleet's capacity keeps the comparison fair:
        # iso-headcount scenarios would confound the composition effect with
        # a several-fold difference in total carrying capacity.
        mix_seeds = 3 if fast else CAPACITY_MIX_SEEDS
        total_fleet = n_dedicated + n_crowd_max
        mix_raw, mix_summary = capacity_mix_experiment(
            core_policies, params, SimulatorG, CAPACITY_MIX_DED_SHARES,
            total_fleet, n_seeds=mix_seeds, base_seed=8000,
            mode=CAPACITY_MIX_MODE,
            ref_composition=(n_dedicated, n_crowd_max), **sim_common)

        # Comments 6/7: holdout-period validation (calibrate on early days,
        # validate on late days) + pre-delivery-overhead evidence.
        hold_seeds = 3 if fast else HOLDOUT_SEEDS
        hold_policies = {}
        for name in HOLDOUT_POLICIES:
            if name in all_policies:
                hold_policies[name] = _make_policies(
                    hybrid_threshold, queueing_target)[name]
        if hold_policies:
            holdout = holdout_validation(
                hold_policies, SimulatorG, city=CITY,
                train_fraction=HOLDOUT_TRAIN_FRACTION, n_seeds=hold_seeds,
                base_seed=9000, duration=SIMULATION_MINUTES)

        # Baseline-tuning robustness check (revision analysis): re-run the
        # core comparison with the COST-optimal baseline thresholds instead of
        # the delay-optimal ones, on its own seed range.  This shows that the
        # OnlineMatching advantage does not hinge on the tuning criterion.
        if RUN_COST_TUNED_CHECK and tuned_cost:
            ct_seeds = 3 if fast else COST_TUNED_SEEDS
            ct_hybrid = int(tuned_cost.get('Hybrid', hybrid_threshold))
            ct_queue = float(tuned_cost.get('Queueing', queueing_target))
            ct_policies = {
                'Hybrid(cost-tuned)': HybridThresholdPolicy(
                    backlog_threshold=ct_hybrid),
                'Queueing(cost-tuned)': QueueingThresholdPolicy(
                    target_wait_min=ct_queue),
                'OnlineMatching': OnlineMatchingPolicy(),
            }
            log.info('Cost-tuned baseline check: Hybrid B=%d, Queueing w=%.1f '
                     '(%d seeds)', ct_hybrid, ct_queue, ct_seeds)
            cost_tuned_raw, cost_tuned_summary = run_replications(
                ct_policies, params, SimulatorG, n=ct_seeds,
                base_seed=COST_TUNED_BASE_SEED, **sim_common)
            cost_tuned_pairwise = pairwise_test_table(
                cost_tuned_raw, ['avg_delay_min', 'on_time_rate',
                                 'total_cost'])
            if not cost_tuned_pairwise.empty:
                cost_tuned_pairwise = cost_tuned_pairwise[
                    (cost_tuned_pairwise['policy_a'] == 'OnlineMatching')
                    | (cost_tuned_pairwise['policy_b'] == 'OnlineMatching')]

    # ----------------------------------------------------------------------
    # Expensive strong baselines (revision analysis: the previous revision
    # reported no non-heuristic benchmark at all).  Rollout (an ADP-style
    # one-step lookahead over the learned marginal-cost base policy) is too
    # expensive for the main batch, so it runs here as a paired benchmark
    # against OnlineMatching / Hybrid / Queueing on its own seed range.  The
    # OR-Tools VRPTW baseline, when the solver is available, is already part
    # of the main comparison above.
    # ----------------------------------------------------------------------
    if RUN_EXPENSIVE_BASELINES and not fast:
        exp_seeds = EXPENSIVE_BASELINE_SEEDS
        exp_policies = {
            'Hybrid': HybridThresholdPolicy(backlog_threshold=hybrid_threshold),
            'Queueing': QueueingThresholdPolicy(target_wait_min=queueing_target),
            'OnlineMatching': OnlineMatchingPolicy(),
            'Rollout': RolloutPolicy(horizon=ROLLOUT_HORIZON,
                                     n_samples=ROLLOUT_N_SAMPLES,
                                     enable_threshold=ROLLOUT_ENABLE_THRESHOLD),
        }
        log.info('Running expensive baselines (Rollout) block: %d policies x '
                 '%d seeds', len(exp_policies), exp_seeds)
        expensive_raw, expensive_summary = run_replications(
            exp_policies, params, SimulatorG, n=exp_seeds,
            base_seed=EXPENSIVE_BASELINE_BASE_SEED, **sim_common)
        expensive_pairwise = pairwise_test_table(
            expensive_raw, ['avg_delay_min', 'on_time_rate', 'total_cost'])
        if not expensive_pairwise.empty:
            expensive_pairwise = expensive_pairwise[
                (expensive_pairwise['policy_a'] == 'OnlineMatching')
                | (expensive_pairwise['policy_b'] == 'OnlineMatching')]

    # ------------------------------------------------------------------
    # Validation and theory
    # ------------------------------------------------------------------
    emp = empirical_kpis(city=CITY)
    # The empirical KPIs reflect the real mixed fleet; validate against a
    # mixed policy rather than the dedicated-only counterfactual.
    # OnlineMatching gives the closest fit to observed LaDe delivery times.
    baseline_policy = 'OnlineMatching' if 'OnlineMatching' in results else list(results.keys())[0]
    df_validation = validate_against_empirical(
        summary_df, city=CITY, duration_min=SIMULATION_MINUTES,
        baseline_policy=baseline_policy)
    df_theory_all = theoretical_bounds(params, n_dedicated, n_crowd_max,
                                       SIMULATION_MINUTES, SURGE_HOURS, SURGE_MULTIPLIER)
    # Keep only the fluid lower bound in the main output.  The M/M/c and
    # Erlang-A bounds assume a stationary single-station queue, which is too
    # stylized for this mixed-fleet dispatching system and often yields an
    # unstable (rho >= 1) approximation; report them only in the Excel sheet
    # for reference and use the feasible fluid bound as the main benchmark.
    df_theory = df_theory_all[df_theory_all['bound'] == 'Fluid'].copy()
    # Drop columns that are undefined for the fluid bound (rho, expected delay,
    # abandonment, ...) so the reported main table contains no NaN cells.
    df_theory = df_theory.dropna(axis=1, how='all')

    # ------------------------------------------------------------------
    # Consolidated parameter table (reviewer comment 4): every parameter with
    # its value, source category and description.
    # ------------------------------------------------------------------
    param_rows = [
        ('simulation_minutes', SIMULATION_MINUTES, 'assumed', 'Operating horizon (12 h, e.g. 08:00-20:00)'),
        ('warmup_minutes', WARMUP_MINUTES, 'assumed', 'Warm-up period excluded from KPIs'),
        ('n_dedicated', n_dedicated, 'calibrated', 'Dedicated couriers (fleet sized for peak-hour target utilization; split share from data)'),
        ('n_crowd_max', n_crowd_max, 'calibrated', 'Maximum crowdsourced couriers (fleet sized for peak-hour target utilization; split share from data)'),
        ('fleet_target_util', FLEET_TARGET_UTIL, 'assumed', 'Target peak-hour utilization used for fleet sizing'),
        ('dedicated_capacity', DEDICATED_CAPACITY, 'assumed', 'Dedicated courier carrying capacity (orders)'),
        ('crowd_capacity', CROWD_CAPACITY, 'assumed', 'Crowdsourced courier carrying capacity (orders)'),
        ('speed_mean', params['speed_mean'], 'estimated',
         f"Courier speed, lognormal fit ({params.get('speed_source', 'trajectory GPS')})"),
        ('speed_std', params['speed_std'], 'estimated', 'Speed standard deviation'),
        ('service_mean', params['service_mean'], 'estimated',
         f"Doorstep service time ({params.get('service_source', 'trajectory dwell fit')})"),
        ('service_std', params['service_std'], 'estimated', 'Service-time standard deviation'),
        ('pickup_overhead_mean', params['pickup_overhead_mean'], 'calibrated', 'Pre-delivery overhead mean (calibrated to accept-to-delivery time; see holdout evidence)'),
        ('pickup_overhead_std', params['pickup_overhead_std'], 'calibrated', 'Pre-delivery overhead standard deviation'),
        ('arrival_rate_scale', params['arrival_rate_scale'], 'calibrated', 'Arrival-rate scale matching empirical daily order volume'),
        ('deadline_min_mean', DEADLINE_MIN, 'assumed', 'Mean of per-order promised deadline offset (truncated normal)'),
        ('deadline_min_std', DEADLINE_STD, 'assumed', 'Std of per-order deadline offset'),
        ('deadline_min_clip', DEADLINE_MIN_CLIP, 'assumed', 'Lower truncation of the deadline offset'),
        ('on_time_threshold_min', ON_TIME_THRESHOLD_MIN, 'assumed', 'Fixed platform on-time threshold used in KPIs and Eq. (1)'),
        ('deadline_buffer_min', DEADLINE_BUFFER_MIN, 'assumed', 'Safety buffer b in the marginal cost (Eq. (8))'),
        ('tw_width_mean', params['tw_width_mean'], 'estimated', 'Mean pickup time-window width'),
        ('cost_delay_per_min', COST_DELAY_PER_MIN, 'assumed', 'Delay cost per minute (alpha)'),
        ('cost_deadline_violation', COST_DEADLINE_VIOLATION, 'assumed', 'Fixed deadline-violation penalty (beta)'),
        ('cost_km', COST_KM, 'assumed', 'Travel cost per km (gamma)'),
        ('cost_dedicated_per_hour', COST_DEDICATED_PER_HOUR, 'assumed', 'Dedicated courier hourly wage (w_D)'),
        ('cost_crowd_per_hour', COST_CROWD_PER_HOUR, 'assumed', 'Crowdsourced courier hourly wage (w_C)'),
        ('dual_lr', DUAL_LR, 'manual', 'Dual-price weight rho in the marginal cost (Eq. (8))'),
        ('dual_price_step', DUAL_PRICE_STEP, 'manual', 'Dual-price increment per assignment'),
        ('dual_price_decay', DUAL_PRICE_DECAY, 'manual', 'Dual-price decay for other feasible couriers per assignment'),
        ('dual_price_step_decay', DUAL_PRICE_STEP_DECAY, 'manual', 'Dual-price decay per time step'),
        ('dual_price_inactive_decay', DUAL_PRICE_INACTIVE_DECAY, 'manual', 'Dual-price decay per minute while crowd courier is offline'),
        ('hybrid_backlog_threshold', hybrid_threshold, 'tuned', 'Hybrid baseline backlog threshold B (grid-searched, delay-optimal)'),
        ('queueing_target_wait_min', queueing_target, 'tuned', 'Queueing baseline target waiting time (grid-searched, delay-optimal)'),
        ('hybrid_backlog_threshold_cost_optimal', tuned_cost.get('Hybrid'), 'tuned', 'Hybrid threshold selected by total cost (robustness check)'),
        ('queueing_target_wait_min_cost_optimal', tuned_cost.get('Queueing'), 'tuned', 'Queueing target wait selected by total cost (robustness check)'),
        ('circuity_factor', CIRCUITY_FACTOR, 'assumed', 'Road-circuity multiplier for Haversine distances (robustness-checked)'),
        ('n_stations', len(params['stations']), 'estimated', 'Number of pickup stations'),
        ('n_aois', len(params['aoi_probs']), 'estimated', 'Number of destination AOIs'),
    ]
    param_rows += [(f'crowd_avail_h{h}', float(v), 'estimated',
                    'Crowd availability probability by hour')
                   for h, v in enumerate(params['crowd_avail'])]
    params_df = pd.DataFrame(param_rows, columns=['param', 'value', 'source',
                                                  'description'])

    # ------------------------------------------------------------------
    # Prepare publication-ready tables
    # ------------------------------------------------------------------
    # Main comparison table uses the multi-seed robust means so that the Excel
    # sheet is consistent with the LaTeX metrics table.
    metrics_main = summary_df.pivot(index='policy', columns='metric', values='mean')
    metrics_main = metrics_main.reset_index().rename_axis(None, axis=1)

    # Reduce metrics table to the most important columns for the main paper.
    key_metric_cols = [
        'policy', 'avg_delay_min', 'on_time_rate', 'on_time_rate_promised',
        'max_backlog', 'total_cost', 'crowd_served', 'dedicated_utilization',
        'crowd_utilization', 'total_km', 'completed_total',
    ]
    metrics_reduced = metrics_main[[c for c in key_metric_cols if c in metrics_main.columns]]

    # Pairwise table: only show comparisons involving the proposed method.
    proposed = 'OnlineMatching'
    pairwise_reduced = pairwise[
        (pairwise['policy_a'] == proposed) | (pairwise['policy_b'] == proposed)
    ].copy() if not pairwise.empty else pairwise

    # Sensitivity table: keep only key mean/std columns to avoid 47-column tables.
    key_sens_cols = ['policy', 'N_CROWD_MAX']
    for col in ['avg_delay_min', 'on_time_rate', 'max_backlog', 'total_cost']:
        for suffix in ['_mean', '_std']:
            c = col + suffix
            if c in sens_df.columns:
                key_sens_cols.append(c)
    sens_reduced = sens_df[[c for c in key_sens_cols if c in sens_df.columns]]

    def _compact_summary(summary, metrics=('avg_delay_min', 'on_time_rate',
                                           'total_cost')):
        """Pivot a long-form replication summary to mean (std) per metric.

        Scenario-aware: ablation summaries carry a ``scenario`` column, which
        is kept as part of the row index.
        """
        sub = summary[summary['metric'].isin(metrics)]
        idx = ['scenario', 'policy'] if 'scenario' in summary.columns else ['policy']
        piv = sub.pivot_table(index=idx, columns='metric', values='mean')
        piv_std = sub.pivot_table(index=idx, columns='metric', values='std')
        out = pd.DataFrame(index=piv.index)
        for m in metrics:
            if m in piv.columns:
                out[m] = [f'{v:.2f} ({s:.2f})' if pd.notna(s) else f'{v:.2f}'
                          for v, s in zip(piv[m], piv_std[m])]
        return out.reset_index()

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    sheets = {
        'metrics': metrics_main,
        'metrics_single_seed': df_metrics_single_seed,
        'params': params_df,
        'robustness': summary_df,
        'pairwise': pairwise,
        'sensitivity': sens_df,
        'validation': df_validation,
        'theory': df_theory,
        'theory_all': df_theory_all,
    }
    if tuning_summary is not None:
        sheets['baseline_tuning'] = tuning_summary
    if abl_summary is not None:
        sheets['ablation'] = abl_summary
        sheets['ablation_pairwise'] = abl_pairwise
        if abl_effects is not None:
            sheets['ablation_effects'] = abl_effects
    if reg_summary is not None:
        sheets['regimes'] = reg_summary
        sheets['regimes_pairwise'] = reg_pairwise
    if circ_df is not None:
        sheets['circuity'] = circ_df
    if mix_summary is not None:
        sheets['capacity_mix'] = mix_summary
    if holdout is not None:
        sheets['holdout_validation'] = holdout['validation']
        sheets['holdout_evidence'] = holdout['evidence']
        sheets['holdout_split'] = holdout['split']
    if cost_tuned_summary is not None:
        sheets['cost_tuned_baselines'] = cost_tuned_summary
        sheets['cost_tuned_pairwise'] = cost_tuned_pairwise
    if expensive_summary is not None:
        sheets['expensive_baselines'] = expensive_summary
        sheets['expensive_baselines_pairwise'] = expensive_pairwise
    path = save_excel(sheets)
    format_excel_best(path)
    log.info('Excel saved to %s', path)

    # Paper main table uses the multi-seed robust means instead of the single-seed snapshot.
    save_latex_table(metrics_reduced,
                     'metrics.tex',
                     caption=f'Policy comparison (mean over {reps} random seeds)',
                     label='tab:metrics')
    # Full robustness results go to the appendix; the main paper uses metrics.tex.
    save_latex_table(summary_df, 'robustness_full.tex',
                     caption=f'Robustness analysis over {reps} seeds (appendix)',
                     label='tab:robustness_full')
    save_latex_table(pairwise_reduced, 'pairwise.tex',
                     caption='Pairwise Wilcoxon tests: OnlineMatching vs. baselines '
                             '(Holm-corrected p-values)',
                     label='tab:pairwise')
    save_latex_table(sens_reduced, 'sensitivity.tex',
                     caption='Sensitivity to maximum crowd fleet size',
                     label='tab:sensitivity')
    save_latex_table(df_validation, 'validation.tex',
                     caption='Validation against empirical LaDe KPIs',
                     label='tab:validation')
    save_latex_table(df_theory, 'theory.tex',
                     caption='Fluid lower bound on total delivery time',
                     label='tab:theory')
    save_latex_table(params_df, 'params.tex',
                     caption='Consolidated parameter table with source categories',
                     label='tab:params_full')
    if tuning_summary is not None:
        tune_cols = [c for c in ['baseline', 'parameter', 'value',
                                 'avg_delay_min_mean', 'avg_delay_min_std',
                                 'on_time_rate_mean', 'total_cost_mean',
                                 'selected_by_delay', 'selected_by_cost',
                                 'pareto_optimal']
                     if c in tuning_summary.columns]
        save_latex_table(tuning_summary[tune_cols], 'baseline_tuning.tex',
                         caption='Baseline threshold tuning (grid search, '
                                 'mean over tuning seeds; delay-optimal and '
                                 'cost-optimal selections and Pareto-optimal '
                                 'candidates are marked)',
                         label='tab:baseline_tuning')
    if abl_summary is not None:
        save_latex_table(_compact_summary(abl_summary), 'ablation.tex',
                         caption='Ablation of OnlineMatching cost components '
                                 'under the base, stress, extreme, contention '
                                 'and heterogeneous-deadline regimes '
                                 '(mean (std) over seeds)',
                         label='tab:ablation')
        if abl_pairwise is not None and not abl_pairwise.empty:
            save_latex_table(abl_pairwise, 'ablation_pairwise.tex',
                             caption='Ablation: Wilcoxon tests vs. the full model '
                                     'under each regime (Holm-corrected p-values)',
                             label='tab:ablation_pairwise')
        if abl_effects is not None and not abl_effects.empty:
            save_latex_table(abl_effects, 'ablation_effects.tex',
                             caption='Component-contribution analysis: paired '
                                     'differences vs. the full model with '
                                     'Holm-corrected Wilcoxon p-values and '
                                     'rank-biserial effect sizes, per regime '
                                     '(positive mean_diff = the removed '
                                     'component helped)',
                             label='tab:ablation_effects')
    if reg_summary is not None:
        save_latex_table(_compact_summary(reg_summary), 'regimes.tex',
                         caption='Policy comparison across congestion regimes '
                                 '(base / stress / extreme; mean (std) over seeds)',
                         label='tab:regimes')
        if reg_pairwise is not None and not reg_pairwise.empty:
            reg_pw = reg_pairwise[
                (reg_pairwise['policy_a'] == proposed)
                | (reg_pairwise['policy_b'] == proposed)]
            save_latex_table(reg_pw, 'regimes_pairwise.tex',
                             caption='Regime comparison: Wilcoxon tests vs. '
                                     'OnlineMatching per regime '
                                     '(Holm-corrected p-values)',
                             label='tab:regimes_pairwise')
    if circ_df is not None:
        circ_cols = [c for c in ['policy', 'CIRCUITY_FACTOR',
                                 'avg_delay_min_mean', 'avg_delay_min_std',
                                 'on_time_rate_mean', 'total_cost_mean']
                     if c in circ_df.columns]
        save_latex_table(circ_df[circ_cols], 'circuity.tex',
                         caption='Robustness to the road-circuity multiplier',
                         label='tab:circuity')
    if mix_summary is not None:
        mix_cols = [c for c in ['policy', 'ded_share', 'n_dedicated',
                                'n_crowd_max', 'total_capacity',
                                'avg_delay_min_mean',
                                'avg_delay_min_std', 'on_time_rate_mean',
                                'total_cost_mean']
                    if c in mix_summary.columns]
        save_latex_table(mix_summary[mix_cols], 'capacity_mix.tex',
                         caption='Dedicated/crowd capacity composition at '
                                 'constant total carrying capacity '
                                 '(iso-capacity scenarios)',
                         label='tab:capacity_mix')
    if cost_tuned_summary is not None:
        save_latex_table(_compact_summary(cost_tuned_summary),
                         'cost_tuned_baselines.tex',
                         caption='Robustness check: core comparison with '
                                 'cost-optimal baseline thresholds '
                                 '(mean (std) over seeds)',
                         label='tab:cost_tuned_baselines')
        if cost_tuned_pairwise is not None and not cost_tuned_pairwise.empty:
            save_latex_table(cost_tuned_pairwise,
                             'cost_tuned_pairwise.tex',
                             caption='Cost-tuned baselines: Wilcoxon tests vs. '
                                     'OnlineMatching (Holm-corrected p-values)',
                             label='tab:cost_tuned_pairwise')
    if expensive_summary is not None:
        save_latex_table(_compact_summary(expensive_summary),
                         'expensive_baselines.tex',
                         caption='Strong-baseline benchmark: Rollout '
                                 '(ADP-style lookahead) vs. the core policies '
                                 '(mean (std) over seeds)',
                         label='tab:expensive_baselines')
        if expensive_pairwise is not None and not expensive_pairwise.empty:
            save_latex_table(expensive_pairwise,
                             'expensive_baselines_pairwise.tex',
                             caption='Strong baselines: Wilcoxon tests vs. '
                                     'OnlineMatching (Holm-corrected p-values)',
                             label='tab:expensive_baselines_pairwise')
    if holdout is not None:
        save_latex_table(holdout['validation'], 'holdout.tex',
                         caption='Holdout-period validation against empirical '
                                 'LaDe KPIs',
                         label='tab:holdout')
        save_latex_table(holdout['evidence'], 'overhead_evidence.tex',
                         caption='Data-based evidence for the pre-delivery '
                                 'overhead parameter',
                         label='tab:overhead_evidence')

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_backlog(backlogs, backlog_labels, SIMULATION_MINUTES)
    plot_metrics_bar(summary_df)
    plot_utilization(results)
    plot_crowd_availability(params)
    # Summary figures restored from the original submission (Minor 2:
    # labels use the standardized "fulfillment time" terminology).
    plot_delay_vs_crowd_utilization(metrics_main, policies=list(core_policies))
    plot_hourly_crowd_availability(params)
    plot_relative_improvement_bar(metrics_main)
    # Keep original config names for metadata, but actual fleet sizes are logged above.
    n_dedicated_used = n_dedicated
    n_crowd_max_used = n_crowd_max
    plot_pareto_frontier(raw_df, x='avg_delay_min', y='total_cost')
    plot_cdf_delays(raw_df)
    plot_sensitivity_heatmap(sens_df, 'N_CROWD_MAX',
                             metric='avg_delay_min_mean')
    plot_sensitivity_lines(sens_df, 'N_CROWD_MAX',
                           operating_point=n_crowd_max)
    if abl_summary is not None:
        plot_ablation(abl_summary)
    if reg_summary is not None:
        plot_regimes(reg_summary)
    if tuning_summary is not None:
        plot_baseline_tuning(tuning_summary)
    if circ_df is not None:
        plot_circuity(circ_df)
    if mix_summary is not None:
        plot_capacity_mix(mix_summary)

    # ------------------------------------------------------------------
    # Metadata and final log (reviewer minor comment 4: report the exact
    # statistical procedure and randomization scheme)
    # ------------------------------------------------------------------
    stat_main = statistical_procedure(reps, base_seed=SEED)
    log.info('Statistical protocol: n=%d, seeds=%s, test=%s, correction=%s',
             stat_main['n_replications'],
             f"{stat_main['base_seed']}..{stat_main['base_seed'] + reps - 1}",
             stat_main['test'], stat_main['multiple_testing'])
    save_metadata(__import__('config'), extra={
        'duration_seconds': time.time() - t0,
        'n_policies': len(policies),
        'policies_in_main_comparison': sorted(policies.keys()),
        'city': CITY,
        'road_network_nodes': len(road_network.nodes) if road_network else 0,
        'n_dedicated_used': n_dedicated_used,
        'n_crowd_max_used': n_crowd_max_used,
        'statistical_procedure_main': stat_main,
        'baselines': {
            'ortools_available': bool(ortools_ok),
            'ortools_import_preflight': bool(_ORTOOLS_AVAILABLE),
            'ortools_vrp_in_main_comparison': 'ORToolsVRP' in policies,
            'rollout_block': {
                'enabled': bool(RUN_EXPENSIVE_BASELINES and not fast),
                'ran': expensive_summary is not None,
                'seeds': (EXPENSIVE_BASELINE_SEEDS if not fast else 3),
                'seed_range': [EXPENSIVE_BASELINE_BASE_SEED,
                               EXPENSIVE_BASELINE_BASE_SEED
                               + (EXPENSIVE_BASELINE_SEEDS if not fast else 3) - 1],
                'horizon': ROLLOUT_HORIZON,
                'n_samples': ROLLOUT_N_SAMPLES,
                'enable_threshold': ROLLOUT_ENABLE_THRESHOLD,
            },
        },
        'tuned_baseline_thresholds': {
            'hybrid_backlog_threshold': hybrid_threshold,
            'queueing_target_wait_min': queueing_target,
            'selection_criterion': 'min mean fulfilment time (delay-optimal); '
                                   'cost-optimal values reported alongside',
            'hybrid_backlog_threshold_cost_optimal': tuned_cost.get('Hybrid'),
            'queueing_target_wait_min_cost_optimal': tuned_cost.get('Queueing'),
            'tuning_seeds': (3 if fast else BASELINE_TUNING_SEEDS),
            'tuning_seed_range': [5000, 5000 + (3 if fast else BASELINE_TUNING_SEEDS) - 1],
        },
        'cost_tuned_check': {
            'enabled': bool(RUN_COST_TUNED_CHECK and tuned_cost),
            'ran': cost_tuned_summary is not None,
            'seeds': (3 if fast else COST_TUNED_SEEDS),
            'seed_range': [COST_TUNED_BASE_SEED,
                           COST_TUNED_BASE_SEED + (3 if fast else COST_TUNED_SEEDS) - 1],
        },
        'extended_experiments': {
            'enabled': RUN_EXTENDED_EXPERIMENTS,
            'ablation_seeds': (3 if fast else ABLATION_SEEDS),
            'regime_seeds': (3 if fast else REGIME_SEEDS),
            'circuity_grid': CIRCUITY_GRID,
            'circuity_seeds': (3 if fast else CIRCUITY_SEEDS),
            'capacity_mix_ded_shares': CAPACITY_MIX_DED_SHARES,
            'capacity_mix_seeds': (3 if fast else CAPACITY_MIX_SEEDS),
            'capacity_mix_mode': CAPACITY_MIX_MODE,
            'holdout_train_fraction': HOLDOUT_TRAIN_FRACTION,
            'holdout_seeds': (3 if fast else HOLDOUT_SEEDS),
        },
    })

    # ------------------------------------------------------------------
    # Comment 7: cross-city external-validity experiment.  This lives in the
    # standalone cross_city.py module (it reruns the full calibration per
    # city and writes its own cross_city.xlsx / cross_city.tex / figure).
    # It runs last so that the main outputs are already saved even if the
    # cross-city batch fails.
    # ------------------------------------------------------------------
    if RUN_CROSS_CITY and not fast:
        log.info('Running cross-city validation (see cross_city.py)')
        try:
            from cross_city import main as cross_city_main
            cross_city_main()
        except Exception as e:
            log.warning('Cross-city experiment failed: %s', e)

    log.info('All done in %.1fs', time.time() - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crowdsourced delivery simulation')
    parser.add_argument('--fast', action='store_true',
                        help='Run a reduced fast experiment')
    parser.add_argument('--smoke', action='store_true',
                        help='Alias for --fast (quick smoke test of the full pipeline)')
    parser.add_argument('--policies', nargs='+', default=None,
                        help='Subset of policies to run')
    parser.add_argument('--n-replications', type=int, default=None,
                        help='Number of random seeds')
    args = parser.parse_args()
    main(fast=args.fast or args.smoke, selected_policies=args.policies,
         n_replications=args.n_replications)
