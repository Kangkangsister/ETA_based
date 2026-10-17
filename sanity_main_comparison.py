"""Quick sanity check: does the revised OnlineMatching still beat the tuned
baselines in the main comparison after the revision-round-2 cost redesign?
Runs a small seed set on real data (not a substitute for the full run).
"""

import logging
import time

import pandas as pd

from base import setup_logging, set_seed, empirical_kpis, calibrate_arrival_scale, \
    estimate_fleet_sizes, pairwise_tests
from config import (SEED, CITY, SIMULATION_MINUTES, N_DEDICATED, N_CROWD_MAX,
                    FLEET_TARGET_UTIL, SURGE_HOURS, SURGE_MULTIPLIER,
                    APPLY_SURGE_MULTIPLIER)
from data.loader import load_delivery, load_pickup, load_trajectory, derive_params
from simulator.core import SimulatorG
from experiments.replication import run_replications
from policies.greedy import DedicatedOnlyPolicy, HybridThresholdPolicy, NearestPolicy
from policies.queueing import QueueingThresholdPolicy
from policies.matching import OnlineMatchingPolicy

log = logging.getLogger(__name__.split('.')[0])

if __name__ == '__main__':
    setup_logging()
    set_seed(SEED)
    delivery = load_delivery()
    pickup = load_pickup()
    traj = load_trajectory()
    params = derive_params(delivery, pickup, traj)
    emp = empirical_kpis(city=CITY)
    calibrate_arrival_scale(params, emp, SIMULATION_MINUTES)
    n_ded, n_crowd = estimate_fleet_sizes(params, N_DEDICATED, N_CROWD_MAX,
                                          simulation_minutes=SIMULATION_MINUTES,
                                          target_util=FLEET_TARGET_UTIL)
    params['n_dedicated'] = n_ded
    params['n_crowd_max'] = n_crowd
    sim_common = dict(duration=SIMULATION_MINUTES, n_dedicated=n_ded,
                      n_crowd_max=n_crowd, surge_hours=SURGE_HOURS,
                      surge_multiplier=SURGE_MULTIPLIER,
                      apply_surge_multiplier=APPLY_SURGE_MULTIPLIER,
                      arrival_rate_scale=params['arrival_rate_scale'])
    # Tuned baseline thresholds from the previous full run (metadata.json).
    policies = {
        'DedicatedOnly': DedicatedOnlyPolicy(),
        'Hybrid': HybridThresholdPolicy(backlog_threshold=30),
        'Nearest': NearestPolicy(),
        'Queueing': QueueingThresholdPolicy(target_wait_min=2.0),
        'OnlineMatching': OnlineMatchingPolicy(),
    }
    raw, summary = run_replications(policies, params, SimulatorG, n=10,
                                    base_seed=42, **sim_common)
    key = summary[summary['metric'].isin(['avg_delay_min', 'on_time_rate',
                                          'on_time_rate_promised', 'total_cost'])]
    pd.set_option('display.width', 200)
    print(key.pivot_table(index='policy', columns='metric',
                          values='mean').round(3).to_string())
    print()
    pw = pairwise_tests(raw, 'avg_delay_min', policy_col='policy', seed_col='seed')
    print(pw[pw['policy_b'] == 'OnlineMatching'][
        ['policy_a', 'mean_a', 'mean_b', 'p_value', 'significance']].to_string(index=False))
