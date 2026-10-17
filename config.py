"""Global configuration for the crowdsourced delivery simulation.

All hyper-parameters are centralized here so that experiments are
reproducible and easy to sweep.  Values are chosen to match the LaDe
last-mile delivery dataset (Shanghai by default) and typical instant
delivery platform settings reported in the literature.
"""

from base import DATA_DIR, setup_logging, set_seed, configure_plotting
from base import SIM_START_HOUR, sim_hour  # re-exported for policies/simulator

# ---------------------------------------------------------------------------
# Simulation horizon
# ---------------------------------------------------------------------------
SIMULATION_MINUTES = 720          # 12 hours, e.g. 08:00--20:00
WARMUP_MINUTES = 60               # warm-up period excluded from KPIs

# ---------------------------------------------------------------------------
# Fleet sizing
# ---------------------------------------------------------------------------
# Defaults are placeholders.  main.py overwrites these with a data-driven
# fleet size: the fleet is sized so that peak-hour demand (arrival rate
# calibrated to the empirical order volume) runs at FLEET_TARGET_UTIL
# utilization, given the per-order cycle time (pickup overhead + travel +
# doorstep service).  The dedicated/crowd split follows the share estimated
# from the observed courier population in LaDe.
N_DEDICATED = 25                  # number of dedicated couriers
N_CROWD_MAX = 60                  # maximum number of crowd couriers that can be activated
FLEET_TARGET_UTIL = 0.85          # target peak-hour utilization for fleet sizing

# Dedicated / crowd capacity (number of concurrent orders a courier can carry)
DEDICATED_CAPACITY = 8
CROWD_CAPACITY = 3

# ---------------------------------------------------------------------------
# Sampling limits for data loading (0 means "do not load")
# ---------------------------------------------------------------------------
SAMPLE_DELIVERY = 100000
SAMPLE_PICKUP = 50000
SAMPLE_TRAJ = 200000              # load a sample of trajectory GPS points

# ---------------------------------------------------------------------------
# Order / service parameters
# ---------------------------------------------------------------------------
DEADLINE_MIN = 120                # mean of the per-order promised deadline offset
                                  # (sampled per order from a truncated normal)
DEADLINE_STD = 30                 # std dev of the per-order deadline offset
DEADLINE_MIN_CLIP = 30            # lower truncation bound of the deadline offset
# Fixed on-time threshold used in the main KPIs and in the cost function
# (Eq. (1) in the paper).  This is intentionally distinct from the per-order
# promised deadline d_o = r_o + offset sampled above: the fixed threshold is
# the platform-level service-level target (and matches the empirical LaDe KPI
# definition), while d_o is the order-level promise used by the policies.
ON_TIME_THRESHOLD_MIN = 120
# Deadline safety buffer b in the marginal-cost function (Eq. (8)): the
# deadline-risk term is the expected violation penalty under a normal
# approximation, beta * Phi((C_oc - (d_o - b)) / sigma_c), gated by the
# probability that the promise is still attainable, (1 - Phi((C_oc - d_o) /
# sigma_c)), where sigma_c grows with the courier's committed workload.  b
# defines how many minutes *before* the promised deadline the risk window
# opens: at-risk-but-savable orders are prioritised toward low-variance
# (short-queue) couriers.  (Reviewer comment 2: b = 0 made the original hard
# indicator coincide with the hard penalty region; moreover a deterministic
# indicator is a monotone transform of the ETA and can provably never change
# the argmin assignment.  The gated risk-probability form gives b a precise,
# active role.)
DEADLINE_BUFFER_MIN = 20
TIME_WINDOW_START_MIN = 0         # pickup time window offset relative to creation
TIME_WINDOW_END_MIN = 60

# Pickup/acceptance overhead before the delivery leg.  This captures the time
# from order acceptance until the courier actually starts travelling to the
# customer (assignment, travel to merchant/station, merchant wait, pickup).
# Calibrated so that the OnlineMatching mixed-fleet baseline reproduces the
# observed LaDe average accept-to-delivery time.
PICKUP_OVERHEAD_MEAN = 64.0       # minutes
PICKUP_OVERHEAD_STD = 30.0        # minutes

# ---------------------------------------------------------------------------
# Policy thresholds
# ---------------------------------------------------------------------------
BACKLOG_THRESHOLD = 15            # Hybrid policy: activate crowd when backlog > threshold
SURGE_BACKLOG_THRESHOLD = 25      # Surge pricing: full activation when backlog > threshold

# ---------------------------------------------------------------------------
# Demand calibration
# ---------------------------------------------------------------------------
# The arrival rates are derived directly from the LaDe hourly empirical curve.
# If APPLY_SURGE_MULTIPLIER is True, the below multiplier is applied on top of
# the empirical rates.  By default it is disabled: the real data already embeds
# lunch/dinner peaks.
APPLY_SURGE_MULTIPLIER = False
SURGE_HOURS = [11, 12, 17, 18, 19]
SURGE_MULTIPLIER = 1.0            # multiplies order arrival rate during surge hours
# Global arrival-rate scale.  main.py overwrites this with a data-driven
# estimate that matches the empirical daily order count.
ARRIVAL_RATE_SCALE = 1.0

# ---------------------------------------------------------------------------
# Travel distance approximation
# ---------------------------------------------------------------------------
# The simulator uses Haversine (great-circle) distance because detailed road
# data are unavailable for the full city.  Urban road networks typically make
# real trips longer than the straight-line distance; CIRCUITY_FACTOR scales
# the Haversine distance to approximate road distance (typical urban values
# are 1.2--1.4).  The factor is applied consistently to simulated travel
# times/distances and to the policies' completion-time estimates.
CIRCUITY_FACTOR = 1.0

# ---------------------------------------------------------------------------
# Cost / objective weights (used by cost-aware policies and ADP)
# ---------------------------------------------------------------------------
# In the paper's notation: alpha = COST_DELAY_PER_MIN, beta =
# COST_DEADLINE_VIOLATION, gamma = COST_KM, w_D = COST_DEDICATED_PER_HOUR,
# w_C = COST_CROWD_PER_HOUR.
COST_DELAY_PER_MIN = 1.0          # cost of one minute of delivery delay (alpha)
COST_DEADLINE_VIOLATION = 50.0    # fixed penalty for missing the deadline (beta)
COST_KM = 0.5                     # travel cost per km (gamma)
COST_DEDICATED_PER_HOUR = 30.0    # dedicated courier hourly wage (w_D)
COST_CROWD_PER_HOUR = 25.0        # crowd courier hourly wage (w_C) (when active)

# ---------------------------------------------------------------------------
# Randomness and reproducibility
# ---------------------------------------------------------------------------
SEED = 42
# Reviewer request: at least 30 independent replications per condition so
# that confidence intervals and paired Wilcoxon tests are meaningful.
# Seeds are consecutive integers: base_seed, base_seed+1, ..., base_seed+n-1.
N_REPLICATIONS = 30               # number of random seeds for statistical testing

# ---------------------------------------------------------------------------
# OnlineMatching hyper-parameters (LP-dual water-filling)
# ---------------------------------------------------------------------------
# rho: weight ("learning rate") of the dual price in the marginal cost.
# Per assignment the chosen courier's dual price increases by DUAL_PRICE_STEP
# and all other feasible couriers decay by DUAL_PRICE_DECAY; additionally all
# prices decay by DUAL_PRICE_STEP_DECAY each minute and inactive crowd
# couriers decay by DUAL_PRICE_INACTIVE_DECAY each minute.
# Scale rationale (revision round 2): the marginal cost is expressed in
# minutes, so the dual price must be able to reach minute scale to influence
# the argmin assignment.  With step = 10 and rho = 0.3, one extra assignment
# adds ~3 min-equivalents and a heavily contested courier (~10 assignments in
# a peak hour) accumulates ~30 min -- comparable to the marginal delay of one
# queued order, so the congestion price can actually reroute orders.  The
# decays are scaled by the same factor of 10 to preserve the original
# dynamics.  (The original 1.0/0.08/0.02/0.15 values kept prices ~100x below
# the ETA term, which made the mechanism inactive and the ablation null.)
DUAL_LR = 0.3
DUAL_PRICE_STEP = 10.0
DUAL_PRICE_DECAY = 0.8
DUAL_PRICE_STEP_DECAY = 0.2
DUAL_PRICE_INACTIVE_DECAY = 1.5

# ---------------------------------------------------------------------------
# Extended experiments (revision): seeds and grids
# ---------------------------------------------------------------------------
# The four strategies reported in the paper; extended experiments (sensitivity,
# circuity, capacity mix) run on this core set to keep the protocol identical
# to the main comparison.
CORE_POLICIES = ['DedicatedOnly', 'Hybrid', 'Queueing', 'OnlineMatching']
RUN_EXTENDED_EXPERIMENTS = True   # master switch for the revision experiments
# Cross-city validation (comment 7) lives in the standalone cross_city.py
# module; this switch makes a full ``python main.py`` run include it.
RUN_CROSS_CITY = True

SENSITIVITY_SEEDS = 30            # seeds per level of the N_CROWD_MAX sweep

# Baseline tuning (fair calibration of the baseline thresholds):
# NOTE: the grid must extend well beyond the previously selected value (B=30
# was the upper edge of [5..30], a boundary optimum).  The extended grid
# verifies that the selected threshold is an interior optimum.
BACKLOG_THRESHOLD_GRID = [5, 10, 15, 20, 25, 30, 40, 60, 80, 100]  # Hybrid backlog B
# NOTE: the grid must extend below the previously selected boundary optimum
# (w=2 was the lower edge of [2..30]); the extended grid verifies that the
# selected target wait is an interior optimum.
QUEUEING_WAIT_GRID = [1, 2, 3, 5, 8, 10, 15, 20, 30]         # Queueing target wait (min)
BASELINE_TUNING_SEEDS = 10

# Ablation of the OnlineMatching marginal-cost components:
ABLATION_SEEDS = 30
# The base regime is relatively uncongested, so the marginal-cost components
# rarely bind and a naive ablation shows no significant differences.  A
# stressed scenario (higher demand, tighter promised deadlines) is added so
# that the contribution of each component becomes identifiable.
ABLATION_STRESS_ARRIVAL_MULT = 1.3   # demand multiplier in the stressed scenario
ABLATION_STRESS_DEADLINE_MIN = 90    # tighter promised deadline (min) in the stressed scenario
# Extreme regime on top of stress: near-saturation demand, very tight promised
# deadlines and reduced crowd availability, so that every marginal-cost
# component (dual price, deadline risk, queue load, distance) binds hard.
ABLATION_EXTREME_ARRIVAL_MULT = 1.5  # demand multiplier in the extreme scenario
ABLATION_EXTREME_DEADLINE_MIN = 60   # promised deadline (min) in the extreme scenario
ABLATION_EXTREME_CROWD_AVAIL_MULT = 0.5  # crowd availability multiplier (extreme)
# Contention regime: abundant demand but *scarce crowd supply*, so orders
# compete for the few available couriers.  The dual congestion price is
# designed for exactly this situation and is inactive when supply is ample.
ABLATION_CONTEND_ARRIVAL_MULT = 1.2    # demand multiplier in the contention scenario
ABLATION_CONTEND_CROWD_AVAIL_MULT = 0.3  # crowd availability multiplier (contention)
# Heterogeneous-deadline regime: promised deadlines are drawn from a discrete
# mixture of service levels instead of a single normal, so the deadline-risk
# component can differentiate *between orders* (tight vs loose promises)
# rather than facing a near-uniform deadline for every order.
ABLATION_HETERO_ARRIVAL_MULT = 1.3     # demand multiplier in the hetero-deadline scenario
ABLATION_HETERO_DEADLINE_MIX = ((30, 0.2), (60, 0.3), (120, 0.3), (180, 0.2))
# Orders whose promised deadline offset is at most TIGHT_DEADLINE_MIN form the
# "tight-deadline" stratum reported by the targeted ablation metrics.
TIGHT_DEADLINE_MIN = 60

# Regime comparison: the core policies are re-run under each ablation regime
# (base / stress / extreme) so the paper can report how the relative policy
# ranking and the advantage of OnlineMatching evolve with congestion.
REGIME_SEEDS = 30

# Road-circuity robustness check:
CIRCUITY_GRID = [1.0, 1.2, 1.4]
CIRCUITY_SEEDS = 10

# Capacity-composition scenarios (dedicated share of a fixed total fleet).
# Reviewer comment 8 + revision analysis: keeping the *headcount* constant is
# not a fair comparison because a dedicated courier carries DEDICATED_CAPACITY
# orders while a crowd courier carries only CROWD_CAPACITY; with iso-headcount
# scenarios the ded_share=0.9 case has several times the carrying capacity of
# the ded_share=0.1 case.  CAPACITY_MIX_MODE='capacity' therefore keeps the
# total CARRYING CAPACITY (couriers x per-courier capacity) constant instead;
# 'fleet' reproduces the old iso-headcount behaviour.
CAPACITY_MIX_DED_SHARES = [0.1, 0.25, 1/3, 0.5, 0.67, 0.9]
CAPACITY_MIX_SEEDS = 10
CAPACITY_MIX_MODE = 'capacity'     # 'capacity' (iso-capacity) | 'fleet' (iso-headcount)

# Holdout (temporal) validation: calibrate on the earliest fraction of days
# and validate on the remaining days.
HOLDOUT_TRAIN_FRACTION = 0.75
HOLDOUT_SEEDS = 30
HOLDOUT_POLICIES = ['OnlineMatching', 'Queueing']

# ---------------------------------------------------------------------------
# OR-Tools VRP lookahead (used by the offline baseline)
# ---------------------------------------------------------------------------
VRP_LOOKAHEAD_MIN = 15            # horizon for offline VRP baseline
VRP_MAX_ORDERS = 40               # max orders per VRP solve

# ---------------------------------------------------------------------------
# Multi-city support (sh, hz, cq, jl, yt)
# ---------------------------------------------------------------------------
CITY = 'sh'                       # default city

# Cross-city validation protocol (reviewer comment 7 + revision analysis):
# with only 10 seeds the smallest attainable Wilcoxon p-value is ~0.00195,
# so the cross-city significance tests carried almost no information.  The
# cross-city experiment now uses the SAME protocol as the main comparison:
# 30 evaluation seeds and 10 tuning seeds per city.
CROSS_CITY_SEEDS = 30             # replications per city x policy
CROSS_CITY_TUNE_SEEDS = 10        # tuning seeds per candidate per city

# Per-city speed adjustment factors.  The LaDe detailed GPS trajectory file
# covers Shanghai only, so the raw speed estimate is shared; city-specific
# effective speeds differ systematically (road network density, traffic,
# courier mix).  These multiplicative factors are ASSUMED values (reported as
# such in the parameter table); set them to estimates from city-specific
# trajectory data when available (see data/loader.load_trajectory(city=...),
# which automatically uses a per-city trajectory file if present).
CITY_SPEED_FACTORS = {
    'sh': 1.00,   # estimated directly from the Shanghai GPS trajectories
    'hz': 0.95,   # Hangzhou: dense urban core, slightly slower effective speed
    'cq': 0.80,   # Chongqing: mountainous terrain, slower effective speed
    'jl': 0.90,   # Jilin: smaller city, but winter conditions; slightly slower
    'yt': 0.95,   # Yantai: mid-size coastal city
}

# ---------------------------------------------------------------------------
# Strong baselines (reviewer comment on missing non-heuristic benchmarks)
# ---------------------------------------------------------------------------
# OR-Tools VRPTW: on the (Linux) server ortools imports cleanly; the Windows
# DLL issue is handled by importing ortools before numpy/pandas in main.py.
# If ortools is unavailable the policy is excluded and metadata records it.
RUN_EXPENSIVE_BASELINES = True    # run the Rollout (ADP) benchmark block
# Rollout is computationally heavy, so it runs as a separate paired benchmark
# block against OnlineMatching / Hybrid / Queueing on its own seed range
# (non-overlapping with every other experiment).
EXPENSIVE_BASELINE_SEEDS = 10
EXPENSIVE_BASELINE_BASE_SEED = 12000
ROLLOUT_HORIZON = 20              # lookahead horizon (minutes)
ROLLOUT_N_SAMPLES = 3             # candidate couriers evaluated per order
ROLLOUT_ENABLE_THRESHOLD = 10     # rollout only when <= this many pending orders

# Baseline-tuning fairness check (revision analysis): the primary tuning
# criterion minimises mean fulfilment time (the paper's service-level focus),
# but to show the conclusion is not an artefact of the selection criterion the
# main comparison is RE-RUN with the cost-optimal baseline thresholds and
# reported alongside (delay-optimal vs cost-optimal baselines).
RUN_COST_TUNED_CHECK = True
COST_TUNED_SEEDS = 30
COST_TUNED_BASE_SEED = 13000

# ---------------------------------------------------------------------------
# Road network (optional; disables if False for speed)
# ---------------------------------------------------------------------------
USE_ROAD_NETWORK = False
ROAD_NETWORK_MAX_ROWS = 50000     # number of road segments to parse

# ---------------------------------------------------------------------------
# Plotting and output
# ---------------------------------------------------------------------------
FIGURE_STYLE = 'paper'            # 'paper' | 'notebook'
USE_LATEX = False                 # set to True if a LaTeX distribution is available
