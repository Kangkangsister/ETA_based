"""Base utilities: logging, seeding, plotting style, statistics,
Excel/LaTeX export, theoretical bounds, and reproducibility metadata.
"""

import json
import logging
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

warnings.filterwarnings('ignore')

import warnings

ROOT = Path(__file__).parent
# The repository is nested as <root>/G/G; data is expected at <root>/logistics_data,
# i.e. a sibling of the outer G directory.
_DEFAULT_DATA_DIRS = [
    ROOT.parent.parent / 'logistics_data',
    ROOT.parent / 'logistics_data',
]
_DATA_DIR_FROM_ENV = os.environ.get('LOGISTICS_DATA_DIR')
if _DATA_DIR_FROM_ENV:
    DATA_DIR = Path(_DATA_DIR_FROM_ENV)
else:
    DATA_DIR = next((d for d in _DEFAULT_DATA_DIRS if d.exists()),
                    _DEFAULT_DATA_DIRS[0])
OUT_DIR = ROOT / 'output'

if not DATA_DIR.exists():
    warnings.warn(f"Data directory not found: {DATA_DIR}. "
                  f"Set LOGISTICS_DATA_DIR or place data at {_DEFAULT_DATA_DIRS[0]}.")

STYLE = 'seaborn-v0_8-whitegrid'
PALETTE = 'viridis'


def setup_logging():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(str(ROOT.name))
    log.setLevel(logging.INFO)
    if log.handlers:
        log.handlers.clear()
    fh = logging.FileHandler(OUT_DIR / 'log.txt', mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
    log.addHandler(sh)
    return log


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def configure_plotting(style=STYLE, palette=PALETTE, use_latex=False):
    try:
        plt.style.use(style)
    except Exception:
        plt.style.use('default')
    if palette:
        sns.set_palette(palette)
    rcParams['figure.dpi'] = 120
    rcParams['savefig.dpi'] = 300
    rcParams['figure.figsize'] = (8, 5)
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 13
    rcParams['legend.fontsize'] = 10
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    if use_latex:
        rcParams['text.usetex'] = True
        rcParams['font.family'] = 'serif'


def save_excel(sheets, filename='results.xlsx'):
    path = OUT_DIR / filename
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return path


def save_latex_table(df, filename, caption=None, label=None):
    """Export a DataFrame to a LaTeX table file."""
    path = OUT_DIR / 'tables'
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filename
    # Format floats and replace NaN with an em-dash so LaTeX compiles cleanly.
    latex = df.to_latex(index=False, float_format='%.3f', na_rep='—',
                        caption=caption, label=label,
                        escape=False)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    return file_path


def savefig(fig, name):
    path = OUT_DIR / 'figures'
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / name
    fig.savefig(file_path, bbox_inches='tight')
    plt.close(fig)
    return file_path


def _package_versions():
    """Return versions of key runtime dependencies without importing heavy DLLs."""
    packages = ['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn',
                'networkx', 'sklearn', 'ortools']
    versions = {}
    try:
        from importlib.metadata import version as pkg_version
    except Exception:
        pkg_version = None
    for pkg in packages:
        try:
            if pkg_version is not None:
                versions[pkg] = pkg_version(pkg)
            else:
                versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = None
    return versions


def save_metadata(config_module, extra=None):
    """Record runtime metadata to ensure reproducibility."""
    path = OUT_DIR / 'metadata.json'
    meta = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cwd': str(Path.cwd()),
        'python': sys.version,
        'data_dir': str(DATA_DIR),
        'data_dir_exists': DATA_DIR.exists(),
        'dependencies': _package_versions(),
        'config': {k: v for k, v in vars(config_module).items()
                   if not k.startswith('_') and k.isupper()},
    }
    if extra:
        meta.update(extra)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Simulation clock helpers
# ---------------------------------------------------------------------------

# The operating window starts at this hour of day (the 12 h horizon
# SIMULATION_MINUTES=720 then covers SIM_START_HOUR .. SIM_START_HOUR+12,
# i.e. 08:00--20:00).  Every consumer of hour-of-day profiles (arrival
# rates, crowd availability, surge hours) MUST go through ``sim_hour`` so
# that simulation minutes are mapped to the correct hour of day.
SIM_START_HOUR = 8


def sim_hour(t_min):
    """Map a simulation minute (0-based) to the hour of day (0--23)."""
    return (SIM_START_HOUR + int(t_min) // 60) % 24


# ---------------------------------------------------------------------------
# Publication-quality statistics
# ---------------------------------------------------------------------------

def confidence_interval(values, level=0.95):
    """Two-sided confidence interval using Student's t (small samples)."""
    from scipy import stats
    a = np.asarray(values, dtype=float)
    n = len(a)
    if n < 2:
        return (np.nan, np.nan)
    m = float(a.mean())
    s = float(a.std(ddof=1))
    if n < 30:
        z = stats.t.ppf((1 + level) / 2.0, df=n - 1)
    else:
        z = 1.96
    h = z * s / math.sqrt(n)
    return (m - h, m + h)


def paired_wilcoxon(values_a, values_b):
    """Return Wilcoxon signed-rank test p-value for paired samples."""
    from scipy import stats
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        return np.nan
    try:
        stat, p = stats.wilcoxon(a, b, alternative='two-sided')
        return float(p)
    except Exception:
        return np.nan


def holm_correction(pvalues):
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Returns an array of adjusted p-values that control the family-wise error
    rate.  NaN values are left unchanged.
    """
    pvals = np.asarray(pvalues, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    valid_mask = ~np.isnan(pvals)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return pvals
    valid_p = pvals[valid_idx]
    sort_order = np.argsort(valid_p)
    sorted_p = valid_p[sort_order]
    # Holm step-down: the i-th smallest p-value (0-based) is multiplied by
    # (n - i); the adjusted sequence must then be non-decreasing, enforced
    # with a cumulative MAXIMUM (using a cumulative minimum here would
    # anti-conservatively shrink every adjusted p to the smallest one).
    adjusted = np.maximum.accumulate(sorted_p * np.arange(len(sorted_p), 0, -1))
    adjusted = np.minimum(adjusted, 1.0)
    # Re-inject corrected values in original positions.
    out = pvals.copy()
    out[valid_idx[sort_order]] = adjusted
    return out


def significance_stars(p):
    """Return significance markers for a p-value."""
    if np.isnan(p):
        return 'n/a'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def replicate(name, run_func, n=30, seeds=None, base_seed=42):
    """Run a function over multiple seeds and summarize statistics."""
    if seeds is None:
        seeds = list(range(base_seed, base_seed + n))
    rows = []
    for seed in seeds:
        m = run_func(seed)
        if not isinstance(m, dict):
            m = dict(value=float(m))
        m['seed'] = seed
        rows.append(m)
    df = pd.DataFrame(rows)
    summary = []
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'seed' in numeric:
        numeric.remove('seed')
    for col in numeric:
        vals = df[col].values
        lo, hi = confidence_interval(vals)
        summary.append(dict(
            policy=name, metric=col,
            mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)),
            min=float(np.min(vals)), max=float(np.max(vals)),
            ci_lower=lo, ci_upper=hi,
        ))
    return df, pd.DataFrame(summary)


def sensitivity_sweep(param_name, values, run_func):
    rows = []
    for v in values:
        m = run_func(v)
        if not isinstance(m, dict):
            m = dict(value=float(m))
        m[param_name] = v
        rows.append(m)
    return pd.DataFrame(rows)


def pairwise_tests(df, metric, policy_col='policy', seed_col='seed'):
    """Compute pairwise Wilcoxon tests for a metric across policies.

    P-values are Holm-Bonferroni corrected within the family of pairwise
    comparisons for this metric to control the family-wise error rate.
    """
    policies = df[policy_col].unique()
    records = []
    for i, pi in enumerate(policies):
        for pj in policies[i + 1:]:
            vi = df[(df[policy_col] == pi) & (df[seed_col].notna())][metric].values
            vj = df[(df[policy_col] == pj) & (df[seed_col].notna())][metric].values
            p = paired_wilcoxon(vi, vj)
            records.append(dict(
                metric=metric, policy_a=pi, policy_b=pj,
                mean_a=float(np.mean(vi)), mean_b=float(np.mean(vj)),
                p_value_raw=p
            ))
    out = pd.DataFrame(records)
    if not out.empty:
        out['p_value'] = holm_correction(out['p_value_raw'].values)
        out['significance'] = out['p_value'].apply(significance_stars)
    return out


# ---------------------------------------------------------------------------
# Excel formatting
# ---------------------------------------------------------------------------

def format_excel_best(path, sheet_names=None):
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except Exception:
        return
    if not Path(path).exists():
        return
    wb = load_workbook(path)
    targets = sheet_names if isinstance(sheet_names, (list, tuple)) else (
        [sheet_names] if sheet_names else wb.sheetnames)
    header_font = Font(bold=True, color='FFFFFF')
    header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
    header_align = Alignment(horizontal='center', vertical='center', wrap_text=True)
    best_font = Font(bold=True, color='C00000')
    thin = Side(style='thin', color='CCCCCC')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for sheet_name in targets:
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        if ws.max_row < 1 or ws.max_column < 1:
            continue
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_align
            cell.border = border
        for col_idx in range(1, ws.max_column + 1):
            max_len = 0
            for row_idx in range(1, ws.max_row + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.border = border
                val = str(cell.value) if cell.value is not None else ''
                max_len = max(max_len, len(val))
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max(max_len + 2, 10), 50)
        ws.freeze_panes = 'A2'
        if ws.max_row < 2:
            continue
        header = [cell.value for cell in ws[1]]
        for col_idx, col_name in enumerate(header, start=1):
            cells = []
            for row_idx in range(2, ws.max_row + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                try:
                    cells.append((float(cell.value), cell))
                except Exception:
                    pass
            if not cells:
                continue
            cname = str(col_name).lower()
            minimize = any(k in cname for k in (
                'delay', 'cost', 'carbon', 'mae', 'rmse', 'std', 'pending',
                'backlog', 'overtime', 'failed', 'load_lost', 'distance',
                'error', 'cascade_fraction'))
            maximize = any(k in cname for k in (
                'on_time', 'service_level', 'gc', 'reward', 'accuracy',
                'rate', 'completed', 'served', 'utilization', 'available'))
            if minimize:
                best_val, best_cell = min(cells, key=lambda x: x[0])
            elif maximize:
                best_val, best_cell = max(cells, key=lambda x: x[0])
            else:
                continue
            best_cell.font = best_font
    wb.save(path)


# ---------------------------------------------------------------------------
# Empirical and theoretical benchmarks
# ---------------------------------------------------------------------------

def _haversine_km(x1, y1, x2, y2):
    """Great-circle distance in km between two lon/lat points."""
    phi1, phi2 = math.radians(y1), math.radians(y2)
    dphi = math.radians(y2 - y1)
    dlambda = math.radians(x2 - x1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 6371.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _expected_trip_km(params):
    """Demand-weighted mean station->AOI trip distance in km."""
    rates = params.get('rates', {})
    station_coords = params.get('station_coords', {})
    aoi_coords = params.get('aoi_coords', {})
    aoi_probs = params.get('aoi_probs', {})
    if not rates or not station_coords or not aoi_coords:
        return 2.0
    total_rate = sum(sum(rates[s]) for s in rates)
    if total_rate <= 0:
        return 2.0
    num = den = 0.0
    for s in rates:
        if s not in station_coords:
            continue
        w_s = sum(rates[s]) / total_rate
        sx, sy = station_coords[s]
        for a, (ax, ay) in aoi_coords.items():
            w = w_s * aoi_probs.get(a, 0.0)
            num += w * _haversine_km(sx, sy, ax, ay)
            den += w
    return num / den if den > 0 else 2.0


def effective_cycle_time_min(params):
    """Per-order courier occupancy time in minutes (the queueing 'service time').

    A courier is occupied by an order for the full cycle: pickup/acceptance
    overhead + travel to the dropoff + doorstep service.  Queueing formulas
    (Erlang-C thresholds, theoretical bounds, fleet sizing) MUST use this
    cycle time as 1/mu; using only the doorstep ``service_mean`` overestimates
    the fleet throughput by an order of magnitude and makes activation
    thresholds never bind.
    """
    cycle = params.get('effective_cycle_min')
    if cycle is not None and np.isfinite(cycle):
        return float(cycle)
    speed = max(1.0, params.get('speed_mean', 15.0))
    trip_km = params.get('mean_trip_km')
    if trip_km is None or not np.isfinite(trip_km):
        trip_km = _expected_trip_km(params)
    return (params.get('pickup_overhead_mean', 0.0)
            + trip_km / speed * 60.0
            + max(1.0, params.get('service_mean', 5.0)))


def estimate_fleet_sizes(params, n_dedicated_default, n_crowd_max_default,
                         simulation_minutes=720, target_util=0.85):
    """Size the fleet so peak-hour demand runs at a target utilization.

    The number of simultaneously busy couriers required at the peak hour is

        required = lambda_peak [orders/min] * cycle_time [min] / target_util,

    where the cycle time is the per-order courier occupancy (pickup overhead
    + demand-weighted travel + doorstep service) and lambda_peak uses the
    calibrated arrival-rate scale.  The dedicated/crowd split follows the
    share estimated from the observed courier population; the crowd pool is
    additionally inflated by the peak-hour crowd availability so that the
    *available* crowd capacity matches the requirement.

    Falls back to the legacy observed-population estimate (scaled by the mean
    crowd availability) when arrival rates are unavailable.
    """
    rates = params.get('rates')
    if rates:
        hours = [sim_hour(i * 60) for i in range(simulation_minutes // 60)]
        scale = params.get('arrival_rate_scale', 1.0)
        hourly_totals = {h: sum(rates[s][h] for s in rates) for h in hours}
        peak_hour = max(hourly_totals, key=hourly_totals.get)
        lam_peak = hourly_totals[peak_hour] / 60.0 * scale
        cycle = effective_cycle_time_min(params)
        required = lam_peak * cycle / max(0.1, target_util)
        est_ded = params.get('n_dedicated_est', 0)
        est_crowd = params.get('n_crowd_est', 0)
        if est_ded + est_crowd > 0:
            share = est_ded / (est_ded + est_crowd)
        else:
            share = n_dedicated_default / max(1, n_dedicated_default + n_crowd_max_default)
        n_dedicated = max(1, int(round(required * share)))
        crowd_avail = params.get('crowd_avail', [0.5] * 24)
        avail_peak = max(0.05, float(crowd_avail[peak_hour]))
        n_crowd_max = max(0, int(round((required - n_dedicated) / avail_peak)))
        return n_dedicated, n_crowd_max

    # Legacy fallback: scale the observed courier population by the mean of
    # the hourly crowd availability curve.
    avail_mean = float(np.mean(params.get('crowd_avail', [0.5])))
    n_dedicated = int(params.get('n_dedicated_est', n_dedicated_default) * avail_mean)
    n_crowd_max = int(params.get('n_crowd_est', n_crowd_max_default) * avail_mean)
    return max(1, n_dedicated), max(0, n_crowd_max)


def empirical_kpis(data_dir=DATA_DIR, city='sh', sample=200000, ds_keep=None):
    """Compute empirical KPIs from the LaDe delivery file.

    The full file is read (three lightweight columns only) so the KPIs are
    exact; a file-head (``nrows``) read would bias every KPI toward the
    earliest days because LaDe files are chronologically ordered.  ``ds_keep``
    optionally restricts the computation to a set of `ds` day labels (used by
    the holdout-period validation).  ``sample`` is accepted for backward
    compatibility and ignored.
    """
    path = data_dir / f'delivery_{city}.csv'
    if not path.exists():
        return dict()
    df = pd.read_csv(path, usecols=['accept_time', 'delivery_time', 'ds'],
                     encoding='utf-8', low_memory=False)
    if ds_keep is not None:
        ds_keep = set(ds_keep)
        df = df[df['ds'].isin(ds_keep)]
    df['accept_time'] = pd.to_datetime(df['accept_time'], format='%m-%d %H:%M:%S', errors='coerce')
    df['delivery_time'] = pd.to_datetime(df['delivery_time'], format='%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['accept_time', 'delivery_time'])
    df['service_min'] = (df['delivery_time'] - df['accept_time']).dt.total_seconds() / 60.0
    df['service_min'] = df['service_min'].clip(lower=2.0, upper=180.0)
    df['hour'] = df['accept_time'].dt.hour
    n_days = max(1, df['ds'].nunique())
    hourly = (df.groupby('hour').size() / n_days).to_dict()
    # Fixed platform-level on-time threshold (matches config.ON_TIME_THRESHOLD_MIN;
    # kept literal here to avoid a circular import between base and config).
    ON_TIME_MIN = 120
    return dict(
        avg_delivery_time_min=float(df['service_min'].mean()),
        on_time_rate=float((df['service_min'] <= ON_TIME_MIN).mean()),
        max_delivery_time_min=float(df['service_min'].max()),
        median_delivery_time_min=float(df['service_min'].median()),
        p90_delivery_time_min=float(df['service_min'].quantile(0.9)),
        total_orders=int(len(df)),
        n_days=int(n_days),
        hourly_demand_mean=float(np.mean(list(hourly.values()))),
        hourly_demand_peak=int(max(hourly.values())) if hourly else 0,
    )


def calibrate_arrival_scale(params, emp, simulation_minutes, clip=(0.5, 2.0)):
    """Scale arrival rates so simulated order volume matches the empirical count.

    The expected raw number of arrivals over the simulation horizon is matched
    to the empirical daily order count scaled to the same horizon.  Returns the
    scale factor (also stored in ``params['arrival_rate_scale']``).
    """
    emp_daily = emp.get('total_orders', 0) / max(1, emp.get('n_days', 1))
    if emp_daily <= 0:
        return float(params.get('arrival_rate_scale', 1.0))
    sim_hours = [sim_hour(i * 60) for i in range(simulation_minutes // 60)]
    expected_raw = sum(sum(params['rates'][s][h] for s in params['stations'])
                       for h in sim_hours)
    target = emp_daily * len(sim_hours) / 24.0
    scale = target / max(1.0, expected_raw)
    scale = float(np.clip(scale, clip[0], clip[1]))
    params['arrival_rate_scale'] = scale
    return scale


def statistical_procedure(n_replications, base_seed=42):
    """Human- and machine-readable description of the statistical protocol.

    Recorded in metadata.json and the run log so that the randomization and
    testing procedure is reported unambiguously (reviewer minor comment 4).
    """
    seeds = list(range(base_seed, base_seed + n_replications))
    return {
        'n_replications': int(n_replications),
        'seed_rule': 'seeds are consecutive integers base_seed .. base_seed+n-1',
        'base_seed': int(base_seed),
        'seeds': seeds,
        'pairing': ('paired by identical random seed across policies, with '
                    'common random numbers: demand (arrivals, destinations, '
                    'deadlines and per-order pre-sampled dynamics) and '
                    'crowd-availability uniforms are drawn from independent '
                    'streams keyed only by the seed, so all policies face '
                    'identical demand/supply realisations; policy-internal '
                    'randomness uses a separate stream'),
        'test': 'Wilcoxon signed-rank test, two-sided',
        'multiple_testing': 'Holm-Bonferroni step-down correction within each metric',
        'significance_level': 0.05,
        'confidence_interval': ("95% CI on the mean; Student's t (df=n-1) for n<30, "
                                "z=1.96 for n>=30"),
    }


def mm_delay_bound(arrival_rate_per_min, service_rate_per_min, c):
    """Erlang-C expected waiting time for an M/M/c queue."""
    lam = arrival_rate_per_min
    mu = service_rate_per_min
    if c <= 0 or mu <= 0 or lam <= 0:
        return dict(rho=np.nan, expected_delay_min=np.nan,
                    expected_system_time_min=np.nan, stable=False)
    rho = lam / (c * mu)
    if rho >= 1.0:
        return dict(rho=rho, expected_delay_min=np.nan,
                    expected_system_time_min=np.nan, stable=False)
    A = lam / mu

    def erlang_c(A, m):
        # Log-domain Erlang-C to avoid overflow for large fleets.
        log_L = m * math.log(A) - math.lgamma(m + 1) + math.log(m / (m - A))
        terms = [i * math.log(A) - math.lgamma(i + 1) for i in range(m)]
        mx = max(terms)
        log_S = mx + math.log(sum(math.exp(t - mx) for t in terms))
        log_ratio = log_S - log_L
        return 0.0 if log_ratio > 700 else 1.0 / (1.0 + math.exp(log_ratio))

    C = erlang_c(A, c)
    Wq = C / (c * mu - lam)
    W = Wq + 1.0 / mu
    return dict(rho=rho, expected_delay_min=float(Wq),
                expected_system_time_min=float(W), stable=True)


def erlang_a_delay_bound(arrival_rate_per_min, service_rate_per_min, c, patience_rate_per_min=1 / 120.0):
    """Erlang-A (M/M/c+M) expected waiting time with customer impatience.

    Approximation from Garnett et al. (2002) for moderate abandonment.
    """
    lam = arrival_rate_per_min
    mu = service_rate_per_min
    theta = patience_rate_per_min
    if c <= 0 or mu <= 0 or lam <= 0:
        return dict(rho=np.nan, expected_delay_min=np.nan, stable=False)
    rho = lam / (c * mu)
    if rho >= 1.0:
        return dict(rho=rho, expected_delay_min=np.nan, stable=False)
    A = lam / mu
    # Halfin-Whitt regime approximation
    beta = (c - A) / math.sqrt(A) if A > 0 else 0.0
    alpha = theta / mu
    # Delay probability approximation
    from scipy import stats
    phi = stats.norm.pdf(beta)
    Phi = stats.norm.cdf(beta)
    # Approximate probability of positive delay
    Pw = (1 - beta * Phi / phi) if phi > 0 else 0.0
    Pw = max(0.0, min(1.0, Pw))
    # Expected waiting time conditional on not abandoning
    Wq = Pw / (c * mu * (1 - rho) + lam * (theta / (theta + mu)) + 1e-9)
    return dict(rho=rho, expected_delay_min=float(Wq), stable=True,
                abandonment_rate=theta, delay_probability=float(Pw))


def fluid_lower_bound(total_demand, capacity_couriers, horizon_min, speed_kmh, avg_trip_km, service_mean):
    """Simple fluid lower bound on total delivery time.

    ``capacity_couriers`` is the average number of simultaneously available
    couriers over the horizon (e.g. ``n_dedicated + n_crowd_max * mean_availability``).
    If all capacity were perfectly utilized, the minimum possible delay is
    bounded by travel time plus service time.
    """
    if capacity_couriers <= 0 or total_demand <= 0:
        return dict(min_delay_min=np.nan, min_total_min=np.nan,
                    capacity_total_min=np.nan, feasible=False)
    # Total courier-minutes available over the horizon.
    capacity_total_min = capacity_couriers * horizon_min
    travel_time_min = avg_trip_km / max(1.0, speed_kmh) * 60.0
    min_per_order = travel_time_min + service_mean
    min_total_min = total_demand * min_per_order
    feasible = min_total_min <= capacity_total_min
    return dict(min_delay_min=float(min_per_order),
                min_total_min=float(min_total_min),
                capacity_total_min=float(capacity_total_min),
                feasible=float(feasible))


def lp_relaxation_bound(orders, couriers, dist_fn, speed_kmh, service_mean):
    """Linear programming relaxation for a single-period assignment.

    This is a lightweight offline lower bound used to compare online
    policies.  A full implementation is available in validation.py.
    """
    try:
        from scipy.optimize import linprog
    except Exception:
        return dict(status='missing_solver', objective=np.nan)
    n_o = len(orders)
    n_c = len(couriers)
    if n_o == 0 or n_c == 0:
        return dict(status='trivial', objective=0.0)
    # Cost = travel time + service time
    costs = []
    for o in orders:
        for c in couriers:
            d = dist_fn(c['x'], c['y'], o['x'], o['y'])
            costs.append(d / speed_kmh * 60.0 + service_mean)
    # min c^T x, subject to each order assigned, each courier <= capacity
    from config import DEDICATED_CAPACITY, CROWD_CAPACITY
    caps = [DEDICATED_CAPACITY if c['kind'] == 'dedicated' else CROWD_CAPACITY
            for c in couriers]
    A_eq = np.zeros((n_o, n_o * n_c))
    for i in range(n_o):
        for j in range(n_c):
            A_eq[i, i * n_c + j] = 1.0
    b_eq = np.ones(n_o)
    A_ub = np.zeros((n_c, n_o * n_c))
    for j in range(n_c):
        for i in range(n_o):
            A_ub[j, i * n_c + j] = 1.0
    b_ub = np.array(caps, dtype=float)
    bounds = [(0, 1) for _ in range(n_o * n_c)]
    res = linprog(costs, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    return dict(status=res.status, objective=float(res.fun) if res.success else np.nan,
                n_orders=n_o, n_couriers=n_c)



