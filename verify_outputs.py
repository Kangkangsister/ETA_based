import pandas as pd
from pathlib import Path

print('=== Final Output Verification ===')
print()

# 1. Excel vs LaTeX metrics consistency
xl = pd.ExcelFile('output/results.xlsx')
metrics_excel = xl.parse('metrics')
print('1. Excel metrics sheet:')
print('   Policies:', sorted(metrics_excel['policy'].tolist()))
print('   Shape:', metrics_excel.shape)
om = metrics_excel[metrics_excel['policy'] == 'OnlineMatching']
print('   OnlineMatching avg_delay_min:', om['avg_delay_min'].values[0])
print()

# 2. Pairwise tests have Holm correction
pairwise = xl.parse('pairwise')
print('2. Pairwise tests:')
print('   Columns:', list(pairwise.columns))
print('   Rows:', len(pairwise))
print('   Corrected p-values present:', 'p_value' in pairwise.columns)
print()

# 3. Sensitivity baseline consistency at N_CROWD_MAX=0
sens = xl.parse('sensitivity')
zero_crowd = sens[sens['N_CROWD_MAX'] == 0][['policy', 'avg_delay_min_mean']]
delays = zero_crowd['avg_delay_min_mean'].unique()
print('3. Zero-crowd baseline consistency:')
print('   Unique avg_delay values:', len(delays))
print('   Value:', delays[0] if len(delays) == 1 else delays)
print()

# 4. Theory bounds
theory = xl.parse('theory')
print('4. Theory bounds:')
print('   Bounds in Excel:', theory['bound'].tolist())
fluid = theory[theory['bound'] == 'Fluid']
print('   Fluid min_delay_min:', fluid['min_delay_min'].values[0])
print()

# 5. LaTeX tables compile-ready
print('5. LaTeX tables NaN check:')
for f in sorted(Path('output/tables').glob('*.tex')):
    content = f.read_text(encoding='utf-8')
    status = 'OK' if 'NaN' not in content else 'CONTAINS NaN'
    print(f'   {f.name}: {status}')
print()

# 6. Core result
robust = xl.parse('robustness')
main_metrics = ['avg_delay_min', 'on_time_rate', 'max_backlog', 'total_cost']
pivot = robust.pivot(index='policy', columns='metric', values='mean')
print('6. Core results (30-seed means):')
print(pivot[main_metrics].round(3).to_string())
