"""Build the analysis-ready dataset from the two data-entry files.

Input  : data/entry_2025_all_cases.csv   (535 children, all items as entered in 2025)
         data/entry_2024_lbc_subset.csv   (311 left-behind children, earlier entry of the same questionnaires)
Output : data/analysis_data.csv

Cleaning rules (as described in the manuscript's Method section):
  1. Item responses outside the response scale (FAD 1-4; SWLS and UWES-S 1-7) are set to missing.
  2. FAD item 13: the 2025 entry of this item is unusable (it disagrees with the 2024 entry for 202 of the
     311 children who appear in both files and correlates only weakly with its subscale). The 2024 values are
     used for those 311 children; the item is left missing for everyone else (imputed in the analyses).
  3. Two children carried an invalid code (3) for 'parents divorced' in the 2025 entry; the 2024 entry has 2
     (= not divorced) for both, which is used.
  4. Children who reported not being left behind skipped the three migration items; those cells are blank.
  Demographic variables are already coarsened in the entry files (see docs/measures.md).
Scale scores in this file are complete-case sums (blank if any item is missing); the analysis scripts recompute
them within each imputed dataset.
"""
import numpy as np, pandas as pd, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

FAD = [f'b{i}' for i in range(1, 31)]; SWLS = [f'c{i}' for i in range(1, 6)]; UWES = [f'f{i}' for i in range(1, 18)]
new = pd.read_csv('data/entry_2025_all_cases.csv'); old = pd.read_csv('data/entry_2024_lbc_subset.csv')
d = new.copy()

# 1. out-of-range responses -> missing
for c in FAD:
    d.loc[(d[c] < 1) | (d[c] > 4), c] = np.nan
for c in SWLS + UWES:
    d.loc[(d[c] < 1) | (d[c] > 7), c] = np.nan

# 2. item 13 from the 2024 entry
b13_2024 = old.set_index('id')['b13']
d['b13_source'] = np.where(d.id.isin(b13_2024.index), '2024 entry', 'missing (imputed in analyses)')
d['b13'] = d.id.map(b13_2024)
d.loc[(d.b13 < 1) | (d.b13 > 4), 'b13'] = np.nan

# 3. invalid divorce code corrected from the 2024 entry
div_2024 = old.set_index('id')['parents_divorced']
bad = ~d.parents_divorced.isin([1, 2])
d.loc[bad, 'parents_divorced'] = d.loc[bad, 'id'].map(div_2024)
assert d.parents_divorced.isin([1, 2]).all()

# derived analysis variables
d['lbc'] = (d.lbc_self_report == 1).astype(int)
d['girl'] = (d.gender == 2).astype(int)
d['divorced'] = (d.parents_divorced == 1).astype(int)
d['grandparent'] = d.caregiver_grandparent

# complete-case scale scores (blank if any item missing)
sub = {'fad_total': FAD, 'affective_interaction': FAD[0:8], 'positive_communication': FAD[8:13], 'self_centredness_rev': FAD[13:19],
       'problem_solving': FAD[19:25], 'family_rules': FAD[25:30], 'life_satisfaction': SWLS, 'engagement_total': UWES,
       'vigour': UWES[0:6], 'dedication': UWES[6:11], 'absorption': UWES[11:17]}
for k, its in sub.items():
    d[k] = d[its].sum(axis=1, min_count=len(its))

cols = ['id', 'lbc', 'girl', 'age', 'grade', 'divorced', 'grandparent', 'both_parents_migrated', 'separation_over_5y',
        'n_siblings', 'birth_order', 'b13_source'] + FAD + SWLS + UWES + list(sub)
out = d[cols].copy()
for c in out.columns:
    if c != 'b13_source':
        out[c] = out[c].astype('Int64')
out.to_csv('data/analysis_data.csv', index=False)
print('analysis_data.csv:', d.shape[0], 'rows;', int(d.lbc.sum()), 'left-behind;', 'item 13 from 2024 entry for', int((d.b13_source == '2024 entry').sum()), 'children')
print('missing item cells (52 items):', int(d[FAD + SWLS + UWES].isna().sum().sum()), '(of which b13:', int(d.b13.isna().sum()), ')')
