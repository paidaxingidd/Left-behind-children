# Values that round differently when the analyses are re-run from the public file

The article's numbers come from the authors' full data file. The public file top-codes the number of siblings at 5 (a predictor in the imputation model) and dichotomises the migration variables, so imputed values differ slightly. Of the 1,732 numbers that the pipeline regenerates for Tables 1–4 and S1–S9, the following round differently in the second decimal (all others are identical at two decimals; chi-square values differ by at most 0.41). Generated on 2026-09-30 by comparing `manuscript_tables.json` (authors' run) with `public_manuscript_tables.json` (public run).

| Where | Quantity | Article (authors' run) | Public re-run |
|---|---|---|---|
| Table 1 | `table1.lbc.sib_m` | 3.1848 | 3.1564 |
| Table 1 | `table1.lbc.sib_sd` | 1.1088 | 1.0404 |
| Table 1 | `table1.non.sib_m` | 3.4071 | 3.3274 |
| Table 1 | `table1.non.sib_sd` | 1.3068 | 1.1530 |
| Table 2 | `table2.M[0]` | 90.61 | 90.60 |
| Table 2 | `table2.SD[2]` | 3.6659 | 3.6648 |
| Table S4 / Method (CFA fit) | `S4.fad5.chi2` | 775.70 | 775.60 |
| Table S4 / Method (CFA fit) | `S4.fad1.chi2` | 1557.50 | 1557.10 |
| Method (reliability) | `reliability.pc.omega` | 0.6752 | 0.6749 |
| Table S5 | `S5.one.chi2` | 4555.86 | 4555.57 |
| Table S5 | `S5.nine.chi2` | 2161.07 | 2161.02 |
| Table S5 | `S5.nine_method.chi2` | 1894.31 | 1894.30 |
| Table 4 | `table4_desc[fad].sd0` | 13.46 | 13.45 |
| Table 4 | `table4_desc[pc].sd0` | 3.8464 | 3.8389 |
| Table 4 | `table4_desc[pc].d` | -0.2342 | -0.2356 |
| Table S6 | `S6_A[E2 Siblings + age + grade FE].c` | 0.3055 | 0.3046 |
| Table S6 note | `S6_F2_fit.fit_pooled.chi2` | 1465.62 | 1465.69 |
| Table S7 | `S7[a_es].p` | 0.0952 | 0.0948 |
| Table S7 | `S7[jointdiff_pc_es].p` | 0.9572 | 0.9548 |
| Table S8 | `S8[C1 + age x LS, age x FF, age x LBC].a_non_lo` | 0.1244 | 0.1251 |
| Table S8 | `S8[C1 + age x LS, age x FF, age x LBC].a_diff_p` | 0.3224 | 0.3252 |
| Table S8 | `S8[C1 + age x LS, age x FF, age x LBC].b_non_lo` | 0.2350 | 0.2349 |
| Table S8 | `S8[C1 + age x LS, age x FF, age x LBC].cp_diff_p` | 0.9072 | 0.9048 |
| Table S8 | `S8[C2 Grade fixed effects].b_non_lo` | 0.2451 | 0.2449 |
| Table S8 | `S8[C2 Grade fixed effects].cp_LBC_lo` | 0.1450 | 0.1450 |
| Table S8 | `S8[C2 Grade fixed effects].cp_diff_p` | 0.7552 | 0.7544 |
| Table S8 | `S8[C3 Grade 6 only].a_diff_p` | 0.6928 | 0.6976 |
| Table S8 | `S8[C3 Grade 6 only].cp_non_p` | 0.0344 | 0.0352 |
| Table S8 | `S8[C3b Grades 5-6 only].a_diff` | 0.0957 | 0.0948 |
| Table S8 | `S8[C3b Grades 5-6 only].b_diff_p` | 0.0152 | 0.0148 |
| Table S8 | `S8[C4 LBC weighted to non-LBC grade distribution].a_diff_lo` | -0.0949 | -0.0956 |
| Table S8 | `S8[C4 LBC weighted to non-LBC grade distribution].a_diff_p` | 0.2848 | 0.2876 |
| Table S8 | `S8[C5 Within-group standardized (separate models)].cp_LBC_hi` | 0.3451 | 0.3448 |
| Table S9 | `A3.none.fit.chi2` | 775.70 | 775.60 |
| Table S9 | `A3.neg.fit.chi2` | 689.08 | 689.01 |
| Table S9 | `A3.neg.factor_corr.PC-PS` | 0.6900 | 0.7000 |
| Table S9 | `A3.both.fit.chi2` | 587.69 | 587.62 |
| Table S9 | `A3.both.pos_method_loadings[0]` | -0.3500 | -0.3600 |
| Table S9 | `A3_wording.one_factor.chi2` | 1557.50 | 1557.10 |
| Table S9 | `A3_wording.two_wording_factors.chi2` | 968.22 | 967.83 |
| Table S9 | `A3_riifa_fit.chi2` | 765.08 | 765.13 |

Notes: `table1.*.sib_m` / `sib_sd` differ because the public file top-codes the number of siblings; the "Excluding separations of ≤ 6 months" row of Table S3 is not reproducible from the public file (duration of separation is dichotomised).