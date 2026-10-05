Analysis pipeline (Python >= 3.10; numpy, scipy, pandas, openpyxl). Run from this directory:
  python3 step1_validate.py      # multiple imputation (m = 20) + Tables 2 and 3 of the article
  python3 step2_measurement.py   # measurement / covariate sensitivity analyses; FAD CFA with wording method factor
  python3 step2b_riifa.py        # FAD random-intercept (acquiescence) model
  python3 step3_h3.py            # Hypothesis 3: joint five-dimension model, composites, equivalence intervals
  python3 step4_lbc_compare.py   # left-behind vs non-left-behind comparison and its robustness checks
  python3 step5_latent.py        # reliability-corrected and latent-variable SEM versions (bootstrap ~10 min)
  python3 step6_report.py        # collects all results into an Excel workbook
  python3 step7_manuscript_tables.py  # every number in the article's tables (Tables 1-4, S1-S9, reliabilities, text statistics) -> public_manuscript_tables.json (~8 min)
Default input: ../../data/analysis_data.csv (LBC_MODE=public). Output files are written to this directory with the prefix "public_".
Reproducibility: the public file top-codes the number of siblings (which enters the imputation model) and dichotomises the
migration variables, so imputed values differ slightly from the authors' run; every estimate agrees with the article to within
+/-0.005 (a few values round differently in the second decimal; sibling descriptives in Table 1 reflect the top-coding and
chi-square values differ by up to 0.41; see ../../docs/reproducibility_differences.md), and the "<= 6 months" row of Table S3 cannot be reproduced.
sem.py is a small maximum-likelihood SEM engine (RAM notation) written for this project; fit statistics follow the usual SEM-software
conventions (chi-square = (N-1)*F_ML). savreader.py reads SPSS .sav files and is only needed for the authors' internal files.
Authors' run (full data, LBC_MODE=final): place the 2025 entry file as data_xinzeng.sav in this directory, run
  python3 00_extract_b13_2024.py "<2024 entry file>.sav"   (writes b13_2024.json)
and then the steps above with LBC_MODE=final (output files without prefix). This is the run the article reports.
