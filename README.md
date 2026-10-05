# Family functioning, life satisfaction and learning engagement among rural left-behind children in China — data and code

This repository contains the de-identified data and the analysis code for the article
*Family functioning, life satisfaction, and learning engagement among rural left-behind children: a mediation study with a classmate reference group* by Weiyao Xiao, Ting Lan and Jiayin Xing (journal details to be added on publication).

## Contents
```
data/
  entry_2025_all_cases.csv    535 children, all questionnaire items as entered in 2025 (raw; contains out-of-range
                              values and an erroneous entry of FAD item 13 — see docs/measures.md)
  entry_2024_lbc_subset.csv   311 left-behind children, an earlier (2024) entry of the same questionnaires
  analysis_data.csv           analysis-ready file built from the two entry files by code/01_build_analysis_data.py
  codebook.csv                variable names, labels and codes
docs/
  measures.md                 instruments, item–subscale mapping, scoring, data notes
  reproducibility_differences.md  cells that round differently when re-run from the public file
code/
  01_build_analysis_data.py   raw entry files -> analysis_data.csv (all cleaning rules, documented in the script)
  python/                     analysis pipeline used for the reported results (multiple imputation, mediation
                              models, bootstrap, CFA/SEM); run in order step1 … step7 (see python/README.txt).
                              It also holds the aggregate output of two runs: files without a prefix come from
                              the authors' run on the full data files (the numbers reported in the article);
                              files prefixed public_ come from a re-run on data/analysis_data.csv.
LICENSE_DATA.txt              CC BY 4.0 (data and documentation)
LICENSE_CODE.txt              MIT (code)
```

## Sample
Children in Grades 3–6 of four rural primary schools in Yudu County, Jiangxi Province, China, surveyed in class in June 2023 (N = 535; 422 self-identified left-behind children and 113 classmates who were not left behind). The data were collected by the research team of the second author as part of the project ‘Understanding Social-Emotional and Behavioral Outcomes in Rural Left-Behind Children: The Perspective of Family–School Cooperation’ (2023 continuing project of the 14th Five-Year Plan for Education Science in Jiangxi Province, grant 23QN062). Written parental consent and child assent were obtained; the consent form stated that de-identified data may be shared for research. Ethics approval: Ethics Committee of Nanchang Normal University (approval No. 2023QN013).

## De-identification
Names were never recorded. The original questionnaire numbers have been replaced by consecutive IDs (the same IDs link the three data files); school is not recorded in the data; the three migration items are given only as the binary variables used in the analyses; number of children in the family and birth order have been top-coded; caregiver codes not on the questionnaire have been blanked. No item responses have been altered.

## Reproducing the results
1. `python code/01_build_analysis_data.py` rebuilds `data/analysis_data.csv` from the two entry files (already included).
2. `cd code/python && python step1_validate.py` reproduces Tables 2 and 3 of the article; `step2 … step5` produce the supplementary analyses, `step6_report.py` collects them into an Excel workbook, and `step7_manuscript_tables.py` regenerates every number in Tables 1–4 and S1–S9 (`public_manuscript_tables.json`). Python ≥ 3.10 with numpy, scipy, pandas, openpyxl; `LBC_MODE=public` is the default. Bootstrap resampling uses fixed seeds.
3. Reproducibility note: to protect participants, the public file top-codes the number of siblings (5 = five or more) and reports migration type, duration of separation and caregiver as binary variables. Because the number of siblings is a predictor in the imputation model, the imputed values differ slightly from those in the authors' run; every estimate in the article is reproduced to within ±0.005 (a few values round differently in the second decimal), except that the sibling descriptives in Table 1 reflect the top-coding and chi-square values differ by up to 0.41; the migration rows of Table 1 are reproduced in their dichotomised form, and the '≤ 6 months' sensitivity row of Table S3 cannot be reproduced from the public file. The cells that round differently are listed in `docs/reproducibility_differences.md`.

## How to cite
Xiao, W., Lan, T., & Xing, J. (2026). Data and code for "Family functioning, life satisfaction, and learning engagement among rural left-behind children" [Data set and code]. GitHub. https://github.com/paidaxingidd/Left-behind-children

## Contact
Weiyao Xiao, School of Education, Nanchang Normal University, Nanchang, China (xiaoweiyao@ncnu.edu.cn)
