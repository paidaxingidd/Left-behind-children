"""Authors' internal step (not needed for the public data): extract FAD item 13 from the 2024 SPSS entry file
(各题得分原始分数 Z分（留守）.sav, 311 left-behind children) into b13_2024.json, keyed by questionnaire number.
The 2025 entry file (各题得分原始分数 Z分（新增）.sav) carries an erroneous re-entry of this item; see docs/measures.md.
Usage:  python3 00_extract_b13_2024.py "各题得分原始分数 Z分（留守）.sav"
Then copy the 2025 file to data_xinzeng.sav in this directory and run the pipeline with LBC_MODE=final."""
import sys, json, numpy as np
from savreader import read_sav

src = sys.argv[1] if len(sys.argv) > 1 else 'data_liushou2024.sav'
d, meta = read_sav(src)
num = np.asarray(d['num'], float); b13 = np.asarray(d['b13'], float)
out = {int(n): float(v) for n, v in zip(num, b13) if not np.isnan(n) and not np.isnan(v) and 1 <= v <= 4}
json.dump(out, open('b13_2024.json', 'w'), indent=0)
print(f'{len(out)} values of item 13 written to b13_2024.json from {src}')
