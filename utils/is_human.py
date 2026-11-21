"""
databench에서 데이터셋의 한 행이 개인인지 판정할 단서를 모음.
event_hit이 비어 있고 person_hit이 채워진 것들이 1차 후보 데이터셋
output: outputs/unit_check.csv
"""

import pathlib, re
import pandas as pd

# 행이 사람이 아님을 시사하는 단어 (거래·사건·게시물 단위)
EVENT = r"invoice|transaction|incident|order|trip|tweet|review|comment|post|" \
        r"claim|flight|game|match|visit|session|log|event|call|ticket"
# 행이 사람임을 시사하는 단어
PERSON = r"age|gender|sex|birth|salary|income|wage|weight|height|bmi|" \
         r"glucose|cholesterol|customer|employee|patient|respondent|student"

rows = []
for p in sorted(pathlib.Path("data/databench/data").glob("*/all.parquet")):
    df = pd.read_parquet(p)
    cols = " ".join(df.columns).lower()

    # 중복 판정: id 계열 컬럼이 고유한가
    dup = ""
    for c in df.columns:
        if re.fullmatch(r"(id|.*_id|.*id)", c.lower()) and df[c].nunique() == len(df):
            dup = c
            break

    rows.append({
        "dataset": p.parent.name,
        "n": len(df),
        "person_hit": ", ".join(sorted(set(re.findall(PERSON, cols)))[:5]),
        "event_hit": ", ".join(sorted(set(re.findall(EVENT, cols)))[:5]),
        "unique_id": dup,
        "all_cols": " | ".join(df.columns[:15]),
    })

out = pd.DataFrame(rows).sort_values("n", ascending=False)
out.to_csv("outputs/unit_check.csv", index=False, encoding="utf-8-sig")
print(out[["dataset", "n", "person_hit", "event_hit", "unique_id"]].to_string(index=False))
