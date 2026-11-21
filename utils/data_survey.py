"""
databench 데이터셋 테이블(80개) 전체를 조사함
각 테이블마다 조사: 이름, 행 수, 열 수, 수치형 열(값이 숫자로 저장된 열) 수, 그리고 수치형 열 이름(col_name) 몇 개 
실험 자체가 불가능한 데이터(행 수가 너무 적을 때 등)를 걸러내는 용도
HF README의 데이터셋 목록를 우선 확인할 것 (https://huggingface.co/datasets/cardiffnlp/databench)
output: outputs/survey.csv
"""

import pathlib
import pandas as pd
import pathlib
pathlib.Path("outputs").mkdir(exist_ok=True)

rows = []
for p in sorted(pathlib.Path("data/databench/data").glob("*/all.parquet")):
    df = pd.read_parquet(p)
    num = df.select_dtypes("number").columns
    rows.append({
        "dataset": p.parent.name,
        "n": len(df),
        "cols": df.shape[1],
        "num_cols": len(num),
        "numeric": ", ".join(num[:6]),
    })

s = pd.DataFrame(rows).sort_values("n", ascending=False)
s.to_csv("outputs/survey.csv", index=False, encoding="utf-8-sig")
print(s.to_string(index=False))
