"""DataBench (cardiffnlp/databench) 전체 다운로드.

테이블 80종(all.parquet) + QA(qa/semeval) + info.yml 을 받아
data/ 아래에 실험용으로 정리한다.

사용법
    python data_fetch.py all        # 전부 (기본값)
    python data_fetch.py tables     # 테이블 80종만
    python data_fetch.py qa         # QA 만
    python data_fetch.py verify     # 내려받은 것 검증만

결과 트리
    data/
      databench/data/<id>/all.parquet      원본 그대로 (snapshot)
      databench/data/<id>/qa.parquet
      databench/data/<id>/info.yml
      tables/<id>.parquet                  80개, all.parquet 사본
      qa/qa_train.parquet
      qa/semeval_train.parquet
      qa/semeval_dev.parquet
      qa/semeval_test.parquet
      manifest.csv                         (id, rows, cols, columns)

주의
  - REV 는 커밋 해시로 고정하는 것을 권장. "main" 은 재현성 없음.
  - all.parquet = 전체 행. sample.parquet = 축소판(실험에 쓰지 말 것).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import pandas as pd

REPO_ID = "cardiffnlp/databench"
REPO_TYPE = "dataset"
REV = os.environ.get("DATABENCH_REV", "main")  # 커밋 해시로 고정 권장

ROOT = pathlib.Path("data")
SNAP = ROOT / "databench"
TABLES = ROOT / "tables"
QADIR = ROOT / "qa"

QA_CONFIGS = [
    ("qa", "train"),
    ("semeval", "train"),
    ("semeval", "dev"),
    ("semeval", "test"),
]


# ---------------------------------------------------------------- tables
def fetch_tables(force: bool = False) -> list[str]:
    """테이블 80종 + info.yml 스냅샷. 이미 있으면 건너뛴다."""
    from huggingface_hub import snapshot_download

    TABLES.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=REV,
        local_dir=str(SNAP),
        allow_patterns=["data/*/all.parquet", "data/*/info.yml"],
        force_download=force,
    )
    print(f"[snapshot] {path}")

    ids = []
    for d in sorted((SNAP / "data").iterdir()):
        src = d / "all.parquet"
        if not src.is_file():
            print(f"[skip] {d.name}: all.parquet 없음", file=sys.stderr)
            continue
        dst = TABLES / f"{d.name}.parquet"
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            pd.read_parquet(src).to_parquet(dst, index=False)
        ids.append(d.name)

    print(f"[tables] {len(ids)}개 → {TABLES}/")
    if len(ids) != 80:
        print(f"[warn] 테이블이 80개가 아닙니다: {len(ids)}개", file=sys.stderr)
    return ids


# -------------------------------------------------------------------- qa
def fetch_qa(force: bool = False) -> None:
    """QA 4종(qa/train, semeval/train|dev|test)을 parquet으로 저장."""
    from datasets import load_dataset

    QADIR.mkdir(parents=True, exist_ok=True)
    for name, split in QA_CONFIGS:
        out = QADIR / f"{name}_{split}.parquet"
        if out.exists() and not force:
            n = len(pd.read_parquet(out))
            print(f"[qa] {out.name}: 이미 있음 ({n}행)")
            continue
        try:
            ds = load_dataset(REPO_ID, name=name, split=split, revision=REV)
            df = ds.to_pandas()
            df.to_parquet(out, index=False)
            print(f"[qa] {out.name}: {len(df)}행")
        except Exception as e:  # noqa: BLE001
            print(f"[qa] {name}/{split} 실패: {type(e).__name__}: {e}", file=sys.stderr)


# ---------------------------------------------------------------- verify
def verify() -> bool:
    """개수·스키마·결측을 확인하고 manifest.csv 를 쓴다."""
    ok = True

    # 테이블
    files = sorted(TABLES.glob("*.parquet"))
    print(f"\n[verify] tables: {len(files)}개 (기대 80)")
    if len(files) != 80:
        ok = False

    rows = []
    for f in files:
        df = pd.read_parquet(f)
        rows.append(
            {
                "dataset": f.stem,
                "rows": len(df),
                "cols": df.shape[1],
                "columns": " | ".join(map(str, df.columns)),
            }
        )
    if rows:
        man = pd.DataFrame(rows)
        man.to_csv(ROOT / "manifest.csv", index=False, encoding="utf-8-sig")
        print(f"  총 행수 {man['rows'].sum():,} · 평균 {man['rows'].mean():.0f}행 · "
              f"컬럼 {man['cols'].min()}~{man['cols'].max()}개")
        print(f"  manifest → {ROOT / 'manifest.csv'}")
        empty = man[man["rows"] == 0]
        if len(empty):
            print(f"  [warn] 빈 테이블: {list(empty['dataset'])}", file=sys.stderr)
            ok = False

    # QA
    print(f"\n[verify] qa:")
    for name, split in QA_CONFIGS:
        f = QADIR / f"{name}_{split}.parquet"
        if not f.exists():
            print(f"  {f.name}: 없음", file=sys.stderr)
            ok = False
            continue
        df = pd.read_parquet(f)
        need = {"dataset", "question", "answer"}
        miss = need - set(df.columns)
        flag = "" if not miss else f"  [warn] 컬럼 누락 {miss}"
        print(f"  {f.name}: {len(df):>5}행 · 컬럼 {list(df.columns)}{flag}")
        if miss:
            ok = False
        # QA의 dataset 값이 실제 테이블과 맞는지
        if "dataset" in df.columns and files:
            unknown = set(df["dataset"].unique()) - {f.stem for f in files}
            if unknown:
                print(f"    [warn] 테이블 없는 dataset 참조: {sorted(unknown)}", file=sys.stderr)

    print(f"\n[verify] {'통과' if ok else '문제 있음'}")
    return ok


# ------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser(description="DataBench 전체 다운로드")
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["all", "tables", "qa", "verify"])
    ap.add_argument("--force", action="store_true", help="이미 받은 파일도 다시 받기")
    a = ap.parse_args()

    ROOT.mkdir(exist_ok=True)
    if a.cmd in ("all", "tables"):
        fetch_tables(force=a.force)
    if a.cmd in ("all", "qa"):
        fetch_qa(force=a.force)
    if a.cmd in ("all", "verify"):
        sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()