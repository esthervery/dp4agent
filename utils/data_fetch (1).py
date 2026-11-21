"""HF에서 databench질문 세트 + 테이블을 data/ 에 저장"""
import pathlib
import pandas as pd
from datasets import load_dataset

from huggingface_hub import snapshot_download

# 전체 데이터를 받아오는 함수 -> python -c "from data_fetch import fetch_all; fetch_all()"
def fetch_all():
    """저장소 전체 스냅샷 (all.parquet만)."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id="cardiffnlp/databench",
        repo_type="dataset",
        revision=REV,
        local_dir="data/databench",
        allow_patterns=["data/*/all.parquet"],
        # info 파일까지 저장하려면 allow_patterns=["data/*/all.parquet", "data/*/info.yml", "*.pdf"]
    )
    print(path)
    return path

# 메타데이터를 받아오는 함수
def fetch_meta():
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id="cardiffnlp/databench", repo_type="dataset", revision=REV,
        local_dir="data/databench",
        allow_patterns=["data/*/info.yml", "*.pdf", "README.md"],
    )
    print(path)

DATA = pathlib.Path("data"); DATA.mkdir(exist_ok=True)
REV = "main"   # 나중에 커밋 해시로 고정

def fetch_qa():
    for name, split in [("qa", "train"), ("semeval", "dev"), ("semeval", "train")]:
        try:
            ds = load_dataset("cardiffnlp/databench", name=name,
                              split=split, revision=REV)
            out = DATA / f"qa_{name}_{split}.parquet"
            ds.to_pandas().to_parquet(out)
            print(f"[qa] {out.name}: {len(ds)}행")
        except Exception as e:
            print(f"[qa] {name}/{split} 실패: {e}")

def fetch_table(dataset_id: str) -> pd.DataFrame:
    out = DATA / f"tbl_{dataset_id}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    url = f"hf://datasets/cardiffnlp/databench/data/{dataset_id}/all.parquet"
    df = pd.read_parquet(url)      # all.parquet (sample 아님)
    df.to_parquet(out)
    print(f"[table] {dataset_id}: {df.shape}")
    return df

if __name__ == "__main__":
    fetch_qa()