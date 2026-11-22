# -*- coding: utf-8 -*-
"""파일럿 CSV 를 '다시 채점'한다 — LLM 재호출 없이.

저장된 `raw`(응답 원문)만 있으면 후처리 조합을 바꿔가며 실행률·정답률을
재계산할 수 있다. API 호출 0회, 비용 0.

사용:
    # lab/ 에서
    python rescore_pilot.py result/pilot_A_245.csv
    python rescore_pilot.py result/pilot_A_245.csv --tables ../data/tables

출력: 콘솔 표 + result/rescore_<원본이름>.csv
"""
from __future__ import annotations
import argparse, os, pathlib, re, sys, textwrap

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp1_lib as L  # noqa: E402


# ---------------------------------------------------------------- 변형 정의
def _pp_ok():
    return L._pp()


def body_from(raw: str, *, strip_fence: bool, truncate: bool) -> str:
    """응답 → 본문. 펜스 제거 / return 절단을 각각 켜고 끌 수 있다."""
    lines = str(raw).split("\n")
    if strip_fence:
        lines = [l for l in lines if not re.match(r"^\s*```", l)]
    out = []
    for l in lines:
        out.append(l)
        if truncate and re.match(r"^\s*return\b", l):
            break
    return "\n".join(out).strip("\n")


def src_pp(body: str, columns: list[str]) -> str:
    """현행: PP 1회 (업스트림과 동일)."""
    return (f"def answer(df: pd.DataFrame):\n"
            f"    df.columns = {list(columns)!r}\n"
            f"{_pp_ok()(body)}\n")


def src_no_pp(body: str, columns: list[str]) -> str:
    """후처리 없음: 본문을 4칸으로 감싸기만."""
    return (f"def answer(df: pd.DataFrame):\n"
            f"    df.columns = {list(columns)!r}\n"
            f"{textwrap.indent(textwrap.dedent(body), '    ')}\n")


def src_dedent_pp(body: str, columns: list[str]) -> str:
    """이중 보정(이전 버전): dedent+4칸 후 PP."""
    return src_pp(textwrap.indent(textwrap.dedent(body), "    ").strip("\n"), columns)


VARIANTS = {
    "현행 (PP, 펜스제거, return절단)": dict(fence=True, trunc=True, src=src_pp),
    "PP 없음": dict(fence=True, trunc=True, src=src_no_pp),
    "return 절단 없음": dict(fence=True, trunc=False, src=src_pp),
    "PP 없음 + 절단 없음": dict(fence=True, trunc=False, src=src_no_pp),
    "이중보정(dedent+PP)": dict(fence=True, trunc=True, src=src_dedent_pp),
    "펜스제거 안 함": dict(fence=False, trunc=True, src=src_pp),
}


def run_variant(body: str, df: pd.DataFrame, cols: list[str], src_fn):
    if not body.strip():
        return None, "EmptyBody"
    try:
        code = src_fn(body, cols)
        ns: dict = {"pd": pd, "np": np, "re": re}
        exec(compile(code, "<rescore>", "exec"), ns)  # noqa: S102
        v = ns["answer"](df.copy())
        if isinstance(v, (pd.Series, np.ndarray, list, tuple)):
            v = v[0] if len(v) else None
        if isinstance(v, np.generic):
            v = v.item()
        return v, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--tables", default=os.environ.get("DATABENCH_TABLES", ""))
    args = ap.parse_args()

    if args.tables:
        L.DATA = pathlib.Path(args.tables)
    src_csv = pathlib.Path(args.csv)
    d = pd.read_csv(src_csv, encoding="utf-8-sig")
    if "raw" not in d:
        raise SystemExit("raw 열이 없습니다 — 저장된 응답이 있어야 재채점할 수 있습니다.")
    ok_call = d[d.get("call_error", pd.Series([""] * len(d))).fillna("") == ""].copy()
    print(f"[입력] {src_csv.name} · 전체 {len(d)} · 호출성공 {len(ok_call)}\n")

    tables: dict[str, pd.DataFrame] = {}
    rows = []
    for _, r in ok_call.iterrows():
        ds = r["dataset"]
        if ds not in tables:
            tables[ds] = L._load_table(ds)
        df, cols = tables[ds], list(tables[ds].columns)
        ref = r.get("answer_ref")
        for name, cfg in VARIANTS.items():
            body = body_from(r["raw"], strip_fence=cfg["fence"], truncate=cfg["trunc"])
            v, err = run_variant(body, df, cols, cfg["src"])
            rows.append({
                "variant": name, "n": r["n"], "rep": r["rep"],
                "condition": r.get("condition"),
                "exec_ok": err is None, "err": err or "",
                "answer": v, "answer_ref": ref,
                "correct": bool(L.close(v, ref)),
                "null": v is None,
            })
    o = pd.DataFrame(rows)

    def blk(g):
        return pd.Series({
            "호출 수": len(g),
            "실행 성공률 %": round(g["exec_ok"].mean() * 100, 1),
            "값 None %": round(g["null"].mean() * 100, 1),
            "정답률 %": round(g["correct"].mean() * 100, 1),
            "정답 수": int(g["correct"].sum()),
            "IndentationError": int(g["err"].str.startswith("IndentationError").sum()),
            "SyntaxError": int(g["err"].str.startswith("SyntaxError").sum()),
            "기타 실행오류": int(((~g["exec_ok"]) & ~g["err"].str.startswith(("IndentationError", "SyntaxError"))).sum()),
        })

    summ = o.groupby("variant", sort=False).apply(blk, include_groups=False)
    print(summ.to_string())

    out = src_csv.parent / f"rescore_{src_csv.stem}.csv"
    o.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    main()
