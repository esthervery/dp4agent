# -*- coding: utf-8 -*-
"""추가 실험 1 — 공통 모듈 (파일럿 + 본 실험).

설계 확정 사항
  · 프롬프트 : P2 = TD + CD + SR (AILS 템플릿 그대로, 예시 9개)
  · 호출     : one-call, 에러 픽싱 없음 (실행 실패는 실패로 기록)
  · 온도     : 기본 0.2 (본 실험에서 0 / 0.2 / 1.0 sweep)
  · 반복     : 문항 우선(question-major) — 같은 문항의 N회를 연속 실행

이 모듈이 하는 일
  build_prompt(...)   : 문항 1건 → P2 프롬프트 + HARNESS(SamEval의 클로드 코드 프리필 대체 지시문)
  call_llm(...)       : OpenAI Chat Completions 1회 호출 (finish_reason 수집)
  extract_body(...)   : 응답에서 함수 본문 추출
  run_code(...)       : 본문 실행 → 반환값
  signature(...)      : 생성 코드의 (컬럼, 필터, 집계) 서명
  score_one(...)      : 정답 비교
  run_pilot(...)      : 파일럿 실행 + CSV 저장

의존: core/ (AILS 저장소), pandas, openai, python-dotenv
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import sys
import textwrap
import time
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 설정
MODEL_DEFAULT = "gpt-6-luna"
TEMP_DEFAULT = 0.2
MAXTOK_DEFAULT = 300          # 파일럿 B(추론 끔). A는 2000
TIMEOUT_S = 120

# 파일럿 갈래
#   A : reasoning_effort low/medium/high, T=1.0 고정, max_completion_tokens=2000
#   B : reasoning_effort none,          T=0/0.2/1.0,  max_completion_tokens=300
PILOT_A = dict(efforts=["low", "medium", "high"], temps=[1.0], max_tokens=2000)
PILOT_B = dict(efforts=["none"],                  temps=[0.0, 0.2, 1.0], max_tokens=300)

EXEMPLAR_INDICES = [
    (17, 140), (0, 132), (28, 286), (31, 303), (4, 246),
    (24, 175), (20, 176), (8, 141), (14, 12),
]

ROOT = pathlib.Path(__file__).resolve().parent          # lab/
REPO = ROOT.parent                                      # dp4agent/ (core/ 가 있는 곳)
if (REPO / "core").exists():
    sys.path.insert(0, str(REPO))

DATA = ROOT / "data" / "tables"

RESULT_DIR = ROOT / "result"                     # 실험 산출물 저장 위치
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _out(path) -> pathlib.Path:
    """산출물 경로를 lab/result/ 아래로 정규화한다(상대 이름만 주면 여기로 저장)."""
    p = pathlib.Path(path)
    if p.parent in (pathlib.Path("."), ROOT):
        p = RESULT_DIR / p.name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------- 하네스 (Claude 프리필 대체)
# 업스트림은 지시문 없이 Claude 어시스턴트 프리필만 사용:
# ClaudeMessageEmbeddingPromptGenerator.__call__ -> chat.add_assistant_message(txt[:-1])
# 아래 문장이 프리필이 하던 역할을 대신한다. 프리필 방식이 이미 보장하던 사실만 적는다.
HARNESS_VERSION = "prefill_emul_v2"

HARNESS = (
    "\n"
    "# Continue the code above from the exact point where it stops.\n"
    "# The line above is unfinished: complete the column list first, "
    "then write the body of the function.\n"
    "# Start the body at the indentation of the line above (4 spaces).\n"
    "# Do not open a new function: continue the current one and do not restart with `def`.\n"
    "# End with a single `return` statement.\n"
)

_PP = None


def _pp():
    """SamEval 후처리기 TillReturnLinePostProcessorMultipleIndents (lazy)."""
    global _PP
    if _PP is None:
        from core.postprocessors import (
            TillReturnLinePostProcessorMultipleIndents as _C,
        )
        _PP = _C(None, prefix=4,
                 first_prefix="    # The columns used to answer the question: ")
    return _PP


# ------------------------------------------------------------ 프롬프트
_PG_CACHE: dict[str, Any] = {}


def _load_table(name: str) -> pd.DataFrame:
    """data/tables/<id>.parquet 우선, 없으면 databench 원본 경로."""
    for p in (DATA / f"{name}.parquet",
              REPO / "data" / "tables" / f"{name}.parquet",
              ROOT / "data" / "databench" / "data" / name / "all.parquet",
              REPO / "data" / "databench" / "data" / name / "all.parquet"):
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(f"테이블 없음: {name}")


def _load_exemplars() -> list[tuple[dict, str]]:
    """annotations_cot.json + 로컬 semeval_train.parquet → 예시 40개.

    HF `load_dataset` 을 쓰지 않는다(네트워크·인덱스 드리프트 회피).
    """
    cands = [p for p in (ROOT / "annotation" / "annotations_cot.json",
                         REPO / "annotation" / "annotations_cot.json") if p.exists()]
    if not cands:
        raise FileNotFoundError(
            f"annotations_cot.json 없음. 찾은 위치: {ROOT}, {REPO}")
    ann = json.loads(cands[0].read_text(encoding="utf-8"))

    qa_cands = [p for p in (ROOT / "data" / "qa" / "semeval_train.parquet",
                            REPO / "data" / "qa" / "semeval_train.parquet",
                            ROOT / "data" / "databench" / "qa" / "semeval_train.parquet",
                            REPO / "data" / "databench" / "qa" / "semeval_train.parquet")
                if p.exists()]
    if not qa_cands:
        raise FileNotFoundError(
            "semeval_train.parquet 없음. 다음 중 한 곳에 두세요:\n"
            f"  {REPO / 'data' / 'qa' / 'semeval_train.parquet'}\n"
            "  (python data_fetch.py qa 실행)")
    tr_p = qa_cands[0]
    tr = pd.read_parquet(tr_p)
    rows = [dict(r) for _, r in tr.iterrows()]
    return [(rows[e["index"]], e["annotation"]) for e in ann["exemplars"]]


def build_prompt(row: dict) -> str:
    """문항 1건(dataset, question, answer, columns_used, column_types) → P2 프롬프트."""
    import core.prompt_generators as PG

    if "p2" not in _PG_CACHE:
        ex = _load_exemplars()
        shots = [ex[i] for i in (x[0] for x in EXEMPLAR_INDICES)]
        zs = PG.ZeroShotDetailedTypesRowsExValuesNullsOneLineRowsVol2(_load_table, num_rows=5)
        fs = PG.FewShotChainOfThoughtBuilderWithTypesRowsAndPredictingResultingTypesVol2(
            _load_table, zs, PG.ExemplarBuilder(_load_table, zs), shots=shots)
        _PG_CACHE["p2"] = PG.ClaudeMessageEmbeddingPromptGenerator(_load_table, fs)

    q = {
        "dataset": row["dataset"],
        "question": row["question"],
        "answer": row.get("answer", ""),
        "columns_used": row.get("columns_used") or [],
        "column_types": row.get("column_types") or [],
    }
    payload = _PG_CACHE["p2"](q)
    return _flatten(payload) + HARNESS


def _flatten(payload: Any) -> str:
    """Claude 스타일 payload({'messages':[{'content':[{'text':..}]}]}) → 평문."""
    if isinstance(payload, str):
        return payload
    parts = []
    for m in payload.get("messages", []):
        c = m.get("content", "")
        if isinstance(c, list):
            parts.append("".join(x.get("text", "") for x in c))
        else:
            parts.append(str(c))
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------- 호출
def make_client(api_key: str | None = None):
    from openai import OpenAI

    if api_key is None:
        try:
            from dotenv import load_dotenv
            for envf in (ROOT / ".env", REPO / ".env"):
                if envf.exists():
                    load_dotenv(envf)
        except ImportError:
            pass
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 없음 — .env 또는 인자로 전달")
    return OpenAI(api_key=api_key, timeout=TIMEOUT_S)


def call_llm(client, prompt: str, model: str = MODEL_DEFAULT,
             temperature: float = TEMP_DEFAULT, max_tokens: int = MAXTOK_DEFAULT,
             reasoning_effort: str | None = "none") -> dict:
    """1회 호출.

    gpt-6-luna(Chat Completions) 제약 — 실측 확인
      · max_tokens 미지원 → max_completion_tokens 사용
      · reasoning_effort != "none" 이면 temperature 는 1.0 만 허용
    """
    t0 = time.time()
    kw = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
        temperature=temperature,
        timeout=TIMEOUT_S,
    )
    if reasoning_effort:
        kw["reasoning_effort"] = reasoning_effort
    r = client.chat.completions.create(**kw)
    ch = r.choices[0]
    u = r.usage
    det = getattr(u, "prompt_tokens_details", None) if u else None
    return {
        "text": ch.message.content or "",
        "finish_reason": ch.finish_reason,
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None),
        "cached_tokens": getattr(det, "cached_tokens", None),
        "system_fingerprint": getattr(r, "system_fingerprint", None),
        "latency_s": round(time.time() - t0, 2),
    }


def probe_capabilities(client, model: str = MODEL_DEFAULT) -> pd.DataFrame:
    """(reasoning_effort × temperature) 어느 칸이 열려 있는지 실측."""
    grid = [(None, 0.0), (None, 0.2), (None, 1.0)] + [
        (e, t) for e in ("none", "low", "medium", "high") for t in (0.0, 0.2, 1.0)
    ]
    out = []
    for eff, t in grid:
        kw = dict(model=model, messages=[{"role": "user", "content": "say ok"}],
                  max_completion_tokens=300, temperature=t)
        if eff:
            kw["reasoning_effort"] = eff
        try:
            r = client.chat.completions.create(**kw)
            out.append({"effort": eff, "temperature": t, "ok": True,
                        "out_tokens": r.usage.completion_tokens, "msg": ""})
        except Exception as e:  # noqa: BLE001
            out.append({"effort": eff, "temperature": t, "ok": False,
                        "out_tokens": None, "msg": str(e)[:100]})
    return pd.DataFrame(out)


# -------------------------------------------------------------- 코드 실행


def extract_body(text: str) -> str:
    """응답 → 함수 본문 (첫 `return` 줄까지).
    들여쓰기 보정은 SamEval 후처리기가 담당한다.
    """
    lines = text.split("\n")
    out = []
    for l in lines:
        out.append(l)
        if re.match(r"^\s*return\b", l):
            break
    return "\n".join(out).strip("\n")


def build_source(body: str, columns: list[str]) -> str:
    """본문 → 실행 소스. SamEval TillReturnLinePostProcessorMultipleIndents 경유.

    이 후처리기는 줄마다 prefix(4)를 빼고 다시 4칸을 붙이므로 입력 첫 줄이 4칸에
    있어야 중첩이 보존된다(업스트림은 Claude 프리필로 이 상태를 보장한다).
    """
    # 1) 공통 들여쓰기 제거로 첫 줄을 0에 맞춘다
    # 2) 4칸으로 감싸 PP 가 기대하는 상태를 만든다
    # 들여쓰기 보정은 PP 몫(업스트림과 동일하게 1회). 
    block = _pp()(body)
    return (f"def answer(df: pd.DataFrame):\n"
            f"    df.columns = {list(columns)!r}\n"
            f"{block}\n")


def run_code(body: str, df: pd.DataFrame, columns: list[str], timeout_s: int = TIMEOUT_S):
    """본문 실행 → (value, error). 예외는 문자열로 반환."""
    if not body.strip():
        return None, "EmptyBody"
    src = build_source(body, columns)
    ns: dict[str, Any] = {"pd": pd, "np": np, "re": re}
    try:
        exec(compile(src, "<agent>", "exec"), ns)  # noqa: S102
        val = ns["answer"](df.copy())
        if isinstance(val, (pd.Series, np.ndarray, list, tuple)):
            val = val[0] if len(val) else None
        if isinstance(val, (np.generic,)):
            val = val.item()
        return val, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"[:200]


# ---------------------------------------------------------------- 서명
AGG_FUNCS = {
    "mean", "median", "sum", "min", "max", "count", "size", "std", "var",
    "nunique", "idxmax", "idxmin", "mode", "value_counts", "nlargest", "nsmallest",
    "quantile", "describe", "groupby", "sum", "len",
}
_OPS = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=", ast.Eq: "==", ast.NotEq: "!="}


def signature(body: str, columns: list[str]) -> dict:
    """생성 코드에서 (컬럼, 필터, 집계) 를 뽑는다.

    실행 실패와 무관하게 계산되므로, 실패 문항도 일관성 분석에 쓸 수 있다.
    """
    sig = {"columns": [], "filters": [], "aggs": [], "ok": False}
    try:
        src = build_source(body, columns)
        tree = ast.parse(src)
    except SyntaxError:
        return sig

    for node in ast.walk(tree):
        # df['col'] / df.col
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                sig["columns"].append(sl.value)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "df" and node.attr not in AGG_FUNCS \
                and node.attr not in {"columns", "shape", "index", "dtypes", "values"}:
            sig["columns"].append(node.attr)
        # 비교 → 필터
        elif isinstance(node, ast.Compare):
            left = _name_of(node.left)
            op = _OPS.get(type(node.ops[0]), "?")
            for comp in node.comparators:
                sig["filters"].append(f"{left}{op}{_const(comp)}")
        # 집계 함수
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in AGG_FUNCS:
            sig["aggs"].append(node.func.attr)

    sig["columns"] = sorted(set(sig["columns"]))
    sig["filters"] = sorted(set(sig["filters"]))
    sig["aggs"] = sorted(set(sig["aggs"]))
    sig["ok"] = True
    return sig


def _name_of(node) -> str:
    if isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant):
            return str(sl.value)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return "?"


def _const(node) -> str:
    if isinstance(node, ast.Constant):
        v = node.value
        return f"{v:.6g}" if isinstance(v, float) else str(v)
    return "?"


def sig_key(sig: dict) -> str:
    return json.dumps([sig["columns"], sig["filters"], sig["aggs"]], ensure_ascii=False)


# ---------------------------------------------------------------- 채점
def close(a, b) -> bool:
    try:
        fa, fb = float(a), float(b)
        if np.isnan(fa) or np.isnan(fb):
            return False
        return abs(fa - fb) <= max(0.01, abs(fa) * 0.01)
    except Exception:
        return str(a).strip().lower() == str(b).strip().lower()


# ---------------------------------------------------------------- 파일럿
def run_pilot(client, questions: pd.DataFrame, ref: pd.DataFrame, out_csv: pathlib.Path,
              condition: str, n_rep: int = 2, model: str = MODEL_DEFAULT,
              temperature: float = TEMP_DEFAULT, max_tokens: int = MAXTOK_DEFAULT,
              reasoning_effort: str | None = "none", verbose: bool = True) -> pd.DataFrame:
    """한 조건(condition)을 문항 우선 × n_rep 회 실행. append + 재개 지원.

    재개 키 = (condition, n, rep). 갈래 A/B 를 같은 CSV 에 쌓아도 충돌하지 않는다.
    """
    out_csv = _out(out_csv)
    ref_map = dict(zip(ref["n"], ref["answer_ref"]))
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv, encoding="utf-8-sig")
        # 하네스 판이 다른 기존 행은 재개 대상에서 제외한다.
        # (조건명을 그대로 두고 CSV 를 이어 써도 스킵되지 않는다)
        if "harness" in prev:
            prev = prev[prev["harness"].fillna("") == HARNESS_VERSION]
        elif "condition" in prev:
            prev = prev.iloc[0:0]
        if "condition" in prev:
            done = set(zip(prev["condition"], prev["n"], prev["rep"]))
        if verbose:
            print(f"[resume] {out_csv.name}: 기존 {len(prev)}행 · "
                  f"이 조건 완료 {sum(1 for c, _, _ in done if c == condition)}건")

    rows = []
    for _, q in questions.iterrows():
        n = int(q["n"])
        df = _load_table(q["dataset"])
        cols = list(df.columns)
        prompt = build_prompt(q.to_dict())
        for rep in range(1, n_rep + 1):
            if (condition, n, rep) in done:
                continue
            base = {"condition": condition, "n": n, "rep": rep,
                    "dataset": q["dataset"], "A": q.get("A"), "B": q.get("B"),
                    "harness": HARNESS_VERSION, "model": model,
                    "effort": reasoning_effort, "temperature": temperature,
                    "max_tokens": max_tokens}
            try:
                res = call_llm(client, prompt, model, temperature, max_tokens, reasoning_effort)
            except Exception as e:  # noqa: BLE001
                rows.append({**base, "call_error": f"{type(e).__name__}: {e}"[:160]})
                if verbose:
                    print(f"  {condition} n={n:>3} rep={rep} CALL-FAIL {str(e)[:60]}")
                continue

            body = extract_body(res["text"])
            val, err = run_code(body, df, cols)
            sg = signature(body, cols)
            hit = close(val, ref_map.get(n))
            rows.append({
                **base,
                "answer_ref": ref_map.get(n), "answer_agent": val,
                "correct": hit, "exec_ok": err is None, "exec_err": err or "",
                "sig_ok": sg["ok"], "sig_key": sig_key(sg),
                # 하네스 준수 여부와 "예외는 없지만 값이 None" 인 누수 기록
                "def_written": bool(re.search(r"^\s*def\s+\w+\s*\(", body, re.M)),
                "null_answer": val is None,
                "finish_reason": res["finish_reason"],
                "text_len": len(res["text"]), "body_len": len(body),
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "reasoning_tokens": res["reasoning_tokens"],
                "cached_tokens": res["cached_tokens"],
                "system_fingerprint": res["system_fingerprint"],
                "latency_s": res["latency_s"],
                "call_error": "", "body": body, "raw": res["text"],
            })
            if verbose:
                print(f"  {condition} n={n:>3} rep={rep} {'O' if hit else 'X'}"
                      f" exec={'ok' if err is None else 'FAIL'}"
                      f" fr={res['finish_reason']:<8} out={res['completion_tokens']}"
                      f" rea={res['reasoning_tokens']}")

    if rows:
        df_new = pd.DataFrame(rows)
        if out_csv.exists():
            df_new = pd.concat([pd.read_csv(out_csv, encoding="utf-8-sig"), df_new],
                               ignore_index=True)
        df_new.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return pd.read_csv(out_csv, encoding="utf-8-sig")


def run_branch(client, questions: pd.DataFrame, ref: pd.DataFrame, out_csv: pathlib.Path,
               branch: str, n_rep: int = 2, model: str = MODEL_DEFAULT,
               verbose: bool = True) -> pd.DataFrame:
    """갈래 A/B 를 한 번에. 조건 라벨은 f"{branch}_e{effort}_T{temp}"."""
    out_csv = _out(out_csv)
    cfg = PILOT_A if branch.upper() == "A" else PILOT_B
    for eff in cfg["efforts"]:
        for t in cfg["temps"]:
            cond = f"{branch.upper()}_e{eff}_T{t:g}"
            print(f"[{cond}] max_tokens={cfg['max_tokens']}")
            run_pilot(client, questions, ref, out_csv, condition=cond, n_rep=n_rep,
                      model=model, temperature=t, max_tokens=cfg["max_tokens"],
                      reasoning_effort=eff, verbose=verbose)
    return pd.read_csv(out_csv, encoding="utf-8-sig")


def summarize(runs: pd.DataFrame, by_condition: bool = False):
    """요약. by_condition=True 면 조건별 표를 돌려준다."""
    if "call_error" not in runs:
        return {} if not by_condition else pd.DataFrame()

    def block(d: pd.DataFrame) -> dict:
        ok = d[d["call_error"].fillna("") == ""]
        if not len(ok):
            return {"호출 수": len(d), "호출 실패": len(d)}
        per_q = ok.groupby("n")
        pin = ok["prompt_tokens"].fillna(0)
        cin = ok["cached_tokens"].fillna(0)
        cout = ok["completion_tokens"].fillna(0)
        cost = float((float((pin - cin).sum()) * 0.10 + float(cin.sum()) * 0.01
                      + float(cout.sum()) * 0.50) / 1e6)
        return {
            "호출 수": len(d),
            "호출 실패": int((d["call_error"].fillna("") != "").sum()),
            "finish=length": int((ok["finish_reason"] == "length").sum()),
            "본문 비어있음": int((ok["body_len"] == 0).sum()),
            "실행 성공률 %": round(ok["exec_ok"].mean() * 100, 1),
            "값 None %": round(ok["null_answer"].mean() * 100, 1) if "null_answer" in ok else None,
            "하네스 위반 %": round(ok["def_written"].mean() * 100, 1) if "def_written" in ok else None,
            "서명 추출 %": round(ok["sig_ok"].mean() * 100, 1),
            "정답률 %": round(ok["correct"].mean() * 100, 1),
            "값 일관성 %": round(per_q["answer_agent"].apply(
                lambda s: s.astype(str).value_counts().max() / len(s)).mean() * 100, 1),
            "의도 일관성 %": round(per_q["sig_key"].apply(
                lambda s: s.value_counts().max() / len(s)).mean() * 100, 1),
            "출력토큰 평균": round(cout.mean(), 1),
            "사고토큰 평균": round(ok["reasoning_tokens"].fillna(0).mean(), 1),
            "캐시hit 평균": round(cin.mean(), 1),
            "fingerprint 수": ok["system_fingerprint"].nunique(),
            "비용 USD": round(cost, 4),
        }

    if by_condition:
        rows = []
        for c, d in runs.groupby("condition", sort=False):
            rows.append({"condition": c, **block(d)})
        return pd.DataFrame(rows)
    return block(runs)