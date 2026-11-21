"""
SR(D) 와 ST(D) 를 모두 제거한 DataBench 파이프라인 래퍼.

프롬프트 구성 성분 (행 의존성 기준)
--------------------------------------------------------------
T0  SCH(D)  스키마 정보      컬럼 인덱스, 컬럼 이름, pandas dtype, 원소 타입
T1  CAT(D)  카테고리 정보    범주형 컬럼의 선언된 카테고리 (테이블 규칙 파생)
T2  NR(D)   데이터 규모 정보 number_of_rows
T3  ST(D)   통계 정보        결측 개수 + 분포 통계(min/max/mean/std 등)
T4  SR(D)   원본 예시 정보   원본 데이터의 첫 N 행

행 비의존 (T0, T1, T2) / 행 의존 (T3, T4)

이 래퍼가 만드는 프롬프트는 T_I(D), I = {0, 1} 이다.
  - SR(D) 제거 : 원본 행 블록을 프롬프트에서 삭제
  - ST(D) 제거 : 컬럼 설명에서 'Non-Null CounT' 필드를 삭제
                 (분포 통계는 AILS-NTUA 원본에 존재하지 않으므로 함께 사라짐)
  - NR(D)      : number_of_rows 는 원본에서 SR 블록 안에만 등장하므로 SR 제거와
                 동시에 프롬프트에서 사라진다.

결과적으로 프롬프트는 D 의 행에 의존하는 값을 하나도 포함하지 않는다.
(단, 'Types of Elements' 와 'Values'(카테고리) 는 실제 행을 스캔해 만들 수 있으나
 본 연구에서는 스키마에 정의된 행 불변 정보로 간주한다.)
"""

import sys

# core/prompt_generators.py 는 f-string 치환식 안에서 백슬래시를 쓴다(PEP 701).
if sys.version_info < (3, 12):
    raise RuntimeError(
        f"core/* 모듈은 Python 3.12+ 가 필요합니다 "
        f"(현재 {sys.version_info.major}.{sys.version_info.minor})."
    )

import ast
import os
import re
import textwrap
import threading

import numpy as np
import pandas as pd

try:                       # .env 가 없는 환경에서도 import 는 성공해야 한다
    from dotenv import load_dotenv
except ImportError:        # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

from openai import OpenAI

import core.executors as executors
import core.pipelines as pipelines
import core.postprocessors as postprocessors
import core.prompt_generators as prompt_generators
import core.utils as utils

# main.py 와 동일한 exemplar 선택
EXEMPLAR_INDICES = [(17, 140), (0, 132), (28, 286), (31, 303), (4, 246),
                    (24, 175), (20, 176), (8, 141), (14, 12)]
ANNOTATIONS_FILE = "annotation/annotations_cot.json"

# CD 블록 헤더 (원본 / ST 제거 후)
CD_HEADER_ORIGINAL = "#,Column,Non-Null CounT,Dtype,Types of Elements,Values"
CD_HEADER_NO_ST = "#,Column,Dtype,Types of Elements,Values"


# ======================================================================
# 1. 컬럼 설명(CD) 생성기 — ST(D) 제거판 / 원본 유지판
# ======================================================================
def _cd_row_values(df: pd.DataFrame) -> list[str]:
    """컬럼별 (이름, dtype, 원소 타입, 카테고리 문자열) 을 순서대로 만든다.

    행 개수(non-null count)는 여기서 계산하지 않는다.
    """
    out = []
    for col in df.columns:
        dtype = df[col].dtype
        python_inner_types = df[col].apply(lambda x: type(x)).unique().tolist()

        values = ""
        if dtype.name == "category":
            unique_values = df[col].cat.categories.tolist()
            if len(unique_values) <= 5:
                values = f"All values are {unique_values}"
            else:
                values = f"5 example values are {unique_values[:5]}"

        out.append((str(col), str(dtype), str(python_inner_types), values))
    return out


def custom_info_no_st(df: pd.DataFrame) -> str:
    """ST(D) 를 포함하지 않는 컬럼 설명.

    원본 custom_info_csv_2 와 비교했을 때 'Non-Null CounT' 필드만 빠진다.
    """
    output = [CD_HEADER_NO_ST]
    for i, (name, dtype, ptypes, values) in enumerate(_cd_row_values(df)):
        output.append(f"{i},{name},{dtype},{ptypes},{values}")
    return "\n".join(output)


def custom_info_with_st(df: pd.DataFrame) -> str:
    """비교용. 원본 custom_info_csv_2 와 동일한 출력 (ST 포함)."""
    output = [CD_HEADER_ORIGINAL]
    for i, (name, dtype, ptypes, values) in enumerate(_cd_row_values(df)):
        non_null_count = int(df[name].notnull().sum())
        output.append(f"{i},{name},{non_null_count},{dtype},{ptypes},{values}")
    return "\n".join(output)


# ======================================================================
# 2. Zero-shot 생성기 — SR 없음 / ST 없음
# ======================================================================
class WithoutSTandSRZeroShot(prompt_generators.ZeroShotGenerator):
    """SR(D) 블록과 ST(D) 필드를 모두 제거한 zero-shot 프롬프트 생성기.

    원본 ZeroShotDetailedTypesRowsExValuesNullsOneLineRowsVol2 에서
      - 'The first N rows from the dataframe' 블록  → SR(D)
      - 컬럼 설명의 'Non-Null CounT' 필드            → ST(D) 의 결측 정보
    를 제거했다. 레이아웃(공백/개행)은 FewShot 이 이어붙일 때 깨지지 않도록 유지한다.
    """

    def __init__(self, loader, num_rows: int = 0, lite: bool = False):
        super().__init__(loader)
        self.num_rows = 0          # 행 블록 없음. 인자는 YAML 호환용.
        self.lite = lite

    def __call__(self, row) -> str:
        df = self.loader(row["dataset"])
        info = "\n        ".join(custom_info_no_st(df).split("\n"))
        return (
            "# TODO: complete the following function. It should give the answer to: "
            + row["question"] + "\n"
            "def answer(df: pd.DataFrame):\n"
            '    """\n'
            "        " + info + "\n"
            '    """\n'
            "\n\n"
            "    df.columns = " + repr(list(df.columns)) + "\n"
            "    "
        )

    def __str__(self):
        return f"WithoutSTandSRZeroShot(no sample rows, no missing-count, lite={self.lite})"


class SchemaOnlyWithSTZeroShot(prompt_generators.ZeroShotGenerator):
    """비교용 베이스라인. SR 은 없지만 ST(결측 개수)는 남아 있는 상태.

    dp_agent_wrap_no_SR.SchemaOnlyZeroShot 과 동일한 동작이다. 프롬프트 diff 확인용.
    """

    def __init__(self, loader, num_rows: int = 0, lite: bool = False):
        super().__init__(loader)
        self.num_rows = 0
        self.lite = lite

    def __call__(self, row) -> str:
        df = self.loader(row["dataset"])
        info = "\n        ".join(custom_info_with_st(df).split("\n"))
        return (
            "# TODO: complete the following function. It should give the answer to: "
            + row["question"] + "\n"
            "def answer(df: pd.DataFrame):\n"
            '    """\n'
            "        " + info + "\n"
            '    """\n'
            "\n\n"
            "    df.columns = " + repr(list(df.columns)) + "\n"
            "    "
        )

    def __str__(self):
        return f"SchemaOnlyWithSTZeroShot(sample rows removed, missing-count kept, lite={self.lite})"


# ======================================================================
# 3. OpenAI Answerer (프리필 → user 메시지 접기)
# ======================================================================
_CONTINUE_INSTRUCTION = """\
You are completing a Python function. Below is an unfinished code block.
Continue it EXACTLY where it stops and output ONLY the continuation.

Rules:
- Do not repeat any part of the given code.
- Do not write explanations and do not use markdown code fences.
- Your first line continues the last, unfinished line of the given code.
- Every following line of the function body is indented by exactly 4 spaces
  (nested blocks add 4 more).
- Finish with a single top-level `return` statement.
"""

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*$")

_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


class OpenAIAnswerer:
    """Claude 스타일 payload({'messages': [...]})를 받아 OpenAI Chat Completions 로 보낸다."""

    def __init__(self, client=None, model="gpt-4o-mini", temperature=0.0,
                 top_p=1.0, max_gen_len=300, reasoning=None,
                 pit=0.0, pot=0.0):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        self.reasoning = (
            reasoning if reasoning is not None
            else model.startswith(_REASONING_PREFIXES)
        )
        self.pit = pit
        self.pot = pot

    @staticmethod
    def _flatten(payload) -> str:
        if isinstance(payload, str):
            return _CONTINUE_INSTRUCTION + "\n" + payload
        user_parts, assistant_parts = [], []
        for m in payload["messages"]:
            text = "".join(c["text"] for c in m["content"])
            (user_parts if m["role"] == "user" else assistant_parts).append(text)
        chunks = [_CONTINUE_INSTRUCTION]
        if user_parts:
            chunks.append("\n\n".join(user_parts))
        chunks.append("### Unfinished code (continue from the very end):\n"
                      + "\n\n".join(assistant_parts))
        return "\n\n".join(chunks)

    @staticmethod
    def _normalize(text: str) -> str:
        lines = [l for l in text.split("\n") if not _FENCE.match(l)]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return "    return None"
        head, tail = lines[0].strip(), lines[1:]
        if not tail:
            return head
        body = textwrap.indent(textwrap.dedent("\n".join(tail)), "    ")
        return head + "\n" + body

    def __call__(self, prompt, custom_gen_parameters=None):
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": self._flatten(prompt)}],
        }
        if self.reasoning:
            kwargs["max_completion_tokens"] = self.max_gen_len
        else:
            kwargs["max_tokens"] = self.max_gen_len
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        if custom_gen_parameters:
            kwargs.update(custom_gen_parameters)

        try:
            r = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f"ERROR: Can't invoke '{self.model}'. Reason: {e}")
            return "", 0, 0

        text = r.choices[0].message.content or ""
        usage = getattr(r, "usage", None)
        return (
            self._normalize(text),
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    def __str__(self):
        return (f"OpenAIAnswerer(Answerer)\n\tmodel: {self.model}"
                f"\n\ttemperature: {self.temperature}\n\ttop_p: {self.top_p}"
                f"\n\tmax_gen_len: {self.max_gen_len}\n\treasoning: {self.reasoning}")


# ======================================================================
# 4. 스레드 기반 실행기 (Windows spawn/pickle 회피)
# ======================================================================
class InlineStatementExecutor(executors.Executor):
    def __init__(self, loader, timeout: int = 60):
        super().__init__(loader)
        self.timeout = timeout

    def __call__(self, inp):
        statement, dataset = inp
        try:
            df = self.loader(dataset)
        except Exception as e:
            return f"__CODE_ERROR__: table load failed: {e}"

        code = "def answer(df):\n" + statement + "\n"
        box = {}

        def work():
            try:
                ns = {"pd": pd, "np": np, "ast": ast}
                exec(code, ns, ns)
                box["ans"] = ns["answer"](df)
            except Exception as e:
                box["err"] = f"__CODE_ERROR__: {type(e).__name__}: {e}"

        t = threading.Thread(target=work, daemon=True)
        t.start()
        t.join(self.timeout)
        if t.is_alive():
            return "__TIMEOUT__"
        if "err" in box:
            return box["err"]
        return box.get("ans")

    def __str__(self):
        return (f"InlineStatementExecutor(Executor)\n\tloader: {self.loader}"
                f"\n\ttimeout: {self.timeout}\n")


# ======================================================================
# 5. 재시도 의미를 명확히 한 상위 파이프라인
# ======================================================================
class SafeErrorFixingPipeline(pipelines.ErrorFixingAndTimeoutPipeline):
    """num_fix_attempts == 실제 에러 수정 LLM 호출 횟수. 타임아웃은 같은 실행기에서 재시도."""

    def __init__(self, main_pipeline, error_fix_pipeline,
                 max_timeout: int = 300, num_fix_attempts: int = 2):
        super().__init__(main_pipeline, error_fix_pipeline,
                         max_timeout=max_timeout, num_attempts=num_fix_attempts)
        self.num_fix_attempts = num_fix_attempts

    @staticmethod
    def _bad(x) -> bool:
        s = str(x)
        return s.startswith("__CODE_ERROR__") or s.startswith("__TIMEOUT__")

    def run_one(self, inp):
        row = inp
        prompt, raw, post, out, itok, otok = self.main_pipeline.run_one(row)
        prompts, raws, posts = [prompt], [raw], [post]
        n_calls = 1

        if str(out).startswith("__TIMEOUT__"):
            prev = self.main_pipeline.executor.timeout
            self.main_pipeline.executor.timeout = self.max_timeout
            try:
                p2, r2, s2, o2, i2, t2 = self.main_pipeline.run_one(row)
            finally:
                self.main_pipeline.executor.timeout = prev
            n_calls += 1
            itok += i2
            otok += t2
            tag = "TIMEOUT FIXED" if not str(o2).startswith("__TIMEOUT__") else "TIMEOUT NOT FIXED"
            prompts += [tag, str(p2)]
            raws += [tag, str(r2)]
            posts += [tag, str(s2)]
            if tag == "TIMEOUT FIXED":
                out, post = o2, s2

        n_fix = 0
        while self._bad(out) and n_fix < self.num_fix_attempts:
            pf, rf, sf, of, i3, t3 = self.error_fix_pipeline.run_one((row, post, out))
            n_fix += 1
            n_calls += 1
            itok += i3
            otok += t3
            fixed = not self._bad(of)
            tag = "ERROR FIXED" if fixed else "ERROR NOT FIXED"
            prompts += [tag, str(pf)]
            raws += [tag, str(rf)]
            posts += [tag, str(sf)]
            out, post = of, sf
            if fixed:
                break

        self.last_trace = {"n_llm_calls": n_calls, "n_fix_attempts": n_fix,
                           "final_error": self._bad(out)}
        return prompts, raws, posts, out, itok, otok

    def run_one_traced(self, row) -> dict:
        prompts, raws, posts, out, itok, otok = self.run_one(row)
        return {"output": out, "code": posts[-1], "raw": raws[-1],
                "input_tokens": itok, "output_tokens": otok, **self.last_trace}

    def __str__(self):
        return (f"SafeErrorFixingPipeline(num_fix_attempts={self.num_fix_attempts}, "
                f"max_timeout={self.max_timeout})\n\t{self.main_pipeline}\n\t{self.error_fix_pipeline}")


# ======================================================================
# 6. 조립
# ======================================================================
def assert_local_table(name: str, loader=None) -> pd.DataFrame:
    """generic_load_table 은 로컬 실패를 삼키고 HuggingFace 로 폴백하므로 미리 확인한다."""
    path = os.path.join("competition", name, "all.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{os.path.abspath(path)} 가 없습니다. 노트북 CWD 가 저장소 루트인지 확인하세요."
        )
    return (loader or utils.generic_load_table)(name)


def build_pipe_nost(model="gpt-4o-mini", temperature=0.0, max_gen_len=300,
                    fix_max_gen_len=1000, num_fix_attempts=2,
                    exec_timeout=60, max_timeout=300,
                    lite=False, debug=False, client=None):
    """SR(D), ST(D) 를 모두 제거한 파이프라인.

    client 를 넘기지 않으면 OpenAI() 를 만든다. 프롬프트만 검사할 때는
    임의의 더미 객체를 넘겨 API 키 없이 조립할 수 있다.
    """
    load_dotenv()
    loader = utils.generic_load_sample if lite else utils.generic_load_table

    _, exemplars = utils.annotation_reader(ANNOTATIONS_FILE)
    shots = [exemplars[i] for i in (x[0] for x in EXEMPLAR_INDICES)]

    zs = WithoutSTandSRZeroShot(loader, lite=lite)
    exemplar_builder = prompt_generators.ExemplarBuilder(loader, zs)
    fewshot = prompt_generators.FewShotChainOfThoughtBuilderWithTypesRowsAndPredictingResultingTypesVol2(
        loader, zs, exemplar_builder, shots=shots)
    pg = prompt_generators.ClaudeMessageEmbeddingPromptGenerator(loader, fewshot)

    post_main = postprocessors.TillReturnLinePostProcessorMultipleIndents(
        loader, prefix=4,
        first_prefix="    # The columns used to answer the question: ")

    main_pipe = pipelines.Pipeline(
        pg,
        OpenAIAnswerer(client, model=model, temperature=temperature,
                       max_gen_len=max_gen_len),
        post_main,
        InlineStatementExecutor(loader, timeout=exec_timeout),
        debug=debug,
    )

    post_fix = postprocessors.TillReturnLinePostProcessorMultipleIndents(
        loader, prefix=4, first_prefix="    ")

    fix_pipe = pipelines.ErrorFixPipeline(
        prompt_generators.ErrorFixingGeneratorClaudeVol2(
            loader, WithoutSTandSRZeroShot(loader, lite=lite)),
        OpenAIAnswerer(client, model=model, temperature=temperature,
                       max_gen_len=fix_max_gen_len),
        post_fix,
        InlineStatementExecutor(loader, timeout=exec_timeout),
        debug=debug,
    )

    return SafeErrorFixingPipeline(main_pipe, fix_pipe,
                                   max_timeout=max_timeout,
                                   num_fix_attempts=num_fix_attempts)


# ======================================================================
# 7. 검증 / 확인용 헬퍼
# ======================================================================
def payload_text(payload) -> str:
    """Claude 스타일 payload 에서 텍스트만 뽑는다."""
    if isinstance(payload, str):
        return payload
    out = []
    for m in payload["messages"]:
        for c in m["content"]:
            out.append(c.get("text", ""))
    return "\n".join(out)


def full_prompt(pipe, row) -> str:
    """main pipeline 이 실제로 만들어내는 최종 프롬프트 문자열."""
    pg = pipe.main_pipeline.prompt_generator
    return payload_text(pg.prompt_generator(row))


def find_cd_blocks(text: str) -> list[list[str]]:
    """프롬프트 안의 CD 블록(헤더 1줄 + 컬럼 줄들)을 모두 추출한다."""
    blocks, cur = [], None
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("#,Column"):
            if cur is not None:
                blocks.append(cur)
            cur = [s]
        elif cur is not None:
            if s == "" or s.startswith("The first") or s.startswith("df.columns"):
                blocks.append(cur)
                cur = None
            else:
                cur.append(s)
    if cur is not None:
        blocks.append(cur)
    return blocks


def assert_no_sample_rows(pipe_or_gen, row=None) -> str:
    """최종 프롬프트(예시 9개 포함)에 SR 이 없는지 확인."""
    if hasattr(pipe_or_gen, "main_pipeline"):
        p = full_prompt(pipe_or_gen, row)
    else:
        p = pipe_or_gen(row)
    assert "rows from the dataframe" not in p, "SR 이 남아 있습니다"
    n = p.count("The columns used to answer the question")
    return p


def assert_no_st(pipe_or_gen, row=None) -> dict:
    """최종 프롬프트에 ST(결측 개수)가 없는지 확인하고 블록 수를 돌려준다."""
    if hasattr(pipe_or_gen, "main_pipeline"):
        p = full_prompt(pipe_or_gen, row)
    else:
        p = pipe_or_gen(row)

    assert "Non-Null" not in p, "ST(결측 개수) 헤더가 남아 있습니다"
    assert CD_HEADER_ORIGINAL not in p, "원본 CD 헤더가 남아 있습니다"

    blocks = find_cd_blocks(p)
    for b in blocks:
        assert b[0] == CD_HEADER_NO_ST, f"예상 밖 CD 헤더: {b[0]}"

    return {"n_cd_blocks": len(blocks), "header": CD_HEADER_NO_ST,
            "n_columns": sum(len(b) - 1 for b in blocks)}


def cd_block_diff(df) -> str:
    """같은 테이블에 대해 ST 포함 / 미포함 CD 블록을 나란히 보여준다."""
    return ("--- ST 포함 (WithoutSRZeroShot, 기존 dp_agent_wrap_no_SR) ---\n"
            + custom_info_with_st(df)
            + "\n\n--- ST 미포함 (WithoutSTandSRZeroShot, 이 파일) ---\n"
            + custom_info_no_st(df))