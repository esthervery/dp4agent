"""
dp_budget.py
============================================================================
차분Privacy 질의 모듈.

- run_seq_once()  : KCI 시행 1회 — 질의를 1건씩 던지며, 신뢰구간 상대폭이
                    threshold 미만으로 streak_required회 연속 유지되면 정지.
- run_seq_n()     : 위 시행을 seed만 바꾸어 n_reps회 반복(기본 500회).
                    numpy 벡터화로 시행 축을 한 번에 계산하므로 빠르고,
                    각 시행의 결과와 전체 통계(요약 dict)를 반환.
- plot_seq_once() : 1회 시행의 궤적 시각화.
- plot_seq_dist() : n회 반복 결과의 분포 시각화(히스토그램).

실험 설정(데이터셋, eps 목록, 상수)은 experiments.ipynb에서 담당하고,
이 모듈은 순수 함수만 담는다. 원래 노트북 전역변수(ALPHA, true_mean, COL)를
읽어 쓰던 부분은 함수 인자로 대체했다.
============================================================================
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats



# ============================================================
# [유틸] 커널 재시작 없이 dp_budget.py를 새로 고치는 함수
# dp_budget.py를 고친 뒤 노트북에서 restart()만 호출하면
# 최신 내용으로 다시 로드된다. (모듈 안에 정의되어 있으므로
# 호출한 쪽(노트북)의 네임스페이스를 자동으로 찾아 갱신한다.)
import importlib
import inspect
import sys


def restart(verbose=True):
    """dp_budget 모듈을 다시 로드하고, 호출한 노트북의
    from-import 이름들도 새 버전의 객체로 갱신한다."""
    ns = inspect.currentframe().f_back.f_globals   # 호출자(노트북)의 전역

    # 1) 로드된 적 있는 dp_budget 관련 모듈을 전부 다시 로드
    to_reload = [m for m in sys.modules
                 if m == "dp_budget" or m.startswith("dp_budget.")]
    if not to_reload:                       # 아직 한 번도 import 안 된 경우
        import dp_budget
        to_reload = ["dp_budget"]

    import dp_budget
    fresh = importlib.reload(dp_budget)     # 2) 모듈 자체를 다시 실행

    # 3) 이전에 from dp_budget import ... 로 노트북 전역에 꺼내둔 이름들을
    #    새 버전의 객체로 덮어쓴다 (여기가 reload만으로는 안 되는 부분)
    refreshed = []
    for name in dir(fresh):
        if name.startswith("_"):
            continue
        obj = getattr(fresh, name)
        # 함수/클래스/상수만 — 모듈이나 서브모듈 객체는 건너뜀
        if hasattr(obj, "__module__") and getattr(obj, "__module__", "") == fresh.__name__:
            if name in ns:                  # 노트북에 이미 꺼내둔 이름만
                ns[name] = obj
                refreshed.append(name)
    # 4) from-import 목록을 다시 실행하는 것과 동일한 효과를 위해,
    #    노트북에서 흔히 쓰는 전체 이름도 강제로 재바인딩
    for name in ["run_seq_once", "run_seq_n", "plot_seq_once", "plot_seq_dist",
                 "run_inv_once", "run_inv_n", "plot_inv_summary", "plot_inv_dist",
                 "check_s_rate", "check_s_rate_sweep",
                 "plot_srate_summary", "plot_srate_curve",
                 "run_inv_hybrid", "plot_hybrid_search", "plot_hybrid_consumption",
                 "run_hybrid_sweep", "summarize_hybrid_sweep",
                 "plot_hybrid_sweep_hist", "plot_hybrid_ridgeline",
                 "plot_hybrid_spread"]:
        if hasattr(fresh, name):
            ns[name] = getattr(fresh, name)
            if name not in refreshed:
                refreshed.append(name)

    if verbose:
        print(f"dp_budget.py 다시 로드 완료 — 갱신된 이름 {len(refreshed)}개:")
        print("  ", ", ".join(sorted(refreshed)))
    return fresh



# ---------------------------------------------------------------------------
# [KCI 시행 1회 — 스칼라 버전]
# ---------------------------------------------------------------------------
def run_seq_once(delta, eps, true_mean, seed=0, threshold=0.01,
                 streak_required=5, maxq=500, alpha=0.05,
                 verbose=True, verbose_upto=20, every=10):
    """KCI 시행 1회.

    질의를 1개씩 받아 누적 통계를 갱신하고, 정지 조건 충족 시 중단.
    정지 조건: 신뢰구간 상대폭(2·moe/|mean|)이 threshold 미만인 상태가
    streak_required회 연속으로 유지되는 첫 k.

    인자
      delta     : 질의 민감도
      eps       : 질의 1건당 예산
      true_mean : 참평균 (원래는 노트북 전역변수 → 인자로 전달)
      alpha     : 0.05이면 95% 신뢰구간 (원래는 노트북 전역변수 ALPHA)

    반환
      history DataFrame:
        k, resp, mean, moe, ci_lo, ci_hi, relw, streak, covered, resp_in
    """
    r = np.random.default_rng(seed)
    b = delta / eps

    s1 = s2 = 0.0          # Σx, Σx²
    streak = 0
    hist = []              # 시각화용 기록

    if verbose:
        print(f"delta={delta:,.4f}  eps={eps}  b={b:,.4f}  "
              f"threshold={threshold}  streak_required={streak_required}\n")

    for k in range(1, maxq + 1):
        v = true_mean + r.laplace(0.0, b)      # 질의 1회 = eps 소모
        s1 += v
        s2 += v * v

        mean = s1 / k
        if k < 2:
            moe = relw = lo = hi = np.nan  # 질의가 1회뿐이면 신뢰구간이 만들어지지 않음
            hit = False
        else:
            # 현재까지 수집한 데이터로 평균의 신뢰구간을 계산하고, 신뢰구간이 충분히 좁아졌는지 판단
            var  = max((s2 - k * mean**2) / (k - 1), 0.0)       # 분산
            se   = np.sqrt(var) / np.sqrt(k)                    # 표준 오차
            moe  = stats.t.ppf(1 - alpha/2, df=k-1) * se        # moe는 오차한계. CI: [mean-moe, mean+moe]
            lo, hi = mean - moe, mean + moe                     # 신뢰구간 하한/상한
            relw = 2 * moe / abs(mean)
            hit  = relw < threshold

        # 신뢰구간 폭 안에 실제 평균(모평균)이 포함되는지 여부 확인
        if np.isfinite(moe):
            covered = (abs(mean - true_mean) <= moe)
        else:
            covered = pd.NA
        resp_in = (lo <= v <= hi) if np.isfinite(moe) else pd.NA

        streak = streak + 1 if hit else 0
        hist.append((k, v, mean, moe, lo, hi, relw, streak, covered, resp_in))

        # 1~verbose_upto회는 매번, 이후로는 every 배수마다(또는 정지 시점에) 출력
        if verbose and (k <= verbose_upto or k % every == 0 or streak >= streak_required):
            flag = "*" * streak
            mark = "-- " if covered is pd.NA else ("IN " if covered else "OUT")
            ci   = ("         --  ~          -- " if not np.isfinite(moe)
                    else f"{lo:11.2f} ~ {hi:11.2f}")
            print(f"k={k:>4} | resp={v:11.2f} | mean={mean:10.2f} | "
                  f"CI=[{ci}] | moe={moe:9.2f} | relw={relw:8.5f} | "
                  f"{mark} | streak={streak} {flag}")

        if streak >= streak_required:
            if verbose:
                print(f"\n>>> 정지: k={k}, eps_total={eps*k:.2f}, "
                      f"추정={mean:,.2f}, width={moe:.2f}, "
                      f"CI=[{lo:,.2f} ~ {hi:,.2f}]")
            break
    else:
        if verbose:
            print(f"\n>>> {maxq}회 내 미수렴")

    df = pd.DataFrame(hist, columns=["k","resp","mean","moe",
                                     "ci_lo","ci_hi","relw","streak","covered","resp_in"])
    df["covered"] = df["covered"].astype("boolean")
    df["resp_in"] = df["resp_in"].astype("boolean")
    return df


# ---------------------------------------------------------------------------
# [벡터화 엔진 — 내부 헬퍼]
# ---------------------------------------------------------------------------
def _t_table(maxq, alpha):
    """k=1..maxq 각각의 t 임계값(t.ppf(1-alpha/2, df=k-1)). k=1은 NaN."""
    t = np.full(maxq, np.nan)
    if maxq >= 2:
        t[1:] = stats.t.ppf(1 - alpha/2, df=np.arange(1, maxq))
    return t


def _seq_batch(true_mean, delta, eps, threshold, streak_required, maxq,
               alpha, seeds, keep_matrix=False):
    """KCI 시행들을 시행 축으로 벡터화한다(내부 헬퍼).

    스칼라 구현(run_seq_once)과의 동등성
      - 각 시행은 자기 rng로 laplace(0, scale, size=maxq)를 한 번에 뽑는다. 이는
        스칼라 호출 maxq번과 같은 난수열을 만든다.
      - 상대 폭은 k >= 2일 때만 유한하다. k=1은 NaN으로 남아 hit=False가 되고,
        이는 스칼라 버전과 정확히 같다.
      - streak은 마지막 실패 위치를 추적해 계산한다. run[i] = i - last_fail[i]가
        곧 연속 충족 횟수다.

    반환 dict: k_stop(절단이면 0), converged, eps_total, relw_stop,
               covered_stop, k_last, 그리고 선택적으로 relw / covered 행렬.
    """
    seeds = np.atleast_1d(np.asarray(seeds, dtype=np.int64))
    n = seeds.size
    scale = delta / eps
    t = _t_table(maxq, alpha)

    noise = np.empty((n, maxq), dtype=float)
    for i, s in enumerate(seeds):
        noise[i] = np.random.default_rng(int(s)).laplace(0.0, scale, size=maxq)
    resp = true_mean + noise
    del noise

    csum = np.cumsum(resp, axis=1)
    k = np.arange(1, maxq + 1, dtype=float)
    cmean = csum / k
    csq = np.cumsum(resp * resp, axis=1)
    var = np.empty_like(resp)
    var[:, 0] = np.nan
    var[:, 1:] = (csq[:, 1:] - k[1:] * cmean[:, 1:] ** 2) / (k[1:] - 1)
    std = np.sqrt(np.clip(var, 0.0, None))
    del csq, var

    se = std / np.sqrt(k)
    moe = t[None, :] * se
    width = 2.0 * moe
    with np.errstate(invalid="ignore", divide="ignore"):
        relw = np.where(np.abs(cmean) > 0, width / np.abs(cmean), np.inf)
    del width, moe, se, std

    hit = np.isfinite(relw) & (relw < threshold)

    # streak: 마지막 실패 위치를 추적
    idx = np.arange(maxq)
    last_fail = np.maximum.accumulate(np.where(hit, -1, idx), axis=1)
    run = idx[None, :] - last_fail
    ok = run >= streak_required

    any_ok = ok.any(axis=1)
    stop0 = np.where(any_ok, ok.argmax(axis=1), maxq - 1)      # 0부터 시작
    converged = any_ok

    ci_half = t[None, :] * np.sqrt(np.clip(
        (np.cumsum(resp * resp, axis=1) - k * cmean ** 2) / np.maximum(k - 1, 1), 0, None)
    ) / np.sqrt(k)
    covered_mat = (cmean - ci_half <= true_mean) & (true_mean <= cmean + ci_half)
    del ci_half

    rows = np.arange(n)
    relw_stop = relw[rows, stop0]
    covered_stop = covered_mat[rows, stop0]
    k_stop = np.where(converged, stop0 + 1, 0)

    out = {
        "k_stop": k_stop.astype(int),
        "converged": converged,
        "eps_total": np.where(converged, eps * k_stop, np.nan),
        "relw_stop": relw_stop,
        "covered_stop": covered_stop,
        "k_last": np.where(converged, k_stop, maxq),
    }
    if keep_matrix:
        out["relw"] = relw
        out["covered"] = covered_mat
    return out


# ---------------------------------------------------------------------------
# [반복 요약 헬퍼]
# ---------------------------------------------------------------------------
def _summarize(values, label):
    """NaN(절단)을 제외한 분포 요약. n_valid는 항상 함께 반환한다."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"{label}_n_valid": 0, f"{label}_mean": np.nan, f"{label}_median": np.nan,
                f"{label}_std": np.nan, f"{label}_q25": np.nan, f"{label}_q75": np.nan,
                f"{label}_iqr": np.nan, f"{label}_q95": np.nan}
    q25, q50, q75, q95 = np.quantile(v, [0.25, 0.5, 0.75, 0.95])
    return {
        f"{label}_n_valid": int(v.size),
        f"{label}_mean": float(v.mean()),
        f"{label}_median": float(q50),
        f"{label}_std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        f"{label}_q25": float(q25),
        f"{label}_q75": float(q75),
        f"{label}_iqr": float(q75 - q25),
        f"{label}_q95": float(q95),
    }


# ---------------------------------------------------------------------------
# [KCI 시행 n회 반복 — seed를 바꾸어 개별 시행 + 통계]
# ---------------------------------------------------------------------------
def run_seq_n(true_mean, delta, eps, n_reps=500, seed0=2026,
              threshold=0.01, streak_required=5, maxq=500, alpha=0.05,
              keep_trajectories=False):
    """KCI 시행을 독립 시드로 n_reps회 반복하고 각 시행 결과 + 통계를 반환.

    seed는 seed0 + i (i = 0..n_reps-1). 각 시행은 run_seq_once()와
    동일한 규칙(동일한 난수열)으로 독립 수행되며, 벡터화로 한 번에 계산된다.

    여기서 나오는 eps_total_mean이 논문에서 KCI 방식이 제안하는 "총예산"이다.
    절단된 시행은 평균에서 제외되므로, 이 값은 반드시 converged_rate와 함께
    읽어야 한다.

    인자
      true_mean         : 참평균
      delta             : 질의 민감도
      eps               : 질의 1건당 예산
      n_reps            : 반복 횟수(논문 설정: 500)
      seed0             : i번째 시행은 seed0 + i를 쓴다
      keep_trajectories : 시행별 상대 폭 행렬도 반환할지(메모리 주의)

    반환
      (trials_df, summary_dict, traj_matrix 또는 None)
      trials_df : rep, seed, converged, k_stop, eps_total, relw_stop, covered
      summary   : eps_total_* 집계 + converged_rate + covered_rate + censored_rate
    """
    if n_reps < 1:
        raise ValueError("n_reps는 1 이상이어야 합니다")

    seeds = seed0 + np.arange(n_reps, dtype=np.int64)
    t0 = time.perf_counter()
    out = _seq_batch(true_mean, delta, eps, threshold, streak_required,
                     maxq, alpha, seeds, keep_matrix=keep_trajectories)
    elapsed = time.perf_counter() - t0

    trials = pd.DataFrame({
        "rep": np.arange(1, n_reps + 1),
        "seed": seeds,
        "converged": out["converged"],
        "k_stop": np.where(out["converged"], out["k_stop"], np.nan),
        "k_last": out["k_last"],
        "eps_total": out["eps_total"],
        "relw_stop": out["relw_stop"],
        "covered": out["covered_stop"],
    })

    summary = {
        "n_reps": int(n_reps),
        "eps": float(eps),
        "threshold": float(threshold),
        "streak_required": int(streak_required),
        "maxq": int(maxq),
        "converged_rate": float(np.mean(out["converged"])),
        "censored_rate": float(1.0 - np.mean(out["converged"])),
        "covered_rate": float(np.mean(out["covered_stop"])),
        "elapsed_sec": round(elapsed, 3),
        **_summarize(out["eps_total"], "eps_total"),
        **_summarize(out["k_stop"], "k_stop"),
    }
    traj = out.get("relw") if keep_trajectories else None
    return trials, summary, traj


# ---------------------------------------------------------------------------
# [시각화 — 1회 시행 궤적]
# ---------------------------------------------------------------------------
def plot_seq_once(h, title="", true_mean=None, k_stop=None, ylabel=""):
    """run_seq_once()의 반환값(히스토리 DataFrame)을 시각화한다."""
    if true_mean is None:
        raise ValueError("true_mean을 지정하세요 (원래는 노트북 전역변수 참조였음)")

    # fig는 액자, ax는 실제 그림 그리는 도화지
    fig, ax = plt.subplots(figsize=(11, 5))
    ok  = h[h.covered == True]
    bad = h[h.covered == False]

    # 데이터를 안정적인 것과 불안정한 것으로 나누어 시각화
    for g, c, lab in [(ok, "tab:blue", "Stable"), (bad, "tab:red", "Unstable")]:
        if len(g):
            # x축 = 질의 수(k), y축 = 추정 평균, 오차막대 = 신뢰구간 반폭(moe)
            ax.errorbar(g.k, g["mean"], yerr=g.moe,
                        fmt="o", ms=3, color=c,
                        ecolor=c, elinewidth=1.2, capsize=0, alpha=0.85, label=lab)

    # 기준선(실제 평균) 그리기
    ax.axhline(true_mean, ls="--", c="green", lw=1.2, label="True Mean")
    if k_stop is not None:
        ax.axvline(k_stop, ls="--", c="k", lw=1, label=f"Converged at {k_stop}")

    r = h.covered.mean()
    ax.set_title(f"{title} | Stable={r:.1%}, Unstable={1-r:.1%}")
    ax.set_xlabel("Query Count"); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# [시각화 — n회 반복 결과 분포]
# ---------------------------------------------------------------------------
def plot_seq_dist(trials, title=""):
    """run_seq_n()의 반환값 trials_df를 시각화한다(k_stop / eps_total 히스토그램)."""
    ok = trials[trials.converged]
    n_conv = len(ok)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].hist(ok["k_stop"], bins=30, color="tab:blue", alpha=0.8)
    if len(ok):
        axes[0].axvline(ok["k_stop"].median(), ls="--", c="k", lw=1.2,
                        label=f"median={ok['k_stop'].median():.0f}")
        axes[0].legend()
    axes[0].set_title(f"k_stop (converged {n_conv}/{len(trials)})")
    axes[0].set_xlabel("k_stop"); axes[0].set_ylabel("시행 수")
    axes[0].grid(alpha=0.3)

    axes[1].hist(ok["eps_total"], bins=30, color="tab:green", alpha=0.8)
    if len(ok):
        axes[1].axvline(ok["eps_total"].mean(), ls="--", c="k", lw=1.2,
                        label=f"mean={ok['eps_total'].mean():.2f}")
        axes[1].legend()
    axes[1].set_title("eps_total = eps × k_stop")
    axes[1].set_xlabel("eps_total"); axes[1].set_ylabel("시행 수")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, y=1.02)
    plt.tight_layout(); plt.show()




# ===========================================================================
# [역산 방식 — CAPRI형 역산 알고리즘] (step7 #261106 새로운 방법론 판)
#   라운드(바깥) : 후보 ε_total → k_max = floor(ε_total / ε)
#   시행(안쪽)   : num_runs번 독립 시행 (시행 내 step 재시도 없음)
#   판정         : 성공 비율 ≥ success_rate_threshold 이면 합격
#   출력         : 합격 라운드에서 '성공한 시행'의 실제 소모 ε 평균(기대 소모량)
#                  + 합격 라운드의 k_eval·ε(보장 예산)은 info로 함께 반환
# ===========================================================================

# ---------------------------------------------------------------------------
# [공용 헬퍼 — 원본 노트북 셀 9/11에서 이동]
# ---------------------------------------------------------------------------
def _validate_inputs(eps, threshold, streak_required, maxq):
    if eps <= 0:
        raise ValueError("eps는 0보다 커야 합니다.")
    if threshold <= 0:
        raise ValueError("threshold는 0보다 커야 합니다.")
    if streak_required < 1:
        raise ValueError("streak_required는 1 이상이어야 합니다.")
    if maxq < 2:
        raise ValueError("maxq는 2 이상이어야 합니다.")


def _statistics(values, true_mean, alpha=0.05):
    """현재까지의 응답값으로 누적 통계와 95% CI를 계산한다."""
    values = np.asarray(values, dtype=float)

    # 응답 값 개수로 k를 계산함
    k = len(values)
    cumulative_mean = float(values.mean())

    if k < 2:
        return {
            "k": k, "response": float(values[-1]),
            "cumulative_mean": cumulative_mean,
            "sample_std": np.nan, "standard_error": np.nan,
            "moe": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
            "ci_width": np.nan, "relative_width": np.nan,
            "hit": False, "covered": pd.NA,
        }

    sample_std = float(values.std(ddof=1))
    standard_error = sample_std / np.sqrt(k)
    t_value = stats.t.ppf(1 - alpha / 2, df=k - 1)
    moe = float(t_value * standard_error)
    ci_lo = cumulative_mean - moe
    ci_hi = cumulative_mean + moe
    ci_width = float(2 * moe)
    relative_width = float(ci_width / abs(cumulative_mean)) if cumulative_mean != 0 else np.inf
    covered = bool(ci_lo <= true_mean <= ci_hi)

    return {
        "k": k, "response": float(values[-1]),
        "cumulative_mean": cumulative_mean,
        "sample_std": sample_std, "standard_error": standard_error,
        "moe": moe, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "ci_width": ci_width, "relative_width": relative_width,
        "hit": False, "covered": covered,
    }


def _overlap_from_ci(ci_lo, ci_hi, delta):
    """원본 CI를 -delta만큼 이동한 가상 인접 CI와 비교한다."""
    if not (np.isfinite(ci_lo) and np.isfinite(ci_hi)):
        return {"adj_ci_lo": np.nan, "adj_ci_hi": np.nan,
                "overlap_width": np.nan, "overlap_ratio": np.nan,
                "width_to_delta": np.nan}
    adj_ci_lo = ci_lo - delta
    adj_ci_hi = ci_hi - delta
    overlap_width = max(0.0, min(ci_hi, adj_ci_hi) - max(ci_lo, adj_ci_lo))
    ci_width = ci_hi - ci_lo
    return {
        "adj_ci_lo": adj_ci_lo,
        "adj_ci_hi": adj_ci_hi,
        "overlap_width": overlap_width,
        "overlap_ratio": overlap_width / ci_width if ci_width > 0 else np.nan,
        "width_to_delta": ci_width / delta if delta > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# [역산 시행 1회 — CAPRI형 역산 알고리즘]
# ---------------------------------------------------------------------------
def _inv_simulate_trial(true_mean, delta, eps, k_max, threshold, streak_required,
                        alpha, rng, keep_rows=False):
    """시행 1회: 질의를 1건씩 추가하며 streak 판정. step 재시도 없음."""
    scale = delta / eps
    responses, rows = [], []
    streak = 0
    stopped_k, last = None, None

    for k in range(1, k_max + 1):
        responses.append(float(true_mean + rng.laplace(0.0, scale)))
        row = _statistics(responses, true_mean, alpha)
        row.update(_overlap_from_ci(row["ci_lo"], row["ci_hi"], delta))

        hit = bool(np.isfinite(row["relative_width"])
                   and row["relative_width"] < threshold)
        streak = streak + 1 if hit else 0
        row.update({"query_k": k, "hit": hit, "streak_count": streak,
                    "actual_eps_total": eps * k})
        if keep_rows:
            rows.append(row)
        last = row

        if streak >= streak_required:
            stopped_k = k
            break

    return stopped_k, last, rows


def run_inv_once(true_mean, delta, eps, initial_eps_total, step,
                 threshold=0.01, streak_required=5,
                 success_rate_threshold=0.95, num_runs=50, max_rounds=20,
                 maxq=500, seed0=2026, alpha=0.05, eps_decimals=2,
                 keep_trajectories="final", verbose=False):
    """CAPRI형 역산 알고리즘의 단일 수행. (원본명: run_inverse_live)

    라운드(바깥) : 후보 ε_total → k_max = floor(ε_total / ε)
    시행(안쪽)   : num_runs번 독립 시행 (시행 내 step 재시도 없음)
    판정         : 성공 비율 ≥ success_rate_threshold 이면 합격
    출력         : 합격 라운드에서 '성공한 시행'의 실제 소모 ε 평균(기대 소모량)
                   + 합격 라운드의 k_eval·ε(보장 예산)은 info로 함께 반환

    인자
      true_mean : 참평균 (원본은 data["true_mean"] → 인자로 전달)
      delta     : 질의 민감도 (원본은 data["delta"] → 인자로 전달)

    반환: (trial_results, round_summary, trajectories, final_eps, info)
      - trial_results : 마지막(합격/종료) 라운드의 시행별 결과
      - round_summary : 라운드별 후보 예산, k_eval, 성공 수, 성공 비율
      - trajectories  : 질의 시점 궤적 (keep_trajectories 설정에 따름)
      - final_eps     : 성공 시행의 실제 소모 ε 평균 (기대 소모량, 반올림)
      - info          : status, eps_total_guaranteed(보장 예산), success_rate 등
    """
    _validate_inputs(eps, threshold, streak_required, maxq)
    if initial_eps_total <= 0:
        raise ValueError("initial_eps_total은 0보다 커야 합니다.")
    if step <= 1:
        raise ValueError("step은 1보다 커야 합니다.")
    if max_rounds < 1:
        raise ValueError("max_rounds는 1 이상이어야 합니다.")
    if num_runs < 1:
        raise ValueError("num_runs는 1 이상이어야 합니다.")
    if not (0 < success_rate_threshold <= 1):
        raise ValueError("success_rate_threshold는 (0, 1] 범위여야 합니다.")
    if keep_trajectories not in ("none", "final", "all"):
        raise ValueError("keep_trajectories는 'none', 'final', 'all' 중 하나입니다.")

    keep_rows = keep_trajectories != "none"
    round_rows, all_traj = [], []
    last_trials, last_traj, last_round = None, [], None
    status = "max_rounds_exceeded"
    eps_total = float(initial_eps_total)

    # ---------------------------------------------------------------- 라운드
    for round_no in range(1, max_rounds + 1):
        candidate = float(eps_total)
        k_max = int(candidate // eps)

        if k_max < 2:                       # 판정 불가 → 성공 비율 0으로 보고 증가
            round_rows.append({
                "round": round_no, "candidate_eps_total": candidate,
                "k_max": k_max, "k_eval": k_max, "eps_total_eval": eps * k_max,
                "n_success": 0, "success_rate": 0.0, "passed": False,
                "maxq_reached": False, "note": "k_max<2",
            })
            eps_total *= step
            continue

        k_eval = min(k_max, maxq)
        maxq_reached = k_max >= maxq

        # ------------------------------------------------------------ 시행
        trials, traj = [], []
        for rep in range(1, num_runs + 1):
            seed = int(seed0 + rep - 1)
            round_seed = int(seed + round_no * 1_000_003)   # 라운드마다 새 시드
            rng = np.random.default_rng(round_seed)

            stopped_k, last, rows = _inv_simulate_trial(
                true_mean, delta, eps, k_eval, threshold, streak_required,
                alpha, rng, keep_rows=keep_rows)
            ok = stopped_k is not None

            trials.append({
                "method": "INVERSE", "round": round_no, "rep": rep,
                "seed": seed, "round_seed": round_seed,
                "eps": float(eps), "k_eval": k_eval,
                "eps_total_eval": float(eps * k_eval),
                "success": ok,
                "k_stop": int(stopped_k) if ok else np.nan,
                "actual_eps_total": float(eps * stopped_k) if ok else np.nan,
                "cumulative_mean": last["cumulative_mean"],
                "ci_width": last["ci_width"],
                "relative_width": last["relative_width"],
                "covered": last["covered"],
                "threshold": float(threshold),
                "streak_required": int(streak_required),
            })
            if keep_rows and rows:
                df = pd.DataFrame(rows)
                df["round"], df["rep"], df["round_seed"] = round_no, rep, round_seed
                df["success"] = ok
                traj.append(df)

        trials_df = pd.DataFrame(trials)
        n_success = int(trials_df["success"].sum())
        rate = n_success / num_runs
        passed = rate >= success_rate_threshold

        round_rows.append({
            "round": round_no, "candidate_eps_total": candidate,
            "k_max": k_max, "k_eval": k_eval, "eps_total_eval": eps * k_eval,
            "n_success": n_success, "success_rate": rate, "passed": passed,
            "maxq_reached": maxq_reached, "note": "",
        })
        if keep_trajectories == "all":
            all_traj.extend(traj)
        last_trials, last_traj, last_round = trials_df, traj, round_rows[-1]

        if verbose:
            print(f"[round {round_no:>3}] cand={candidate:>8.4f} | k_eval={k_eval:>4} | "
                  f"success={n_success}/{num_runs} ({rate:.3f}) | pass={passed}")

        if passed:
            status = "converged"
            break
        if maxq_reached:                    # maxq까지 가도 임계 미달
            status = "maxq_reached"
            break
        eps_total *= step

    # ---------------------------------------------------------------- 출력
    round_summary = pd.DataFrame(round_rows)
    trial_results = last_trials if last_trials is not None else pd.DataFrame()

    if keep_trajectories == "final" and last_traj:
        trajectories = pd.concat(last_traj, ignore_index=True)
    elif keep_trajectories == "all" and all_traj:
        trajectories = pd.concat(all_traj, ignore_index=True)
    else:
        trajectories = pd.DataFrame()

    if status == "converged":
        succ = trial_results.loc[trial_results["success"], "actual_eps_total"]
        final_eps_raw = float(succ.mean())
        final_eps = round(final_eps_raw, eps_decimals)
        eps_total_guaranteed = float(last_round["eps_total_eval"])
    else:
        final_eps_raw = final_eps = eps_total_guaranteed = np.nan

    info = {
        "status": status,
        "final_eps": final_eps,                       # 기대 소모량 (출력)
        "final_eps_raw": final_eps_raw,
        "eps_total_guaranteed": eps_total_guaranteed, # 보장 예산
        "success_rate": last_round["success_rate"] if last_round else np.nan,
        "n_success": last_round["n_success"] if last_round else 0,
        "num_runs": num_runs,
        "rounds_used": len(round_rows),
        "success_rate_threshold": success_rate_threshold,
        "eps": float(eps), "threshold": float(threshold),
        "streak_required": int(streak_required),
    }

    if verbose:
        print(f"\n[요약] status={status} | 성공 비율={info['success_rate']} | "
              f"기대 소모량 ε={final_eps} | 보장 예산 ε={eps_total_guaranteed}")

    return trial_results, round_summary, trajectories, final_eps, info


# ---------------------------------------------------------------------------
# [역산 n회 반복 — seed0을 바꾸어 독립 수행 + 통계]
# ---------------------------------------------------------------------------
def run_inv_n(true_mean, delta, eps, initial_eps_total, step,
              n_reps=20, seed0_base=2026, seed_stride=10_000_019,
              threshold=0.01, streak_required=5, success_rate_threshold=0.95,
              num_runs=50, max_rounds=20, maxq=500, alpha=0.05,
              eps_decimals=2, verbose=False):
    """역산 알고리즘(run_inv_once)을 seed0을 바꾸어 n_reps회 독립 수행하고
    각 시행 결과 + 통계 요약을 반환한다.

    시드 규칙: i번째 시행의 seed0 = seed0_base + i * seed_stride
    (시드 블록이 겹치지 않도록 통합본 §8의 seed_stride 규칙을 따른다)

    인자
      n_reps : 역산 알고리즘 전체 반복 횟수 (역산 1회 = num_runs번의 내부 시행 포함)

    반환
      (trials_df, summary)
      trials_df : seed_idx, seed0, status, rounds_used, success_rate,
                  n_success, final_eps, eps_total_guaranteed
      summary   : converged_rate + final_eps_* 집계 + eps_total_guaranteed_* 집계
    """
    rows = []
    t0 = time.perf_counter()

    for i in range(n_reps):
        s0 = int(seed0_base + i * seed_stride)
        trial_results, round_summary, trajectories, final_eps, info = run_inv_once(
            true_mean, delta, eps, initial_eps_total, step,
            threshold=threshold, streak_required=streak_required,
            success_rate_threshold=success_rate_threshold,
            num_runs=num_runs, max_rounds=max_rounds, maxq=maxq,
            seed0=s0, alpha=alpha, eps_decimals=eps_decimals,
            keep_trajectories="none", verbose=False)

        rows.append({
            "seed_idx": i + 1, "seed0": s0,
            "status": info["status"],
            "rounds_used": info["rounds_used"],
            "success_rate": info["success_rate"],
            "n_success": info["n_success"],
            "final_eps": info["final_eps"],
            "eps_total_guaranteed": info["eps_total_guaranteed"],
        })
        if verbose:
            print(f"[seed {i+1:>3}/{n_reps}] seed0={s0} | {info['status']:<20} | "
                  f"보장={info['eps_total_guaranteed']} | 기대={info['final_eps']}")

    trials = pd.DataFrame(rows)
    elapsed = time.perf_counter() - t0

    summary = {
        "n_reps": int(n_reps),
        "num_runs": int(num_runs),          # 역산 1회당 내부 시행 수
        "eps": float(eps),
        "threshold": float(threshold),
        "streak_required": int(streak_required),
        "success_rate_threshold": float(success_rate_threshold),
        "converged_rate": float(trials["eps_total_guaranteed"].notna().mean()),
        "elapsed_sec": round(elapsed, 3),
        **_summarize(trials["final_eps"], "final_eps"),
        **_summarize(trials["eps_total_guaranteed"], "eps_guaranteed"),
        **_summarize(trials["success_rate"], "success_rate"),
    }
    return trials, summary


# ---------------------------------------------------------------------------
# [시각화 — 역산 1회 결과 (원본 셀 52에서 이동)]
# ---------------------------------------------------------------------------
def plot_inv_search(round_summary, info, ax=None):
    """① 라운드별 후보 예산 대비 성공 비율."""
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    rs = round_summary
    ax.plot(rs["eps_total_eval"], rs["success_rate"], "-o", ms=4, lw=1,
            color="tab:blue", label="Success rate per round")
    ax.axhline(info["success_rate_threshold"], ls="--", color="gray",
               label=f"τ = {info['success_rate_threshold']}")
    if info["status"] == "converged":
        acc = rs[rs["passed"]].iloc[-1]
        ax.plot(acc["eps_total_eval"], acc["success_rate"], "*", ms=14,
                color="tab:red", label=f"Accepted: ε_total = {acc['eps_total_eval']:.2f}")
    ax.set_xlabel("Candidate total budget (k_eval · ε)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Budget search")
    ax.legend(fontsize=8, loc="lower right")
    return ax


def plot_inv_consumption(trial_results, info, ax=None, seed=0):
    """② 합격 라운드의 실제 소모 ε: ECDF(전체 시행 기준) + strip plot."""
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    n = len(trial_results)
    vals = np.sort(trial_results.loc[trial_results["success"],
                                     "actual_eps_total"].to_numpy(float))
    if vals.size == 0:
        ax.text(0.5, 0.5, "No successful trials", ha="center", transform=ax.transAxes)
        return ax

    # ECDF: 분모 = 전체 시행 수 → 곡선의 최종 높이 = 성공 비율
    y = np.arange(1, vals.size + 1) / n
    ax.step(np.r_[vals[0], vals], np.r_[0, y], where="post",
            color="tab:blue", lw=1.5, label="ECDF (all trials)")

    # strip plot (개별 시행)
    rng = np.random.default_rng(seed)
    ax.scatter(vals, -0.06 + rng.uniform(-0.02, 0.02, vals.size),
               s=6, alpha=0.4, color="tab:blue")

    mean_eps = info["final_eps_raw"]
    guar = info["eps_total_guaranteed"]
    p_mean = np.sum(vals <= mean_eps) / n
    ax.axvline(mean_eps, color="tab:green", ls="-",
               label=f"Expected consumption (output) = {mean_eps:.2f}  [{p_mean:.2f}]")
    ax.axvline(guar, color="tab:red", ls="--",
               label=f"Guaranteed budget = {guar:.2f}  [{info['success_rate']:.2f}]")
    ax.axhline(info["success_rate_threshold"], color="gray", ls=":", lw=1)

    ax.set_xlabel("Actual consumed ε (ε · k_stop)")
    ax.set_ylabel("Fraction of trials finished")
    ax.set_ylim(-0.1, 1.03)
    ax.set_title(f"Consumed budget (failed: {n - vals.size}/{n})")
    ax.legend(fontsize=7, loc="upper left")
    return ax


def plot_inv_trajectories(trajectories, info, n_show=30, ax=None, seed=0):
    """③ 합격 라운드의 상대폭 궤적 (일부 시행만 표시)."""
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    if trajectories.empty:
        ax.text(0.5, 0.5, "No trajectories stored", ha="center", transform=ax.transAxes)
        return ax
    tr = trajectories[trajectories["round"] == trajectories["round"].max()]
    reps = tr["rep"].unique()
    rng = np.random.default_rng(seed)
    show = rng.choice(reps, size=min(n_show, len(reps)), replace=False)

    for rep in show:
        d = tr[tr["rep"] == rep]
        ok = bool(d["success"].iloc[0])
        ax.plot(d["query_k"], d["relative_width"], lw=0.8, alpha=0.5,
                color="tab:blue" if ok else "tab:red")
        if ok:
            ax.plot(d["query_k"].iloc[-1], d["relative_width"].iloc[-1],
                    "o", ms=3, color="black")

    ax.axhline(info["threshold"], color="gray", ls="--",
               label=f"threshold = {info['threshold']}")
    ax.set_yscale("log")
    ax.set_xlabel("Number of queries k")
    ax.set_ylabel("Relative CI width")
    ax.set_title(f"Trajectories (streak = {info['streak_required']}; "
                 f"blue: success, red: fail)")
    ax.legend(fontsize=8)
    return ax


def plot_inv_summary(trial_results, round_summary, trajectories, info,
                     n_show=30, figsize=(15, 4)):
    """①②③을 한 줄로."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_inv_search(round_summary, info, ax=axes[0])
    plot_inv_consumption(trial_results, info, ax=axes[1])
    plot_inv_trajectories(trajectories, info, n_show=n_show, ax=axes[2])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# [시각화 — 역산 n회 반복 결과 분포]
# ---------------------------------------------------------------------------
def plot_inv_dist(trials, title=""):
    """run_inv_n()의 반환값 trials_df를 시각화한다(기대 소모량 / 보장 예산 히스토그램)."""
    ok = trials[trials["final_eps"].notna()]
    n_conv = len(ok)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].hist(ok["final_eps"], bins=20, color="tab:blue", alpha=0.8)
    if len(ok):
        axes[0].axvline(ok["final_eps"].mean(), ls="--", c="k", lw=1.2,
                        label=f"mean={ok['final_eps'].mean():.2f}")
        axes[0].legend()
    axes[0].set_title(f"final_eps, 기대 소모량 (converged {n_conv}/{len(trials)})")
    axes[0].set_xlabel("final_eps"); axes[0].set_ylabel("시행 수")
    axes[0].grid(alpha=0.3)

    axes[1].hist(ok["eps_total_guaranteed"], bins=20, color="tab:orange", alpha=0.8)
    if len(ok):
        axes[1].axvline(ok["eps_total_guaranteed"].mean(), ls="--", c="k", lw=1.2,
                        label=f"mean={ok['eps_total_guaranteed'].mean():.2f}")
        axes[1].legend()
    axes[1].set_title("eps_total_guaranteed, 보장 예산")
    axes[1].set_xlabel("eps_total_guaranteed"); axes[1].set_ylabel("시행 수")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, y=1.02)
    plt.tight_layout(); plt.show()




# ===========================================================================
# [총예산 기준 성공률 검정 — step7 셀 33 "토탈 엡실론을 입력하고 시행 성공률 계산" 판]
#   총 예산(eps_total)을 입력으로 받아, 그 예산 안에서 KCI 정지 조건
#   (상대폭 < threshold를 streak_required회 연속 충족)을 만족하는 시행의
#   비율을 독립적인 n번의 시행으로 통계적으로 추정한다.
# ===========================================================================

def _kci_stop_times(true_mean, delta, eps, threshold, streak_required,
                    num_runs, k_cap, seed0=0, alpha=0.05, min_queries=2):
    """KCI 정지 규칙을 시행별로 k_cap개 질의까지 시뮬레이션.
    반환: seeds, stop_k(정지 못하면 0), relw 행렬, 누적평균 행렬"""
    k_cap = int(k_cap)
    seeds = seed0 + np.arange(num_runs)              # 시행 시드 = seed0 + rep - 1
    if k_cap < max(2, min_queries):                  # CI 계산 불가 → 전부 미수렴
        return seeds, np.zeros(num_runs, dtype=int), None, None

    scale = delta / eps
    X = np.empty((num_runs, k_cap))
    for r, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        X[r] = true_mean + rng.laplace(0.0, scale, size=k_cap)

    k = np.arange(1, k_cap + 1)
    Y = X - X[:, :1]                                 # 수치 안정용 이동(분산 불변)
    S1 = np.cumsum(Y, axis=1)
    S2 = np.cumsum(Y ** 2, axis=1)
    mean = X[:, :1] + S1 / k

    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.maximum((S2 - S1 ** 2 / k) / (k - 1), 0.0)   # 표본분산(ddof=1)
        tcrit = np.full(k_cap, np.nan)
        tcrit[1:] = stats.t.ppf(1 - alpha / 2, k[1:] - 1)
        moe = tcrit * np.sqrt(var / k)
        relw = 2 * moe / np.abs(mean)

    hit = relw < threshold                           # NaN → False
    hit[:, :max(min_queries, 2) - 1] = False

    streak = np.zeros(num_runs, dtype=int)
    stop_k = np.zeros(num_runs, dtype=int)
    for j in range(k_cap):
        streak = np.where(hit[:, j], streak + 1, 0)
        newly = (stop_k == 0) & (streak >= streak_required)
        stop_k[newly] = j + 1
    return seeds, stop_k, relw, mean


def _wilson(n_succ, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = n_succ / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (c - h, c + h)


def check_s_rate(true_mean, delta, eps, eps_total, threshold,
                 streak_required, num_runs=1000, seed0=0,
                 alpha=0.05, maxq=200, min_queries=2,
                 return_trials=True):
    """총예산 eps_total 하에서 KCI 시행 중 정지 규칙을 만족(수렴)한 횟수·비율 계산.
    (원본명: kci_success_under_budget)"""
    k_max = int(np.floor(eps_total / eps + 1e-9))    # 3.0/0.1 = 29.999... 방지
    k_eval = min(k_max, maxq)

    seeds, stop_k, relw, mean = _kci_stop_times(
        true_mean, delta, eps, threshold, streak_required,
        num_runs, k_eval, seed0, alpha, min_queries)

    conv = stop_k > 0
    n_conv = int(conv.sum())
    spent = np.where(conv, stop_k * eps, k_eval * eps)
    lo, hi = _wilson(n_conv, num_runs)

    summary = {
        "eps_total_input": eps_total,
        "k_max": k_max,
        "k_eval": k_eval,
        "capped_by_maxq": k_max > maxq,
        "num_runs": num_runs,
        "n_converged": n_conv,
        "n_not_converged": num_runs - n_conv,
        "success_rate": n_conv / num_runs,
        "success_rate_ci95": (lo, hi),
        "mean_eps_spent_converged": float(spent[conv].mean()) if n_conv else np.nan,
        "median_eps_spent_converged": float(np.median(spent[conv])) if n_conv else np.nan,
    }

    if not return_trials:
        return summary, None

    idx = np.arange(num_runs)
    last = np.where(conv, stop_k, k_eval) - 1
    trials = pd.DataFrame({
        "trial": idx + 1,
        "seed": seeds,
        "converged": conv,
        "stop_k": np.where(conv, stop_k, np.nan),
        "eps_spent": spent,
        "final_relw": relw[idx, last] if relw is not None else np.nan,
        "final_mean": mean[idx, last] if mean is not None else np.nan,
    })
    return summary, trials


def check_s_rate_sweep(true_mean, delta, eps, eps_totals, threshold,
                       streak_required, num_runs=1000, seed0=0,
                       alpha=0.05, maxq=200, min_queries=2):
    """maxq까지 한 번만 시뮬레이션하고, 여러 총예산에 대한 수렴 비율을 한꺼번에 계산.
    (원본명: kci_success_sweep)"""
    _, stop_k, _, _ = _kci_stop_times(
        true_mean, delta, eps, threshold, streak_required,
        num_runs, maxq, seed0, alpha, min_queries)

    rows = []
    for B in eps_totals:
        k_max = int(np.floor(B / eps + 1e-9))
        k_eval = min(k_max, maxq)
        conv = (stop_k > 0) & (stop_k <= k_eval)
        n_conv = int(conv.sum())
        lo, hi = _wilson(n_conv, num_runs)
        rows.append({
            "eps_total": B, "k_max": k_max, "capped_by_maxq": k_max > maxq,
            "n_converged": n_conv, "success_rate": n_conv / num_runs,
            "ci95_lo": lo, "ci95_hi": hi,
            "mean_eps_spent_converged":
                float((stop_k[conv] * eps).mean()) if n_conv else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# [시각화 — check_s_rate 결과]
#   통합본 시각화 카탈로그의 컨벤션을 따른다:
#   - ECDF를 주 그림으로 (plot_kci_ecdf: "이 한 패널에서 세 가지를 읽는다")
#   - 절단(미수렴) 시행은 항상 별도로 표시 (plot_kci_hist의 절단 막대 규칙)
#   - 기준선(τ, threshold)은 점선 (plot_inverse_search 규칙)
# ---------------------------------------------------------------------------
def plot_srate_summary(summary, trials, title=""):
    """check_s_rate()의 반환값을 3패널로 시각화한다.

      ① 성공률 게이지 막대 + Wilson 95% 신뢰구간
      ② 실제 소모 ε의 ECDF (곡선 최종 높이 = 성공 비율, 절단 시행 포함)
      ③ 수렴 시행의 정지 시점(k_stop) 히스토그램 + 예산 상한선
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── ① 성공률 게이지 ──────────────────────────────────────────────
    ax = axes[0]
    rate = summary["success_rate"]
    lo, hi = summary["success_rate_ci95"]
    ax.bar([0], [rate], width=0.6, color="tab:blue", alpha=0.8,
           label=f"success rate = {rate:.3f}")
    ax.errorbar([0], [rate],
                yerr=[[max(0.0, rate - lo)], [max(0.0, hi - rate)]], fmt="none",
                ecolor="k", capsize=5, lw=1.2, label=f"Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
    ax.axhline(0.95, ls="--", c="gray", lw=1, label="τ = 0.95 (참고)")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    ax.set_ylabel("성공 비율")
    ax.set_title(f"① 총예산 {summary['eps_total_input']}에서의 성공률 "
                 f"(n={summary['num_runs']})")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8, loc="lower right")

    # ── ② 실제 소모 ε ECDF ──────────────────────────────────────────
    ax = axes[1]
    vals = np.sort(trials["eps_spent"].to_numpy(float))
    n = len(vals)
    y = np.arange(1, n + 1) / n
    ax.step(np.r_[vals[0], vals], np.r_[0, y], where="post",
            color="tab:blue", lw=1.5, label="ECDF (전체 시행)")
    # 곡선 최종 높이 = 성공 비율이 되도록 전체 시행이 분모
    ax.axhline(rate, ls=":", c="tab:blue", lw=1, alpha=0.7)
    if summary["n_not_converged"]:
        cens = trials.loc[~trials["converged"], "eps_spent"]
        ax.scatter(cens, np.full(len(cens), -0.04), marker="|", s=80,
                   color="tab:red", alpha=0.5, label=f"미수렴(예산 소진) {len(cens)}개")
    ax.axvline(summary["mean_eps_spent_converged"], color="tab:green", ls="-",
               label=f"수렴 시행 평균 소모 = {summary['mean_eps_spent_converged']:.2f}")
    ax.set_xlabel("실제 소모 ε (ε · k_stop)")
    ax.set_ylabel("누적 비율")
    ax.set_ylim(-0.08, 1.03)
    ax.set_title("② 소모 예산 ECDF (최종 높이 = 성공 비율)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    # ── ③ 정지 시점 분포 ────────────────────────────────────────────
    ax = axes[2]
    ok = trials[trials["converged"]]
    ax.hist(ok["stop_k"], bins=30, color="tab:blue", alpha=0.8,
            label=f"수렴 시행 (n={len(ok)})")
    ax.axvline(summary["k_eval"], color="tab:red", ls="--", lw=1.2,
               label=f"예산 상한 k_eval = {summary['k_eval']}")
    if len(ok):
        ax.axvline(ok["stop_k"].median(), color="k", ls=":", lw=1.2,
                   label=f"중앙값 = {ok['stop_k'].median():.0f}")
    ax.set_xlabel("정지 시점 k_stop (질의 수)")
    ax.set_ylabel("시행 수")
    ax.set_title("③ 수렴 시행의 정지 시점 분포")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(title, y=1.02)
    plt.tight_layout(); plt.show()


def plot_srate_curve(sweep_df, title=""):
    """check_s_rate_sweep()의 반환값을 시각화한다.

      총예산 → 성공률 곡선 + Wilson 95% 신뢰구간 오차막대 + τ 기준선.
      역산 방식의 예산 탐색 그림(plot_inverse_search)과 나란히 비교할 수 있다.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d = sweep_df

    # Wilson CI 오차막대
    yerr = np.vstack([(d["success_rate"] - d["ci95_lo"]).to_numpy(),
                      (d["ci95_hi"] - d["success_rate"]).to_numpy()])
    ax.errorbar(d["eps_total"], d["success_rate"], yerr=yerr,
                fmt="-o", ms=4, lw=1.2, color="tab:blue",
                ecolor="tab:blue", elinewidth=0.8, capsize=2, alpha=0.9,
                label="성공 비율 (Wilson 95% CI)")

    # maxq로 잘린 구간은 점선으로 표시(해당 예산에서 전체 k를 평가하지 못함)
    capped = d[d["capped_by_maxq"]]
    if len(capped):
        ax.scatter(capped["eps_total"], capped["success_rate"],
                   facecolors="none", edgecolors="tab:red", s=70, lw=1.5,
                   label="maxq 절단 구간(참고용)")

    ax.set_xlabel("총 예산 ε_total")
    ax.set_ylabel("성공 비율")
    ax.set_ylim(-0.03, 1.06)
    ax.set_title(title or "총예산별 KCI 수렴(정지 규칙 충족) 비율")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout(); plt.show()



# ===========================================================================
# [예산 역산 — 지수 탐색 + 이진 탐색 하이브리드]
#   후보를 eps_total이 아니라 정수 질의 수 k 축에서 직접 진행한다.
#     f(k) = "k개 질의 예산에서 num_runs개 독립 시행의 성공 비율 >= tau ?"
#   성공 비율은 k에 대해 단조 증가이므로 f는 "거짓…거짓…참" 판정 함수다.
#   1단계(지수): k_init에서 출발해 f가 처음 참이 되는 k를 2배씩 뻗으며 포착
#                → 답은 (직전 실패 k, 최초 성공 k] 구간 안에 갇힌다
#   2단계(이진): 구간의 중앙값을 평가해 불변식 f(lo)=0, f(hi)=1을 유지하며
#                hi-lo=1이 될 때까지 조인다 → 최초 성공 k를 정확히 출력
#   seed_policy:
#     "fresh" — 후보 평가마다 새 시드(원본 CAPRI의 라운드별 독립 평가 유지)
#     "fixed" — 탐색 전체에서 같은 시행 시드 재사용(접두어 재사용, 판정이
#               결정적이 되어 f가 정확히 단조)
# ===========================================================================
def _eval_k(true_mean, delta, eps, threshold, streak_required, alpha,
            num_runs, k, seed0_eval, tau):
    """후보 k 하나를 평가한다. 반환: (성공 비율, 합격 여부, stop_k 배열)"""
    _, stop_k, _, _ = _kci_stop_times(
        true_mean, delta, eps, threshold, streak_required,
        num_runs, k, seed0_eval, alpha)
    conv = stop_k > 0
    rate = float(conv.mean())
    return rate, bool(rate >= tau), stop_k


def run_inv_hybrid(true_mean, delta, eps, threshold, streak_required,
                   tau=0.95, num_runs=100, maxq=500, k_init=2, beta=2,
                   seed0=2026, alpha=0.05, seed_policy="fresh",
                   seed_stride=1_000_003):
    """예산 역산을 k 축 하이브리드 탐색(지수 → 이진)으로 해결한다.

    원본 run_inv_once의 (initial_eps_total, step) 입력은 사라지고
    (k_init, beta)로 대체된다. 이진 단계는 hi-lo=1까지 조여
    성공 비율이 tau를 처음 넘는 최소 질의 수를 정확히 출력한다.

    인자
      k_init      : 최초 후보 질의 수 (기본 2 — k<2면 신뢰구간이 만들어지지 않음)
      beta        : 지수 단계 배율 (기본 2 = doubling search)
      seed_policy : "fresh"(후보 평가마다 새 시드) | "fixed"(시행 시드 고정)
      seed_stride : fresh 정책에서 평가 번호에 곱해지는 시드 보폭

    반환 dict
      status          : "converged" | "infeasible"
      k_min           : 성공 비율 >= tau 인 최소 질의 수 (최종 출력값)
      eps_guaranteed  : k_min * eps (보장 예산)
      eps_expected    : k_min 평가에서 성공 시행들의 실제 소모 eps 평균
      evals           : 후보 평가 기록 DataFrame (round, k, phase, success_rate, passed)
      queries_used    : 탐색 전체에서 소모한 총 질의 수 (num_runs × k의 합)
    """
    if seed_policy not in ("fresh", "fixed"):
        raise ValueError("seed_policy는 'fresh' 또는 'fixed'여야 합니다.")
    if k_init < 2:
        raise ValueError("k_init은 2 이상이어야 합니다 (k<2면 신뢰구간 없음).")
    if k_init > maxq:
        raise ValueError("k_init이 maxq보다 클 수 없습니다.")
    if beta <= 1:
        raise ValueError("beta는 1보다 커야 합니다.")

    evals, queries = [], 0
    eval_no = 0

    def evaluate(k):
        nonlocal eval_no, queries
        eval_no += 1
        # fresh: 평가마다 완전히 새로운 시드 블록 / fixed: 항상 같은 시드 블록
        s0 = seed0 if seed_policy == "fixed" else seed0 + eval_no * seed_stride
        rate, passed, stop_k = _eval_k(true_mean, delta, eps, threshold,
                                       streak_required, alpha, num_runs, k, s0, tau)
        queries += num_runs * k
        evals.append({"round": eval_no, "k": k, "success_rate": rate,
                      "passed": passed, "seed0_eval": s0})
        return passed, stop_k

    # ------------------------------------------------ 1단계: 지수 탐색 (괄호 치기)
    k_lo = None                              # f=0 확정점 (없으면 k_init이 곧 답)
    k_hi = k_init
    passed, stop_k_hi = evaluate(k_hi)
    if passed:
        # k_init에서 이미 합격 → k_init이 탐색 공간의 최소값이므로 곧 답
        conv = stop_k_hi > 0
        return {
            "status": "converged", "k_min": k_init,
            "eps_guaranteed": k_init * eps,
            "eps_expected": float((stop_k_hi[conv] * eps).mean()),
            "evals": pd.DataFrame(evals), "queries_used": queries,
            "stop_k": stop_k_hi,
        }

    while not passed:
        k_lo = k_hi                          # 실패 확정점 갱신
        k_hi = min(k_hi * beta, maxq)        # 2배씩 뻗되 상한(maxq) 클리핑
        if k_hi == k_lo:                     # 이미 maxq였는데 실패 → 불가능
            return {"status": "infeasible", "k_min": None,
                    "eps_guaranteed": np.nan, "eps_expected": np.nan,
                    "evals": pd.DataFrame(evals), "queries_used": queries,
                    "stop_k": None}
        passed, stop_k_hi = evaluate(k_hi)

    # ------------------------------------------------ 2단계: 이진 탐색 (조이기)
    # 불변식: f(k_lo)=0, f(k_hi)=1  → 답은 항상 (k_lo, k_hi] 안에 있다
    stop_k_min = stop_k_hi
    while k_hi - k_lo > 1:
        mid = (k_lo + k_hi) // 2
        passed_mid, stop_k_mid = evaluate(mid)
        if passed_mid:
            k_hi, stop_k_min = mid, stop_k_mid    # 성공 → 상한을 mid로 당김
        else:
            k_lo = mid                            # 실패 → 하한을 mid로 올림

    k_min = k_hi                             # 최초로 성공한 최소 질의 수
    conv = stop_k_min > 0
    return {
        "status": "converged", "k_min": int(k_min),
        "eps_guaranteed": float(k_min * eps),
        "eps_expected": float((stop_k_min[conv] * eps).mean()),
        "evals": pd.DataFrame(evals), "queries_used": queries,
        "stop_k": stop_k_min,
    }


# ===========================================================================
# [시각화 — run_inv_hybrid 결과]
#   통합본 시각화 카탈로그의 컨벤션을 따른다:
#   - 예산 탐색 그림: plot_inverse_search(성공률 곡선 + τ선 + 합격 별표)의
#     확장판 — 지수/이진 두 단계를 마커로 구분하고 궤적을 선으로 연결
#   - 소모량 ECDF: plot_inverse_consumption의 규칙(전체 시행 분모, 기대/보장선)
#   - 판정 함수의 단조성 확인: 기록된 (k, passed) 점들이 "거짓…거짓…참"인지
#     한눈에 검증하는 패널(이진 탐색의 전제 조건 점검)
# ===========================================================================
def plot_hybrid_search(res, title=""):
    """run_inv_hybrid() 결과의 탐색 과정을 2패널로 시각화한다.

      ① 탐색 궤적: 후보 k 평가 순서(지수 단계 → 이진 단계), 합격/불합격,
         최종 k_min 표시
      ② 판정 함수: 가로축 k, 세로축 성공 비율 — 기록된 평가 점들 + τ 기준선
         (기록된 점들이 "거짓…거짓…참"으로 단조인지 눈으로 확인)
    """
    ev = res["evals"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── ① 탐색 궤적 (평가 순서) ─────────────────────────────────────
    ax = axes[0]
    seq = ev["round"].to_numpy()
    ks = ev["k"].to_numpy()
    passed = ev["passed"].to_numpy(dtype=bool)
    # 지수/이진 단계 구분: 첫 합격 라운드 이후가 이진 단계
    first_pass = int(np.argmax(passed))
    ax.plot(seq[:first_pass + 1], ks[:first_pass + 1], "-o", ms=5,
            color="tab:orange", label="지수 단계 (×beta)")
    if first_pass + 1 < len(seq):
        ax.plot(seq[first_pass:], ks[first_pass:], "-s", ms=5,
                color="tab:blue", label="이진 단계 (중앙값)")
    ax.scatter(seq[passed], ks[passed], s=150, marker="*",
               color="tab:red", zorder=5, label="합격 평가")
    if res["k_min"] is not None:
        ax.axhline(res["k_min"], ls="--", c="green", lw=1.2,
                   label=f"k_min = {res['k_min']}")
        ax.scatter([seq[-1]], [res["k_min"]], s=120, marker="D",
                   color="green", zorder=5)
    ax.set_xlabel("평가 순서 (라운드)")
    ax.set_ylabel("후보 질의 수 k")
    ax.set_title("① 탐색 궤적: 지수 단계로 경계 포착 → 이진 단계로 조임")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # ── ② 판정 함수의 단조성 ────────────────────────────────────────
    ax = axes[1]
    ax.scatter(ks[~passed], ev["success_rate"].to_numpy()[~passed],
               s=45, color="tab:red", label="실패 (성공률 < τ)")
    ax.scatter(ks[passed], ev["success_rate"].to_numpy()[passed],
               s=45, color="tab:blue", marker="^", label="합격 (성공률 ≥ τ)")
    ax.axhline(0.95, ls="--", c="gray", lw=1)
    if res["k_min"] is not None:
        ax.axvline(res["k_min"], ls="--", c="green", lw=1.2,
                   label=f"k_min = {res['k_min']}")
    ax.set_xscale("log", base=2)          # 지수 단계 후보들이 등간격으로 보이게
    ax.set_xlabel("후보 질의 수 k (log2 눈금)")   # 아래첨자 2는 폰트에 없을 수 있어 log2로 표기
    ax.set_ylabel("성공 비율")
    ax.set_title("② 판정 함수 f(k): 단조 증가 확인 (이진 탐색의 전제)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(title or (f"하이브리드 예산 역산 "
                           f"[{res.get('status', '?')}]"), y=1.02)
    plt.tight_layout(); plt.show()
    return fig


def plot_hybrid_consumption(res, title=""):
    """run_inv_hybrid() 최종 후보(k_min) 평가의 소모 예산을 시각화한다.

      소모 ε의 ECDF + 기대 소모량(평균)·보장 예산 수직선.
      plot_inverse_consumption / plot_srate_summary ② 패널과 같은 규칙.
      res["stop_k"]가 있어야 그린다(항상 반환됨).
    """
    if res.get("stop_k") is None:
        raise ValueError("stop_k 기록이 없습니다(status가 infeasible이면 소모량 그림이 없다).")
    stop_k = res["stop_k"]
    n = len(stop_k)
    eps = res["eps_guaranteed"] / res["k_min"]          # 역산: eps = 보장예산 / k_min
    spent = np.where(stop_k > 0, stop_k * eps, res["k_min"] * eps)
    conv = stop_k > 0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = np.sort(spent)
    y = np.arange(1, n + 1) / n
    ax.step(np.r_[vals[0], vals], np.r_[0, y], where="post",
            color="tab:blue", lw=1.5, label="ECDF (전체 시행)")
    if (~conv).any():
        ax.scatter(spent[~conv], np.full((~conv).sum(), -0.04), marker="|",
                   s=80, color="tab:red", alpha=0.5,
                   label=f"예산 소진(미수렴) {(~conv).sum()}개")
    ax.axvline(res["eps_expected"], color="tab:green", ls="-",
               label=f"기대 소모량 = {res['eps_expected']:.2f}")
    ax.axvline(res["eps_guaranteed"], color="tab:red", ls="--",
               label=f"보장 예산 = {res['eps_guaranteed']:.2f}")
    ax.set_xlabel("실제 소모 ε (ε · k_stop)")
    ax.set_ylabel("누적 비율")
    ax.set_ylim(-0.08, 1.03)
    ax.set_title(title or (f"k_min={res['k_min']} 후보의 소모 예산 "
                           f"(총 {res['queries_used']:,} 질의로 탐색)"))
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.show()
    return fig


# ===========================================================================
# [하이브리드 역산 × num_runs/시드 스윕] — 원본 run_inverse_seed_sweep(§8)의
#   하이브리드(fixed) 판 이식본.
#   목적: (1) 시드를 바꿨을 때 k_min이 얼마나 움직이는가(재현성)
#         (2) num_runs(판정 시행 수)를 키우면 그 움직임이 줄어드는가
#   하이브리드 자체는 고정 시드(fixed)로 판정을 결정적으로 만들므로, 여기서의
#   "시드"는 판정 표본을 이루는 시행 시드 블록(seed0)을 가리킨다.
#   시드 규칙: i번째 스윕 시드 = seed0_base + i * seed_stride (원본 §8 동일)
# ===========================================================================
def _hybrid_swarm_offsets(values, width=0.30, bin_w=None):
    """같은 값을 가진 점들이 겹치지 않게 옆으로 밀어내는 오프셋(beeswarm)."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return np.zeros(0)
    if bin_w is None:
        rng_ = np.ptp(v)
        bin_w = rng_ / 25 if rng_ > 0 else 1.0
    keys = np.round(v / bin_w).astype(np.int64)
    off = np.zeros(n)
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        m = idx.size
        if m > 1:
            kk = np.arange(m) - (m - 1) / 2
            off[idx] = kk * (width / (m - 1))
    return off


def run_hybrid_sweep(true_mean, delta, eps, threshold, streak_required,
                     num_runs_list=(10, 50, 200), n_seeds=20,
                     tau=0.95, maxq=500, k_init=2, beta=2,
                     seed0_base=2026, seed_stride=10_000_019, alpha=0.05):
    """num_runs × 시드 격자에서 하이브리드 역산(fixed)을 돌린다.

    반환
      df : num_runs, seed_idx, seed0, status, k_min, eps_guaranteed,
           eps_expected, n_evals, queries_used 열의 기록
    """
    rows = []
    for num_runs in num_runs_list:
        for i in range(n_seeds):
            s0 = int(seed0_base + i * seed_stride)
            res = run_inv_hybrid(true_mean, delta, eps, threshold, streak_required,
                                 tau=tau, num_runs=num_runs, maxq=maxq,
                                 k_init=k_init, beta=beta, seed0=s0,
                                 alpha=alpha, seed_policy="fixed")
            rows.append({
                "num_runs": int(num_runs), "seed_idx": i + 1, "seed0": s0,
                "status": res["status"],
                "k_min": res["k_min"],
                "eps_guaranteed": res["eps_guaranteed"],
                "eps_expected": res["eps_expected"],
                "n_evals": len(res["evals"]),
                "queries_used": res["queries_used"],
            })
    return pd.DataFrame(rows)


def summarize_hybrid_sweep(df):
    """스윕 결과의 num_runs별 요약 — 보장 예산(ε)의 흩어짐이 num_runs와 함께 줄어드는지."""
    g = df.groupby("num_runs")["eps_guaranteed"]     # 주 지표: 보장 예산(ε)
    s = g.agg(eps_guaranteed_n="count", eps_guaranteed_median="median",
              eps_guaranteed_mean="mean", eps_guaranteed_std="std",
              eps_guaranteed_min="min", eps_guaranteed_max="max")
    s["eps_guaranteed_iqr"] = g.quantile(0.75) - g.quantile(0.25)
    # 보조 지표: k_min(질의 수) 기준 요약도 함께
    gk = df.groupby("num_runs")["k_min"]
    s["k_min_median"] = gk.median()
    s["k_min_std"] = gk.std()
    s["converged_rate"] = df.groupby("num_runs")["eps_guaranteed"].apply(
        lambda x: x.notna().mean())
    s["queries_mean"] = df.groupby("num_runs")["queries_used"].mean()
    return s.reset_index()



# ---------------------------------------------------------------------------
# [시각화 — 스윕 결과]
# ---------------------------------------------------------------------------
def plot_hybrid_sweep_beeswarm(df, title=None):
    """num_runs별 작은 패널에 시드별 k_min 점을 옆으로 펼쳐 그린다.

    시드가 많으면 점이 겹치므로 beeswarm처럼 밀어낸다. 패널마다 중앙값 선을
    넣어 흩어짐과 그 축소를 함께 읽는다 (원본 plot_seed_sweep_beeswarm 규칙).
    """
    d = df.dropna(subset=["k_min"]).copy()
    nrs = sorted(d["num_runs"].unique())
    fig, axes = plt.subplots(1, len(nrs), figsize=(3.4 * len(nrs), 3.5),
                             sharex=True, squeeze=False)
    axes = axes[0]
    lo, hi = d["k_min"].min(), d["k_min"].max()
    pad = (hi - lo) * 0.15 if hi > lo else 1.0
    for ax, nr in zip(axes, nrs):
        g = d[d["num_runs"] == nr]
        jit = _hybrid_swarm_offsets(g["k_min"].to_numpy(), width=0.55)
        ax.plot(g["k_min"].to_numpy(), 0.0 + jit, "o", ms=5.5,
                color="tab:blue", alpha=0.85, markeredgecolor="white")
        med = g["k_min"].median()
        ax.axvline(med, color="tab:red", ls="--", lw=1.2,
                   label=f"중앙값 {med:.0f}")
        ax.set_ylim(-1, 1)
        ax.set_xlim(max(0, lo - pad), hi + pad)
        ax.set_title(f"num_runs = {nr}")
        ax.set_yticks([])
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_xlabel("k_min (최소 질의 수)")
    fig.suptitle(title or "시드별 k_min 흩어짐 — num_runs에 따른 수렴", y=1.03)
    plt.tight_layout(); plt.show()
    return fig


def plot_hybrid_sweep_heatmap(df, title=None):
    """시드(행) × num_runs(열) 격자에 k_min을 색으로 — 이상 시드가 한눈에 보인다."""
    d = df.dropna(subset=["k_min"])
    pivot = d.pivot_table(index="seed_idx", columns="num_runs", values="k_min")
    fig, ax = plt.subplots(figsize=(1.2 * len(pivot.columns) + 3, 0.35 * len(pivot) + 3))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(r) for r in pivot.index])
    ax.set_xlabel("num_runs"); ax.set_ylabel("시드 (seed_idx)")
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v = pivot.iloc[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                        color="white" if im.norm(v) < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="k_min")
    ax.set_title(title or "시드 × num_runs 격자의 k_min (비수렴 시드는 결측)")
    plt.tight_layout(); plt.show()
    return fig



# ===========================================================================
# [시각화 — 스윕 결과, 원본 카탈로그 방식 이식판]
#   - plot_hybrid_sweep_hist  : 셀 46 plot_seed_sweep_hist의 k_min 판
#   - plot_hybrid_ridgeline   : 셀 12 plot_ridgeline의 k_min 판 (능선)
#   - plot_hybrid_spread      : 셀 12 plot_num_runs_spread의 k_min 판 (보조 그림)
#   k_min은 정수이므로 "관측된 값 자체를 격자로" 쓰는 원본 규칙이 더 단순해진다.
# ===========================================================================
def plot_hybrid_sweep_hist(df, title=None):
    """num_runs별 보장 예산 히스토그램 — 시드 분포가 한 칸으로 모이는 과정.

    구간 설정이 핵심: 보장 예산은 후보 예산 격자(ε 간격의 이산값)에만 놓이므로
    **관측된 격자 자체를 구간으로** 쓴다 — 임의 bins가 사실을 왜곡할 일이 없다.
    세 패널은 x·y축을 공유해 "왼쪽은 넓게 퍼지고 오른쪽은 한 칸에 몰린다"가
    패널 간 비교로 읽힌다. (원본 plot_seed_sweep_hist 이식판)
    """
    d = df.dropna(subset=["eps_guaranteed"]).copy()
    nrs = sorted(d["num_runs"].unique())
    lattice = np.sort(d["eps_guaranteed"].unique())    # 관측된 후보 예산 격자 = 구간
    gaps = np.diff(lattice)
    width = float(min(gaps)) * 0.9 if gaps.size else 0.1   # 격자 간격의 90% — 붙어서 읽히게

    fig, axes = plt.subplots(1, len(nrs), figsize=(3.0 * len(nrs), 3.4),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    ymax = max((d.loc[d["num_runs"] == n, "eps_guaranteed"] == L).sum()
               for n in nrs for L in lattice)

    # x축을 격자 전체가 아니라 관측이 몰린 곳(중앙값 ± 4·SD)으로 한정해
    # 빈 공간을 없앤다. 패널 간 "퍼짐→수렴" 비교는 sharex 덕에 유지된다.
    med_all = d["eps_guaranteed"].median()
    sd_all = max(d["eps_guaranteed"].std(ddof=1), 1e-9)
    x_lo = max(lattice.min(), med_all - 4 * sd_all)
    x_hi = min(lattice.max(), med_all + 4 * sd_all)

    for ax, nr in zip(axes, nrs):
        v = d.loc[d["num_runs"] == nr, "eps_guaranteed"]
        counts = np.array([(v == L).sum() for L in lattice])
        colors = ["tab:blue" if c == counts.max() else "lightsteelblue" for c in counts]
        ax.bar(lattice, counts, width=width, color=colors,
               edgecolor="0.25", lw=0.5)
        ax.set_title(f"num_runs={nr}\n중앙 {v.median():.2f} / SD {v.std(ddof=1):.2f}",
                     fontsize=8.5)
        ax.set_xlabel("보장 총예산 ε (k_min × eps)", fontsize=8.5)
        ax.set_xlim(x_lo - width, x_hi + width)
        ax.set_ylim(0, ymax * 1.08)
        ax.grid(alpha=0.3, axis="y")

    axes[0].set_ylabel("시드 수")
    fig.suptitle(title or f"num_runs가 커질수록 시드별 보장 예산이 한 칸으로 모인다 "
                          f"(시드 {len(d) // len(nrs)}개씩)", fontsize=10)
    fig.tight_layout()
    return fig, axes


def plot_hybrid_ridgeline(df, col="eps_guaranteed", n_list=None, bw=None,
                          min_bw=0.05, overlap=1.6, normalize="global",
                          cmap="cividis", title=None):
    """num_runs별 보장 예산 분포를 겹쳐 놓은 능선 (원본 plot_ridgeline 이식판).

    그룹 수가 적당하고 경향(좁아짐)이 뚜렷할 때 가장 잘 읽힌다. 겹침이 정보의
    일부를 가리므로, 좁아짐의 정량적 확인은 plot_hybrid_spread에서 따로 한다.
    """
    n_list = n_list or sorted(df["num_runs"].unique())
    data = {n: df.loc[df["num_runs"] == n, col].dropna().to_numpy() for n in n_list}
    allv = np.concatenate([v for v in data.values() if v.size])
    lo, hi = allv.min() - 1.0, allv.max() + 1.0
    x = np.linspace(lo, hi, 600)

    def kde(v):
        h = bw or max(1.06 * v.std(ddof=1) * v.size ** -0.2 if v.size > 1 else 0,
                      min_bw)
        z = (x[:, None] - v[None, :]) / h
        return np.exp(-0.5 * z ** 2).sum(axis=1) / (v.size * h * np.sqrt(2 * np.pi))

    dens = {n: kde(v) if v.size else np.zeros_like(x) for n, v in data.items()}
    gmax = max(d.max() for d in dens.values()) if dens else 1.0
    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.95, len(n_list)))

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for i, n in enumerate(n_list):
        dd = dens[n] / (gmax if normalize == "global" else max(dens[n].max(), 1e-12))
        y = i + overlap * dd
        ax.fill_between(x, i, y, color=colors[i], alpha=0.85, zorder=len(n_list) - i)
        ax.plot(x, y, color="0.2", lw=1.0, zorder=len(n_list) - i)
        ax.axhline(i, color="0.3", lw=0.8, zorder=0)
        if data[n].size:
            med = np.median(data[n])
            ax.vlines(med, i, i + overlap * np.interp(med, x, dd), colors="0.15",
                      linestyles="--", lw=1.0, zorder=len(n_list) + 1)
            n_nan = df.loc[df["num_runs"] == n, col].isna().sum()
            ax.text(hi, i + 0.35, f"n={n}\n중앙 {med:.2f}"
                    + (f"\n결측={n_nan}" if n_nan else ""), fontsize=7.5, va="bottom")

    ax.set_yticks(range(len(n_list)))
    ax.set_yticklabels([f"num_runs={n}" for n in n_list])
    ax.set_xlim(lo, hi + 0.1 * (hi - lo))
    ax.set_xlabel("보장 총예산 ε (k_min × eps)")
    ax.set_title(title or "num_runs가 커질수록 보장 예산 분포가 좁아짐", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_hybrid_spread(summary, title=None):
    """num_runs에 대한 보장 예산의 표준편차와 IQR (로그축, 보조 그림).

    ridgeline이 모양을 보여준다면, 이 그림은 좁아짐이 단조인지 확인한다
    (원본 plot_num_runs_spread 이식판). 로그축 mathtext 눈금의 U+2212
    문제를 피하기 위해 눈금 라벨을 일반 숫자로 바꾼다.
    """
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    s = summary.reset_index() if "num_runs" not in summary.columns else summary.copy()
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.plot(s["num_runs"], s["eps_guaranteed_std"], "-o", color="tab:blue", lw=1.6, ms=4,
            label="표준편차")
    ax.plot(s["num_runs"], s["eps_guaranteed_iqr"], "-s", color="tab:orange", lw=1.5, ms=4,
            label="사분위범위")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("num_runs (로그축)")
    ax.set_ylabel("보장 예산의 흩어짐 (로그축)")
    ax.set_title(title or "num_runs에 따른 보장 예산의 흩어짐", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax
