from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import polars as pl

from simple_lab_test.utils import get_taxi_data

# scipy는 선택 의존성(없어도 동작하도록)
try:
    from scipy import optimize, stats
except Exception:  # pragma: no cover
    optimize = None
    stats = None


_DT_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass
class WindowSpec:
    start: str
    end: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # (min_lon, min_lat, max_lon, max_lat)


class SelfCorrectingProcess:
    """
    Self-Correcting Process 분석/학습용 클래스 (univariate).

    Model (대표 형태):
      λ(t) = exp( μ t - α N(t) )

    - μ > 0 : 시간 경과에 따라 log-intensity가 증가하는 기울기
    - α > 0 : 이벤트 발생 시 log-intensity를 낮추는 교정 강도
    - N(t) : t까지 발생한 이벤트 수 (counting process)

    주의:
      - 이벤트 시점에서의 강도는 보통 left-limit를 사용:
          λ(t_i) = exp( μ t_i - α (i-1) )  (t_i 직전에는 N = i-1)
      - 적분항은 구간별(piecewise)로 정확히 계산 가능.

    제공 기능:
      - 데이터 전처리(pick_dt -> pick_ts -> pick_sec)
      - 시간/공간 필터
      - log-likelihood 계산(폐형식)
      - MLE 파라미터 적합(scipy.optimize 필요)
      - time-rescaling 진단(z ~ Exp(1), u ~ Uniform(0,1))
      - inversion 기반 시뮬레이션(얇게/정확하게, thinning 불필요)
    """

    def __init__(self, df: pl.DataFrame, spec: WindowSpec, time_unit: str = "hour"):
        self.df_raw = df
        self.spec = spec
        self.df = self._preprocess(df)

        self.time_unit = time_unit
        self.time_scale = {"sec": 1.0, "min": 60.0, "hour": 3600.0}[time_unit]

        # windowed df
        self.df_win = self._filter_window(self.df, spec)

        # t: relative time in *seconds* -> scale to chosen unit
        t_sec = self._event_times_relative_sec(self.df_win, spec)
        self.t = t_sec / self.time_scale

        # T: window length in *seconds* -> scale to chosen unit
        T_sec = self._parse_dt_sec(self.spec.end) - self._parse_dt_sec(self.spec.start)
        self.T_sec = float(T_sec)
        self.T = float(T_sec) / self.time_scale

        self.params_: Optional[Dict[str, float]] = None

    # -----------------------------
    # Preprocess / Filter
    # -----------------------------
    @staticmethod
    def _preprocess(df: pl.DataFrame) -> pl.DataFrame:
        if "pick_dt" not in df.columns:
            raise ValueError("Input df must contain 'pick_dt' column.")

        out = (
            df.with_columns(
                pl.col("pick_dt").str.strptime(pl.Datetime, format=_DT_FMT, strict=False).alias("pick_ts")
            )
            .filter(pl.col("pick_ts").is_not_null())
            .with_columns(pl.col("pick_ts").dt.epoch("s").cast(pl.Int64).alias("pick_sec"))
        )
        return out

    @staticmethod
    def _parse_dt_sec(s: str) -> int:
        return int(datetime.strptime(s, _DT_FMT).timestamp())

    def _filter_window(self, df: pl.DataFrame, spec: WindowSpec) -> pl.DataFrame:
        start_sec = self._parse_dt_sec(spec.start)
        end_sec = self._parse_dt_sec(spec.end)
        if end_sec <= start_sec:
            raise ValueError("WindowSpec.end must be > WindowSpec.start")

        out = df.filter((pl.col("pick_sec") >= start_sec) & (pl.col("pick_sec") < end_sec))

        if spec.bbox is not None:
            if not all(c in out.columns for c in ["pick_lon", "pick_lat"]):
                raise ValueError("bbox filtering requires 'pick_lon' and 'pick_lat'.")
            min_lon, min_lat, max_lon, max_lat = spec.bbox
            out = out.filter(
                (pl.col("pick_lon") >= min_lon) & (pl.col("pick_lon") <= max_lon) &
                (pl.col("pick_lat") >= min_lat) & (pl.col("pick_lat") <= max_lat)
            )
        return out

    def _event_times_relative_sec(self, df_win: pl.DataFrame, spec: WindowSpec) -> np.ndarray:
        """
        window start를 0으로 두고 relative seconds로 변환.
        """
        start_sec = self._parse_dt_sec(spec.start)

        t_abs = (
            df_win.select(pl.col("pick_sec"))
            .sort("pick_sec")
            .to_series()
            .to_numpy()
            .astype(np.int64)
        )

        if t_abs.size == 0:
            return np.array([], dtype=np.float64)

        # 1개라도 반환은 하되, 이후 fit 단계에서 막는 방식
        t_rel = (t_abs - start_sec).astype(np.float64)
        t_rel = t_rel[t_rel >= 0]

        if t_rel.size == 0:
            return np.array([], dtype=np.float64)

        return t_rel

    # -----------------------------
    # Core: log-likelihood (closed-form)
    # -----------------------------
    @staticmethod
    def _safe_logsumexp_exp_term(mu: float, a: float, b: float) -> float:
        """
        compute log( exp(mu*b) - exp(mu*a) ) safely, for mu>0 and b>a.
        값 자체가 매우 클 수 있으므로 로그로 계산할 때 사용.
        """
        if b <= a:
            raise ValueError("require b>a")
        mb = mu * b
        ma = mu * a
        # exp(mb) - exp(ma) = exp(mb) * (1 - exp(ma-mb))
        # log(...) = mb + log(1 - exp(ma-mb))
        x = ma - mb  # negative
        # when x is very negative, exp(x) ~ 0
        return mb + np.log1p(-np.exp(x))

    @classmethod
    def log_likelihood(cls, t: np.ndarray, T: float, mu: float, alpha: float) -> float:
        """
        Self-correcting log-likelihood on [0, T], with event times t (relative, sorted).

        λ(t) = exp(mu*t - alpha*N(t))

        Event intensity uses left-limit:
          for i-th event at t[i], N(t[i]-) = i
          if i is 0-indexed, N(t_i-) = i
          => log λ(t_i) = mu*t_i - alpha*i

        LL = Σ_{i=0..n-1} (mu*t_i - alpha*i)  -  ∫_0^T exp(mu*u - alpha*N(u)) du

        Integral is piecewise where N(u)=k on [t_k, t_{k+1}), with t_0=0, t_{n}=T boundary.
        ∫_{a}^{b} exp(mu*u - alpha*k) du = exp(-alpha*k) * (exp(mu*b) - exp(mu*a)) / mu   (mu>0)
        """
        if mu <= 0 or alpha <= 0:
            return -np.inf
        if T <= 0:
            return -np.inf
        if t.size == 0:
            # no events: LL = - ∫_0^T exp(mu*u) du
            # = -(exp(mu*T)-1)/mu
            return -float((np.exp(mu * T) - 1.0) / mu)

        # ensure sorted and within [0,T)
        if np.any(t < 0) or np.any(t >= T):
            return -np.inf

        n = int(t.size)

        # term1: sum log intensities at event times (left-limit)
        # i is 0..n-1
        i = np.arange(n, dtype=np.float64)
        term1 = float(np.sum(mu * t - alpha * i))

        # term2: integral
        # intervals: [0,t0) with k=0, [t0,t1) with k=1, ..., [t_{n-1}, T) with k=n
        # careful: after last event, N(u)=n on [t_{n-1},T)
        times = np.concatenate([[0.0], t, [float(T)]])  # length n+2
        integral = 0.0
        for k in range(0, n + 1):
            a = float(times[k])
            b = float(times[k + 1])
            if b <= a:
                continue
            # compute (exp(mu*b)-exp(mu*a))/mu with overflow-awareness
            # prefer direct if safe; otherwise log-space
            mb = mu * b
            ma = mu * a
            if mb < 700:  # exp(700) ~ 1e304 near float limit
                diff = (np.exp(mb) - np.exp(ma)) / mu
                integral += float(np.exp(-alpha * k) * diff)
            else:
                # log(diff) = log(exp(mu*b)-exp(mu*a)) - log(mu)
                log_diff = cls._safe_logsumexp_exp_term(mu, a, b) - np.log(mu)
                # integral add exp(-alpha*k) * exp(log_diff)
                integral += float(np.exp(-alpha * k + log_diff))

        return float(term1 - integral)

    # -----------------------------
    # Fit (MLE)
    # -----------------------------
    def fit_mle(
        self,
        *,
        init_mu: Optional[float] = None,
        init_alpha: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        scipy.optimize가 있을 때 MLE로 (mu, alpha) 추정.

        - positivity를 위해 log-parameterization 사용:
            mu = exp(x0), alpha = exp(x1)
        """
        if self.t.size < 2:
            raise ValueError(f"MLE 적합에는 이벤트가 최소 2개 이상 필요합니다. (현재 {self.t.size}개)")

        if optimize is None:
            raise RuntimeError("scipy.optimize가 필요합니다. (pip install scipy)")

        t = self.t
        T = self.T  # <-- 스케일된 T 사용

        if T <= 0:
            raise ValueError("관측 구간 T가 0 이하입니다.")

        n_events = int(t.size)
        rate_emp = n_events / T  # events per chosen time unit (hour/min/sec)

        mu0 = float(init_mu) if init_mu is not None else max(1e-6, 0.1 * rate_emp)
        alpha0 = float(init_alpha) if init_alpha is not None else 0.1
        x0 = np.log([mu0, alpha0]).astype(np.float64)

        def objective(x: np.ndarray) -> float:
            mu = float(np.exp(x[0]))
            alpha = float(np.exp(x[1]))

            # overflow 방지 가드: mu*T가 너무 크면 패널티
            if mu * T > 50:
                return 1e30 + (mu * T - 50) * 1e10

            ll = self.log_likelihood(t, T, mu, alpha)
            if not np.isfinite(ll):
                return 1e30
            return -ll

        res = optimize.minimize(objective, x0=x0, method="L-BFGS-B")

        mu_hat = float(np.exp(res.x[0]))
        alpha_hat = float(np.exp(res.x[1]))

        out = {
            "mu": mu_hat,
            "alpha": alpha_hat,
            "success": bool(res.success),
            "message": str(res.message),
            "nll": float(res.fun),
        }
        self.params_ = out
        return out

    # -----------------------------
    # Time-rescaling diagnostics
    # -----------------------------
    @staticmethod
    def _rescaling_z(t: np.ndarray, mu: float, alpha: float) -> np.ndarray:
        """
        Time-rescaling theorem:
          z_i = ∫_{t_{i-1}}^{t_i} λ(u) du  should be i.i.d. Exp(1)

        For self-correcting:
          Between events (i-1)-th to i-th, N(u) = i-1 (0-indexed i).
          With boundaries t_-1 = 0.

        For interval [a,b] with N=k:
          z = exp(-alpha*k) * (exp(mu*b)-exp(mu*a))/mu
        """
        n = t.size
        if n < 2:
            return np.array([], dtype=np.float64)

        times = np.concatenate([[0.0], t])  # start boundary
        z = np.zeros(n, dtype=np.float64)

        for i in range(n):
            a = float(times[i])          # t_{i-1} (with t_{-1}=0)
            b = float(t[i])              # t_i
            k = i                        # N(u)=i on [t_{i-1}, t_i) ? careful:
            # Using event-index convention:
            # - On [0,t0): N=0  => i=0 interval uses k=0
            # - On [t0,t1): N=1 => i=1 interval uses k=1
            # So for interval ending at t[i], k=i.
            # (This includes the first interval as k=0.)
            if b <= a:
                z[i] = 0.0
                continue
            z[i] = np.exp(-alpha * k) * (np.exp(mu * b) - np.exp(mu * a)) / mu

        return z

    def diagnose_time_rescaling(self, mu: float, alpha: float) -> Dict[str, Any]:
        """
        - z_i ~ Exp(1)
        - u_i = 1 - exp(-z_i) ~ Uniform(0,1)
        """
        z = self._rescaling_z(self.t, mu, alpha)
        # 첫 구간 포함 여부는 분석 목적에 따라 다르나, 여기서는 포함합니다.
        u = 1.0 - np.exp(-z)

        out: Dict[str, Any] = {
            "n": int(z.size),
            "mean_z": float(z.mean()) if z.size else None,
            "var_z": float(z.var(ddof=1)) if z.size >= 2 else None,
            "mean_u": float(u.mean()) if u.size else None,
            "var_u": float(u.var(ddof=1)) if u.size >= 2 else None,
        }

        if stats is not None and z.size:
            D_exp, p_exp = stats.kstest(z, "expon", args=(0, 1.0))
            D_unif, p_unif = stats.kstest(u, "uniform", args=(0, 1.0))
            out["ks_exp"] = {"KS_D": float(D_exp), "p_value": float(p_exp)}
            out["ks_unif"] = {"KS_D": float(D_unif), "p_value": float(p_unif)}
        else:
            out["ks_note"] = "scipy 미설치 또는 데이터 부족으로 KS test 생략"

        return out

    # -----------------------------
    # Simulation: exact inversion (no thinning)
    # -----------------------------
    @staticmethod
    def simulate_inversion(
        *,
        mu: float,
        alpha: float,
        T: float,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Exact sequential simulation using inversion on integrated intensity.

        At time t with k past events (N(t)=k), intensity is:
          λ(u) = exp(mu*u - alpha*k), u >= t until next event occurs.

        Draw E ~ Exp(1), solve for next time s>0:
          ∫_{t}^{t+s} exp(mu*u - alpha*k) du = E
          exp(-alpha*k) * (exp(mu*(t+s)) - exp(mu*t)) / mu = E
          exp(mu*(t+s)) = exp(mu*t) + mu*E*exp(alpha*k)
          t_next = (1/mu) * log( exp(mu*t) + mu*E*exp(alpha*k) )

        Iterate until t_next >= T.
        """
        if mu <= 0 or alpha <= 0:
            raise ValueError("require mu>0 and alpha>0")
        if T <= 0:
            return np.array([], dtype=np.float64)

        rng = np.random.default_rng(seed)
        t = 0.0
        k = 0  # number of past events
        events: List[float] = []

        while True:
            E = float(rng.exponential(scale=1.0))  # Exp(1)
            base = np.exp(mu * t)
            inc = mu * E * np.exp(alpha * k)
            t_next = (1.0 / mu) * np.log(base + inc)

            if t_next >= T:
                break

            events.append(t_next)
            t = t_next
            k += 1

        return np.array(events, dtype=np.float64)

    # -----------------------------
    # Convenience run
    # -----------------------------
    def run_full(
        self,
        *,
        fit: bool = True,
        init_mu: Optional[float] = None,
        init_alpha: Optional[float] = None,
        sim_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        - fit_mle(optional)
        - diagnostics(time-rescaling)
        - simulate (same T)
        """
        T = self.T  # 스케일된 T
        T_sec = self.T_sec  # 원래 초 단위
        n_events = int(self.t.size)

        if fit:
            params = self.fit_mle(init_mu=init_mu, init_alpha=init_alpha)
        else:
            if self.params_ is None:
                raise ValueError("fit=False인데 params_가 없습니다. 먼저 fit_mle를 호출하세요.")
            params = self.params_

        diag = self.diagnose_time_rescaling(params["mu"], params["alpha"])

        # simulate도 반드시 같은 단위의 T를 넣어야 함
        sim_times = self.simulate_inversion(mu=params["mu"], alpha=params["alpha"], T=T, seed=sim_seed)

        return {
            "model": "SelfCorrecting(univariate)",
            "time_unit": self.time_unit,
            "window": {"start": self.spec.start, "end": self.spec.end, "bbox": self.spec.bbox},
            "T": float(T),  # 스케일된 시간 단위
            "T_seconds": float(T_sec),  # 원래 초
            "n_events_observed": n_events,
            "params": params,
            "diagnostics": diag,
            "simulated_times_from0": sim_times,  # self.time_unit 기준
            "n_events_simulated": int(sim_times.size),
        }
if __name__ == "__main__":
    df = get_taxi_data()
    spec = WindowSpec(
        start="2015-01-10 20:00:00",
        end="2015-01-10 21:00:00",
        bbox=None,  # 필요 시 (min_lon, min_lat, max_lon, max_lat)
    )

    scp = SelfCorrectingProcess(df, spec)

    # 1) MLE + 진단 + 시뮬
    out = scp.run_full(fit=True, sim_seed=42)
    print(out["params"])
    print(out["diagnostics"])

    # 2) 특정 파라미터로 LL만 계산
    T = scp._parse_dt_sec(spec.end) - scp._parse_dt_sec(spec.start)
    ll = SelfCorrectingProcess.log_likelihood(scp.t, scp.T, mu=1e-4, alpha=0.2)
    print("LL:", ll)
