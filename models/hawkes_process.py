from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import polars as pl

try:
    from scipy import optimize, stats
except Exception:  # scipy가 없으면 fit/KS는 제한됨
    optimize = None
    stats = None


_DT_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass
class WindowSpec:
    start: str
    end: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # (min_lon, min_lat, max_lon, max_lat)


class HawkesProcess:
    """
    Univariate Hawkes Process (Exponential kernel) 분석/학습용 클래스.

    Model:
      λ(t) = μ + α * Σ_{t_i < t} exp(-β (t - t_i))

    Parameters:
      μ   : base intensity (events/sec), μ > 0
      α   : excitation magnitude, α >= 0
      β   : decay rate, β > 0
      branching ratio n = α/β  (권장: n < 1)

    제공 기능:
      - 데이터 전처리(pick_dt -> pick_ts -> pick_sec)
      - 시간/공간 필터
      - log-likelihood 계산(빠른 재귀식)
      - MLE 파라미터 적합(scipy.optimize 필요)
      - time-rescaling 진단(z_i ~ Exp(1), u_i ~ Uniform(0,1))
      - Ogata thinning 기반 시뮬레이션
    """

    def __init__(self, df: pl.DataFrame, spec: WindowSpec):
        self.df_raw = df
        self.spec = spec
        self.df = self._preprocess(df)

        # windowed df & event times (relative seconds)
        self.df_win = self._filter_window(self.df, spec)
        self.t = self._event_times_relative_sec(self.df_win, spec)

        # fitted parameters
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

        if t_abs.size < 2:
            raise ValueError("이벤트가 너무 적습니다(최소 2개 이상 권장).")

        t_rel = (t_abs - start_sec).astype(np.float64)
        # safety: window start 이전이 없도록
        t_rel = t_rel[t_rel >= 0]
        if t_rel.size < 2:
            raise ValueError("필터 후 유효 이벤트가 너무 적습니다.")
        return t_rel

    # -----------------------------
    # Core: intensity at events & log-likelihood
    # -----------------------------
    @staticmethod
    def _compute_R_exp_kernel(t: np.ndarray, beta: float) -> np.ndarray:
        """
        R_i = Σ_{j<i} exp(-β (t_i - t_j))  (event i at t[i])
        Exponential kernel에서 O(N) 재귀로 계산.

        Recurrence:
          R_0 = 0
          R_i = exp(-β Δ_i) * (1 + R_{i-1}),  i>=1
        """
        n = t.size
        R = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            dt = t[i] - t[i - 1]
            decay = np.exp(-beta * dt)
            R[i] = decay * (1.0 + R[i - 1])
        return R

    @classmethod
    def log_likelihood(cls, t: np.ndarray, T: float, mu: float, alpha: float, beta: float) -> float:
        """
        Univariate exponential Hawkes log-likelihood on [0, T].

        LL = Σ log( μ + α R_i )  - μ T  - (α/β) Σ (1 - exp(-β (T - t_i)))

        where R_i = Σ_{j<i} exp(-β (t_i - t_j))
        """
        if mu <= 0 or alpha < 0 or beta <= 0:
            return -np.inf
        if T <= 0:
            return -np.inf

        R = cls._compute_R_exp_kernel(t, beta)
        lam_at_events = mu + alpha * R
        if np.any(lam_at_events <= 0):
            return -np.inf

        term1 = np.sum(np.log(lam_at_events)) # 실제 사건이 발생한 시에서의 강도 합
        term2 = mu * T                        # 기저 강도의 적분 합
        term3 = (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - t))) # 자기 흥 효과에 대한 적분항
        return float(term1 - term2 - term3)

    # -----------------------------
    # Fit (MLE) with penalty for stability
    # -----------------------------
    def fit_mle(
        self,
        *,
        init_mu: Optional[float] = None,
        init_alpha: Optional[float] = None,
        init_beta: Optional[float] = None,
        enforce_branching_lt1: bool = True,
        penalty_weight: float = 1e6,
    ) -> Dict[str, float]:
        """
        scipy.optimize가 있을 때 MLE로 (μ, α, β) 추정.

        - positivity를 위해 log-parameterization 사용:
            μ = exp(x0), α = exp(x1), β = exp(x2)
        - (선택) α/β >= 1이면 큰 패널티 부여
        """
        if optimize is None:
            raise RuntimeError("scipy.optimize가 필요합니다. (pip install scipy)")

        t = self.t
        T = float(self._parse_dt_sec(self.spec.end) - self._parse_dt_sec(self.spec.start))
        if T <= 0:
            raise ValueError("관측 구간 T가 0 이하입니다.")

        # 초기값(대략적으로)
        n_events = t.size
        rate_emp = n_events / T  # events/sec

        mu0 = float(init_mu) if init_mu is not None else max(1e-6, 0.5 * rate_emp)
        alpha0 = float(init_alpha) if init_alpha is not None else max(1e-6, 0.5 * rate_emp)
        beta0 = float(init_beta) if init_beta is not None else 1.0 / max(1.0, (T / max(1, n_events)))  # 러프

        x0 = np.log([mu0, alpha0, beta0]).astype(np.float64)

        def objective(x: np.ndarray) -> float:
            mu = float(np.exp(x[0]))
            alpha = float(np.exp(x[1]))
            beta = float(np.exp(x[2]))

            ll = self.log_likelihood(t, T, mu, alpha, beta)
            if not np.isfinite(ll):
                return 1e30

            # 안정성 패널티(α/β < 1 권장)
            if enforce_branching_lt1:
                br = alpha / beta
                if br >= 1.0:
                    # (br-1)^2 형태로 급격히 벌점
                    ll -= penalty_weight * (br - 1.0) ** 2

            return -ll  # minimize negative log-likelihood

        res = optimize.minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
        )

        mu_hat = float(np.exp(res.x[0]))
        alpha_hat = float(np.exp(res.x[1]))
        beta_hat = float(np.exp(res.x[2]))
        br = alpha_hat / beta_hat

        out = {
            "mu": mu_hat,
            "alpha": alpha_hat,
            "beta": beta_hat,
            "branching_ratio": float(br),
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
    def _rescaling_z_exp_kernel(t: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
        """
        Time-rescaling theorem:
          z_i = ∫_{t_{i-1}}^{t_i} λ(u) du  should be i.i.d. Exp(1)

        For exponential Hawkes:
          Between events, past-sum decays exponentially.

        Let H_{i-1} = Σ_{j<=i-1} exp(-β (t_{i-1} - t_j)) = 1 + R_{i-1}
        Then for Δ = t_i - t_{i-1}:
          z_i = μ Δ + (α/β) * H_{i-1} * (1 - exp(-β Δ))

        Also for first interval (from 0 to t0):
          z_0 = μ * t0  (no history)
        """
        n = t.size
        if n < 2:
            return np.array([], dtype=np.float64)

        R = HawkesProcess._compute_R_exp_kernel(t, beta)
        z = np.zeros(n, dtype=np.float64)

        # first interval [0, t0)
        z[0] = mu * t[0]

        # subsequent intervals
        for i in range(1, n):
            dt = t[i] - t[i - 1]
            H_prev = 1.0 + R[i - 1]
            z[i] = mu * dt + (alpha / beta) * H_prev * (1.0 - np.exp(-beta * dt))

        return z

    def diagnose_time_rescaling(self, mu: float, alpha: float, beta: float) -> Dict[str, Any]:
        """
        - z_i ~ Exp(1)
        - u_i = 1 - exp(-z_i) ~ Uniform(0,1)
        """
        z = self._rescaling_z_exp_kernel(self.t, mu, alpha, beta)
        # 통상 i=1..N 구간을 쓰거나, 첫 구간 포함 여부는 선택인데 여기서는 포함(z[0] 포함)
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
    # Simulation: Ogata thinning (exponential kernel)
    # 비균질 포아송 프로세스(Non-homogeneous Poisson Process)를 시뮬레이션하는 표준 기법
    # -----------------------------
    @staticmethod
    def simulate_ogata(
        *,
        mu: float,
        alpha: float,
        beta: float,
        T: float,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Ogata thinning for univariate exponential Hawkes (monotone-decay between events).

        State:
          h(t) = Σ exp(-β (t - t_i))  (t_i < t)
        Then λ(t) = μ + α h(t)

        Between events, h decays: h(t+s) = h(t) * exp(-β s)
        Upper bound M can be λ(t) because intensity decreases between events (if no new event).

        Steps:
          - sample s ~ Exp(M)
          - t <- t + s
          - compute λ(t) and accept with prob λ(t)/M
          - if accepted: h <- h*exp(-β s) + 1
            else:        h <- h*exp(-β s)
        """
        if mu <= 0 or alpha < 0 or beta <= 0:
            raise ValueError("Invalid params: require mu>0, alpha>=0, beta>0")
        if T <= 0:
            return np.array([], dtype=np.float64)

        rng = np.random.default_rng(seed)
        t = 0.0
        h = 0.0
        events: List[float] = []

        while t < T:
            M = mu + alpha * h
            if M <= 0:
                break

            s = float(rng.exponential(scale=1.0 / M))
            t_candidate = t + s
            if t_candidate >= T:
                break

            # decay h to candidate time
            h_candidate = h * np.exp(-beta * s)
            lam_candidate = mu + alpha * h_candidate

            # accept?
            if float(rng.random()) <= (lam_candidate / M):
                events.append(t_candidate)
                # jump: add new event
                h = h_candidate + 1.0
                t = t_candidate
            else:
                # reject: just move time and keep decayed h
                h = h_candidate
                t = t_candidate

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
        init_beta: Optional[float] = None,
        sim_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        - fit_mle(optional)
        - diagnostics(time-rescaling)
        - simulate (same T)
        """
        T = float(self._parse_dt_sec(self.spec.end) - self._parse_dt_sec(self.spec.start))
        n_events = int(self.t.size)

        if fit:
            params = self.fit_mle(
                init_mu=init_mu,
                init_alpha=init_alpha,
                init_beta=init_beta,
                enforce_branching_lt1=True,
            )
        else:
            if self.params_ is None:
                raise ValueError("fit=False인데 params_가 없습니다. 먼저 fit_mle를 호출하세요.")
            params = self.params_

        diag = self.diagnose_time_rescaling(params["mu"], params["alpha"], params["beta"])
        sim_times = self.simulate_ogata(
            mu=params["mu"],
            alpha=params["alpha"],
            beta=params["beta"],
            T=T,
            seed=sim_seed,
        )

        return {
            "model": "Hawkes(univariate, exp-kernel)",
            "window": {"start": self.spec.start, "end": self.spec.end, "bbox": self.spec.bbox},
            "T_seconds": float(T),
            "n_events_observed": n_events,
            "params": params,
            "diagnostics": diag,
            "simulated_times_sec_from0": sim_times,
            "n_events_simulated": int(sim_times.size),
        }