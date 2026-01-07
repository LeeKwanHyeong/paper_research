from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import polars as pl

# scipy는 선택 의존성(없어도 동작하도록)
try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


_DT_FMT = "%Y-%m-%d %H:%M:%S"

@dataclass
class WindowSpec:
    start: str  # "YYYY-MM-DD HH:MM:SS"
    end: str    # "YYYY-MM-DD HH:MM:SS"
    bbox: Optional[Tuple[float, float, float, float]] = None  # (min_lon, min_lat, max_lon, max_lat)


class PoissonBase:
    """
    공통 전처리/필터/타입 안정화를 담당하는 베이스.

    - pick_dt(String) -> pick_ts(Datetime) -> pick_sec(Int64 epoch seconds)
    - 분석 시간창/공간창 필터
    - epoch seconds 기반으로 모든 시간차 계산 (total_seconds() 의존 제거)
    """

    def __init__(self, df: pl.DataFrame):
        self.df_raw = df
        self.df = self._preprocess(df)

    @staticmethod
    def _preprocess(df: pl.DataFrame) -> pl.DataFrame:
        if "pick_dt" not in df.columns:
            raise ValueError("Input df must contain 'pick_dt' column (String).")

        out = (
            df.with_columns(
                pl.col("pick_dt").str.strptime(pl.Datetime, format=_DT_FMT, strict=False).alias("pick_ts")
            )
            .filter(pl.col("pick_ts").is_not_null())
            .with_columns(
                pl.col("pick_ts").dt.epoch("s").cast(pl.Int64).alias("pick_sec")
            )
        )
        return out

    @staticmethod
    def _parse_dt_sec(s: str) -> int:
        return int(datetime.strptime(s, _DT_FMT).timestamp())

    def filter_window(self, spec: WindowSpec) -> pl.DataFrame:
        start_sec = self._parse_dt_sec(spec.start)
        end_sec = self._parse_dt_sec(spec.end)
        if end_sec <= start_sec:
            raise ValueError("WindowSpec.end must be > WindowSpec.start")

        out = self.df.filter((pl.col("pick_sec") >= start_sec) & (pl.col("pick_sec") < end_sec))

        if spec.bbox is not None:
            if not all(col in out.columns for col in ["pick_lon", "pick_lat"]):
                raise ValueError("bbox filtering requires 'pick_lon' and 'pick_lat' columns.")
            min_lon, min_lat, max_lon, max_lat = spec.bbox
            out = out.filter(
                (pl.col("pick_lon") >= min_lon) & (pl.col("pick_lon") <= max_lon) &
                (pl.col("pick_lat") >= min_lat) & (pl.col("pick_lat") <= max_lat)
            )

        return out

    @staticmethod
    def _sorted_event_times_sec(df_win: pl.DataFrame) -> np.ndarray:
        if df_win.height == 0:
            return np.array([], dtype=np.int64)
        return (
            df_win.select(pl.col("pick_sec"))
            .sort("pick_sec")
            .to_series()
            .to_numpy()
            .astype(np.int64)
        )

    @staticmethod
    def _ensure_min_events(event_times_sec: np.ndarray, n_min: int = 2) -> None:
        if event_times_sec.size < n_min:
            raise ValueError(f"이벤트가 너무 적습니다(최소 {n_min}개 이상 권장).")

    @staticmethod
    def _ks_test(dist_name: str, data: np.ndarray, args: tuple) -> Optional[Dict[str, float]]:
        """
        SciPy가 있으면 KS-test 수행, 없으면 None 반환.
        """
        if stats is None:
            return None
        D, p = stats.kstest(data, dist_name, args=args)
        return {"KS_D": float(D), "p_value": float(p)}


# ============================================================
# HPP (Homogeneous Poisson Process) Analyzer
# ============================================================

class HPPAnalyzer(PoissonBase):
    """
    HPP: λ(t)=λ (상수)

    - λ MLE: N/T
    - inter-arrival ~ Exp(λ) 진단 + KS
    - bin-count mean≈var 진단 (0-bin 포함)
    - HPP 시뮬레이션(Exp inter-arrival 누적)
    """

    @staticmethod
    def estimate_lambda(event_times_sec: np.ndarray) -> float:
        """
        λ_hat = N / T  (events/sec)
        T: first->last span (seconds)
        """
        HPPAnalyzer._ensure_min_events(event_times_sec, 2)
        t0 = int(event_times_sec[0])
        t1 = int(event_times_sec[-1])
        T = float(t1 - t0)
        if T <= 0:
            raise ValueError("시간 범위 T가 0 이하입니다.")
        return float(event_times_sec.size) / T

    @staticmethod
    def interarrival_seconds(event_times_sec: np.ndarray) -> np.ndarray:
        HPPAnalyzer._ensure_min_events(event_times_sec, 2)
        return np.diff(event_times_sec).astype(np.float64)

    @staticmethod
    def diagnose_exponential(dt: np.ndarray, lam: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "emp_mean_dt": float(dt.mean()),
            "emp_var_dt": float(dt.var(ddof=1)) if dt.size >= 2 else None,
            "theo_mean_dt": float(1.0 / lam),
            "theo_var_dt": float(1.0 / (lam ** 2)),
        }
        # KS test (optional)
        ks = PoissonBase._ks_test("expon", dt, args=(0, 1.0 / lam))
        out["ks"] = ks
        return out

    @staticmethod
    def bin_counts_full(
        event_times_sec: np.ndarray,
        start_sec: int,
        end_sec: int,
        bin_size_sec: int,
    ) -> np.ndarray:
        """
        [start_sec, end_sec) 구간을 bin_size_sec로 나눈 후
        모든 bin에 대해 카운트를 반환 (0-bin 포함)
        """
        if end_sec <= start_sec:
            raise ValueError("end_sec must be > start_sec")
        if bin_size_sec <= 0:
            raise ValueError("bin_size_sec must be positive")

        T = end_sec - start_sec
        B = int(np.ceil(T / bin_size_sec))
        if B <= 0:
            return np.zeros(0, dtype=np.int64)

        # bin index: floor((t-start)/bin)
        bins = ((event_times_sec - start_sec) // bin_size_sec).astype(np.int64)
        bins = bins[(bins >= 0) & (bins < B)]

        counts = np.bincount(bins, minlength=B).astype(np.int64)
        return counts

    @staticmethod
    def diagnose_poisson_bin_counts(counts: np.ndarray, lam: float, bin_size_sec: int) -> Dict[str, Any]:
        emp_mean = float(counts.mean()) if counts.size else 0.0
        emp_var = float(counts.var(ddof=1)) if counts.size >= 2 else None
        theo_mean = float(lam * bin_size_sec)

        return {
            "emp_mean": emp_mean,
            "emp_var": emp_var,
            "theo_mean_lambdaDelta": theo_mean,
            "overdispersion_var_over_mean": (float(emp_var / emp_mean) if (emp_var is not None and emp_mean > 0) else None),
            "n_bins": int(counts.size),
        }

    @staticmethod
    def simulate_hpp(lam: float, T_seconds: float, seed: int = 0) -> np.ndarray:
        """
        HPP 시뮬레이션: inter-arrival ~ Exp(lam) 누적 (0..T)
        반환: 상대시간(seconds from 0)
        """
        if lam <= 0:
            raise ValueError("lam must be positive")
        if T_seconds <= 0:
            return np.array([], dtype=np.float64)

        rng = np.random.default_rng(seed)
        t = 0.0
        times: List[float] = []
        while True:
            dt = float(rng.exponential(scale=1.0 / lam))
            t += dt
            if t > T_seconds:
                break
            times.append(t)
        return np.array(times, dtype=np.float64)

    def run(
        self,
        spec: WindowSpec,
        bin_size_sec: int = 60,
        sim_seed: int = 42,
    ) -> Dict[str, Any]:
        df_win = self.filter_window(spec)
        event_times = self._sorted_event_times_sec(df_win)
        self._ensure_min_events(event_times, 2)

        lam = self.estimate_lambda(event_times)
        dt = self.interarrival_seconds(event_times)

        start_sec = self._parse_dt_sec(spec.start)
        end_sec = self._parse_dt_sec(spec.end)

        counts = self.bin_counts_full(event_times, start_sec, end_sec, bin_size_sec)

        # 시뮬레이션은 관측 span 기준(첫 이벤트~마지막 이벤트)로 맞추는 편이 비교가 쉬움
        T_span = float(event_times[-1] - event_times[0])
        sim_times = self.simulate_hpp(lam, T_span, seed=sim_seed)

        return {
            "model": "HPP",
            "window": {"start": spec.start, "end": spec.end, "bbox": spec.bbox},
            "lambda_hat_events_per_sec": float(lam),
            "lambda_hat_events_per_min": float(lam * 60.0),
            "n_events": int(event_times.size),
            "span_seconds_first_to_last": float(T_span),
            "diagnose_exponential": self.diagnose_exponential(dt, lam),
            "bin_size_sec": int(bin_size_sec),
            "bin_counts": counts,
            "diagnose_bin_counts": self.diagnose_poisson_bin_counts(counts, lam, bin_size_sec),
            "simulated_times_sec_from0": sim_times,
            "simulated_n": int(sim_times.size),
        }


# ============================================================
# Piecewise-constant NHPP Analyzer
# ============================================================

class PiecewiseNHPPAnalyzer(PoissonBase):
    """
    piecewise-constant NHPP:
      λ(t) = λ_b  for t in bin b
    where bins partition [start, end)

    - fit: λ_b = N_b / Δ_b  (0-bin 포함)
    - diagnose: time-rescaling theorem
        z_i = ∫_{t_{i-1}}^{t_i} λ(u) du  ~ i.i.d. Exp(1)
        u_i = 1 - exp(-z_i)             ~ i.i.d. Uniform(0,1)
    - simulate: inversion on cumulative intensity (thinning 불필요)
    """

    @staticmethod
    def fit_bins(
        event_times_sec: np.ndarray,
        start_sec: int,
        end_sec: int,
        bin_size_sec: int,
    ) -> Dict[str, Any]:
        if end_sec <= start_sec:
            raise ValueError("end_sec must be > start_sec")
        if bin_size_sec <= 0:
            raise ValueError("bin_size_sec must be positive")

        T = end_sec - start_sec
        B = int(np.ceil(T / bin_size_sec))
        bin_edges = start_sec + np.arange(B + 1, dtype=np.int64) * bin_size_sec
        bin_edges[-1] = end_sec  # 마지막 edge는 end에 맞춤
        bin_lengths = (bin_edges[1:] - bin_edges[:-1]).astype(np.float64)

        # full bin counts(0 포함)
        bins = ((event_times_sec - start_sec) // bin_size_sec).astype(np.int64)
        bins = bins[(bins >= 0) & (bins < B)]
        counts = np.bincount(bins, minlength=B).astype(np.int64)

        lam_bins = counts / bin_lengths  # events/sec
        return {
            "B": int(B),
            "bin_edges_sec": bin_edges,
            "bin_lengths_sec": bin_lengths,
            "counts_bins": counts,
            "lam_bins": lam_bins.astype(np.float64),
        }

    @staticmethod
    def build_cum_intensity(lam_bins: np.ndarray, bin_lengths: np.ndarray) -> np.ndarray:
        cum = np.zeros(lam_bins.size + 1, dtype=np.float64)
        cum[1:] = np.cumsum(lam_bins * bin_lengths)
        return cum

    @staticmethod
    def integrated_intensity_between(
        a: float,
        b: float,
        bin_edges: np.ndarray,
        lam_bins: np.ndarray,
        bin_lengths: np.ndarray,
        cum_int: np.ndarray,
    ) -> float:
        """
        ∫_a^b λ(t) dt for piecewise-constant λ
        a,b: absolute seconds (a<b)
        """
        if b <= a:
            return 0.0

        B = lam_bins.size
        ia = int(np.searchsorted(bin_edges, a, side="right") - 1)
        ib = int(np.searchsorted(bin_edges, b, side="right") - 1)
        ia = max(0, min(B - 1, ia))
        ib = max(0, min(B - 1, ib))

        if ia == ib:
            return float(lam_bins[ia] * (b - a))

        # partial ia
        end_ia = float(bin_edges[ia + 1])
        part_a = float(lam_bins[ia] * (end_ia - a))

        # full middle
        if ib >= ia + 2:
            full_mid = float(cum_int[ib] - cum_int[ia + 1])
        else:
            full_mid = 0.0

        # partial ib
        start_ib = float(bin_edges[ib])
        part_b = float(lam_bins[ib] * (b - start_ib))

        return part_a + full_mid + part_b

    @classmethod
    def time_rescaling_z(
        cls,
        event_times_sec: np.ndarray,
        bin_edges: np.ndarray,
        lam_bins: np.ndarray,
        bin_lengths: np.ndarray,
    ) -> np.ndarray:
        cls._ensure_min_events(event_times_sec, 2)
        cum_int = cls.build_cum_intensity(lam_bins, bin_lengths)

        z = np.zeros(event_times_sec.size - 1, dtype=np.float64)
        for i in range(1, event_times_sec.size):
            a = float(event_times_sec[i - 1])
            b = float(event_times_sec[i])
            z[i - 1] = cls.integrated_intensity_between(
                a, b, bin_edges, lam_bins, bin_lengths, cum_int
            )
        return z

    @staticmethod
    def diagnose_rescaling(z: np.ndarray) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n": int(z.size),
            "mean_z": float(z.mean()) if z.size else None,
            "var_z": float(z.var(ddof=1)) if z.size >= 2 else None,
            "min_z": float(z.min()) if z.size else None,
            "max_z": float(z.max()) if z.size else None,
        }

        # Uniform transform
        u = 1.0 - np.exp(-z)
        out["mean_u"] = float(u.mean()) if u.size else None
        out["var_u"] = float(u.var(ddof=1)) if u.size >= 2 else None

        # KS tests (optional)
        if stats is not None and z.size:
            D_exp, p_exp = stats.kstest(z, "expon", args=(0, 1.0))
            D_unif, p_unif = stats.kstest(u, "uniform", args=(0, 1.0))
            out["ks_exp"] = {"KS_D": float(D_exp), "p_value": float(p_exp)}
            out["ks_unif"] = {"KS_D": float(D_unif), "p_value": float(p_unif)}
        else:
            out["ks_note"] = "scipy 미설치 또는 데이터 부족으로 KS test 생략"

        return out

    @staticmethod
    def simulate_piecewise_nhpp(
        lam_bins: np.ndarray,
        bin_edges: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """
        piecewise inversion 시뮬레이션 (절대 epoch seconds, float)
        """
        rng = np.random.default_rng(seed)
        start = float(bin_edges[0])
        end = float(bin_edges[-1])

        B = lam_bins.size
        t = start
        j = int(np.searchsorted(bin_edges, t, side="right") - 1)
        j = max(0, min(B - 1, j))

        times: List[float] = []

        while t < end:
            e = float(rng.exponential(scale=1.0))  # Exp(1) on intensity scale

            while True:
                if t >= end:
                    break

                left = float(bin_edges[j])
                right = float(bin_edges[j + 1])
                rate = float(lam_bins[j])

                if rate <= 0.0:
                    # 이 bin은 intensity가 0 → 이벤트 불가, 다음 bin으로
                    t = right
                    j += 1
                    if j >= B:
                        t = end
                    continue

                rem = right - t
                cap = rate * rem  # intensity mass in remaining bin

                if e <= cap:
                    t = t + e / rate
                    if t < end:
                        times.append(t)
                    break
                else:
                    e -= cap
                    t = right
                    j += 1
                    if j >= B:
                        t = end
                    continue

        return np.array(times, dtype=np.float64)

    def run(
        self,
        spec: WindowSpec,
        bin_size_sec: int = 300,
        sim_seed: int = 42,
    ) -> Dict[str, Any]:
        df_win = self.filter_window(spec)
        event_times = self._sorted_event_times_sec(df_win)
        self._ensure_min_events(event_times, 2)

        start_sec = self._parse_dt_sec(spec.start)
        end_sec = self._parse_dt_sec(spec.end)

        fit = self.fit_bins(event_times, start_sec, end_sec, bin_size_sec)
        z = self.time_rescaling_z(
            event_times_sec=event_times,
            bin_edges=fit["bin_edges_sec"],
            lam_bins=fit["lam_bins"],
            bin_lengths=fit["bin_lengths_sec"],
        )
        diag = self.diagnose_rescaling(z)

        sim_times = self.simulate_piecewise_nhpp(
            lam_bins=fit["lam_bins"],
            bin_edges=fit["bin_edges_sec"],
            seed=sim_seed,
        )

        return {
            "model": "NHPP(piecewise-constant)",
            "window": {"start": spec.start, "end": spec.end, "bbox": spec.bbox},
            "bin_size_sec": int(bin_size_sec),
            "n_events": int(event_times.size),
            "fit": {
                "B": fit["B"],
                "bin_edges_sec": fit["bin_edges_sec"],
                "bin_lengths_sec": fit["bin_lengths_sec"],
                "counts_bins": fit["counts_bins"],
                "lam_bins_events_per_sec": fit["lam_bins"],
                "lam_bins_events_per_min": fit["lam_bins"] * 60.0,
            },
            "time_rescaling_z": z,
            "diagnostics": diag,
            "simulated_times_sec_epoch": sim_times,
            "simulated_n": int(sim_times.size),
        }
