from datetime import datetime
import numpy as np
from scipy import stats

class PoissonProcess:
    def __init__(self, df: pl.DataFrame, start: str, end: str, bin_size_sec = 60):
        self.df = df
        self.start = start
        self.end = end
        self.bin_size_sec = bin_size_sec

    '''
    특정 시간 구간 + 특정 공간 필터

    Poisson 가정이 성립하려면, rate가 크게 변하지 않는 구간을 잡는 것이 중요.

    Example)
        - 특정 날짜 하루
        - 특정 시간대 (Ex: 20:00 ~ 21:00)
        - 특정 Location bounding box
    '''
    def _filter_window(self, df, start, end, bbox = None):
        start_dt = datetime.strptime(start, '%Y-%m_%d %H:%M:%S')
        end_dt = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')

        out = df.filter(
            (pl.col('pick_ts') >= pl.lit(start_dt)) &
            (pl.col('pick_ts') < pl.lit(end_dt))
        )

        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            out = out.filter(
                (pl.col('pick_lon') >= min_lon) & (pl.col('pick_lon') <= max_lon) &
                (pl.col('pick_lat') >= min_lat) & (pl.col('pick_lat') <= max_lat)
            )

        return out

    '''
    λ(레이트) 추정 (MLE)

    관측 구간 길이를 T(초 단위)라 하면, Homogeneous Poisson process의 MLE는 $\hat{\lambda} = \frac{N}{T}$
    '''
    def _estimate_lambda(self, df_win: pl.DataFrame) -> float:
        # pick_ts ascending
        ts = df_win.sort(pl.col('pick_ts')).sort('pick_ts')
        n = ts.height
        if n < 2:
            raise ValueError('이벤트가 너무 작음(최소 2개 이상 권장).')

        t0 = ts[0, 0]
        t1 = ts[-1, 0]

        T_seconds = float(t1 - t0)
        if T_seconds <= 0:
            raise ValueError('시간 범위 T가 0 이하')
        lam = n / T_seconds
        return lam

    def _get_interarrival_seconds(self, df_win: pl.DataFrame) -> np.ndarray:
        ts = df_win.select(pl.col('pick_ts')).sort('pick_ts').to_series()
        t = ts.to_list()
        dt = np.array([(t[i] -t[i-1]).total_seconds() for i in range(1, len(t))], dtype = float)
        return dt


    '''
    Check A) Inter-arrival time이 Exponential(λ) 인가? </br>
    Poisson Process이면, 이벤트 간격 $\Delta t_i = t_i - t_{i-1}$ 가 i.i.d. Exp(λ) 입니다. </br>
    아래는 KS test + 기본 통계(평균≈1/λ, 분산≈1/λ²) 체크입니다.
    '''
    def _check_exponential(self, dt: np.ndarray, lam: float):
        # base diagnosis
        mean_dt = dt.mean()
        var_dt = dt.var(ddof = 1)

        # theoretical value
        theo_mean = 1.0 / lam
        theo_var = 1.0 / (lam ** 2)

        out = {
            'emp_mean_dt': mean_dt,
            'emp_var_dt': var_dt,
            'theo_mean_dt': theo_mean,
            'theo_var_dt': theo_var,
        }
        return out

    def _ks_test_exponential(self, dt: np.ndarray, lam: float):
        # Exp(scale = 1/lam)
        # scipy의 expon은 scale 파라미터 사용
        D, p = stats.kstest(dt, 'expon', args = (0, 1.0/lam))
        return {'KS_D': float(D), 'p_value': float(p)}

    '''
    Check B) Bin-count가 Poisson(λΔ)인가? (Mean≈Var 확인 포함)</br>
    관측 구간을 동일 길이 bin(예: 1분/5분/10분)으로 나누고, 각 bin의 이벤트 수를 세면:</br>
    $N_j \sim \text{Poisson}(\lambda \Delta)$
    '''
    def _bin_counts(self, df_win: pl.DataFrame, bin_size_sec: int = 60) -> np.ndarray:
        tmp = df_win.select([(pl.col('pick_ts').dt.epoch('s')).alias('tsec')]).sort('tsec')
        t0 = tmp[0, 0]
        tmp = tmp.with_columns(
            ((pl.col('tsec') - t0) // bin_size_sec).cast(pl.Int64).alias('bin')
        )

        counts = tmp.group_by('bin').len().sort('bin')['len'].to_numpy()
        return counts

    def _check_poisson_counts(self, counts: np.ndarray, lam: float, bin_size_sec: int):
        emp_mean = counts.mean()
        emp_var = counts.var(ddof = 1)
        theo_mean = lam * bin_size_sec

        return {
            'emp_mean': float(emp_mean),
            'emp_var': float(emp_var),
            'theo_mean_lambdaDelta': float(theo_mean),
            'overdispersion_var_over_mean': float(emp_var / emp_mean) if emp_mean > 0 else None
        }

    def _simulate_poisson_process(self, lam: float, T_seconds: float, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        t = 0.0
        times = []
        while True:
            dt = rng.exponential(scale = 1.0/lam)
            t += dt
            if t > T_seconds:
                break
            times.append(t)
        return np.array(times, dtype = float)


    def run_homogeneous_poisson_process(
            self
    ):
        df = self.df

        df_win = self._filter_window(
            df,
            start = start,
            end = end,
            bbox = None
        )

        lam = self._estimate_lambda(df_win)
        dt = self._get_interarrival_seconds(df_win)

        print("lambda_hat (events/sec):", lam)
        print("lambda_hat (events/min):", lam * 60)

        print("Exponential check:", check_exponential(dt, lam))
        print("KS test:", ks_test_exponential(dt, lam))

        counts = self._bin_counts(df_win, bin_size_sec = self.bin_size_sec)
        print('Poisson count check:', self._check_poisson_counts(counts, lam, self.bin_size_sec))

        ts_sorted = df_win.select('pick_ts').sort('pick_ts').to_series().to_list()
        T_seconds = (ts_sorted[-1] - ts_sorted[0]).total_seconds()
        sim_times = self._simulate_poisson_process(lam = lam, T_seconds = T_seconds, seed = 42)

        print('Observed N: ', len(ts_sorted))
        print('Simulated N: ', len(sim_times))


