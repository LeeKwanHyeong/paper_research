from __future__ import annotations

"""
Intermittency / Demand Type Detector UseCase (refactored for AnalyticsService)
=============================================================================

## 목적
주간/월간 수요 시계열에서 자재(oper_part_no)별로 다음을 산출함.

- ADI (Average Demand Interval)
- CV / CV2 (변동성 지표)
- zero_ratio (0 비율; 희소성)
- mean_nonzero_gap (수요 발생 간 평균 간격; seq 기반)
- demand_type: smooth / erratic / intermittent / lumpy (+ insufficient / no_demand)

## freq 처리 원칙
- freq는 'weekly' | 'monthly' 중 하나
- 본 UseCase는 입력 df가 이미 해당 freq로 집계되어 있다고 가정함.
  (예: monthly 예측이면 master_target_df 자체가 월 인덱스(yyyymm 등)로 집계되어 들어오거나,
       weekly 테이블을 monthly로 집계한 df가 들어오는 형태)
- 결과 저장/테이블명 suffix 등에 freq를 사용 가능

## 분류 기준 (Syntetos–Boylan 계열에서 널리 쓰는 threshold)
- ADI threshold: 1.32
- CV^2 threshold: 0.49
  - ADI < 1.32 and CV2 < 0.49  -> Smooth
  - ADI < 1.32 and CV2 >= 0.49 -> Erratic
  - ADI >= 1.32 and CV2 < 0.49 -> Intermittent
  - ADI >= 1.32 and CV2 >= 0.49-> Lumpy

## 입력 스키마 (tb_master_target)
Schema([
    ('oper_part_no', String),
    ('demand_dt', Int64),   # weekly: yyyyww / monthly: yyyymm (권장) 또는 내부 규약
    ('demand_qty', Float64),
    ('seq', UInt32)         # 1..N (part별 연속 인덱스; gap 계산에 권장)
])

### 주의
- epsilon은 "거의 0"을 0으로 취급하기 위한 기준임.
  예: 20 이하를 0으로 간주하려면 epsilon=20으로 설정.
- mean_nonzero_gap은 yyyyww 파싱보다 안정적인 seq(연속 인덱스)로 계산함.
"""

from dataclasses import dataclass
from typing import Literal

import polars as pl



# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class IntermittentConfig:
    # columns
    id_col: str = "oper_part_no"
    y_col: str = "demand_qty"
    date_col: str = "demand_dt"
    seq_col: str = "seq"

    # classification thresholds
    adi_threshold: float = 1.32
    cv2_threshold: float = 0.49

    # y <= epsilon => 0 취급
    epsilon: float = 0.0

    # 최소 길이
    min_periods: int = 10

    # labels
    no_demand_label: str = "no_demand"
    insufficient_label: str = "insufficient"


class IntermittentDetector:
    """
    AnalyticsService 호출 규약:
      uc = IntermittentDetectorUseCase(freq=fcst_freq, config=IntermittentConfig(...))
      out = uc.run(df=self.master_target_df)
    """

    def __init__(
        self,
        *,
        freq: Literal["weekly", "monthly"] = 'weekly',
        config: IntermittentConfig,
    ):
        if freq not in ("weekly", "monthly"):
            raise ValueError("[IntermittentDetectorUseCase] freq must be 'weekly' or 'monthly'")
        self.freq = freq
        self.cfg = config

    # -------------------------
    # Public API
    # -------------------------
    def run(self, *, df: pl.DataFrame, return_stats: bool = True) -> pl.DataFrame:
        """
        Args:
            df: tb_master_target (already aligned to freq)
            return_stats: True면 지표까지 포함

        Returns:
            part-level DataFrame
        """
        stats = self._compute_stats(df)
        out = self._classify(stats)


        if return_stats:
            cols = [
                self.cfg.id_col,
                "demand_type",
                "is_sparsity",
                "n_periods",
                "n_zero",
                "n_nz",
                "zero_ratio",
                "ADI",
                "CV",
                "CV2",
                "mean_nonzero_gap",
                "median_nonzero_gap",
                "nz_mean",
                "nz_std",
                "freq",
            ]
            cols = [c for c in cols if c in out.columns]
            return out.select(cols)

        # return out.select([self.cfg.id_col, "demand_type", "is_sparsity", "freq"])
        return out

    # -------------------------
    # Validation
    # -------------------------
    def _validate_input(self, df: pl.DataFrame) -> None:
        cfg = self.cfg

        required = [cfg.id_col, cfg.y_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[IntermittentDetector] Missing required columns: {missing}")

        # seq는 optional이지만 있으면 검증
        if cfg.seq_col in df.columns:
            bad_seq = df.select((pl.col(cfg.seq_col) <= 0).any().alias("bad")).item()
            if bool(bad_seq):
                raise ValueError("[IntermittentDetector] 'seq' must be >= 1 for all rows.")

        neg_y = df.select((pl.col(cfg.y_col) < 0).any().alias("neg")).item()
        if bool(neg_y):
            raise ValueError("[IntermittentDetector] demand_qty must be >= 0 for all rows.")

    # -------------------------
    # Stats
    # -------------------------
    def _compute_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        part-level stats:
          - n_periods, n_zero, n_nz
          - zero_ratio
          - nz_mean, nz_std
          - CV, CV2
          - ADI
          - mean_nonzero_gap, median_nonzero_gap (seq 기반; 없으면 null)
        """
        self._validate_input(df)
        cfg = self.cfg

        idc, yc, seqc = cfg.id_col, cfg.y_col, cfg.seq_col
        eps = float(cfg.epsilon)

        lf = df.lazy()

        # nonzero: y > eps
        is_nz = pl.col(yc) > eps

        stats = (
            lf.group_by(idc)
            .agg(
                pl.len().alias("n_periods"),
                ((~is_nz).cast(pl.UInt32)).sum().alias("n_zero"),
                (is_nz.cast(pl.UInt32)).sum().alias("n_nz"),
                pl.col(yc).filter(is_nz).mean().alias("nz_mean"),
                pl.col(yc).filter(is_nz).std(ddof=1).alias("nz_std"),
            )
            .with_columns(
                (pl.col("n_zero") / pl.col("n_periods")).alias("zero_ratio"),
                pl.when(pl.col("n_nz") > 0)
                .then(pl.col("n_periods") / pl.col("n_nz"))
                .otherwise(None)
                .alias("ADI"),
                pl.when((pl.col("nz_mean") > 0) & pl.col("nz_std").is_not_null())
                .then(pl.col("nz_std") / pl.col("nz_mean"))
                .otherwise(None)
                .alias("CV"),
                pl.when((pl.col("nz_mean") > 0) & pl.col("nz_std").is_not_null())
                .then((pl.col("nz_std") / pl.col("nz_mean")) ** 2)
                .otherwise(None)
                .alias("CV2"),
            )
        )

        # mean_nonzero_gap: seq diff mean/median on nonzero events
        if seqc in df.columns:
            gap = (
                lf.select([idc, seqc, yc])
                .filter(is_nz)
                .sort([idc, seqc])
                .with_columns((pl.col(seqc) - pl.col(seqc).shift(1)).over(idc).alias("_gap"))
                .group_by(idc)
                .agg(
                    pl.col("_gap").drop_nulls().mean().alias("mean_nonzero_gap"),
                    pl.col("_gap").drop_nulls().median().alias("median_nonzero_gap"),
                )
            )
            stats = stats.join(gap, on=idc, how="left")
        else:
            stats = stats.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("mean_nonzero_gap"),
                pl.lit(None, dtype=pl.Float64).alias("median_nonzero_gap"),
            )

        return stats.collect()

    # -------------------------
    # Classification
    # -------------------------
    def _classify(self, stats: pl.DataFrame) -> pl.DataFrame:
        cfg = self.cfg

        enough = pl.col("n_periods") >= int(cfg.min_periods)
        adi = pl.col("ADI")
        cv2 = pl.col("CV2")

        demand_type_expr = (
            pl.when(pl.col("n_nz") == 0)
            .then(pl.lit(cfg.no_demand_label))
            .when((~enough) | adi.is_null() | cv2.is_null())
            .then(pl.lit(cfg.insufficient_label))
            .when((adi < cfg.adi_threshold) & (cv2 < cfg.cv2_threshold))
            .then(pl.lit("smooth"))
            .when((adi < cfg.adi_threshold) & (cv2 >= cfg.cv2_threshold))
            .then(pl.lit("erratic"))
            .when((adi >= cfg.adi_threshold) & (cv2 < cfg.cv2_threshold))
            .then(pl.lit("intermittent"))
            .otherwise(pl.lit("lumpy"))
            .alias("demand_type")
        )

        out = (
            stats.lazy()
            .with_columns(
                demand_type_expr
            )
            .with_columns([
                pl.col("demand_type").is_in(["intermittent", "lumpy"]).alias("is_sparsity"),
                pl.lit(self.freq).alias("freq"),
            ])
            .collect()
        )
        return out