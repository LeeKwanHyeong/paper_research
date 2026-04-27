from __future__ import annotations

from collections.abc import Iterable

import polars as pl


def _validate_log_base(log_base: float) -> float:
    """
    Guard the transformed-scale utilities against invalid bases.

    We only support meaningful logarithm bases greater than 1.0 because the
    magnitude-factorized mark definition relies on monotonic log scaling.
    """
    log_base = float(log_base)
    if log_base <= 1.0:
        raise ValueError(f"log_base must be > 1.0, got {log_base}")
    return log_base


def make_marks_log_magnitude(
    df: pl.DataFrame,
    *,
    min_order: int = 0,
    max_order: int | None = None,
    clip_min_qty: float = 1.0,
    log_base: float = 10.0,
) -> pl.DataFrame:
    """
    Positive-demand event sequence -> magnitude-factorized labels.

    Labels:
    - mark: floor(log_base(qty)) based order class
    - scale_residual: fractional part of log_base(qty), expected in [0, 1)
      when not upper-clipped
    - log_qty: continuous transformed-scale target used for
      reconstruction/debugging
    - log10_qty: always kept for debugging/reporting, regardless of `log_base`

    Notes:
    - This is the professor-guided path where mark means "order of magnitude",
      not a hand-crafted quantile bin.
    - If max_order is used and values are clipped to the top class, residual can
      become >= 1.0 for overflow samples. That keeps reconstruction information,
      but removes the strict [0, 1) guarantee for those capped cases.
    """
    log_base = _validate_log_base(log_base)
    ev = (
        df.filter(pl.col("demand_qty") > 0)
        .group_by(["oper_part_no", "seq"], maintain_order=True)
        .agg([
            pl.col("demand_dt").first().alias("demand_dt"),
            pl.col("demand_qty").sum().alias("demand_qty"),
        ])
        .sort(["oper_part_no", "seq"])
        .with_columns([
            pl.col("demand_qty").clip(clip_min_qty, None).alias("demand_qty_clipped"),
        ])
        .with_columns([
            # Keep both the chosen training scale and log10 for debugging so we
            # can compare different binning bases without losing a common axis.
            pl.col("demand_qty_clipped").log(base=log_base).alias("log_qty"),
            pl.col("demand_qty_clipped").log(base=10.0).alias("log10_qty"),
            (pl.col("seq") - pl.col("seq").shift(1).over("oper_part_no"))
            .fill_null(0)
            .cast(pl.Int32)
            .alias("delta_t"),
        ])
        .with_columns([
            pl.col("log_qty").floor().cast(pl.Int32).alias("raw_order"),
        ])
    )

    if min_order is not None:
        ev = ev.with_columns(
            pl.col("raw_order").clip(int(min_order), None).alias("raw_order")
        )

    if max_order is not None:
        ev = ev.with_columns(
            pl.col("raw_order").clip(None, int(max_order)).alias("mark")
        )
    else:
        ev = ev.with_columns(pl.col("raw_order").alias("mark"))

    ev = ev.with_columns([
        (pl.col("log_qty") - pl.col("mark")).cast(pl.Float64).alias("scale_residual"),
        pl.col("log_qty").alias("z"),
    ])

    return ev.select([
        "oper_part_no",
        "demand_dt",
        "seq",
        "delta_t",
        "demand_qty",
        "log_qty",
        "log10_qty",
        "scale_residual",
        "mark",
        "z",
    ])


def summarize_log_magnitude_distribution(
    df: pl.DataFrame,
    *,
    clip_min_qty: float = 1.0,
    log_base: float = 10.0,
) -> pl.DataFrame:
    """
    Summarize raw log-base order distribution before deciding merge rules.

    This is intended for the professor's "rebuild mark/binning" workflow:
    inspect the order classes first, then decide whether the highest orders
    should be merged into one tail class.
    """
    log_base = _validate_log_base(log_base)
    ev = (
        df.filter(pl.col("demand_qty") > 0)
        .group_by(["oper_part_no", "seq"], maintain_order=True)
        .agg(pl.col("demand_qty").sum().alias("demand_qty"))
        .with_columns(
            pl.col("demand_qty")
            .clip(clip_min_qty, None)
            .log(base=log_base)
            .floor()
            .cast(pl.Int32)
            .alias("raw_order")
        )
        .group_by("raw_order")
        .len()
        .sort("raw_order")
        .with_columns([
            (pl.col("len") / pl.col("len").sum()).alias("ratio"),
            (pl.col("len").cum_sum() / pl.col("len").sum()).alias("cum_ratio"),
            pl.lit(log_base).alias("log_base"),
        ])
    )
    return ev


def compare_log_base_distributions(
    df: pl.DataFrame,
    *,
    log_bases: Iterable[float] = (10.0, 4.0, 2.0),
    clip_min_qty: float = 1.0,
) -> pl.DataFrame:
    """
    Stack multiple raw order distributions into one table for quick comparison.

    This is useful when we want to see whether log10 is too coarse and whether
    log4 or log2 spreads the dominant mark-0 population more evenly.
    """
    distributions: list[pl.DataFrame] = []
    for log_base in log_bases:
        dist = summarize_log_magnitude_distribution(
            df,
            clip_min_qty=clip_min_qty,
            log_base=log_base,
        )
        distributions.append(dist)

    if not distributions:
        raise ValueError("log_bases must contain at least one base.")

    return pl.concat(distributions).select([
        "log_base",
        "raw_order",
        "len",
        "ratio",
        "cum_ratio",
    ]).sort(["log_base", "raw_order"])


def suggest_max_order(
    df: pl.DataFrame,
    *,
    clip_min_qty: float = 1.0,
    min_count: int = 100,
    min_coverage: float = 0.999,
    log_base: float = 10.0,
) -> int:
    """
    Suggest an upper order cap for mark classes.

    Heuristic:
    - keep orders while cumulative coverage is below `min_coverage`
    - or while the class has at least `min_count` samples
    - merge anything above the returned order into the top class
    """
    dist = summarize_log_magnitude_distribution(
        df,
        clip_min_qty=clip_min_qty,
        log_base=log_base,
    )
    rows = dist.to_dicts()
    if not rows:
        raise ValueError("No positive-demand events found.")

    suggested = int(rows[-1]["raw_order"])
    for row in rows:
        raw_order = int(row["raw_order"])
        count = int(row["len"])
        coverage = float(row["cum_ratio"])
        suggested = raw_order
        if coverage >= min_coverage and count < min_count:
            break

    return suggested

def make_marks_intermittent_default(df: pl.DataFrame, K: int = 12, p1: float = 0.99, p2: float = 0.999) -> pl.DataFrame:
    # assert K >= 6

    # event 생성
    ev = (
        df.filter(pl.col('demand_qty') > 0) # 0 이상인것들만
          .sort(['oper_part_no', 'seq'])
          .with_columns([
              pl.col('demand_qty').log1p().alias('z'),
             (pl.col('seq') - pl.col('seq').shift(1).over('oper_part_no'))
                .fill_null(0)
                .cast(pl.Int32)
                .alias('delta_t'),
        ])
    )

    q1 = ev.select(pl.col("demand_qty").quantile(p1, "nearest").alias("q1")).item()
    q2 = ev.select(pl.col("demand_qty").quantile(p2, "nearest").alias("q2")).item()
    base = ev.filter(pl.col('demand_qty') <= q1)

    n_base_bins = K - 2
    cut_probs = [i / n_base_bins for i in range(1, n_base_bins)]
    print(cut_probs)

    q_expr = [
        pl.col("z").quantile(p, "nearest").alias(f"q_{i:02d}")
        for i, p in enumerate(cut_probs, start=1)
    ]

    # base가 너무 작거나(이벤트 부족) cuts가 비는 경우를 대비
    if base.height == 0:
        cuts = []
    else:
        row = base.select(q_expr).row(0)
        cuts = sorted(set(float(x) for x in row if x is not None))
        print(cuts)
    # z -> base_bin (0..n_base_bins-1)
    expr = None
    for i, c in enumerate(cuts):
        if expr is None:
            expr = pl.when(pl.col('z') <= c).then(i)
        else:
            expr = expr.when(pl.col('z') <= c).then(i)

    base_bin = (pl.lit(0) if expr is None else expr.otherwise(len(cuts))).clip(0, n_base_bins - 1)
    print(base_bin)
    out = ev.with_columns(
        pl.when(pl.col('demand_qty') <= q1)
          .then(base_bin)
          .when(pl.col('demand_qty') <= q2)
          .then(pl.lit(K-2))
          .otherwise(pl.lit(K-1))
          .cast(pl.Int32)
          .alias('mark')
    )

    return out.select(['oper_part_no', 'demand_dt', 'seq', 'delta_t', 'demand_qty', 'z', 'mark'])

def make_marks_even_quantile(df: pl.DataFrame, K: int = 4) -> pl.DataFrame:
    # 1) Event(>0)만 + 중복 주차 집계
    ev = (
        df.filter(pl.col('demand_qty') > 0)
          .group_by(['oper_part_no', 'seq'], maintain_order=True)
          .agg([
            pl.col('demand_dt').first().alias('demand_dt'),
            pl.col('demand_qty').sum().alias('demand_qty'),
        ])
        .sort(['oper_part_no', 'seq'])
    )

    # 2) z, delta_t
    ev = ev.with_columns([
        pl.col('demand_qty').log1p().alias('z'),
        (pl.col('seq') - pl.col('seq').shift(1).over('oper_part_no'))
          .fill_null(0)
          .cast(pl.Int32)
          .alias('delta_t'),
    ])

    # 3) 첫 이벤트 제거
    ev = (
        ev.with_columns(pl.cum_count('oper_part_no').over('oper_part_no').alias('_rn'))
          .filter(pl.col('_rn') > 0)
          .drop('_rn')
    )

    # 4) Global 균등 빈도 binning (rank 기반)
    #    - ordinal rank: 동일 z라도 순서를 부여하여 bin 균등화
    N = ev.height
    ev = ev.with_columns(
        (
            (pl.col('z').rank('ordinal') - 1) * K / pl.lit(N)
        ).floor().cast(pl.Int32).clip(0, K - 1).alias('mark')
    )

    return ev.select(['oper_part_no', 'demand_dt', 'seq', 'delta_t', 'demand_qty', 'z', 'mark'])
