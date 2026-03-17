import polars as pl

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