import random
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# -----------------------------
# mark remap (동일)
# -----------------------------
def remap_marks_in_samples(samples):
    uniq = sorted({m for marks, _ in samples for m in marks})
    mapping = {m: i for i, m in enumerate(uniq)}
    remapped = [([mapping[m] for m in marks], times) for marks, times in samples]
    return remapped, mapping


# -----------------------------
# Dataset: dt 스케일링 적용
# -----------------------------
class EventSeqDataset(Dataset):
    def __init__(self, samples, *, dt_scale=3600.0, dt_clip_scaled=None):
        """
        dt_scale:
          - 예: 3600이면 dt를 '시간 단위'로 바꿈 (dt_seconds / 3600)
        dt_clip_scaled:
          - 스케일 적용 후 clip. 예: 4.0이면 4시간 이상은 4로 자름
        """
        self.samples = samples
        self.dt_scale = float(dt_scale)
        self.dt_clip_scaled = dt_clip_scaled

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        marks_list, times_list = self.samples[idx]
        assert len(marks_list) == len(times_list)

        dts = [0.0]
        for i in range(1, len(times_list)):
            d_sec = float(times_list[i] - times_list[i - 1])
            d_sec = max(d_sec, 0.0)

            # 1) scale
            d = d_sec / self.dt_scale

            # 2) optional clip (scaled domain)
            if self.dt_clip_scaled is not None:
                d = min(d, float(self.dt_clip_scaled))

            dts.append(d)

        return (
            torch.tensor(marks_list, dtype=torch.long),
            torch.tensor(dts, dtype=torch.float32),
        )


def collate_pad(batch, pad_mark: int):
    marks_list, dts_list = zip(*batch)
    B = len(marks_list)
    Lmax = max(x.size(0) for x in marks_list)

    marks = torch.full((B, Lmax), fill_value=pad_mark, dtype=torch.long)
    dts = torch.zeros((B, Lmax), dtype=torch.float32)
    mask = torch.zeros((B, Lmax), dtype=torch.bool)

    for i, (m, dt) in enumerate(zip(marks_list, dts_list)):
        L = m.size(0)
        marks[i, :L] = m
        dts[i, :L] = dt
        mask[i, :L] = True

    return marks, dts, mask


def make_loaders_from_samples(samples, *, K, batch_size=128, val_ratio=0.1, seed=42,
                              dt_scale=3600.0, dt_clip_scaled=4.0):
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)

    n_val = max(1, int(len(samples) * val_ratio))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    tr_samples = [samples[i] for i in tr_idx]
    val_samples = [samples[i] for i in val_idx]

    train_ds = EventSeqDataset(tr_samples, dt_scale=dt_scale, dt_clip_scaled=dt_clip_scaled)
    val_ds = EventSeqDataset(val_samples, dt_scale=dt_scale, dt_clip_scaled=dt_clip_scaled)

    pad_mark = K  # PAD = K (마지막 인덱스). 모델 num_marks는 K+1.

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_pad(b, pad_mark=pad_mark),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_pad(b, pad_mark=pad_mark),
        drop_last=False,
    )
    return train_loader, val_loader, pad_mark


class RMTPPDataset(Dataset):
    """
    전체 이벤트 시퀀스를 유지한 채,
    샘플(윈도우)을 train/val로 분할하는 RMTPP next-event dataset.

    - lookback L개 이벤트를 입력으로 사용
    - 다음 이벤트(1개)를 target으로 사용
    - split은 자재별 이벤트 index 기준으로 수행:
        k = floor(n * (1 - val_ratio))
        target_idx < k  -> train
        target_idx >= k -> val
    """

    def __init__(
        self,
        marked_df: pl.DataFrame,
        lookback: int = 30,
        val_ratio: float = 0.2,
        mode: str = "train",          # "train" | "val"
        clip_dt_min1: bool = True,
    ):
        assert mode in ("train", "val")
        self.lookback = int(lookback)
        self.val_ratio = float(val_ratio)
        self.mode = mode

        df = marked_df.sort(["oper_part_no", "seq"])

        if clip_dt_min1:
            df = df.with_columns(
                pl.col("delta_t").cast(pl.Int32).clip(1, None).alias("delta_t")
            )

        grouped = (
            df.group_by("oper_part_no")
              .agg([
                  pl.col("delta_t").alias("dt_list"),
                  pl.col("mark").alias("mk_list"),
                  pl.col("scale_residual").alias("val_list") if "scale_residual" in df.columns else pl.lit(None).alias("val_list"),
              ])
        )

        self.parts = grouped["oper_part_no"].to_list()
        self.dt_lists = grouped["dt_list"].to_list()
        self.mk_lists = grouped["mk_list"].to_list()
        self.val_lists = grouped["val_list"].to_list()

        # 전역 인덱스: (part_idx, end_pos)
        self.index = []

        for p_idx, (dt_list, mk_list) in enumerate(zip(self.dt_lists, self.mk_lists)):
            n = len(dt_list)
            # 입력 L개 + target 1개 필요
            if n < self.lookback + 1:
                continue

            # 자재별 split boundary (event index)
            k = int(np.floor(n * (1.0 - self.val_ratio)))  # target_idx >= k -> val

            # end_pos: 입력 마지막 위치, target은 end_pos+1
            for end_pos in range(self.lookback - 1, n - 1):
                target_idx = end_pos + 1

                if self.mode == "train":
                    if target_idx < k:
                        self.index.append((p_idx, end_pos))
                else:  # val
                    if target_idx >= k:
                        self.index.append((p_idx, end_pos))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        p_idx, end_pos = self.index[idx]

        dt = np.asarray(self.dt_lists[p_idx], dtype=np.float32)
        mk = np.asarray(self.mk_lists[p_idx], dtype=np.int64)
        val_list = self.val_lists[p_idx]
        val = None if val_list is None else np.asarray(val_list, dtype=np.float32)

        start = end_pos - self.lookback + 1

        x_dt = dt[start:end_pos + 1]        # (L,)
        x_mk = mk[start:end_pos + 1]        # (L,)
        y_dt = dt[end_pos + 1]              # ()
        y_mk = mk[end_pos + 1]              # ()

        sample = {
            "x_mk": torch.from_numpy(x_mk).long(),
            "x_dt": torch.from_numpy(x_dt).float(),
            "y_mk": torch.tensor(y_mk).long(),
            "y_dt": torch.tensor(y_dt).float(),
            "part_idx": torch.tensor(p_idx).long(),
        }

        if val is not None:
            sample["x_val"] = torch.from_numpy(val[start:end_pos + 1]).float()
            sample["y_val"] = torch.tensor(val[end_pos + 1]).float()

        return sample


class RMTPPWeekLookbackDataset(Dataset):
    """
    TS 방식: lookback을 '이벤트 개수 L'이 아니라 '최근 W주(weeks)'로 정의.

    샘플 구성:
      - context_end = i (0..n-2)
      - input window = events with seq in [seq_i - (W-1), seq_i]  (최근 W주)
      - target event = i+1 을 sequence 맨 끝에 append (teacher forcing)
      - variable length -> max_seq_len로 left-pad, mask 생성

    반환:
      marks: [max_seq_len] (long)  (PAD=pad_id)
      dts:   [max_seq_len] (float) (PAD=0)
      mask:  [max_seq_len] (bool)
    """

    def __init__(
        self,
        marked_df: pl.DataFrame,
        lookback_weeks: int = 52,
        max_seq_len: int = 64,      # (context events within W weeks) + 1(target) 를 담을 최대 길이
        val_ratio: float = 0.2,
        mode: str = "train",        # "train" | "val"  (target index 기준 split)
        pad_id: int | None = None,  # None이면 K_real(=mark.max+1)로 자동 설정
        clip_dt_min1: bool = True,
    ):
        assert mode in ("train", "val")
        self.W = int(lookback_weeks)
        self.max_len = int(max_seq_len)
        self.val_ratio = float(val_ratio)
        self.mode = mode

        df = marked_df.sort(["oper_part_no", "seq"])

        if clip_dt_min1:
            df = df.with_columns(
                pl.col("delta_t").cast(pl.Int32).clip(1, None).alias("delta_t")
            )

        # mark range 확인 → pad_id 자동 설정
        if pad_id is None:
            k_real = int(df.select((pl.col("mark").max() + 1).alias("k")).item())
            pad_id = k_real
        self.pad_id = int(pad_id)

        grouped = (
            df.group_by("oper_part_no")
              .agg([
                  pl.col("seq").cast(pl.Int32).alias("seq_list"),
                  pl.col("delta_t").cast(pl.Float32).alias("dt_list"),
                  pl.col("mark").cast(pl.Int32).alias("mk_list"),
                  pl.col("scale_residual").cast(pl.Float32).alias("val_list") if "scale_residual" in df.columns else pl.lit(None).alias("val_list"),
              ])
        )

        self.parts = grouped["oper_part_no"].to_list()
        self.seq_lists = grouped["seq_list"].to_list()
        self.dt_lists = grouped["dt_list"].to_list()
        self.mk_lists = grouped["mk_list"].to_list()
        self.val_lists = grouped["val_list"].to_list()

        # 샘플 인덱스: (part_idx, context_end_idx=i)
        self.index = []
        for p_idx, seq_list in enumerate(self.seq_lists):
            n = len(seq_list)
            if n < 2:
                continue

            k = int(np.floor(n * (1.0 - self.val_ratio)))  # target_idx >= k -> val
            for i in range(0, n - 1):  # target exists at i+1
                target_idx = i + 1
                if self.mode == "train":
                    if target_idx < k:
                        self.index.append((p_idx, i))
                else:
                    if target_idx >= k:
                        self.index.append((p_idx, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        p_idx, i = self.index[idx]

        seq = np.asarray(self.seq_lists[p_idx], dtype=np.int32)
        dt  = np.asarray(self.dt_lists[p_idx], dtype=np.float32)
        mk  = np.asarray(self.mk_lists[p_idx], dtype=np.int64)
        val_list = self.val_lists[p_idx]
        val = None if val_list is None else np.asarray(val_list, dtype=np.float32)

        t_i = int(seq[i])
        left = t_i - (self.W - 1)

        # context window indices: [j..i] where seq[j] >= left
        # (seq는 정렬되어 있으므로 이진탐색)
        j = int(np.searchsorted(seq, left, side="left"))
        ctx_idx = np.arange(j, i + 1, dtype=np.int32)

        # target append (i+1)
        tgt_idx = i + 1
        all_idx = np.concatenate([ctx_idx, np.array([tgt_idx], dtype=np.int32)], axis=0)

        # 필요하면 최근 max_len만 유지 (left-truncate)
        if len(all_idx) > self.max_len:
            all_idx = all_idx[-self.max_len:]

        mk_seq = mk[all_idx]             # (T,)
        dt_seq = dt[all_idx]             # (T,)
        T = len(all_idx)

        # left-pad to max_len
        pad_len = self.max_len - T
        if pad_len > 0:
            mk_pad = np.full((self.max_len,), self.pad_id, dtype=np.int64)
            dt_pad = np.zeros((self.max_len,), dtype=np.float32)
            # Residual padding uses 0.0 because masked positions are ignored in loss.
            val_pad = np.zeros((self.max_len,), dtype=np.float32)
            mask   = np.zeros((self.max_len,), dtype=np.bool_)

            mk_pad[pad_len:] = mk_seq
            dt_pad[pad_len:] = dt_seq
            if val is not None:
                val_pad[pad_len:] = val[all_idx]
            mask[pad_len:]   = True
        else:
            mk_pad = mk_seq.astype(np.int64)
            dt_pad = dt_seq.astype(np.float32)
            val_pad = val[all_idx].astype(np.float32) if val is not None else np.zeros((self.max_len,), dtype=np.float32)
            mask   = np.ones((self.max_len,), dtype=np.bool_)

        sample = {
            "marks": torch.from_numpy(mk_pad).long(),     # (max_len,)
            "dts":   torch.from_numpy(dt_pad).float(),    # (max_len,)
            "mask":  torch.from_numpy(mask).bool(),       # (max_len,)
            "part_idx": torch.tensor(p_idx).long(),
        }

        if val is not None:
            sample["values"] = torch.from_numpy(val_pad).float()

        return sample


def collate_week_lookback(batch):
    marks = torch.stack([b["marks"] for b in batch], dim=0).long()   # (B,Lmax)
    dts   = torch.stack([b["dts"] for b in batch], dim=0).float()    # (B,Lmax)
    mask  = torch.stack([b["mask"] for b in batch], dim=0).bool()    # (B,Lmax)
    part_idx = torch.stack([b["part_idx"] for b in batch], dim=0).long()
    values = None
    if "values" in batch[0]:
        values = torch.stack([b["values"] for b in batch], dim=0).float()
    return marks, dts, mask, part_idx, values


def collate_next_event(batch):
    # (B, L)
    x_mk = torch.stack([b['x_mk'] for b in batch], dim=0).long()
    x_dt = torch.stack([b['x_dt'] for b in batch], dim=0).float()

    # (B,)
    y_mk = torch.stack([b['y_mk'] for b in batch], dim=0).long()
    y_dt = torch.stack([b['y_dt'] for b in batch], dim=0).float()

    # (B,)
    part_idx = torch.stack([b['part_idx'] for b in batch], dim=0).long()
    x_val = None
    y_val = None
    if "x_val" in batch[0]:
        x_val = torch.stack([b["x_val"] for b in batch], dim=0).float()
        y_val = torch.stack([b["y_val"] for b in batch], dim=0).float()

    return x_mk, x_dt, y_mk, y_dt, part_idx, x_val, y_val


def _clip_dt_min1(df: pl.DataFrame) -> pl.DataFrame:
    # delta_t가 0이 섞여 있으면 학습/시뮬이 흔들리므로 최소 1로 클립
    return df.with_columns(
        pl.col("delta_t").cast(pl.Int32).clip(1, None).alias("delta_t")
    )

def time_split_events(marked_df: pl.DataFrame, val_ratio: float = 0.2) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    part별 이벤트를 시간순으로 split:
      train: 앞쪽 (1-val_ratio)
      val:   뒤쪽 val_ratio
    """
    df = _clip_dt_min1(marked_df).sort(["oper_part_no", "seq"])

    # 각 part별 전체 길이 n, split index k 계산
    meta = (
        df.group_by("oper_part_no")
          .agg(pl.len().alias("n"))
          .with_columns((pl.col("n") * (1.0 - val_ratio)).floor().cast(pl.Int32).alias("k"))
    )

    df2 = df.join(meta, on="oper_part_no", how="left")

    # part별 row_number 생성 후 k 기준 split
    df2 = df2.with_columns(
        pl.cum_count('oper_part_no').over("oper_part_no").alias("rn")  # 0..n-1
    )

    train_df = df2.filter(pl.col("rn") < pl.col("k")).drop(["n", "k", "rn"])
    val_df   = df2.filter(pl.col("rn") >= pl.col("k")).drop(["n", "k", "rn"])

    return train_df, val_df

