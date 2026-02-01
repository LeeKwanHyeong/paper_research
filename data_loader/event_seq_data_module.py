import random
import torch
from torch.utils.data import Dataset, DataLoader

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
