"""Canonical next-event windows, with filtering before parquet materialization."""

import hashlib
import json

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


REQUIRED = ("oper_part_no", "seq", "delta_t", "demand_qty", "chronological_split")


def load_frame(path, *, include_test=False):
    splits = ["train", "validation"] + (["test"] if include_test else [])
    frame = (pl.scan_parquet(path).select(REQUIRED)
             .filter(pl.col("chronological_split").is_in(splits)).collect()
             .sort(["oper_part_no", "seq"]))
    validate_frame(frame)
    return frame


def validate_frame(frame):
    if frame.height == 0 or frame.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("Empty data or null event fields")
    for column in ("seq", "delta_t", "demand_qty"):
        values = frame[column].to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Nonfinite or negative " + column)
    seq = frame["seq"].to_numpy()
    if not np.equal(seq, np.floor(seq)).all() or (seq > np.iinfo(np.int32).max).any():
        raise ValueError("seq must fit the recorded int32 event grid")
    if frame.select(pl.struct("oper_part_no", "seq").n_unique()).item() != frame.height:
        raise ValueError("Duplicate series/sequence identities")
    codes = {"train": 0, "validation": 1, "test": 2}
    for group in frame.partition_by("oper_part_no", maintain_order=True):
        try:
            order = np.array([codes[split] for split in group["chronological_split"]])
        except KeyError as exc:
            raise ValueError("Unknown chronological split") from exc
        if (np.diff(order) < 0).any():
            raise ValueError("Splits are not chronological within a series")


class EventWindows(Dataset):
    """The legacy lookback window and target population, without dummy marks.

    lookback is measured in seq units, not necessarily weeks. Return the
    observed context plus one target, right-padded as in the former objective.
    """

    def __init__(self, frame, *, target_split, lookback, max_seq_len):
        if type(lookback) is not int or lookback < 1 or type(max_seq_len) is not int or max_seq_len < 2:
            raise ValueError("Positive integer lookback and max_seq_len >= 2 are required")
        self.lookback, self.max_seq_len = lookback, max_seq_len
        self.target_split = target_split
        self.groups = frame.sort(["oper_part_no", "seq"]).partition_by("oper_part_no", maintain_order=True)
        self.parts = [g["oper_part_no"][0] for g in self.groups]
        self.sequences = [g["seq"].to_numpy().astype(np.int32) for g in self.groups]
        self.durations = [np.maximum(g["delta_t"].to_numpy().astype(np.int32), 1).astype(np.float32) for g in self.groups]
        self.raw_quantities = [g["demand_qty"].to_numpy().astype(np.float64) for g in self.groups]
        self.quantities = [q.astype(np.float32) for q in self.raw_quantities]
        split_lists = [g["chronological_split"].to_list() for g in self.groups]
        self.index = [(p, i) for p, splits in enumerate(split_lists)
                      for i in range(len(splits) - 1) if splits[i + 1] == target_split]
        if not self.index:
            raise ValueError("No next-event targets for split " + target_split)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        part, context_end = self.index[index]
        seq = self.sequences[part]
        left = np.searchsorted(seq, int(seq[context_end]) - self.lookback + 1, side="left")
        left = max(left, context_end + 2 - self.max_seq_len)
        stop = context_end + 2
        length = stop - left
        dts = torch.zeros(self.max_seq_len)
        qty = torch.zeros(self.max_seq_len)
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        dts[:length] = torch.from_numpy(self.durations[part][left:stop].copy())
        qty[:length] = torch.from_numpy(self.quantities[part][left:stop].copy())
        mask[:length] = True
        return {"dts": dts, "quantities": qty, "mask": mask, "index": index}

    def population(self):
        part = np.array([p for p, _ in self.index], dtype="<i8")
        position = np.array([i + 1 for _, i in self.index], dtype="<i8")
        seq = np.array([self.sequences[p][i] for p, i in zip(part, position)], dtype="<i8")
        # Identity uses original float64 values, not the float32 model inputs.
        qty = np.array([self.raw_quantities[p][i] for p, i in zip(part, position)], dtype="<f8")
        digest = hashlib.sha256(b"hard_lmm_target_identity_v1\0" + self.target_split.encode() + b"\0")
        digest.update(json.dumps([str(p) for p in self.parts], ensure_ascii=False, separators=(",", ":")).encode())
        for values in (part, position, seq):
            digest.update(values.tobytes())
        return {"target_count": len(self), "target_identity_sha256": digest.hexdigest(),
                "target_quantity_sha256": hashlib.sha256(b"hard_lmm_target_quantity_v1\0" + qty.tobytes()).hexdigest()}


def target_outputs(model, batch, *, lambda_log_qty=1.0):
    dts, qty, mask = batch["dts"], batch["quantities"], batch["mask"]
    lengths = mask.sum(1)
    if (lengths < 2).any():
        raise ValueError("A target requires at least one observed history event")
    rows = torch.arange(mask.size(0), device=mask.device)
    target, previous = lengths - 1, lengths - 2
    observed = mask.clone()
    observed[rows, target] = False
    history_qty = qty.clone()
    history_qty[rows, target] = 0
    time_states, quantity_states = model.encode_task_states(dts, history_qty, mask, memory_write_mask=observed)
    true_dt, true_qty = dts[rows, target], qty[rows, target]
    time_hidden, quantity_hidden = time_states[rows, previous], quantity_states[rows, previous]
    quantity = model.quantity_outputs(quantity_hidden, true_qty)
    time_loss = -model.log_observation_dt(time_hidden, true_dt)
    return {"joint_loss": time_loss + lambda_log_qty * quantity["train_loss"],
            "time_loss": time_loss, "log_qty_loss": quantity["log_mse"],
            "true_qty": true_qty, "pred_qty": quantity["point_prediction"],
            "true_dt": true_dt, "pred_dt": model.predict_time_median(time_hidden),
            "history_length": lengths - 1}
