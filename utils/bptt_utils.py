import torch

from models.recurrent_marked_temporal_point_process.rmtpp import RMTPP


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_nll = 0.0
    total_m = 0.0
    total_t = 0.0
    n_batches = 0

    for marks, dts, mask in loader:
        marks = marks.to(device)
        dts = dts.to(device)
        mask = mask.to(device)

        out = model.nll(marks, dts, mask)
        loss = out["nll"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_nll += float(out["nll"].item())
        total_m += float(out["nll_marker"].item())
        total_t += float(out["nll_time"].item())
        n_batches += 1

    return {
        "nll": total_nll / max(n_batches, 1),
        "nll_marker": total_m / max(n_batches, 1),
        "nll_time": total_t / max(n_batches, 1),
        "batches": n_batches,
    }

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_nll = 0.0
    total_m = 0.0
    total_t = 0.0
    n_batches = 0

    for marks, dts, mask in loader:
        marks = marks.to(device)
        dts = dts.to(device)
        mask = mask.to(device)

        out = model.nll(marks, dts, mask)
        total_nll += float(out["nll"].item())
        total_m += float(out["nll_marker"].item())
        total_t += float(out["nll_time"].item())
        n_batches += 1

    return {
        "nll": total_nll / max(n_batches, 1),
        "nll_marker": total_m / max(n_batches, 1),
        "nll_time": total_t / max(n_batches, 1),
        "batches": n_batches,
    }

@torch.no_grad()
def predict_next(model: RMTPP, marks: torch.Tensor, dts: torch.Tensor):
    """
    marks: [1, L], dts: [1, L]
    """
    h = model.forward_hidden(marks, dts)    # [1, L, H]
    h_last = h[:, -1, :]                    # [1, H]

    # next mark distribution
    logits = model.mark_head(h_last)        # [1, K]
    prob = torch.softmax(logits, dim = -1)
    y_hat = torch.argmax(prob, dim = -1)    # [1]

    # next dt sampling
    dt_hat = model.sample_next_dt(h_last)   # [1]
    return y_hat, dt_hat, prob

