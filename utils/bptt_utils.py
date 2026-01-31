import torch

from models.recurrent_marked_temporal_point_process.rmtpp import RMTPP


def train_one_epoch(model: RMTPP, loader, optimizer, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    model.train()
    total = 0.0

    for marks, dts, mask in loader:
        marks = marks.to(device)
        dts = dts.to(device)
        mask = mask.to(device)

        out = model.nll(marks, dts, mask)
        loss = out['nll']

        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5.0)
        optimizer.step()

        total += float(loss.item())

    return total / max(len(loader), 1)

@torch.no_grad()
def eval_epoch(model: RMTPP, loader, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    total = 0.0

    for marks, dts, mask in loader:
        marks = marks.to(device)
        dts = dts.to(device)
        mask = mask.to(device)

        out = model.nll(marks, dts, mask)
        total += float(out['nll'].item())

    return total / max(len(loader), 1)

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

