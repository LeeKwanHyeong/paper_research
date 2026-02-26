import torch

class RMTPPWrapper:
    def __init__(self, core_model):
        self.model = core_model

    @torch.no_grad()
    def predict_next(self, marks: torch.Tensor, dts: torch.Tensor):
        h = self.model.forward(marks, dts)  # [1, L, H]
        h_last = h[:, -1, :]  # [1, H]

        logits = self.model.mark_head(h_last)  # [1, K_model]
        prob = torch.softmax(logits, dim=-1)  # [1, K_model]
        y_hat = torch.argmax(prob, dim=-1)  # [1]

        dt_hat = self.model.sample_next_dt(h_last)  # [1]
        return y_hat, dt_hat, prob