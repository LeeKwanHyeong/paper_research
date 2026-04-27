import torch

class RMTPPWrapper:
    def __init__(self, core_model):
        self.model = core_model

    @torch.no_grad()
    def predict_next(self, marks: torch.Tensor, dts: torch.Tensor):
        h = self.model.forward(marks, dts)  # [1, L, H]
        h_last = h[:, -1, :]  # [1, H]

        logits = self.model.mark_head(h_last)[..., : self.model.cfg.num_marks - 1]  # exclude PAD
        prob = torch.softmax(logits, dim=-1)  # [1, K_model]
        y_hat = torch.argmax(prob, dim=-1)  # [1]

        dt_hat = self.model.sample_next_dt(h_last)  # [1]
        value_hat = None
        qty_hat = None
        if hasattr(self.model, "predict_value"):
            value_hat = self.model.predict_value(h_last)
            qty_hat = self.model.reconstruct_qty(y_hat, value_hat)

        return y_hat, dt_hat, prob, value_hat, qty_hat
