import torch

class RMTPPWrapper:
    def __init__(self, core_model):
        self.model = core_model

    @torch.no_grad()
    def predict_next(self, dts: torch.Tensor, marks: torch.Tensor):
        """
        dts:   [1, L] float
        marks: [1, L] long
        returns:
          dt_next:     [1] float
          mark_logits: [1, K] float
        """
        h = self.model.forward_hidden(marks, dts)   # [1, L, H]
        h_last = h[:, -1, :]                        # [1, H]

        mark_logits = self.model.mark_head(h_last)  # [1, K]
        dt_next = self.model.sample_next_dt(h_last) # [1]  (샘플링)

        return dt_next, mark_logits