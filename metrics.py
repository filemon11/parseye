from supar.utils.metric import Metric

import torch

from typing import Optional, Self

def ae(x : torch.Tensor, y : torch.Tensor) -> float:
    return torch.sum(torch.abs(x.squeeze(-1) - y)).detach().item()

def se(x : torch.Tensor, y : torch.Tensor) -> float:
    return torch.sum(torch.pow(x.squeeze(-1) - y, 2)).detach().item()

class EyeTrackingMetric(Metric):

    def __init__(
        self,
        loss: Optional[float] = None,
        preds: Optional[torch.Tensor] = None,
        golds: Optional[torch.Tensor] = None,
        reverse: bool = True,
        eps: float = 1e-12
    ) -> Self:
        super().__init__(reverse=reverse, eps=eps)

        self.n_ae = 0.0
        self.n_se = 0.0

        if loss is not None:
            self(loss, preds, golds)

    def __call__(
        self,
        loss: float,
        preds: torch.Tensor,
        golds: torch.Tensor,
    ) -> Self:

        self.n += torch.numel(preds)
        self.count += 1
        self.total_loss += float(loss)

        self.n_ae += ae(preds, golds)
        self.n_se += se(preds, golds)
        
        return self

    def __add__(self, other: Self) -> Self:
        metric = self.__class__(eps = self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.total_loss = self.total_loss + other.total_loss

        metric.n_ae = self.n_ae + other.n_ae
        metric.n_se = self.n_se + other.n_se
        
        metric.reverse = self.reverse or other.reverse
        return metric

    @property
    def score(self):
        return self.mse

    @property
    def mse(self):
        return self.n_se / (self.n + self.eps)

    @property
    def mae(self):
        print("absolute error:", self.n_ae, "instances:", self.n)
        return self.n_ae / (self.n + self.eps)

    @property
    def values(self) -> dict:
        return {'MSE': self.mse,
                'MAE': self.mae}
    
    def __repr__(self):
        return f"loss: {self.loss:.4f} - " + ' '.join([f"{key}: {val:6.2}" for key, val in self.values.items()])