import torch.nn


class MAPELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs((y_true - y_pred) / torch.clamp(y_true, 1e-6)))
