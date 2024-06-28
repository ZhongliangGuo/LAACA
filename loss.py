import torch
from torch import nn


def calc_mean_std(feat: torch.Tensor, eps=1e-5):
    # eps is a small value adde y-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward_once(self, pred, target):
        pred_mean, pred_std = calc_mean_std(pred)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(pred_mean, target_mean) + self.mse_loss(pred_std, target_std)

    def forward(self, pred, target):
        assert len(pred) == len(target)
        loss = self.forward_once(pred[0], target[0])
        for i in range(1, len(pred)):
            loss += self.forward_once(pred[i], target[i])
        return loss
