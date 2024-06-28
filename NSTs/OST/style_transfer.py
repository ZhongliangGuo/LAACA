import torch
from torchvision.transforms import ToPILImage


def style_transfer(model, content, style, alpha=1.0):
    with torch.no_grad():
        return ToPILImage()(model(content, style, alpha).clamp_(0, 1)[0])
