import json
import torch
from torch import nn, optim
from torchvision import models
from NSTs.Gatys.losses import StyleLoss, ContentLoss


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_vgg19_features():
    """
    :param pth: the path for pretrained weights, if you didn't download it, just give "pth = None".
    :return: the vgg19 features
    """
    net = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

    print('Loaded pretrained weights')
    return net.features.eval()


def get_style_model_and_losses(cnn: nn.Module,
                               normalization: Normalization,
                               style_img: torch.Tensor,
                               content_img: torch.Tensor,
                               vgg19_config: dict):
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)
        if name in vgg19_config['content_layers']:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        if name in vgg19_config['style_layers']:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses


def get_config(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def run_once(cnn,
             normalization: Normalization,
             style_img: torch.Tensor,
             content_img: torch.Tensor,
             sty_trans_config: dict,
             vgg19_config: dict,
             ):
    input_img = content_img.clone()
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization, style_img, content_img,
                                                                     vgg19_config)
    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    optimizer = optim.LBFGS([input_img])
    run = [0]
    while run[0] <= sty_trans_config['epochs']:
        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= sty_trans_config['style_weight']
            content_score *= sty_trans_config['content_weight']

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            return style_score + content_score

        optimizer.step(closure)
    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img.clone().detach()
