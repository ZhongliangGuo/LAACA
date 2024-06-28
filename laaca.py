import time
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from net import get_vgg
from os.path import join
from loss import StyleLoss
from functools import partial
from torchvision import transforms
from dataset import get_INST_loader


def get_gaussian_kernel(device, k=4):
    sigma = k
    kernel_size = 4 * k + 1
    kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - (kernel_size // 2)
            y = j - (kernel_size // 2)
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(3, 1, 1, 1).to(device)
    return partial(F.conv2d, weight=kernel, padding='same', groups=3)


def get_mask(tensor, kernel, freq='high'):
    """
    :param tensor: original images
    :param kernel: gaussian kernel to get freq images
    :param freq: which freq position
    :return: mask, {high} freq position will be True
    """
    if freq == 'low':
        target = 1
    elif freq == 'high':
        target = 0
    else:
        raise ValueError('parameter freq can only be high or low')
    low_f = torch.clamp(kernel(tensor), 0, 1)
    high_f = torch.clamp(tensor - low_f, 0, 1)
    mask = torch.max(high_f, dim=1)[0].sign().unsqueeze(1)
    mask = (torch.cat([mask, mask, mask], dim=1) == target)
    return mask


def calc_lp(images: torch.Tensor, attacked_images: torch.Tensor):
    """
    :param images: original images
    :param attacked_images: attacked images
    :return: l_inf and l_2 in [batch_size] shape
    """
    assert images.size(0) == attacked_images.size(0)
    batch_size = images.size(0)
    with torch.no_grad():
        perturbation = attacked_images - images
        l_inf = perturbation.abs().view(batch_size, -1).max(dim=1)[0]
        l_2 = torch.norm(perturbation, p=2, dim=[1, 2, 3])
    return l_inf.item(), l_2.item()


class LAACA:
    def __init__(self, label_folder, vgg_normed_path, device, k, image_size=(512, 512),
                 layers=(3, 10, 17, 30)):
        self.loss_fn = StyleLoss()
        self.encoder = get_vgg(vgg_normed_path, layers=layers).to(device).eval()
        self.kernel = get_gaussian_kernel(device, k)
        self.device = device
        img_trans_list = []
        if image_size is not None:
            img_trans_list.append(transforms.Resize(image_size))
        img_trans_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(img_trans_list)
        self.dataloader = get_INST_loader(label_folder, device=self.device, transform=self.transform)
        self.to_img = transforms.ToPILImage()

    def run(self, save_path, alpha=8 / 255, eps=80 / 255, step=40):
        time_list = []
        l_2_list = []
        l_inf_list = []
        pbar = tqdm(total=len(self.dataloader))
        for batch_idx, (data) in enumerate(self.dataloader):
            results = laaca(style_img=data, encoder=self.encoder, loss_fn=self.loss_fn, device=self.device,
                            gaussian_kernel=self.kernel, alpha=alpha, eps=eps, step=step)
            img, time_consume, l_2, l_inf = self.to_img(results['inst_img'][0]), results['time'], results['l_2'], \
                results['l_inf']

            img.save(join(save_path, '{}.jpg'.format(batch_idx)))
            l_2_list.append('{:.4f}\n'.format(l_2))
            l_inf_list.append('{:.4f}\n'.format(l_inf))
            time_list.append('{:.2f}\n'.format(time_consume))

            pbar.update(1)
        pbar.close()
        with open(join(save_path, 'time_list.txt'), mode='w') as f:
            f.writelines(time_list)
            f.writelines(['device: {}'.format(torch.cuda.get_device_name(self.device))])
        with open(join(save_path, 'l_2_list.txt'), mode='w') as f:
            f.writelines(l_2_list)
            f.writelines(['device: {}'.format(torch.cuda.get_device_name(self.device))])
        with open(join(save_path, 'l_inf_list.txt'), mode='w') as f:
            f.writelines(l_inf_list)
            f.writelines(['device: {}'.format(torch.cuda.get_device_name(self.device))])


def laaca(style_img: torch.FloatTensor,
          encoder: nn.Module,
          loss_fn: nn.Module,
          device,
          gaussian_kernel,
          eps=80 / 255,
          alpha: float = 8 / 255,
          step: int = 100):
    cost_his = []
    mask = get_mask(style_img, gaussian_kernel)
    # initialize a perturbation to avoid loss is 0 at the beginning
    perturb = torch.rand_like(style_img, device=device) * 2 / 255
    perturb[~mask] = 0
    inst_img = torch.clamp(style_img + perturb, min=0, max=1)
    t = time.time()
    for i in range(step):
        inst_img.requires_grad = True
        style_features = encoder(style_img)
        inst_features = encoder(inst_img)
        encoder.zero_grad()
        cost = loss_fn(inst_features, style_features)
        cost.backward()
        cost_his.append(cost.item())
        adv_images = inst_img + alpha * inst_img.grad.sign()
        eta = adv_images - style_img
        eta[~mask] = 0
        eta = torch.clamp(eta, min=-eps, max=eps)
        inst_img = torch.clamp(style_img + eta, min=0, max=1).detach_()
    t = time.time() - t
    l_inf, l_2 = calc_lp(style_img, inst_img)
    return {
        'inst_img': inst_img,
        'l_inf': l_inf,
        'l_2': l_2,
        'loss_his': cost_his,
        'time': t,
    }
