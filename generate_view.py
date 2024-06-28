import torch
import os.path
import pandas as pd

from PIL import Image
from os.path import join
from torchvision import transforms


def generate_a_patch(data, save_name, device, methods=('Gatys', 'OST', 'AdaIN', 'CMD', 'EFDM'), space=True):
    header = ['style image', 'style INST image', 'content image']
    for method in methods:
        header += ['NST ' + method, 'INST ' + method]

    f = transforms.Compose([transforms.Resize((512, 512)),
                            transforms.ToTensor()])
    final = torch.Tensor().to(device)
    for idx, row in data.iterrows():
        temp = torch.concat([f(Image.open(row[1])), f(Image.open(row[2]))], dim=1).to(device)
        patch = torch.concat([torch.ones([3, 256, 512]).to(device), f(Image.open(row[3])).to(device),
                              torch.ones([3, 256, 512]).to(device)], dim=1)
        temp = torch.concat([temp, patch], dim=2)
        for i in range(len(methods)):
            patch = torch.concat([f(Image.open(row[2 * i + 4])), f(Image.open(row[2 * i + 5]))], dim=1).to(device)
            temp = torch.concat([temp, patch], dim=2)
        final = torch.concat([final, temp], dim=1)
        if space:
            final = torch.concat([final, torch.ones([3, 10, 512 * (len(methods) + 2)]).to(device)], dim=1)
    transforms.ToPILImage()(final).save(save_name)


def generate_imgs(label_folder, save_folder, device, num_sty_per_patch=5,
                  methods=('Gatys', 'OST', 'AdaIN', 'CMD', 'EFDM'), space=True):
    df = pd.read_csv(join(label_folder, 'view.csv'))
    dfs = [df.iloc[i:i + num_sty_per_patch].reset_index(drop=True) for i in range(0, len(df), num_sty_per_patch)]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for batch_idx, data_batch in enumerate(dfs):
        generate_a_patch(data_batch, join(save_folder, '{}.jpg'.format(batch_idx)), device=device, methods=methods,
                         space=space)
