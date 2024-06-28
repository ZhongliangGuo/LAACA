import os
import pandas as pd

from PIL import Image
from os.path import join
from torch.utils.data import Dataset, DataLoader


class INSTData(Dataset):
    def __init__(self, label_folder, device, transform):
        super().__init__()
        self.transform = transform
        self.device = device
        self.label = pd.read_csv(join(label_folder, 'inst_label.csv'), header=None)

    def __getitem__(self, index):
        img = self.transform(
            Image.open(self.label.iloc[index][0])
        ).to(self.device)
        return img

    def __len__(self):
        return len(self.label)


class NSTData(Dataset):
    def __init__(self, label_folder, device, transform):
        super().__init__()
        self.transform = transform
        self.device = device
        self.label = pd.read_csv(join(label_folder, 'nst_label.csv'))

    def __getitem__(self, index):
        data = {'file_name': 'sty_{}_content_{}'.format(self.label['style_img'][index].split('/')[-1].split('.')[0],
                                                        self.label['content_img'][index].split('/')[-1].split('.')[0])}
        for key in self.label.columns:
            data[key] = self.transform(
                Image.open(self.label[key][index])
            ).to(self.device)
        return data

    def __len__(self):
        return len(self.label)


def get_INST_loader(label_folder, device, transform, batch_size=1):
    dataset = INSTData(label_folder=label_folder, device=device, transform=transform)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)


def get_NST_loader(label_folder, device, transform):
    dataset = NSTData(label_folder=label_folder, device=device, transform=transform)
    return DataLoader(dataset=dataset, batch_size=1, shuffle=False)


def prepare_INST_label(folder):
    data_folder = join(folder, 'style')
    images_path = sorted(os.listdir(data_folder))
    for i, v in enumerate(images_path):
        images_path[i] = join('style', v)
    # images_path = ['{}.jpg'.format(i) for i in range(20)]
    label = pd.DataFrame(images_path)
    label.to_csv(join(folder, 'inst_label.csv'), header=False, index=False)


def prepare_nst_dataset(data_folder):
    style_imgs = ['{}.jpg'.format(i) for i in range(20)]
    INST_imgs = ['{}_INST.jpg'.format(i) for i in range(20)]
    content_imgs = ['{}.jpg'.format(i) for i in range(11)]
    header = ['style_img', 'INST_img', 'content_img']
    label = []
    for idx, style_img in enumerate(style_imgs):
        for content_img in content_imgs:
            label.append([join('style', style_img), join('INST', INST_imgs[idx]), join('content', content_img)])
    label = pd.DataFrame(label)
    label.to_csv(join(data_folder, 'nst_label.csv'), header=header, index=False)


if __name__ == '__main__':
    prepare_INST_label('dataset/')
    prepare_nst_dataset(r'dataset')
