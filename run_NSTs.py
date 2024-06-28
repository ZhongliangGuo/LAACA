import os
import torch
import torch.nn as nn

from tqdm import tqdm
from os.path import join
from torchvision import transforms
from dataset import get_NST_loader
from torchvision.utils import save_image
from NSTs.OST.model import MultiLevelAE_OST


def run_Gatys(device, max_iter, save_folder, label_folder, do_nst=False):
    from NSTs.Gatys.style_transfer import get_config, run_once, Normalization, get_vgg19_features

    cnn = get_vgg19_features().to(device)
    un_tensor = transforms.ToPILImage()
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    sty_trans_config = get_config(os.path.join(current_script_dir, r'NSTs/Gatys/style_transfer.json'))
    sty_trans_config['epochs'] = max_iter
    vgg19_config = get_config(os.path.join(current_script_dir, r'NSTs/Gatys/vgg19.json'))
    transform = transforms.Compose([
        transforms.Resize(sty_trans_config['img_size']),  # scale imported image
        transforms.ToTensor()])
    loader = get_NST_loader(label_folder=label_folder, device=device, transform=transform)
    normalization = Normalization(torch.tensor(sty_trans_config['norm_mean']),
                                  torch.tensor(sty_trans_config['norm_std']))
    print('Starting generate NST and INST samples for Gatys method...')
    pbar = tqdm(total=len(loader))
    for idx, (data) in enumerate(loader):
        if do_nst:
            nst = un_tensor(
                run_once(cnn=cnn, normalization=normalization, style_img=data['style_img'],
                         content_img=data['content_img'],
                         sty_trans_config=sty_trans_config, vgg19_config=vgg19_config)[0])
            nst.save(join(save_folder, 'NST', '{}.jpg'.format(data['file_name'][0])))
        inst = un_tensor(
            run_once(cnn=cnn, normalization=normalization, style_img=data['INST_img'], content_img=data['content_img'],
                     sty_trans_config=sty_trans_config, vgg19_config=vgg19_config)[0])
        inst.save(join(save_folder, 'INST', '{}.jpg'.format(data['file_name'][0])))
        pbar.update(1)
    pbar.close()
    print('Finished for Gatys method.')


def run_Gatys_once(content_pil, style_pil, device, save_folder, max_iter=200, do_nst=False):
    from NSTs.Gatys.style_transfer import get_config, run_once, Normalization, get_vgg19_features

    cnn = get_vgg19_features().to(device)
    un_tensor = transforms.ToPILImage()
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    sty_trans_config = get_config(os.path.join(current_script_dir, r'NSTs/Gatys/style_transfer.json'))
    sty_trans_config['epochs'] = max_iter
    vgg19_config = get_config(os.path.join(current_script_dir, r'NSTs/Gatys/vgg19.json'))
    transform = transforms.Compose([
        transforms.Resize(sty_trans_config['img_size']),  # scale imported image
        transforms.ToTensor()])
    normalization = Normalization(torch.tensor(sty_trans_config['norm_mean']),
                                  torch.tensor(sty_trans_config['norm_std']))
    content_img = transform(content_pil).unsqueeze(0).to(device)
    style_img = transform(style_pil).unsqueeze(0).to(device)
    nst = un_tensor(
        run_once(cnn=cnn, normalization=normalization, style_img=style_img,
                 content_img=content_img,
                 sty_trans_config=sty_trans_config, vgg19_config=vgg19_config)[0])
    nst.save(join(save_folder, 'Gatys.jpg'))


def run_decoder_based(device, save_folder, label_folder, method, do_nst=False):
    from NSTs.decoder_based.style_transfer import NST
    decoder_based_nst = NST(device=device,
                            vgg_normed_path='./NSTs/decoder_based/models/vgg_normalised.pth',
                            adain_decoder_path='./NSTs/decoder_based/models/adain_decoder.pth',
                            efdm_decoder_path='./NSTs/decoder_based/models/efdm_decoder.pth',
                            method=method)

    transform = transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
    ])
    loader = get_NST_loader(label_folder=label_folder, device=device, transform=transform)
    print(f'Starting generate NST and INST samples for {method} method...')
    with torch.no_grad():
        pbar = tqdm(total=len(loader))
        for idx, (data) in enumerate(loader):
            if do_nst:
                nst = decoder_based_nst.style_transfer(data['content_img'], data['style_img'])
                save_image(nst.cpu(), join(save_folder, 'NST', '{}.jpg'.format(data['file_name'][0])))
            inst = decoder_based_nst.style_transfer(data['content_img'], data['INST_img'])
            save_image(inst.cpu(), join(save_folder, 'INST', '{}.jpg'.format(data['file_name'][0])))
            pbar.update(1)
        pbar.close()
        print(f'Finished for {method} method.')


def run_decoder_based_once(content_pil, style_pil, device, save_folder, method):
    from NSTs.decoder_based.style_transfer import NST
    decoder_based_nst = NST(device=device,
                            vgg_normed_path='./NSTs/decoder_based/models/vgg_normalised.pth',
                            adain_decoder_path='./NSTs/decoder_based/models/adain_decoder.pth',
                            efdm_decoder_path='./NSTs/decoder_based/models/efdm_decoder.pth',
                            method=method)

    transform = transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
    ])
    content_img = transform(content_pil).unsqueeze(0).to(device)
    style_img = transform(style_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        nst = decoder_based_nst.style_transfer(content_img, style_img)
        save_image(nst.cpu(), join(save_folder, f'{method}.jpg'))


def run_OST(device, save_folder, label_folder, alpha=0.6, do_nst=False):
    from NSTs.OST.style_transfer import style_transfer
    model = MultiLevelAE_OST(pretrained_path_dir='NSTs/OST/models')
    model = model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    loader = get_NST_loader(label_folder=label_folder, device=device, transform=transform)
    print('Starting generate NST and INST samples for OST method...')
    with torch.no_grad():
        pbar = tqdm(total=len(loader))
        for idx, (data) in enumerate(loader):
            if do_nst:
                nst = style_transfer(model, data['content_img'], data['style_img'], alpha)
                nst.save(join(save_folder, 'NST', '{}.jpg'.format(data['file_name'][0])))
            inst = style_transfer(model, data['content_img'], data['INST_img'], alpha)
            inst.save(join(save_folder, 'INST', '{}.jpg'.format(data['file_name'][0])))
            pbar.update(1)
        pbar.close()
        print('Finished for OST method.')


def run_OST_once(content_pil, style_pil, device, save_folder, alpha=0.6):
    from NSTs.OST.style_transfer import style_transfer
    model = MultiLevelAE_OST(pretrained_path_dir='NSTs/OST/models')
    model = model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    content_img = transform(content_pil).unsqueeze(0).to(device)
    style_img = transform(style_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        nst = style_transfer(model, content_img, style_img, alpha)
        nst.save(join(save_folder, 'OST.jpg'))


def run_SANet(device, save_folder, label_folder, step=1, do_nst=False):
    from NSTs.SANet.style_transfer import get_decoder, get_vgg, Transform, style_transfer
    from torchvision.utils import save_image
    # region prepare net
    decoder = get_decoder().to(device)
    transform = Transform(in_planes=512).to(device)
    vgg = get_vgg().to(device)
    decoder.eval()
    transform.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('./NSTs/SANet/decoder_iter_500000.pth'))
    transform.load_state_dict(torch.load('./NSTs/SANet/transformer_iter_500000.pth'))
    vgg.load_state_dict(torch.load(r'./NSTs/SANet/vgg_normalised.pth'))

    norm = nn.Sequential(*list(vgg.children())[:1])
    enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1
    model = {
        'norm': norm.to(device),
        'enc_1': enc_1.to(device),
        'enc_2': enc_2.to(device),
        'enc_3': enc_3.to(device),
        'enc_4': enc_4.to(device),
        'enc_5': enc_5.to(device),
        'transform': transform.to(device),
        'decoder': decoder.to(device),
    }

    # endregion
    img_tf = transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
    ])
    loader = get_NST_loader(label_folder=label_folder, device=device, transform=img_tf)
    print('Starting generate NST and INST samples for SANet method...')
    with torch.no_grad():
        pbar = tqdm(total=len(loader))
        for idx, (data) in enumerate(loader):
            if do_nst:
                nst = style_transfer(model, data['content_img'], data['style_img'], step)
                save_image(nst, join(save_folder, 'NST', '{}.jpg'.format(data['file_name'][0])))
            inst = style_transfer(model, data['content_img'], data['INST_img'], step)
            save_image(inst, join(save_folder, 'INST', '{}.jpg'.format(data['file_name'][0])))
            pbar.update(1)
        pbar.close()
        print('Finished for SANET method.')


def run_SANet_once(content_pil, style_pil, device, save_folder, step=1):
    from NSTs.SANet.style_transfer import get_decoder, get_vgg, Transform, style_transfer
    from torchvision.utils import save_image
    # region prepare net
    decoder = get_decoder().to(device)
    transform = Transform(in_planes=512).to(device)
    vgg = get_vgg().to(device)
    decoder.eval()
    transform.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('./NSTs/AdaAttN/decoder_iter_500000.pth'))
    transform.load_state_dict(torch.load('./NSTs/AdaAttN/transformer_iter_500000.pth'))
    vgg.load_state_dict(torch.load(r'./NSTs/AdaAttN/vgg_normalised.pth'))

    norm = nn.Sequential(*list(vgg.children())[:1])
    enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1
    model = {
        'norm': norm.to(device),
        'enc_1': enc_1.to(device),
        'enc_2': enc_2.to(device),
        'enc_3': enc_3.to(device),
        'enc_4': enc_4.to(device),
        'enc_5': enc_5.to(device),
        'transform': transform.to(device),
        'decoder': decoder.to(device),
    }

    # endregion
    img_tf = transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
    ])
    content_img = img_tf(content_pil).unsqueeze(0).to(device)
    style_img = img_tf(style_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        nst = style_transfer(model, content_img, style_img, step)
        save_image(nst, join(save_folder, 'SANet.jpg'))
