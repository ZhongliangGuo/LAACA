import argparse
import os.path
import sys
from os.path import join
import torch

from laaca import LAACA
from generate_view import generate_imgs
from run_NSTs import run_Gatys, run_decoder_based, run_OST, run_SANet
from labels.generate_labels import generate_nst, generate_view_label, generate_inst


# region methods for dealing with folders
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_method_dir(path, sub, do_nst):
    create_dir(join(path, sub))
    if do_nst:
        create_dir(join(path, sub, 'NST'))
    create_dir(join(path, sub, 'INST'))


# endregion


if __name__ == '__main__':
    # region create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_content', type=int, default=10, help='number of content images')
    parser.add_argument('--num_style', type=int, default=11, help='number of content images')
    parser.add_argument('--vgg_normed_path', type=str, default=r'./NSTs/decoder_based/models/vgg_normalised.pth')
    parser.add_argument('--adain_decoder_path', type=str, default=r'./NSTs/decoder_based/models/adain_decoder.pth')
    parser.add_argument('--efdm_decoder_path', type=str, default=r'./NSTs/decoder_based/models/efdm_decoder.pth')
    parser.add_argument('--label_folder', type=str, default='labels',
                        help='Path to the folder where contains the inst_label.csv')
    parser.add_argument('--k', type=int, default=4, help='gaussian kernel size')
    parser.add_argument('--alpha', type=float, default=8, help='update step-length, alpha/255')
    parser.add_argument('--epsilon', type=float, default=80, help='update step-length, alpha/255')
    parser.add_argument('--step', type=int, default=100, help='number of iteration')
    parser.add_argument('--layers', type=str, default='(3, 10, 17, 30)')
    parser.add_argument('--im_size', type=int, default=512, nargs='+',
                        help='Image size. Either single int or tuple of int')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the generated images')
    parser.add_argument('--seed', type=int, default=3407, help='random seed for torch')
    parser.add_argument('--do_nst', type=str, default='True', choices=['True', 'False'],
                        help='if will do nst for original images')
    parser.add_argument('--default_nst_folder', type=str, default='./dataset/nst_default',
                        help='Path to the folder where contains pre transferred images')
    args = parser.parse_args()
    args.do_nst = eval(args.do_nst)
    if args.save_path is None:
        args.save_path = 'gen_imgs/alp{}_k{}_eps{}'.format(int(args.alpha), int(args.k), int(args.epsilon))
    args.alpha = args.alpha / 255
    args.epsilon = args.epsilon / 255
    assert args.alpha < 1
    assert args.epsilon < 1
    assert type(args.do_nst) is bool
    img_size = args.im_size  # Integer or list (512, 512)
    if isinstance(img_size, list):
        if len(args.im_size) != 2:
            print("Image size can either be a single int or a list of two ints.")
            sys.exit(0)
    else:
        img_size = (img_size, img_size)
    args.layers = eval(args.layers)
    assert type(args.layers) is tuple
    assert min(args.layers) >= 0
    assert max(args.layers) <= 30
    # endregion
    # the NST methods will be tested
    nst_methods = [
        'Gatys',  # 2015
        'AdaIN',  # 2017 ICCV
        'OST',  # 2019 ICCV
        'SANet',  # 2019 CVPR
        'EFDM'  # 2022 CVPR
    ]
    # region create-folder
    create_dir(args.save_path)
    create_dir(join(args.save_path, 'style_INST'))
    for nst_method in nst_methods:
        create_method_dir(args.save_path, nst_method, args.do_nst)
    # endregion
    # region selecting the device and fixing random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU_name = torch.cuda.get_device_name(device)
    torch.set_default_device(device)
    torch.manual_seed(args.seed)
    # endregion
    # region save configs
    with open(join(args.save_path, 'configs.txt'), mode='w') as f:
        f.writelines(['{} = {}\n'.format(i[0], i[1]) for i in args._get_kwargs()])
    # endregion
    # region main body for LAACA
    generate_inst(args.num_style, dst=r'./labels/inst_label.csv')
    attacker = LAACA(label_folder=args.label_folder,
                     vgg_normed_path=args.vgg_normed_path,
                     device=device,
                     k=args.k,
                     image_size=img_size)
    attacker.run(save_path=join(args.save_path, 'style_INST'),
                 alpha=args.alpha,
                 eps=args.epsilon,
                 step=args.step)
    del attacker
    # endregion
    # region do style transfer
    generate_nst(args.save_path, args.save_path, num_content=args.num_content, num_style=args.num_style)
    if 'Gatys' in nst_methods:
        run_Gatys(device=device, max_iter=200, save_folder=join(args.save_path, 'Gatys'), label_folder=args.save_path,
                  do_nst=args.do_nst)
    if 'AdaIN' in nst_methods:
        run_decoder_based(device, label_folder=args.save_path, save_folder=join(args.save_path, 'AdaIN'),
                          method='adain', do_nst=args.do_nst)
    if 'OST' in nst_methods:
        run_OST(device, label_folder=args.save_path, save_folder=join(args.save_path, 'OST'), do_nst=args.do_nst)
    if 'SANet' in nst_methods:
        run_SANet(device, label_folder=args.save_path, save_folder=join(args.save_path, 'SANet'), do_nst=args.do_nst)
    if 'HM' in nst_methods:
        run_decoder_based(device, label_folder=args.save_path, save_folder=join(args.save_path, 'HM'),
                          method='hm', do_nst=args.do_nst)
    if 'EFDM' in nst_methods:
        run_decoder_based(device, label_folder=args.save_path, save_folder=join(args.save_path, 'EFDM'),
                          method='efdm', do_nst=args.do_nst)
    # endregion
    # region generating long image to show the comparison
    generate_view_label(args.save_path, args.save_path, args.default_nst_folder, num_content=args.num_content,
                        num_style=args.num_style, do_nst=args.do_nst, methods=nst_methods)
    generate_imgs(args.save_path, join(args.save_path, 'views'), device=device, methods=nst_methods,
                  num_sty_per_patch=args.num_content)
    # endregion
