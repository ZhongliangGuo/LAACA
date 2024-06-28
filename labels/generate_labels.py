import pandas as pd
from os.path import join


def generate_inst(num_style=6, src='dataset/style', dst='./inst_label.csv'):
    label = []
    for i in range(num_style):
        label.append(join(src, '{}.jpg'.format(i)))
    df = pd.DataFrame(label, columns=None)
    df.to_csv(dst, header=False, index=False)


def generate_nst(src_INST, save_folder, src_c='dataset/content', src_s='dataset/style', num_content=5, num_style=18):
    """
    :param src_INST: the folder which contains the 'style_INST' folder
    :param src_c: the folder which contains the content images
    :param src_s: the folder which contains the style images
    :param num_content: number of content images
    :param num_style: number of style images
    """
    header = ['style_img', 'INST_img', 'content_img']
    label = []
    for style in range(num_style):
        for content in range(num_content):
            label.append([join(src_s, '{}.jpg'.format(style)),
                          join(src_INST, 'style_INST', '{}.jpg'.format(style)),
                          join(src_c, '{}.jpg'.format(content))])
    df = pd.DataFrame(label, columns=header)
    df.to_csv(join(save_folder, 'nst_label.csv'), index=False)


def generate_view_label(src_INST, save_folder, default_nst_folder, src_c='dataset/content', src_s='dataset/style',
                        num_content=11, num_style=25, do_nst=False, methods=('Gatys', 'OST', 'AdaIN', 'CMD', 'EFDM')):
    header = ['style image', 'style INST image', 'content image']
    nst_path = save_folder if do_nst else default_nst_folder
    for method in methods:
        header += ['NST ' + method, 'INST ' + method]
    label = []
    for style in range(num_style):
        for content in range(num_content):
            temp = [join(src_s, '{}.jpg'.format(style)),
                    join(src_INST, 'style_INST', '{}.jpg'.format(style)),
                    join(src_c, '{}.jpg'.format(content))]
            for method in methods:
                temp += [join(nst_path, method, 'NST', 'sty_{}_content_{}.jpg'.format(style, content)),
                         join(save_folder, method, 'INST', 'sty_{}_content_{}.jpg'.format(style, content))]
            label.append(temp)

    df = pd.DataFrame(label, columns=header)
    df.to_csv(join(save_folder, 'view.csv'))


if __name__ == '__main__':
    generate_inst(24)
