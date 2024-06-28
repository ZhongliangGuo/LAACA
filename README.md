# Locally Adaptive Adversarial Color Attack

This project is for the LAACA we proposed in the paper "[Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack](https://arxiv.org/abs/2401.09673)".

The `dataset` folder contains the content images and style images.

The main implementation is in `laaca.py`.

To run the method with some content images and style images, please use the following bash script:

```bash
python main.py
```

the details for parameters can be found via:

```bash
python main.py --help
```

Our proposed Image Quality Assessment (IQA), **Aesthetic Color Distance Metric** (ACDM), can be found in this [repo](https://github.com/ZhongliangGuo/ACDM).

## Cite

```latex
@article{guo2024artwork,
  title={Artwork protection against neural style transfer using locally adaptive adversarial color attack},
  author={Guo, Zhongliang and Wang, Kaixuan and Li, Weiye and Qian, Yifei and Arandjelovi{\'c}, Ognjen and Fang, Lei},
  journal={arXiv preprint arXiv:2401.09673},
  year={2024}
}
```

