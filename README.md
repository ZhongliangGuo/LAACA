# Locally Adaptive Adversarial Color Attack

This project is for the LAACA we proposed in the paper "[Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack](https://arxiv.org/abs/2401.09673)", which is accepted by **50th European Conference on Artificial Intelligence (ECAI 2024)**.

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

## Environment

```
numpy==2.0.0
pandas==1.5.2
Pillow==9.2.0
torch==2.0.0
torchvision==0.15.1
tqdm==4.65.0
```

## Cite

```latex
@incollection{guo2024artwork,
  title={Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack},
  author={Guo, Zhongliang and Dong, Junhao and Qian, Yifei and Wang, Kaixuan and Li, Weiye and Guo, Ziheng and Wang, Yuheng and Li, Yanli and Arandjelovi{\'c}, Ognjen and Fang, Lei},
  booktitle={ECAI 2024},
  year={2024},
  publisher={IOS Press}
}
```

