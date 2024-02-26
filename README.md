<div align="center">

# MiSSNet: Memory-Inspired Semantic Segmentation Augmentation Network for Class-Incremental Learning in Remote Sensing Images

[![Paper](https://img.shields.io/badge/IEEE-10418153-brightgreen)](https://ieeexplore.ieee.org/document/10418153)



</div>

# Requirements

Please follow requirement.txt

# Using
An example is like miss_3s.sh, which is used in a squeue.

# Citation
```
@ARTICLE{10418153,
  author={Xie, Jiajun and Pan, Bin and Xu, Xia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MiSSNet: Memory-Inspired Semantic Segmentation Augmentation Network for Class-Incremental Learning in Remote Sensing Images}, 
  year={2024},
  volume={62},
  number={},
  pages={1-13},
  keywords={Incremental learning; remote sensing images; semantic segmentation},
  doi={10.1109/TGRS.2024.3360701}}
```

# Concat
谢家骏 Xie Jiajun (<xiejiajuncqc@mail.nankai.edu.cn>)


# Acknowledgement

## Repository
This repository is a modified version of
[PLOP](https://github.com/arthurdouillard/CVPR2021_PLOP). Please cite it.


```
@inproceedings{douillard2021plop,
  title={PLOP: Learning without Forgetting for Continual Semantic Segmentation},
  authors={Douillard, Arthur and Chen, Yifu and Dapogny, Arnaud and Cord, Matthieu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

## Datasets
The benchmarks used in our paper refer to the following articles.

* [Performance Evaluation of Single-Label and Multi-Label Remote Sensing Image Retrieval Using a Dense Labeling Dataset](https://www.mdpi.com/2072-4292/10/6/964) [[dataset](https://sites.google.com/view/zhouwx/dataset#h.p_ebsAS1Bikmkd)]
* [A stepwise domain adaptive segmentation network with covariate shift alleviation for remote sensing imagery](https://ieeexplore.ieee.org/document/9716091) [[dataset](https://github.com/jojolee6513/Marsegdataset)]
* [Ensemble Knowledge Transfer for Semantic Segmentation](https://ieeexplore.ieee.org/document/8354272) [[dataset](https://github.com/ishann/aeroscapes)]

## Experiments
Our experimental part is also supported by the following papers, please cite them if possible.

* [Incremental Learning Techniques for Semantic Segmentation](https://ieeexplore.ieee.org/document/9022296) [[code](https://lttm.dei.unipd.it/paper_data/IL)]
* [Continual Learning With Structured Inheritance for Semantic Segmentation in Aerial Imagery](https://ieeexplore.ieee.org/document/9426950)
* [Historical Information-Guided Class-Incremental Semantic Segmentation in Remote Sensing Images](https://ieeexplore.ieee.org/document/9762919) [[code](https://github.com/RongXueE/HCISS)]
* [Class-Incremental Learning via Dual Augmentation](https://proceedings.neurips.cc/paper/2021/file/77ee3bc58ce560b86c2b59363281e914-Paper.pdf) [[code](https://github.com/Impression2805/IL2A)]
* [Cross-image relational knowledge distillation for semantic segmentation](https://arxiv.org/abs/2204.06986) [[code](https://github.com/winycg/CIRKD)]






