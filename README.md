PR-PL: A Novel Transfer Learning Framework with Prototypical Representation based Pairwise Learning for EEG-Based Emotion Recognition
=
* A Pytorch implementation of our paper "PR-PL: A Novel Prototypical Representation Based Pairwise Learning Framework for Emotion Recognition Using EEG Signals." <br> 
* [IEEE TAFFC](https://ieeexplore.ieee.org/document/10160130)

# Installation:
* Python 3.7
* Pytorch 1.3.1
* NVIDIA CUDA 9.2
* Numpy 1.20.3
* Scikit-learn 0.23.2
* scipy 1.3.1

# Preliminaries
* Prepare dataset: [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html) and [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/index.html)

# Training 
* PR-PL model definition file: model_PR_PL.py 
* Pipeline of the PR-PL: implementation_PR_PL.py
* implementation of domain adversarial training: Adversarial.py
# Dataset prepare
* data_prepare_seed.m
# Usage
* After modify setting (path, etc), just run the main function in the implementation_PR_PL.py
# Acknowledgement
* The implementation code of domain adversarial training is bulit on the [dalib](https://dalib.readthedocs.io/en/latest/index.html) code base 
# Citation
If you find our work helps your research, please kindly consider citing our paper in your publications.
@ARTICLE{10160130,
  author={Zhou, Rushuang and Zhang, Zhiguo and Fu, Hong and Zhang, Li and Li, Linling and Huang, Gan and Li, Fali and Yang, Xin and Dong, Yining and Zhang, Yuan-Ting and Liang, Zhen},
  journal={IEEE Transactions on Affective Computing}, 
  title={PR-PL: A Novel Prototypical Representation Based Pairwise Learning Framework for Emotion Recognition Using EEG Signals}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TAFFC.2023.3288118}}
