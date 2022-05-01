A Novel Transfer Learning Framework with Prototypical Representation based Pairwise Learning for Cross-Subject Cross-Session EEG-Based Emotion Recognition
=
* A Pytorch implementation of our paper "A Novel Transfer Learning Framework with Prototypical Representation based Pairwise Learning for Cross-Subject Cross-Session EEG-Based Emotion Recognition".<br> 
* [arxiv](https://arxiv.org/abs/2202.06509)

# Installation:
* Python 3.7
* Pytorch 1.3.1
* Numpy 1.20.3
* Scikit-learn 0.23.2
* scipy 1.3.1

# Preliminaries
* Prepare dataset: [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html) and [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/index.html)

# Training 
* PR-PL model definition file: model_PR_PL.py 
* Pipeline of the PR-PL: implementation_PR_PL.py
* implementation of domain adversarial traing: Adversarial.py
# Usage
* After modify setting (path, etc), just run the main function in the implementation_PR_PL.py
# Acknowledgement
* The implementation code of domain adversarial training is bulit on the [dalib](https://dalib.readthedocs.io/en/latest/index.html) code base 
# Citation
If you find our work helps your research, please kindly consider citing our paper in your publications.
@misc{https://doi.org/10.48550/arxiv.2202.06509,
  doi = {10.48550/ARXIV.2202.06509},
  
  url = {https://arxiv.org/abs/2202.06509},
  
  author = {Zhou, Rushuang and Zhang, Zhiguo and Yang, Xin and Fu, Hong and Zhang, Li and Li, Linling and Huang, Gan and Dong, Yining and Li, Fali and Liang, Zhen},
  
  keywords = {Human-Computer Interaction (cs.HC), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {A Novel Transfer Learning Framework with Prototypical Representation based Pairwise Learning for Cross-Subject Cross-Session EEG-Based Emotion Recognition},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
