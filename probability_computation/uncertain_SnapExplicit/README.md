Uncertain
=======

Uncertainty-driven recommendation systems in Python.

:warning: This project is currently under development and should be used very carefully.

This project provides pytorch-lightning [:link:](https://github.com/PyTorchLightning/pytorch-lightning) based implementations of several recommendation algorithms:

* Explicit feedback:

    * [Probabilistic Matrix Factorization (PMF)](https://proceedings.neurips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf)
    
    * [Confidence-aware Probabilistic Matrix Factorization (CPMF)](https://ojs.aaai.org/index.php/AAAI/article/view/11251)
    
    * [OrdRec](https://dl.acm.org/doi/abs/10.1145/2043932.2043956)
    
    * [Bernoulli Matrix Factorization (BeMF)](https://arxiv.org/pdf/2006.03481v1.pdf)

* Implicit feedback:

    * [Logistic Matrix Factorization (logMF)](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)
    
    * [Bayesian personalized ranking Matrix Factorization (bprMF)](https://arxiv.org/pdf/1205.2618.pdf)


This branch is a historical snapshot of the full project to assist on reproducilibity of the paper "Estimating and Evaluating the Uncertainty of Rating Predictions and Top-n Recommendations in Recommender Systems".

The notebooks to be followed for reproducibility are under tests/Movielens.ipynb and tests/Netflix.ipynb. In order to run the notebooks, the uncertain folder has to be downloaded and loaded as a python package. Due to data redistribution restrictions we are unable to provide the full datasets. Nevertheless, the data preparation function under uncertain/utils/data.py has been seeded accordingly to guarantee reproducibility. For further assistance on reproducing, please contact me by rasing an issue.
