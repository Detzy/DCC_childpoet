# DCC_childpoet
Deep Continuous Clustering branch adjusted for CHILD POET

## Introduction ##

This is based on the Pytorch implementation of the DCC algorithms presented in the following paper ([paper](http://arxiv.org/abs/1803.01449)):

Sohil Atul Shah and Vladlen Koltun. Deep Continuous Clustering.

If you use this code in your research, please cite their paper.
```
@article{shah2018DCC,
	author    = {Sohil Atul Shah and Vladlen Koltun},
	title     = {Deep Continuous Clustering},
	journal   = {arXiv:1803.01449},
	year      = {2018},
}
```

## Requirement ##

* Python >= 3.6
* [Pytorch](http://pytorch.org/) >= v1.1.0
* [TensorboardX](https://github.com/lanpa/tensorboard-pytorch) >= 2.4.1
* Other requirements as needed, see environment.yml file (not yet created).

## Usage ##
For a full guide on how to use DCC, see the original repo by Atul and Koltun. 
For DCC adapted to CHILD POET, run the code in run_dcc_childpoet.sh
(Remember to change input parameters in the script to suit your use case)