# StatisticalMechanics_MachineLearning__MScThesis

## Aim
This repo contains the code intended to support the developing process of my Master Degree Thesis.

## Scope

Motivated by recent publications concerning the development and formalisation of a theoretical framework in which Deep Learning may fit into, this thesis topic and work is finalised to investigate the evolution of the topology of a neural system structure during the process of learning.

In the full swing of the example provided in [2], some synthetic data set are being cratfed, according to some particular morphology (e.g. binary tree structure and independent clusters, created in a _simuated annealing_ fashion), then these are fed to a neural network. A simple feed forward network is accounted for as model zero.

The main concern is the nucleation of some recurrent structures in the trained network (in the terms of [4], _network motifs_). The interest is to investigate the process of evolution of such motifs and their role (if any), and try to put this phenomenon in relationship with the learning process, that is: How the probabilistic signature of the data fed to the system is encoded in the internal structure of the network itself. A step further that it is in order to be taken is the following: While in the rich pool of previous work on network motifs mining all of the systems are assumed to be _unweighted_ (e.g. [4] and references therein), here one cannot think of neglecting the strengths of connections between neurons. Other than that, what sets this work apart from the formerly mentioned literature, is that the system has undergone an evolutionary process driven first and foremost by the optimization algorithm chosen (here SGD is uniquely adopted). A genetics-inspired evolutionary force here may not be a good choice: Parameters must be updated according to optimum rules.

## Contents

Although Jupyter notebooks render visualisation ease and readability, it is better for the coding process to use Python sources. A Docs folder in which an up-tp-date version of the progress report lays, alongside with the motif dictionary document, taken [here](https://www.weizmann.ac.il/mcb/UriAlon/download/network-motif-software).


## References

1. [Andrew M. Saxe, James L. McClelland, Surya Ganguli, _Exact solutions to the nonlinear dynamics of learning in deep linear neural networks_, 2014](https://arxiv.org/abs/1312.6120 "arXiv")

2. [Andrew M. Saxe, James L. McClelland, Surya Ganguli, _A mathematical theory of semantic development in deep neural networks_, 2018](https://arxiv.org/abs/1810.10531 "arXiv")

3. [Sebastian Musslick, Andrew M. Saxe, Kayhan Ozcimder, Biswadip Dey, Greg Henselman, Jonathan D. Cohen , _Multitasking Capability Versus Learning Efficiency in Neural Network Architectures_, 2017](https://www.researchgate.net/publication/317019423_Multitasking_Capability_Versus_Learning_Efficiency_in_Neural_Network_Architectures "Research Gate")

4. [Nadav Kashtan and Uri Alon, _Spontaneous evolution of modularity and network motifs_, 2005](https://www.pnas.org/content/102/39/13773 "PNAS")
