# Statistical Mechanics of Deep Learning. A Network Science Approach

## Aim
This repo contains the code intended to support the developing process of my Master Degree Thesis.

## Scope

Motivated by recent publications concerning the development and formalisation of a theoretical framework in which Deep Learning may fit into, this thesis topic and work is finalised to investigate the evolution of the topology of a neural system structure during the process of learning.

Network Science is thought to be an interesting standpoint from which to look at the phenomenon from. Inspired by [Ron Milo et al., 2002](https://science.sciencemag.org/content/298/5594/824), a simple neural network is inspected as breeding ground for the emergence of network motifs, once the model is trained on different sets of synthetic data and with different initialisation schemes ([Saxe et al., 2013](https://arxiv.org/abs/1312.6120), [Glorot and Bengio, 2010](http://proceedings.mlr.press/v9/glorot10a.html)).

An endearing biological resemblance with **kinases** dynamics in transduction networks may be appreciated (see [Alon, 2006](https://www.crcpress.com/An-Introduction-to-Systems-Biology-Design-Principles-of-Biological-Circuits/Alon/p/book/9781439837177)).
 
## Contents

`all_sources_new` source codes. The `main.py` scripts launches what necessary 

* `DataSets` folder, which contains the scripts to generate the two data sets used. Number of data examples and other details are hard-coded. In addition, serialized files of previously created data sets are present, in order to fetch them in the simulations.
* `all_pkg` contains the modules I have written to accomplish the task. Other than Machine Learning, as Keras, and standard scientific computing libraries, NumPy, Pandas and so forth, all the code is written from scratch, from the generation of synthetic data to the utilities to read the results of the FANMOD motifs mining program (see below) and results production.
* `Results`. Here the directory relative to one seed is created, therein the results created for each initialisation scheme are stored. Those comprehend the serialized files containing motifs informations, useful to be fetched if needed. In each directory containing the results for the initialisation schemes, results relative to the data set are stored, that is, the `.h5` model from Keras and the `.txt` graph data structures used by the FANMOD program.

This hierarchy is automatically created by a program run, and files are saved in a consistent fashion, in such a way to fetch the needed files in the same run or later. Note that in the `main.py` script the execution is modular: The user tells whether to repeat the preprocessing stage, training stage or whatever. Accordingly, files are created again or fetched.

## Motifs mining tool

The [FANMOD](http://theinf1.informatik.uni-jena.de/motifs/) program has been used, in that it has the capability of inspecting colored netowrks, as it is the case. See the linked page for relevant papers, executable, sources and license.

## Manuscript

Soon it will be uploaded the full manuscript of my thesis.
