# Emergence of Network Motifs in Deep Neural Networks

## Purpose
This repo contains the code intended to support the developing process of my Master Degree Thesis. The same results are used in the article [_Emergence of network motifs in deep neural networks_](https://www.mdpi.com/1099-4300/22/2/204).

## Summary

Motivated by recent publications concerning the development and formalisation of a theoretical framework in which Deep Learning may fit into, this thesis topic and work is finalised to investigate the evolution of the topology of a simple feedforward artificial neural network during the process of learning.

Network Science is thought to be an interesting standpoint. Inspired by [Milo et al. (2002)](https://science.sciencemag.org/content/298/5594/824), a simple multilayer-perceptron model is inspected as breeding ground for the emergence of network motifs, once the model is trained on different sets of synthetic data and with different initialisation schemes ([Saxe et al., 2013](https://arxiv.org/abs/1312.6120), [Glorot and Bengio, 2010](http://proceedings.mlr.press/v9/glorot10a.html)). The question addressed is whether one can appreciate the trace that the learning dynamics leaves in the form of functional modules. The neural network inspected is trained on a multi-label classification task using two different data sets. Another pivotal aspect is the influence of the initialization scheme in relationship with training efficiency. Turns out that non-trivial schemes, such as Orthogonal matrices and Glorot-Bengio schemes, could impress a sharper initial fingerprint to the initial motifs landscape, thus encouraging these motifs to develop.

An endearing biological resemblance with **kinases** dynamics in transduction networks may be appreciated (see [Alon, 2006](https://www.crcpress.com/An-Introduction-to-Systems-Biology-Design-Principles-of-Biological-Circuits/Alon/p/book/9781439837177)). Moreover, signature of diversity can be appreciated for different arrangments of data sets and initialisation schemes. It is not straightforward to understand whether this diversity stems from different initial and boundary conditions or it is a by-product of noise (for example coming from the stochasticity of the SGD algorithm).
 
## Code

### Contents

`srcs` source codes. The `main.py` scripts launches what necessary 

* `DataSets` folder, which contains the scripts to generate the two data sets used. Number of data examples and other details are hard-coded. In addition, serialized files of previously created data sets are present, in order to fetch them in the simulations.
* `utils` contains the modules I have written to accomplish the task. Other than Machine Learning, as Keras, and standard scientific computing libraries, NumPy, Pandas and so forth, all the code is written from scratch, from the generation of synthetic data to the utilities to read the results of the FANMOD motifs mining program (see below) and results production.
* `Results`. Here the directory relative to one seed is created. In `seed_nseed` the directory `synds` is created if the user wants to perform the analyses on the sets of synthetic data. Otherwise, by setting `flow['mnist'] = True`, the analyses are carried out for the MNIST data set. Thus the directory `mnsit` is created in place of `synds`. The subsequent hierarchy is the same, however. In `synds/` the results created for each initialisation scheme are stored in proper directories, alongside with the directories that contain graphical results and serialized objects. In each directory containing the results for the initialisation schemes, results relative to the data set are stored, that is, the `.h5` model from Keras and the `.txt` graph data structures used by the FANMOD program.

This hierarchy is automatically created by a program run, and files are saved in a consistent fashion, in such a way to fetch the needed files in the same run or later. Note that in the `main.py` script the execution is modular: The user instructs the program whether to repeat the any of the stages in the computational workflow. Accordingly, files are created again or fetched.

### Usage

In the `main.py` file, the `flow` dictionary is declared, and then passed as unpackable object. Herein, instructions about the program flow are contained. That is, these instruct the program whether to perform certain chunks of execution, e.g. model initialization, training, preprocess and so forth.

### FANMOD usage in the analyses

The FANMOD software has the advantages to implement wighted (colored) network motifs mining and has a graphical user interface. It is not possible to call it from command line, hence to integrate it in the workflow of the code contained in this repository. The user must tune the workflow instructions in the aforementioned dictionaries in such a way to create the graph files amenable for FANMOD in a first run, run FANMOD so that this latter can generate the needed output files and in a second time the code should be run again, executing the postprocess stage.

In the analyses presented FANMOD has been used setting the number of random networks to compare the actual model with to be 100. Check the `CSV` option in the `Output file` menu. Check `Directed` (default choice) in the `Input file` menu. Browse the text file produced by the code run and set the name of the output file to `env_weight_size_out.csv`. `env` should be replaced by `init`, `tree`, `clus`, respectively the initial configuration, binary tree and independent clusters data sets. Replace `weight` with `u` or `w`, respectively unweighted or weighed network. In this second case check the `Colored edges` option. Replace `size` with `s4` or `s5`, respectively four or five vertices motifs. As an example, the input file for the case of binary tree data set accounting for a weighted graph will be `tree_w_Graph.txt` and it is created automatically by the program. Assume that one wants to mine four nodes motifs in this case. Then the user should call the output file (**and put it in same directory**) `tree_w_s4_out.csv`. 

Note that the directory where this input file is stored is `..\srcs\Results\seed_n\synds\init_scheme\dataset\`. Here there are the `.h5` Keras model export, the FANMOD input file and the output file generated running this latter. In the postprocessing stage, the program is thought to fetch these output files from this directory.


### Remark about some coding choices
The OO-paradigm has been exploited but not abused. For the purposes of many of the functions written, it has been realized that simple function definition were the best trade-off between code readability, architecture clarity and modularity. The definition and usage of reduntant object types in this case would have resolved in an over-complex classes landscape, thus giving problems in an hypothetic further re-use.

### Remarks about the MNIST data set case
It would be interesting to do a comparison with real-world data sets related results. The code has been modified to implement in a painless way the training on thte MNIST data set. However, the major difficulty is the fact that the networks used for the artificial data sets are inevitably different from the net that would absorb the MNIST data samples and classify them, due to the different input and output dimensions.

However, another improvement front related to this necessity would be the **scalability** of the approach presented. Scalability is the Achille's heel of the whole work. Larger graphs require a larger resources deployment. One interesting direction for further research is indeed devoted to this scalability issue. 

## Motifs mining tool

The [FANMOD](http://theinf1.informatik.uni-jena.de/motifs/) program has been used, in that it has the capability of inspecting colored networks, as it is the case. See the linked page for relevant papers, executable, sources and license.

## Manuscript
Published in [Entropy](https://www.mdpi.com/1099-4300/22/2/204)
