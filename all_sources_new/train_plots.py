

from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

import proGraphDataStructure as pg
import keras_ops as ko


"""
This script is called by the main program for the sake of 
producing density plots, referred to model parameters
(estimated) distributions.
"""

ds_dict = {'init' : 'Initial',
           'tree' : 'Tree',
           'clus' : 'Clusters',
           'mvg'  : 'Multitask'}

is_dict = {'orth' : 'Orthogonal',
           'normal' : 'Normal',
           'glorot' : 'Glorot'}

path_fig = r'C:\Users\matte\Desktop\MasterThesis\SKRITTURA\immagini\last_plots\\'


def distributions_plot(path_model, datasets, init_scheme):
    
    model_init = load_model(path_model + r'\init\model_init.h5')
    graph_init = pg.proGraph(model_init)
    Edges_init = graph_init.GetEdges()
    EdgesDF = pd.DataFrame.from_dict(Edges_init, orient = "index",
                                     columns = ["edge", "param"])
    weights_init = EdgesDF['param'].values
    w_i = np.sort(weights_init)
    
    for dataset_id in datasets[1:]:
        ax,fig = plt.subplots(figsize=(5,3))
        model = load_model(path_model + r'\{}\model_{}.h5'.format(dataset_id,dataset_id))
        graph = pg.proGraph(model)
        Edges = graph.GetEdges()
        EdgesDF = pd.DataFrame.from_dict(Edges, orient = "index",
                                         columns = ["edge", "param"])
        weights = EdgesDF['param'].values
        w_p = np.sort(weights)
        bw = 1.06 * w_i.std() * w_i.size ** (-1 / 5.)
        sns.kdeplot(w_i, bw=bw, shade=True, label='Before training')
        bw = 1.06 * w_p.std() * w_p.size ** (-1 / 5.)
        sns.kdeplot(w_p, bw=bw, shade=True, label='After training')
        title = '{} dataset'.format(ds_dict[dataset_id])
        plt.title(title)
        plt.xlabel('Weights')
        plt.ylabel('Estimated density')
        plt.savefig(path_fig + '_kdeplot_{}_{}.png'.format(init_scheme,dataset_id))
        plt.show()
    #end
    
#end





















