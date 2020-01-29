
"""
The flow dictionary is the core of the program behaviour. It tells
whether to perform each chunk of the execution, might it be model 
initialization, training, preprocess, ...

Instructions and details must be hard coded
_________________________________________________________________________________________________
"""


import os
import sys
sys.path.append(os.getcwd() + r'\utils')

import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import streams
import train
import preprocess
import postprocess

seed_value = 450509
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)


flow   = {'mnist'             : True,
          'initialize'        : False,
          'train'             : False,
          'network'           : {240120 : [20,10],
                                 250120 : [20,20],
                                 180112 : [30,30],
                                 450509 : [50,50,20]},
          'preprocess'        : True,
          'postprocess'       : False,
          'serialize'         : False,
          'efficacy'          : False,
          'sp_by'             : ' ',
          'seed'              : seed_value,
          'write_graph'       : False,
          'weighted_graph'    : 'w',
          'motifs_size'       : 5,
          'detail'            : 'msm',
          'save_model'        : True,
          'plot'              : {'train'         : False,
                                 'weights_kdes'  : True,
                                 'preprocess'    : False,
                                 'post_scatter'  : True,
                                 'post_distr'    : True,
                                 'post_bars'     : True,
                                 'post_summary'  : False} }


most_changed_motifs_initscheme = {}
variations_all_motifs = {}
significances_all_motifs = {}

initializations = ['normal','orth','glorot']
# initializations = ['glorot']

if flow['mnist']: 
    datasets = ['init','mnist']
    flow.update({'exclusion_cutoff' : 0.1})
else: 
    datasets = ['init','tree','clus']
    flow.update({'exclusion_cutoff' : 0.55})
#end


path_results, path_summary_plots, \
    path_latex_tables, path_serialized_dataframes = \
        streams.hierarchy(flow['seed'], **flow)

flow.update({'path_splots' : path_summary_plots})


for init_scheme in initializations:
    
    """
    PREPROCESS: Initialization, training, graph construction
    _____________________________________________________________________________________________
    """
    
    path_results_initscheme = path_results + r'\{}'.format(init_scheme)
    path_save_figures = path_results_initscheme + r'\_Figures'
    streams.check_create_directory(path_results_initscheme)
    streams.check_create_directory(path_save_figures)
    
    flow.update({'path_figures'   : path_save_figures})
    flow.update({'path_output'    : path_results_initscheme})
    flow.update({'path_serialize' : path_serialized_dataframes})
    
    if flow['initialize']:
        train.model_init(init_scheme, **flow)
    #end
    
    
    if flow['train']:
        for dataset in datasets[1:]:
            model, accuracy = train.model_train('', dataset, init_scheme, **flow)
        #end
    if flow['plot']['weights_kdes']:
        train.weights_kdes(datasets, init_scheme, **flow)
    #end
    
    if flow['preprocess']:
        binsedges = preprocess.bins_for_scheme(datasets, init_scheme, **flow)
        edges_dfs = [preprocess.spectrum_discretize(binsedges, dataset,
                                                    init_scheme, **flow) for dataset in datasets]
        preprocess.t_test(edges_dfs, datasets, init_scheme, **flow)
    #end
    
    
    """
    POSTPROCESS: motifs mining, graphical results
    _____________________________________________________________________________________________
    """
    
    if flow['postprocess']:
        
        if flow['weighted_graph'] == 'w': 
            offset = 2
        else: 
            offset = 0
        #end
        
        path_initial_net = flow['path_output'] + \
            r'\init\init_{}_s{}_out.csv'.format(flow['weighted_graph'], flow['motifs_size'])
        
        motifs_significances, motifs_variations, motifs_most_changed = \
            postprocess.analysis_launcher(path_initial_net, offset, datasets, **flow)
        
        variations_all_motifs.update({init_scheme : motifs_variations})
        most_changed_motifs_initscheme.update({init_scheme : motifs_most_changed})
        significances_all_motifs.update({init_scheme : motifs_significances})
        
        if flow['serialize']:
            """
            One may wish to store the motifs informations mined and use them later
            """
            
            dataframes = {'variations_all_motifs' : variations_all_motifs,
                          'most_changed_motifs_initscheme' : most_changed_motifs_initscheme,
                          'significances_all_motifs' : significances_all_motifs}
            streams.serialize_dataframes(dataframes, **flow)
        #end
    #end
#end


"""
Summary plots. This part must be separated since the data wrangled here
refer to all the initialization schemes and data sets adopted
_________________________________________________________________________________________________
"""

if flow['plot']['post_summary']:
    if flow['weighted_graph'] == 'w':
        _append = r'w_s{}_{}.pickle'.format(flow['motifs_size'], flow['detail'])
    else:
        _append = r'u_s{}.pickle'.format(flow['motifs_size'])
    #end
    
    handle = open(flow['path_serialize'] + r'\significances_all_motifs_{}'.format(_append), 'rb')
    variations_all_motifs = pickle.load(handle)
    handle.close()
    
    flow['sp_by'] = 'initialization'
    postprocess.significance_profiles(variations_all_motifs, datasets, initializations, **flow)
    flow['sp_by'] = 'dataset'
    postprocess.significance_profiles(variations_all_motifs, datasets, initializations, **flow)
#end

"""
Learning efficacy plots. Here the model is assumed to be trained yet on
each initialization scheme and for each data set.
The model is not saved to .h5 and only the efficacy plots are saved.

This chuck is placed separately from the training calls since here it 
is needed to loop over the datasets and initialization schemes faster.
_________________________________________________________________________________________________
"""

if flow['efficacy']:
    train.efficacy_plots(path_results, datasets, initializations, **flow)
#end

"""
Log memo in which details about threshold and network details are annotated
_________________________________________________________________________________________________
"""

with open(path_results + r'\details_log.txt', 'w') as f:
    f.write('Cut-off: {}'.format(flow['exclusion_cutoff']))
    f.write('\nNetwork architecture:\n')
    f.write('Input layer: 31 units, activation: relu\n')
    for layer in flow['network'][flow['seed']]:
        f.write('Hidden layer: {} units, activation: relu\n'.format(layer))
    #end
    f.write('Output layer: 4 units, activation: softmax')
#end
f.close()

plt.close('all')













    