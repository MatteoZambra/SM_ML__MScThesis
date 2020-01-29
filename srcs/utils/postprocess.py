
"""
This module is the most subtle and tricky part of the coding process

The information generated by the motifs mining tool is huge and obscure.
There is a log file in which the details of each motif detected are provided.

In the scope of the present work is important to gain both quantitative and qualitative
insights out of these informations. Hence the choice of extensive plotting utilities.
The dictionaries ``motifs'' returned by the motifsreader.motifs_load function are the
pivotal data structure used. These are mined and the Z-score values of the instances
gathered under each of the keys ids are compared for each motif, in the initialised and
after-training configurations.
Based on the values of the significance scores and the variations values in the lapse before
and after training, graphical results are produced, in bar plots and significance profiles formats
(see the main text of the manuscript)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

import motifsreader as mr


ds_dict = {'init' : 'Initial',
           'tree' : 'Tree',
           'clus' : 'Clusters',
           'mnist': 'MNIST'}

is_dict = {'orth'   : 'Orthogonal',
           'normal' : 'Normal',
           'glorot' : 'Glorot'}


def analysis_launcher(path_initial_net, offset, datasets, **flow):
                               
    """
    This function calls all the other utilities to obtain explanatory graphics.
    In the execution flow, the most used data types are dictionaries and 
    pandas.DataFrame data structures. It yields the ease of indexing a motif or
    a group of motifs with the isomorphic group key.
    
    Note that when it comes to weighted analyses, for each isomorphism key we
    have a list of motifs, each of which has associated a significance score. 
    Thus it is in order to distinguish the most significant and the most typical motifs.
    
    Most significant: given an isomorphism key, the motif of that group which has 
                      the greatest Z-score is the most significative
    Most typical:     given the same group, the motif which has the Z-score that is
                      closest to the average Z-score for that group
                      
    Input:
        ~ path_initial_net (string) : path where the initialized model lays.
                                      It is used to compare the motifs found in
                                      the trained models
        ~ offset (integer) : the FANMOD programs outputs the results in a .csv
                             file. If the network to be inspected has colors 
                             associated with the edges, then this information is
                             reported, hence two more lines in the header are 
                             written. To let the program flow regardless, an offset
                             must be set, so that the results may be read either way
        ~ datasets (list of strings) : contain the data sets identifyiers
        ~ flow (dictionary) : control the program flow
     
    Returns:
        ~ motifs (pandas.DataFrame) : contains the motifs found
        ~ variations (pandas.DataFrame) : contains the variations in significance 
                                          that each motif experiences in the course 
                                          of learning
        ~ most_changed (pandas.DataFrame) : top changed motifs
    """
    
    motifs_significances = []
    all_motifs_variations = []
    motifs_most_changed = []
    
    motifs_init = mr.motifs_load(path_initial_net, offset, flow['motifs_size'])
    motifs_init = motifs_selection(motifs_init, flow['detail'])
    motifs_init.rename(columns = {'Zscore' : 'init'},
                          inplace = True)
    
    motifs_significances.append(motifs_init)
    
    for dataset in datasets[1:]:
        
        # STEP 0: import motifs as dicts
        path_trained_net = flow['path_output'] +  \
          r'\{}\{}_{}_s{}_out.csv'.format(dataset, dataset, flow['weighted_graph'], flow['motifs_size'])
        motifs_post = mr.motifs_load(path_trained_net, offset, flow['motifs_size'])
        
        # STEP 1: in case of weighted graph,
        #         pick the MaxSignificanceMotif
        motifs_post = motifs_selection(motifs_post, flow['detail'])
        motifs_post.rename(columns = {'Zscore' : dataset}, inplace = True)
        motifs_significances.append(motifs_post)
        
        # STEP 2: scatter plot and selection of 
        #         most changed motifs
        scatter_plot(motifs_init, motifs_post, dataset, **flow)
        
        variations_for_dataset, most_changed_for_dataset = \
                      variations_distributions(motifs_init, motifs_post, dataset, **flow)
        
        all_motifs_variations.append(variations_for_dataset)
        motifs_most_changed.append(most_changed_for_dataset)
    # end
    
    # STEP 3: group data for all datasets
    all_motifs_variations = pd.concat(all_motifs_variations, axis = 1, sort = False)
    all_motifs_variations = all_motifs_variations.sort_index()
    
    motifs_most_changed = pd.concat(motifs_most_changed, axis = 1, sort = False)
    motifs_most_changed = motifs_most_changed.sort_index()
    variations_comparison(motifs_most_changed, **flow)
    
    motifs_significances = pd.concat(motifs_significances, axis = 1, sort = False)
    motifs_significances = motifs_significances.fillna(0)
    motifs_significances = motifs_significances.sort_index()
    
    return motifs_significances, all_motifs_variations, motifs_most_changed
    
#end

def motifs_selection(motifs, detail):

    """
    This function returns a data structure containing the motifs
    requested, whether they are desired to be most significant or
    most typical
    Input:
        ~ motifs (dictionary) : in the format {key : list of motifsreader.MotifObj}
                                See the ``motifsreader.py'' module
        ~ detail (string) : tells whether to look for most significant or
                            most typical motifs
        
    Returns
        ~ selected_motifs (pandas.DataFrame) : contains the selected motifs
    """
    
    columns = ['motifID', 'Zscore']
    
    selected_motifs = []
    
    for key in motifs:
        
        motif_instances_df = pd.DataFrame(columns = columns)
        max_significance_for_key = pd.DataFrame(columns = columns)
        
        if (len(motifs[key]) == 0):
            i = "sub_000"
            motif_instances_df.at[i,'motifID'] = int(key)
            motif_instances_df.at[i,'Zscore']  = 0.0
        #end
        
        for motif_obj in motifs[key]:
            
            i = "sub_" + str(motif_obj.subgrID)
            
            motif_instances_df.at[i,'motifID'] = int(key)
            motif_instances_df.at[i,'Zscore']  = float(motif_obj.Zscore)
        #end
        
        for name in motif_instances_df:
            motif_instances_df[name] = pd.to_numeric(motif_instances_df[name])
        #end
        
        if (detail == 'msm'):
            
            id_max = motif_instances_df['Zscore'][lambda z : z == max(z)].index[0]
            max_significance_for_key = pd.DataFrame(motif_instances_df.loc[id_max])
            max_significance_for_key = max_significance_for_key.T
            selected_motifs.append(max_significance_for_key)
            
        elif (detail == 'mtm'):
            
            zscores = motif_instances_df['Zscore'].values
            average_zscore = np.mean(zscores)
            id_typical = np.argmin(abs(zscores - average_zscore))
            most_typical_for_key = pd.DataFrame(motif_instances_df.iloc[id_typical])
            most_typical_for_key = most_typical_for_key.T
            selected_motifs.append(most_typical_for_key)
            
        #end
        
    #end
    
    selected_motifs = pd.concat(selected_motifs, axis = 0)
    selected_motifs = selected_motifs.astype({'motifID' : int})
    
    selected_motifs = selected_motifs.sort_values(by = ['motifID'])
    selected_motifs = selected_motifs.set_index('motifID')
    
    return selected_motifs
#end

def scatter_plot(motifs_init, motifs_post, dataset, **flow):    
    """
    Produces a scatterplot, in which the x-asis is the initial value of the 
    significance score and the y-axis, conversely, is the final value.
    Each point is a motif instance, if it is located on the right side of 
    the bisector locus, then its significance decreased in the course of 
    learning. On the other hand, of it is located on the left, then its significance
    has increased. This may point out a relevance of the motif in the learning process.
    
    Input:
        ~ motifs_init (dictionariy) : in the format explained above, referred
                                       to the initial configuration
        ~ motifs_post (dictionariy) : same, but for the trained configuration
        ~ dataset (string) : data set specifyier
        ~ flow (dictionary) : as above
        
    Returns:
        nothing
    """
    
    keys = motifs_init.index.tolist()
    if (keys != motifs_post.index.tolist()): 
        print("INCONGRUENCE : keys")
        print('Simply means that some motifs in the initial/trained config do not match those in the final/initial')
        motifs_ = pd.concat([motifs_init, motifs_post], axis = 1, sort = False)
        motifs_ = motifs_.fillna(0)
        zsc_init = motifs_['init'].values.tolist()
        zsc_post = motifs_[dataset].values.tolist()
    else:
        zsc_init = motifs_init['init'].values.tolist()
        zsc_post = motifs_post[dataset].values.tolist()
    #end
    
    if (min(zsc_init) < 0 or min(zsc_post) < 0):
        minest = min(min(zsc_init), min(zsc_post))
    else:
        minest = 0.0
    #end
    
    maxest = max(max(zsc_init), max(zsc_post))
    bisector = np.linspace(minest, maxest, 10)
    
    fig,ax = plt.subplots(figsize=(5,5))
    ax.plot(bisector,bisector, 'k--', linewidth = 2, alpha = 0.3)
    
    for zi,zp in zip(zsc_init,zsc_post):
        ax.scatter(zi,zp, s = 30, color = 'r', alpha = 0.5)
    #end
    
    bisectrLine = Line2D([],[],color = 'k',
                         linestyle = '--',
                         linewidth = 3,
                         alpha = 0.3,
                         label = 'Bisector')
    smallDot    = Line2D([],[],color = 'r',marker = 'o',
                         linestyle = 'None',
                         markersize = 3,
                         label = 'Motif Entity')
    
    ax.legend(handles = [bisectrLine, smallDot], loc = 'lower right',
              prop = {'size' : 10}, fancybox = True)
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel("Initial configuration Z-score")
    ax.set_ylabel("Trained configuration Z-score")
    title = 'Z-score scatter-plot. {} domain, {}-vertices motifs'.format(ds_dict[dataset],str(flow['motifs_size']))
    plt.title(title)
    if (flow['plot']['post_scatter']):
        if (flow['weighted_graph'] == 'u'):
            title = flow['path_figures'] + r'\scatter_s{}_u.png'.format(str(flow['motifs_size']))
        elif (flow['weighted_graph'] == 'w'):
            title = flow['path_figures'] + r'\scatter_s{}_w_{}.png'.format(str(flow['motifs_size']), flow['detail'])
        #end
        plt.savefig(title, dpi = 300, bbox_inches = 'tight')
    #end
    plt.show()
    plt.close('all')
#end

def variations_distributions(motifs_init, motifs_post, dataset, **flow):
                             
    """
    Here the graphic produced is an histogram. It depicts the distribution of the
    variations in the significance score.
    
    Input:
        same as above
    
    Returns:
        ~ variations_df (pandas.DataFrame) : see above
        ~ motifs_most_changed (pandas.DataFrame) : see above
    """
    
    if (flow['motifs_size'] == 4):
        num_bins_edges = 10
    elif (flow['motifs_size'] == 5):
        num_bins_edges = 40
    elif (flow['motifs_size'] == 6):
        num_bins_edges = 60
    #end
    
    variations_df = pd.DataFrame(columns = [dataset])
    
    keys = motifs_init.index.tolist()
    if (keys != motifs_post.index.tolist()): 
        print("INCONGRUENCE : keys")
        motifs_ = pd.concat([motifs_init, motifs_post], axis = 1, sort = False)
        motifs_ = motifs_.fillna(0)
        variations_df[dataset] = motifs_[dataset] - motifs_['init']
    else:
        variations_df[dataset] = motifs_post[dataset] - motifs_init['init']
    #end
    
    variations_df = variations_df.sort_values(by = [dataset], ascending = False)
    
    values = variations_df.values
    values = values.flatten()

    fig,ax = plt.subplots(figsize = (15,6))
    
    bins_edges = np.linspace(min(values),max(values),num_bins_edges)
    
    hist, _ = np.histogram(values, bins = bins_edges, density = False)
    plt.hist(values, bins = bins_edges, alpha = 0.5)
    ax = plt.gca()
    for tk in ax.get_xticklabels():
        tk.set_rotation(45)
    #end

    
    if (flow['motifs_size'] == 4):
        bins_ticks = bins_edges
    elif (flow['motifs_size'] == 6 or flow['motifs_size'] == 5):
        bins_ticks = []
        for i in range(len(bins_edges)):
            if (i % 2 == 0):
                bins_ticks.append(bins_edges[i])
            #end
        #end
    #end
    
    plt.xticks(bins_ticks)
    plt.xlabel('Z-score variation')
    plt.ylabel('Frequencies')
    title = 'Distribution of Z-score absolute variations. '
    title += r'{} domain, {}-vertices motifs'.format(ds_dict[dataset], str(flow['motifs_size']))
    plt.title(title)
    if (flow['plot']['post_distr']):
        if (flow['weighted_graph'] == 'u'):
            title = flow['path_figures'] + r'\{}_varhist_s{}_u.png'.format(dataset, str(flow['motifs_size']))
        elif (flow['weighted_graph'] == 'w'):
            title = flow['path_figures'] + r'\{}_varhist_s{}_w_{}.png'.format(dataset, 
                                     str(flow['motifs_size']), flow['detail'])
        #end
        plt.savefig(title, dpi=300, bbox_inches = "tight")
    #end
    
    plt.show()
    plt.close('all')

    limits = []
    if (flow['motifs_size'] == 4):
        limits.append(min(values))
    elif (flow['motifs_size'] == 6 or flow['motifs_size'] == 5):
        upper_limit = float(input('Upper limit: '))
        limits.append(upper_limit)
        ask_in = True
        while (ask_in):
            ask_lower = input('Lower limit? (y/n): ')
            if (ask_lower == 'y' or ask_lower == 'Y' or \
                ask_lower == 'n' or ask_lower == 'N'):
                ask_in = False
            #end
        #end
        if (ask_lower == 'y'):
            lower_limit = float(input('Lower limit: '))
            limits.append(lower_limit)
        #end
    #end
    
    if (len(limits) == 1):
        mask = variations_df[dataset] >= limits[0]
        motifs_most_changed = variations_df[mask]
    elif (len(limits) == 2):
        mask = (variations_df[dataset] >= limits[0]) | \
               (variations_df[dataset] <= limits[1])
        motifs_most_changed = variations_df[mask]
    #end
    
    
    return variations_df, motifs_most_changed
#end

def variations_comparison(most_changed_motifs, **flow):
                          
    """
    This produces a bar plot. Each motif is indicized be the 
    key number for an isomorphic group, so for each key one has
    as many bars as the environments. 
    
    Input:
        ~ most_changed_motifs (pandas.DataFrame) : as above
        ~ flow (dictionary) : as above
    
    Returns:
        nothing
    """
    
    
    most_changed_motifs.rename(columns = {'tree' : 'Tree',
                                 'clus' : 'Clusters',
                                 'mvg'  : 'Multitask', 'mnist' : 'MNIST'},
                                 inplace = True)
    var_df = most_changed_motifs.sort_index()
    var_df.plot(kind = 'bar', figsize = (7.5,5))
    ax = plt.gca()
    for tk in ax.get_xticklabels():
        tk.set_rotation(45)
    #end
    ax.set_xlabel("Motif ID")
    ax.set_ylabel("Absolute Variation")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xticklabels(most_changed_motifs.index.tolist())
    title = 'Absolute Z-score variations, s = {}'.format(flow['motifs_size'])
    plt.title(title)
    if (flow['plot']['post_bars']):
        if (flow['weighted_graph'] == 'u'):
            title = flow['path_figures'] + r'\zvar_s{}_u.png'.format(str(flow['motifs_size']))
        elif (flow['weighted_graph'] == 'w'):
            title = flow['path_figures'] + r'\zvar_s{}_w_{}.png'.format(flow['motifs_size'], flow['detail'])
        #end
        plt.savefig(title, dpi=300, bbox_inches = "tight")
    #end
    plt.show()
    
    # plt.show()
    # plt.close('all')
    
    most_changed_motifs = most_changed_motifs.fillna(0)
#end
    


def significance_profiles(all_motifs, datasets, initializations, **flow):    
    """
    This is a pivotal visualization. 
    The motifs keys are displayed on the x-axis and the respective y-axis 
    values are the significance scores, thus yielding a plot similar to
    the ``significance profile'' depicted in Figures 1 and 3 in 
    Milo et al., (2004) Science, Vol. 303, Issue 5663, pp. 1538-1542
    
    Input:
        ~ all_motifs (pandas.DataFrame) : as above
        ~ datasets (list of strings) : as above
        ~ initalizations (list of strings) : contains the initialization schemes
                                             specifyiers
        
    Returns:
        ~ nothing
    """
    
    max_of_df = []
    for item in all_motifs:
        max_of_df.append(max(all_motifs[item].max()))
    #end
    max_of_df = max(max_of_df)
    min_of_df = []
    for item in all_motifs:
        min_of_df.append(min(all_motifs[item].min()))
    #end
    min_of_df = min(min_of_df)
    
    aestetics_datasets = {'init' : {'color'  : 'k',
                                    'marker' : 'o'},
                          'tree' : {'color'  : 'g',
                                    'marker' : '^'},
                          'clus' : {'color'  : 'r',
                                    'marker' : 'v'},
                          'mnist': {'color'  : 'r',
                                    'marker' : 'x'}
                          }
    aestetics_initializations = {'orth'   : {'color'  : 'c',
                                             'marker' : '^'},
                                 'normal' : {'color'  : 'm',
                                             'marker' : 'v'},
                                 'glorot' : {'color'  : 'y',
                                             'marker' : 's'}
                                 }
    
    if (flow['sp_by'] == 'initialization'):
        outer_loop       = initializations
        inner_loop       = datasets
        legend_loop      = initializations
        legend_dict      = ds_dict
        legend_aestetics = aestetics_datasets
        plot_aestetics   = aestetics_datasets
        title_dict       = is_dict
    elif (flow['sp_by'] == 'dataset'):
        outer_loop       = datasets
        inner_loop       = initializations
        legend_loop      = datasets
        legend_dict      = is_dict
        legend_aestetics = aestetics_initializations
        plot_aestetics   = aestetics_initializations
        title_dict       = ds_dict
    #end    
    
    for item_outer in outer_loop:
        
        fig,ax = plt.subplots(figsize = (10,5))
        
        for item_inner in inner_loop:
            
            if (flow['sp_by'] == 'initialization'):
                keys   = all_motifs[item_outer][item_inner].index.tolist()
                values = all_motifs[item_outer][item_inner].values.tolist()
            else:
                keys   = all_motifs[item_inner][item_outer].index.tolist()
                values = all_motifs[item_inner][item_outer].values.tolist()
            #end
            
            xspan = np.arange(1,len(keys)+1)
            ax.scatter(xspan,values, s = 30, 
                       color  = plot_aestetics[item_inner]['color'],
                       marker = plot_aestetics[item_inner]['marker'],
                       alpha = 0.5)
            ax.plot(xspan,values, lw = 2,
                    color = plot_aestetics[item_inner]['color'],
                    alpha = 0.5)
            
            handles = []
            for legend_loop in inner_loop:
                handles.append(Line2D([],[],
                                      color  = legend_aestetics[legend_loop]['color'], 
                                      marker = legend_aestetics[legend_loop]['marker'],
                                      linestyle = '-',
                                      markersize = 5,
                                      label = legend_dict[legend_loop]))
            
            ax.legend(handles = handles, prop = {'size' : 10},
                      loc = 'upper left', fancybox = True)
            
            ax.set_ylim((min_of_df-1,max_of_df+1))
            
            if (flow['sp_by'] == 'initialization'):
                title = 'Significance profile for fixed {} initialisation scheme'.format(title_dict[item_outer])
            else:
                title = 'Significance profile for fixed {} environment'.format(title_dict[item_outer])
            #end
            
            ax.set_title(title,fontsize = 15)
            ax.set_xticks(xspan)
            ax.set_xticklabels(keys)
            ax.set_xlabel('Motif ID')
            ax.set_ylabel('Significance profile')
            
            if (item_inner == inner_loop[-1]):
                if (flow['weighted_graph'] == 'u'):
                    title = flow['path_splots'] + r'\s{}_u_sp_{}_{}.png'.format(str(flow['motifs_size']),
                            flow['sp_by'], item_outer)
                elif (flow['weighted_graph'] == 'w'):
                    title = flow['path_splots'] + r'\s{}_w_sp_{}_{}_{}.png'.format(str(flow['motifs_size']),
                            flow['sp_by'], item_outer, flow['detail'])
                #end
                plt.savefig(title, dpi = 300, bbox_inches = 'tight')
            #end
        #end
    #end
    
    plt.show()
    plt.close('all')
#end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    