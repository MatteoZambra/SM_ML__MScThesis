

import graphds
import streams

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_ind

from keras.models import load_model


ds_dict = {'init' : 'Initial',
           'tree' : 'Tree',
           'clus' : 'Clusters',
           'mnist': 'MNIST'}

is_dict = {'normal' : 'Normal',
           'orth'   : 'Orthogonal',
           'glorot' : 'Glorot'}




def bins_for_scheme(datasets, init_scheme, **flow):
    """
    All the weights from initial configuration to all trained configurations are
    gathered in an array.
    Gaussian fit performed. All the weights {w : p(w) >= cutoff * max(fitted_gaussian_density)} 
    are removed, in that are the most weak and are thought not to be relevant.
    The tails are sliced in this way: 1/5, the part closest to the mean (zero),
    are set to the ``mild'' category. The rest of the tails are ``strongly positive''
    or ``negative'', depending on whether is the right or left tail.
    
    The bins of the histogram thus obtained are returned and subsequently used 
    for categorisation, see the ``specturmSplit'' module
    
    Input:
        ~ datasets (list of string) : contains the data sets specifiers
        ~ init_scheme (string) : initialization specifier
        ~ flow (dictionary) : flow control
    
    Returns
        ~ bins_edges (list of floats) : which identify the position of the 
                                        subdivisions
    """
    
    print('Bins creation. {} initialization'.format(init_scheme))
    
    is_dict = {'orth'   : 'Orthogonal',
               'normal' : 'Normal',
               'glorot' : 'Glorot'}
    
    weights_array = np.array([])
    
    for dataset in datasets:
        
        print('{} data set'.format(dataset))
        model = load_model(flow['path_output'] + r'\{}\model_{}.h5'.format(dataset,dataset))
        graph = graphds.proGraph(model)
        Edges = graph.GetEdges()
        EdgesDF = pd.DataFrame.from_dict(Edges, orient = 'index',
                                         columns = ['edge','src', 'trg', 'param'])
        weights = EdgesDF['param'].values
        weights_array = np.concatenate((weights_array, weights))
    #end
    
    weights_array = np.sort(weights_array)
    
    (mu,sigma) = norm.fit(weights_array)
    
    N,bins = np.histogram(weights_array, density = True)
    fitted_curve = norm.pdf(weights_array, mu, sigma)
    
    if flow['plot']['preprocess']:
        fig,ax = plt.subplots(figsize = (10,5))
        ax.hist(weights_array,bins = 100, density = True, alpha = 0.2)
        ax.plot(weights_array, fitted_curve, 'k', lw = 2, alpha = 0.3)
    #end
    
    third_positive_fraction = 1/5.
    third_negative_fraction = 1/5.
    
    threshold = flow['exclusion_cutoff'] * max(fitted_curve)
    bins_prev = []
    
    for i in range(1, weights_array.size):
        if ((fitted_curve[i-1] <  threshold and fitted_curve[i] >= threshold) or \
            (fitted_curve[i-1] >= threshold and fitted_curve[i] <  threshold)):
            
            bins_prev.append(weights_array[i])
        #end
    #end
    
    min_ = min(weights_array) + 0.1 * min(weights_array)
    max_ = max(weights_array) + 0.1 * max(weights_array)
    third_negative = third_negative_fraction * (min_ - bins_prev[0]) + bins_prev[0]
    third_positive = third_positive_fraction * (max_ - bins_prev[1]) + bins_prev[1]
    
    bins_edges = [min_, third_negative, bins_prev[0], bins_prev[1], third_positive, max_]
    
    if flow['plot']['preprocess']:
        N_,bins_hist = np.histogram(weights_array,bins = bins_edges, density = True)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        N__,bins_hist__,patches = ax.hist(weights_array, bins_hist, density = True, alpha = 0.5)
        patches[0].set_facecolor('r')
        patches[1].set_facecolor('y')
        patches[2].set_facecolor('lightgray')
        patches[3].set_facecolor('y')
        patches[4].set_facecolor('g') 
        
        legend_elements = [Line2D([0],[0], color = 'k', lw = 2, alpha = 0.3, label = 'Gaussian fit'),
                           Patch(facecolor = 'b', edgecolor = 'b', alpha = 0.2, label = 'Weights'),
                           Patch(facecolor = 'r', edgecolor = 'r', alpha = 0.5,
                                 label = 'Negative'),
                           Patch(facecolor = 'y', edgecolor = 'y', alpha = 0.5,
                                 label = 'Mild (+/-)'),
                           Patch(facecolor = 'g', edgecolor = 'g', alpha = 0.5,
                                 label = 'Positive'),
                           Patch(facecolor = 'lightgray', edgecolor = 'lightgray', alpha = 0.5,
                                 label = 'Excluded')]
                           
        ax.legend(handles = legend_elements, loc = 'best',
                  prop = {'size' : 10}, fancybox = True)
        
        plt.xticks(bins_edges, rotation = 45)
        ax.set_xlabel('Weights population')
        ax.set_ylabel('Normalised frequency')
        title = 'Gaussian fit and categories assignment, {} intialisation'.format(is_dict[init_scheme])
        plt.title(title)
        plt.savefig(flow['path_output'] + r'\_Figures\gaussian_fit_weights.png',
                    dpi=300, bbox_inches = 'tight')
        plt.show()
        plt.close('all')
    #end
    
    return bins_edges
#end



def spectrum_discretize(bins_edges, dataset, init_scheme, **flow):
    """
    This is the core of the procedure: the keras model is turned to
    a graph, via the ``proGraphDataStructure'' module functionalities.
    
    Input:
        ~ bins_edges (list of floats) : see above
        ~ datasets (list of strings) : see above
        ~ init_scheme (string) : see above
        ~ flow (dictionary) : see above
                                     
    Returns:
        ~ edges_df (pandas.DataFrame) : contains the edges meta-informations
                                        that is the nodes the edge links, the category associated
                                        to each edge, the connection strength. The category
                                        information is used to classify edges among strongly positive
                                        or negative, mildly positive or negative, negligible.
    """
    
    print('\nWeights specturm discretisation of ' + dataset + ' domain')
    
    model = load_model(flow['path_output'] + r'\{}\model_{}.h5'.format(dataset, dataset))
    path_save_figures = flow['path_output'] + r'\_Figures'
    
    graph = graphds.proGraph(model)
    edges = graph.GetEdges()    
    edges_df = pd.DataFrame.from_dict(edges, orient = 'index',
                                   columns = ['edge', 'src', 'trg', 'param'])
    weights = np.asarray(edges_df['param'])
    if flow['plot']['preprocess']:
        spectrum_split_plot(weights, path_save_figures, dataset, bins_edges)
    #end
    
    edges_df = parameters_categories(edges_df, bins_edges)
    
    
    """
    NOTE: owing to the choice of having categories
        ~ to remove
        ~ mildly positive/negative
        ~ positive
        ~ negative
    and owing to the fact that the histogram exported by the bins_for_scheme function
    are five, this modification is in order, that is
        ~ the category 4 contains mildly positive, the is set to 2, which
        ~ is the category that already contains mildy negative values
        ~ category 3 contains null values, and is set to 4, the category that
          is then removed
        ~ category 5 contains positive value, but now category 3 has been set to 3 and
          category 4 has been set to 2, then it remains to set cat 5 to 3, that of positive
          values
    category 1 and 2, resp. negative and mildly negative, remain untouched.
    """
    
    edges_df.loc[edges_df['cats'] == 4, 'cats'] = 2
    edges_df.loc[edges_df['cats'] == 3, 'cats'] = 4
    edges_df.loc[edges_df['cats'] == 5, 'cats'] = 3
    
    edges_df = edges_df[edges_df['cats'] != 4]
    df_copy = edges_df
    edges_df = edges_df[['src','trg','cats']]
    
    # _edges = {'df_{}'.format(dataset) : edges_df, 'values_{}'.format(dataset) : weights_values}
    
    streams.check_create_directory(flow['path_output'] + r'\{}'.format(dataset))
    
    if flow['write_graph']:
        print('Writing Graph File\n')
        
        filename = flow['path_output'] + r'\{}\_{}_{}_Graph.txt'.format(dataset, dataset, flow['weighted_graph'])
        np.savetxt(filename, edges_df.values, fmt='%d')
    #end

    # return edges_df
    return df_copy
#end
    
def t_test(edges_dfs, datasets, init_scheme, **flow):
    """
    DOC
    """
    weights_init = edges_dfs[0]['param'].values
    
    handle = open(flow['path_output'] + r'\t-tests_partial.txt','w')
    handle.write('{} & Initial & {:.6f} & {:.6f} & -- {} \n'.format(is_dict[init_scheme], np.mean(np.abs(weights_init)),
                                                               weights_init.std(), r'\\'))
    for dataset,df in zip(datasets[1:],edges_dfs[1:]):
        print('{} t-test'.format(dataset))
        weights = df['param'].values
        t_stat, p_value = ttest_ind(weights_init, weights, equal_var = False)
        print('t = {:.6f}, p = {:.6f}'.format(t_stat,p_value))
        handle.write('{} & {} & {:.6f} & {:.6f} & {:.6f} {}\n'.format(is_dict[init_scheme], ds_dict[dataset],
                                                                 np.mean(np.abs(weights)), weights.std(),
                                                                 p_value, r'\\'))
    #end
    handle.close()
    
#end


def spectrum_split_plot(weights, path_save_figures, dataset, bins_edges):
    """
    Histogram plot.
    The number of categories is set before (binsEdges), and accordingly the 
    bars assume a different color
    
    Input:
        ~ weights (numpy.ndarray) : contains the weights values (all the population)
        ~ path_save_figures (string) : path to save the figures
        ~ dataset (string) : see above
        ~ bins_edges (list of floats) : see above
        
    Returns:
        nothing
    """
    
    ds_dict = {'init' : 'Initial',
               'tree' : 'Tree',
               'clus' : 'Clusters',
               'mnist': 'MNIST'}
    
#    k = gaussian_kde(weights, bw_method = 'silverman')
#    x = np.sort(weights)
#    y = np.reshape(k(x).T, weights.size)
    
    fig,ax = plt.subplots(figsize = (8,4))
#    ax.plot(weights,y, 'k', lw = 2, alpha = 0.3)
    N,bins,patches = ax.hist(weights, bins = bins_edges, alpha = 0.7)
    
    patches[0].set_facecolor('r')
    patches[1].set_facecolor('y')
    patches[2].set_facecolor('lightgray')
    patches[3].set_facecolor('y')
    patches[4].set_facecolor('g')
    
    legend_elements = [Patch(facecolor = 'r', edgecolor = 'r',
                             alpha = 0.7, label = str(int(N[0]))),
                       Patch(facecolor = 'y', edgecolor = 'y', 
                             alpha = 0.7, label = str(int(N[1] + N[3]))),
                       Patch(facecolor = 'g', edgecolor = 'g', 
                             alpha = 0.7, label = str(int(N[4]))),
                       Patch(facecolor = 'lightgray', edgecolor = 'lightgray', 
                             alpha = 0.7, label = str(int(N[2])))]
    
    ax.legend(handles = legend_elements, loc = 'best', prop = {'size' : 10},
               fancybox = True)
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xticks(bins_edges, rotation = 45)
    plt.xlabel('Sorted values')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.title('Frequency plot with modified bins, {} environment'.format(ds_dict[dataset]))
    plt.savefig(path_save_figures + '\{}_hist_weights.png'.format(dataset),
                dpi=300, bbox_inches = 'tight')
    plt.show()
    plt.close('all')
#end

        
def parameters_categories(df, bins_edges):
    """
    categories are numerical values, integers.
    Edges and nodes are set to a category according
    to their value.
    The number of categories is the length of the 
    binsEdges array, namely, the 5 slices of the spectrum
    
    Input:
        ~ df (pandas.DataFrame) : the edges, see above
        ~ bins_edges (list of floats) : see above
        
    Returns:
        ~ df pandas.DataFrame) : edges, in which the categories are set
    """

    df['cats'] = pd.Series(np.zeros(df.shape[0]),
                           index = df.index,
                           dtype = np.int32)
    
    for cate in range(1, len(bins_edges)):
        mask = (df['param'] >= bins_edges[cate-1]) & (df['param'] < bins_edges[cate])
        df.loc[mask,'cats'] = cate
        
    return df
#end
