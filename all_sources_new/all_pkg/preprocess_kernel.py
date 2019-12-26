
"""
Path to save images is a global variable
"""

path_save_figs =  r'../figures'
streams.check_create_directory(path_save_figs)


import proGraphDataStructure as pg
import spectrumSplit as ssp
import streams 
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.stats import norm


def bins_for_scheme(path_in_dir, datasets, init_scheme):

    """
    All the weights from initial configuration to all trained configurations are
    gathered in an array.
    Gaussian fit performed. All the weights {w : p(w) >= 0.55 * max(fitted_gaussian_density)} 
    are removed, in that are the most weak and are thought not to be relevant.
    The tails are sliced in this way: 1/5, the part closest to the mean (zero),
    are set to the ``mild'' category. The rest of the tails are ``strongly positive''
    or ``negative'', depending on whether is the right or left tail.
    
    The bins of the histogram thus obtained are returned and subsequently used 
    for categorisation, see the ``specturmSplit'' module
    
    Input:
        ~ path_in_dir           string, points to the directory in which the .h5 model 
                                is saved
        ~ datasets              list of strings, identifying the data set
        ~ init_scheme           string, identifies the initialisation scheme
    
    Returns
        ~ bins_edges            list of floats, which identify the position of the 
                                subdivisions
    """
    
    is_dict = {'orth' : 'Orthogonal',
               'normal' : 'Normal',
               'glorot' : 'Glorot'}
    
    weights_array = np.array([])
    
    for dataset_id in datasets:
        
        model = load_model(path_in_dir + r'\{}\model_{}.h5'.format(dataset_id,dataset_id))
        graph = pg.proGraph(model)
        Edges = graph.GetEdges()
        EdgesDF = pd.DataFrame.from_dict(Edges, orient = "index",
                                         columns = ["edge", "param"])
        weights = EdgesDF['param'].values
        weights_array = np.concatenate((weights_array, weights))
    #end
    
    weights_array = np.sort(weights_array)
    
    (mu,sigma) = norm.fit(weights_array)
    
    fig,ax = plt.subplots(figsize = (10,5))
    N,bins = np.histogram(weights_array, density = True)
    ax.hist(weights_array,bins = 100, normed = True, alpha = 0.2)
    
    fitted_curve = norm.pdf(weights_array, mu, sigma)
    ax.plot(weights_array, fitted_curve, 'k', lw = 2, alpha = 0.3)
    
    threshold = 0.55 * max(fitted_curve)
    bins_prev = []
    
    for i in range(1, weights_array.size):
        if ((fitted_curve[i-1] <  threshold and fitted_curve[i] >= threshold) or \
            (fitted_curve[i-1] >= threshold and fitted_curve[i] <  threshold)):
            
            bins_prev.append(weights_array[i])
        #end
    #end
    
    min_ = min(weights_array) + 0.1 * min(weights_array)
    max_ = max(weights_array) + 0.1 * max(weights_array)
    third_negative = 1/5. * (min_ - bins_prev[0]) + bins_prev[0]
    third_positive = 1/5. * (max_ - bins_prev[1]) + bins_prev[1]
    
    bins_edges = [min_, third_negative, bins_prev[0], bins_prev[1], third_positive, max_]
    
    N_,bins_hist = np.histogram(weights_array,bins = bins_edges, density = True)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    N__,bins_hist__,patches = ax.hist(weights_array, bins_hist, normed = True, alpha = 0.5)
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
    plt.savefig(path_save_figs + r'\{}\gaussian_fit_weights.png'.format(init_scheme),
                dpi=300, bbox_inches = "tight")
    plt.show()
    
    return bins_edges

#end



def spectrum_discretize(path_in_dir, dataset_id, plot,
                        weighted_graph, write_file,
                        init_scheme, bins_edges):
                        
    """
    This is the core of the procedure: the keras model is turned to
    a graph, via the ``proGraphDataStructure'' module functionalities.
    
    Input:
        ~ path_in_dir               same as above
        ~ dataset_id                string, as above
        ~ plot                      dict, contains bools to instruct the program flow
                                    about whether display graphics or not
        ~ weighted_graph            char, instructs the program flow whether the
                                    graph is treated as weighted or not
        ~ write_file                char, instructs the program flow whether to 
                                    write the graph structure to file or not
        ~ init_scheme               as above
        ~ bins_edges                list of floats, where to place the subdivisions
                                    among categories
                                    
    Returns:
        ~ EdgesDF                   pandas.DataFrame, contains the edges meta-informations
                                    that is the nodes the edge links, the category associated
                                    to each edge, the connection strength. The category
                                    information is used to classify edges among strongly positive
                                    or negative, mildly positive or negative, negligible.
    """
    
    print("\nWeights specturm discretisation of " + dataset_id + " domain")
    
    model = load_model(path_in_dir + r'\{}\model_{}.h5'.format(dataset_id,
                       dataset_id))
    streams.check_create_directory(path_in_dir + r'\images')
    path_save_pic = path_save_figs + r'\{}\{}'.format(init_scheme,dataset_id)
    
    graph = pg.proGraph(model)

    Edges = graph.GetEdges()
    
    EdgesDF = pd.DataFrame.from_dict(Edges, orient = "index",
                                   columns = ["edge", "param"])
    
    weights = np.asarray(EdgesDF["param"])
    
    
    ssp.spectrum_split_plot(weights,path_save_pic,dataset_id,bins_edges)
    
    EdgesDF = ssp.CategoriseWeightsBiases(EdgesDF,bins_edges)
    
    
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
    
    EdgesDF.loc[EdgesDF['cats'] == 4, 'cats'] = 2
    EdgesDF.loc[EdgesDF['cats'] == 3, 'cats'] = 4
    EdgesDF.loc[EdgesDF['cats'] == 5, 'cats'] = 3
    
    
    streams.check_create_directory(path_in_dir + r'\{}'.format(dataset_id))
    filename = path_in_dir + r'\{}\{}_{}_Graph.txt'.format(dataset_id,
                                dataset_id, weighted_graph)
    

    if (write_file == "Y" or write_file == "y"):
        print("Writing Graph File\n")
        
        AdjLists = graph.GetAdjLists()
        with open(filename,'w') as f:
            for _,i in enumerate(AdjLists):
                for j in range(len(AdjLists[i])):
                    
                    l = AdjLists[i][j]
#                    tmp = EdgesDF.loc[EdgesDF["edge"] == (i,l)]
                    
                    # we preventively EXCLUDE the elements which have category 4
                    # that is, weak connections
                    if (int(EdgesDF[EdgesDF['edge'] == (i,l)]['cats']) != 4):
                        
                        if (weighted_graph == "u" or weighted_graph == "U"):
                            f.write("%s %s %s" % (i-1, l-1, 1))
                            f.write("\n")
                        else:
                            f.write("%s %s " % (i-1, l-1))
                            
                            """
                            UNCOMMENT to account for nodes values (colors)
                            """
    #                        f.write("%s %s %s " % (int(NodesDF.loc[i,"cats"]),
    #                                               int(NodesDF.loc[l,"cats"]),
    #                                               int(tmp.at[tmp.index[0],"cats"])))
                            cat = int(EdgesDF[EdgesDF['edge'] == (i,l)]['cats'])
                            f.write("%s "%(cat))
                            f.write("\n")
                        #end
                    #end
                #end
            #end
        f.close()
    #end

    return EdgesDF
    
#end