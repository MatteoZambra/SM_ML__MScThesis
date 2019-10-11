

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
#import seaborn as sns
#from scipy.stats import norm, gamma, expon

import motifsreader as mr


ds_dict = {'init' : 'Initial',
           'tree' : 'Tree',
           'clus' : 'Clusters',
           'mvg'  : 'Multitask'}

is_dict = {'orth' : 'Orthogonal',
           'normal' : 'Normal',
           'glorot' : 'Glorot'}



def launcher_weighted_analysis(path_init, path_def, path_figs, datasets,
                               offset, weighted_graph, size, plot, detail):
    
    motifs = []
    variations = []
    most_changed = []
    
    motifs_init = mr.motifs_load(path_init, offset, size)
    motifs_init = motifs_selection(motifs_init, detail)
    motifs_init.rename(columns = {'Zscore' : 'init'},
                          inplace = True)
    
    motifs.append(motifs_init)
    
    for dataset_id in datasets[1:]:
        
        # STEP 0: import motifs as dicts
        path_post = path_def +  \
          r'\{}\{}_{}_s{}_out.csv'.format(dataset_id, dataset_id,
                                          weighted_graph, size)
        motifs_post = mr.motifs_load(path_post, offset, size)
        
        # STEP 1: in case of weighted graph,
        #         pick the MaxSignificanceMotif
        
        motifs_post = motifs_selection(motifs_post, detail)
        motifs_post.rename(columns = {'Zscore' : dataset_id},
                              inplace = True)
        motifs.append(motifs_post)
        
        # STEP 2: scatter plot and selection of 
        #         most changed motifs
        
        scatter_plot(motifs_init, motifs_post,
                     path_figs,
                     dataset_id, size,
                     weighted_graph,
                     plot, detail)
        
        variations_,motifs_most_changed = \
                     variations_distributions(motifs_init,
                                                 motifs_post,
                                                 path_figs,
                                                 dataset_id, size,
                                                 weighted_graph,
                                                 plot, detail)
        variations.append(variations_)
        most_changed.append(motifs_most_changed)
        
    # end
    
    # STEP 3: group data for all datasets
                
    variations = pd.concat(variations, axis = 1, sort = False)
    
    most_changed = pd.concat(most_changed,
                                axis = 1,
                                sort = False)
        
    variations_comparison(most_changed, 
#                                         path_def + r'\images',
                          path_figs,
                          weighted_graph, size, plot, detail)
    
    motifs = pd.concat(motifs, axis = 1, sort = False)
    
    
    return motifs,variations,most_changed
    
#end




def motifs_selection(motifs, detail):
    
    columns = ['motifID', 'Zscore']
    
    selected_motifs = []
    
    for key in motifs:
        
        motif_instances_df = pd.DataFrame(columns = columns)
        max_significance_for_key = pd.DataFrame(columns = columns)
        
        if (len(motifs[key]) == 0):
            i = "sub_000"
            motif_instances_df.at[i,'motifID'] = int(key)
            motif_instances_df.at[i,'Zscore']  = 0.0
        
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


def scatter_plot(motifs_init,motifs_post, path_images, 
                 dataset_id, size, weighted_graph, plot, detail):
    
    keys = motifs_init.index.tolist()
    if (keys != motifs_post.index.tolist()): 
        print("INCONGRUENCE : keys")
        motifs_ = pd.concat([motifs_init, motifs_post], axis = 1, sort = False)
        motifs_ = motifs_.fillna(0)
        zsc_init = motifs_['init'].values.tolist()
        zsc_post = motifs_[dataset_id].values.tolist()
    else:
        zsc_init = motifs_init['init'].values.tolist()
        zsc_post = motifs_post[dataset_id].values.tolist()
    #end
    
    if (min(zsc_init) < 0 or min(zsc_post) < 0):
        minest = min(min(zsc_init), min(zsc_post))
    else:
        minest = 0.0
    #end
    
    maxest = max(max(zsc_init), max(zsc_post))
    bisectrix = np.linspace(minest, maxest, 10)
    
    fig,ax = plt.subplots(figsize=(5,5))
    ax.plot(bisectrix,bisectrix, 'k--', linewidth = 2, alpha = 0.3)
    
    for zi,zp in zip(zsc_init,zsc_post):
        ax.scatter(zi,zp, s = 30, color = 'r', alpha = 0.5)
    #end
    
    bisectrLine = Line2D([],[],color = 'k',
                         linestyle = '--',
                         linewidth = 3,
                         alpha = 0.3,
                         label = 'Bisectrix')
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
    title = 'Z-score scatter-plot. {} domain, {}-vertices motifs'.format(
            ds_dict[dataset_id],str(size))
    plt.title(title)
    if (plot['motifs']):
        if (weighted_graph == 'u'):
            title = path_images + r'\{}_motifsScatter_s{}_u.png'.format(dataset_id,
                                     str(size))
        elif (weighted_graph == 'w'):
            title = path_images + r'\{}_motifsScatter_s{}_w_{}.png'.format(dataset_id, 
                                     str(size), detail)
        #end
        plt.savefig(title, dpi = 300, bbox_inches = 'tight')
    #end
    plt.show()
    
#end


def variations_distributions(motifs_init,motifs_post, path_images, 
                             dataset_id, size, weighted_graph, plot, detail):
    
    if (size == 4):
        num_bins_edges = 10
    elif (size == 6):
        num_bins_edges = 20
    #end
    
    variations_df = pd.DataFrame(columns = [dataset_id])
    
    keys = motifs_init.index.tolist()
    if (keys != motifs_post.index.tolist()): 
        print("INCONGRUENCE : keys")
        motifs_ = pd.concat([motifs_init, motifs_post], axis = 1, sort = False)
        motifs_ = motifs_.fillna(0)
        variations_df[dataset_id] = motifs_[dataset_id] - motifs_['init']
    else:
        variations_df[dataset_id] = motifs_post[dataset_id] - motifs_init['init']
    #end
    
    variations_df = variations_df.sort_values(by = [dataset_id], ascending = False)
    
    values = variations_df.values
    values = values.flatten()
    
#    quantiles = []
#    quantiles.append(np.quantile(values,0.75))
#    if (np.quantile(values,0.25) <= -0.5*quantiles[0]):
#        quantiles.append(np.quantile(values,0.4))
#    #end

    fig,ax = plt.subplots(figsize = (8,3))
    
    bins_edges = np.linspace(min(values),max(values),num_bins_edges)
    
    hist, _ = np.histogram(values, bins = bins_edges, density = False)
    plt.hist(values, bins = bins_edges, alpha = 0.5)

    
    if (size == 4):
        bins_ticks = bins_edges
    elif (size == 6):
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
    title += r'{} domain, {}-vertices motifs'.format(ds_dict[dataset_id],
                                                     str(size))
    plt.title(title)
    if (plot['motifs']):
        if (weighted_graph == 'u'):
            title = path_images + r'\{}_motifsAbsVar_s{}_u.png'.format(dataset_id, 
                                     str(size))
        elif (weighted_graph == 'w'):
            title = path_images + r'\{}_motifsAbsVar_s{}_w_{}.png'.format(dataset_id, 
                                     str(size), detail)
        #end
        plt.savefig(title, dpi=300, bbox_inches = "tight")
    #end
    
    plt.show()

    limits = []
    if (size == 4):
        limits.append(min(values))
    elif (size == 6):
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
        mask = variations_df[dataset_id] >= limits[0]
        motifs_most_changed = variations_df[mask]
    elif (len(limits) == 2):
        mask = (variations_df[dataset_id] >= limits[0]) | \
               (variations_df[dataset_id] <= limits[1])
        motifs_most_changed = variations_df[mask]
    #end
    
    
    return variations_df,motifs_most_changed
    
#end


def variations_comparison(variations, path_images, 
                          weighted_graph, size, plot, detail):
    
    
    variations.rename(columns = {'tree' : 'Tree',
                                 'clus' : 'Clusters',
                                 'mvg'  : 'Multitask'}, inplace = True)
    fig,ax = plt.subplots(figsize = (10,5))
    ax = variations.plot.bar(rot = 0)
    ax.set_xlabel("Motif ID")
    ax.set_ylabel("Absolute Variation")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xticklabels(variations.index.tolist())
    title = 'Absolute Z-score variations, s = {}'.format(size)
    plt.title(title)
    for tk in ax.get_xticklabels():
        tk.set_rotation(45)
    #end
    if (plot['motifs']):
        if (weighted_graph == 'u'):
            title = path_images + r'\all_s{}_u_totZscVar.png'.format(str(size))
        elif (weighted_graph == 'w'):
            title = path_images + r'\all_s{}_w_totZscVar_{}.png'.format(size, detail)
        #end
        plt.savefig(title, dpi=300, bbox_inches = "tight")
    #end
    
    variations = variations.fillna(0)
    
#    plt.rcParams.update(plt.rcParamsDefault)
    fig,ax = plt.subplots(figsize = (5,5))
    cmap = cm.get_cmap('Spectral', 10)
    plt.imshow(variations.corr(), cmap=cmap)
    labels = variations.columns.tolist()
    labels = ['',labels[0],'',labels[1],'',labels[2],'']
    span = np.arange(3)
    ax.xticks = span
    ax.yticks = span
    ax.set_xticklabels(labels,rotation=45, fontsize=10)
    ax.set_yticklabels(labels,rotation=45,fontsize=10)
#    print(ax.get_xticks())
    plt.title('Total variations correlations. s = {}'.format(size))
    plt.colorbar()
    if (plot['motifs']):
        if (weighted_graph == 'u'):
            title = path_images + r'\_s{}_u_cov.png'.format(str(size))
        elif (weighted_graph == 'w'):
            title = path_images + r'\_s{}_w_cov_{}.png'.format(str(size), detail)
        #end
        plt.savefig(title, dpi = 300, bbox_inches = 'tight')
    #end
    plt.show()

#end
    
def LaTeX_source_export(motifs_df, weighted_graph, size, seed, detail, variations):
    """
    for the laziest
    
    It works, but for the ``variations'' case it requires some further refinements
    """
    if (variations):
        num_cols = 3
        cols = 'c | c | c'
        file_title_descr = 'variations'
    else:
        num_cols = 4
        cols = 'c | c | c | c'
        for key in motifs_df:
            motifs_df[key].rename(columns = ds_dict, inplace = True)
        #end
        file_title_descr = 'absvals'
    #end
    
    
    path_latex = # *** absolute path where figures are wanted to be saved ***
    if (weighted_graph == 'u'):
        rest = r'\_seed_{}_s{}_u_latex_source_{}.txt'.format(seed,
                   size, file_title_descr)
    elif (weighted_graph == 'w'):
        rest = r'\_seed_{}_s{}_w_latex_source_{}_{}.txt'.format(seed,
                   size, file_title_descr, detail)
    #end
    
    filename = path_latex + rest
    
    output_source = open(filename,'w')
    
    output_source.write('\\begin{table}\n')
    output_source.write('\\begin{center}\n')
    output_source.write('\t\\begin{tabular}{ ' + cols \
                        + ' }\n\t\t\hline\n\t\t\hline\n')
    
    output_source.write('\t\t')
    for dataset_id in motifs_df['orth'].keys().tolist():
        output_source.write('{} '.format(dataset_id))
        if (dataset_id == 'Multitask'):
            output_source.write('\\\ \n')
        elif (dataset_id == 'Initial' or \
              dataset_id == 'Tree' or \
              dataset_id == 'Clusters'):
            output_source.write('& ')
        #end
    #end
    
    for df_insch in motifs_df:
        
        
        output_source.write('\t\t\hline\n\t\t')
        output_source.write('\multicolumn{' + str(num_cols) + '}{c}{' + \
                            is_dict[df_insch] + '} \\\ \n')
        
        for dataset_id in motifs_df[df_insch]:
            
            df_dataset = motifs_df[df_insch][dataset_id]
            df_dataset = df_dataset.fillna(0)
            df_dataset = df_dataset.sort_values(ascending = False)
            
            
            if (variations):
                keys_list = df_dataset.index.tolist()
                keys_list = keys_list[:2]
                caption_char = '\Delta Z = '
            else:
                keys_list = df_dataset.index.tolist()
                keys_list = keys_list[:2]
                caption_char = 'Z = '
            #end
            
            for key in keys_list:
                if (df_dataset[key] != 0.0):
                    if (key == keys_list[-1]):
                        if (dataset_id == 'Multitask'):
                            sep_char = '\\\ \t'
                        else:
                            sep_char = '&'
                        #end
                    else:
                        sep_char = '~'
                    #end
                    piece1 = '\t\t\includegraphics[scale=0.1]{immagini/simul/motifs/'
                    piece2 = str(key) + '} ' + sep_char + '\t% ' + str(df_dataset[key]) + \
                             ' ' + dataset_id + '\n'
                    output_source.write(piece1 + piece2)
                #end
            #end
            output_source.write('\t\t%end of dataset ' + dataset_id + '\n')
        #end
        
        for dataset_id in motifs_df[df_insch]:
            
            df_dataset = motifs_df[df_insch][dataset_id]
            df_dataset = df_dataset.fillna(0)
            df_dataset = df_dataset.sort_values(ascending = False)
            
            if (variations):
                keys_list = df_dataset.index.tolist()
                keys_list = keys_list[:2]
            else:
                keys_list = df_dataset.index.tolist()
                keys_list = keys_list[:2]
            #end
            
            for key in keys_list:
                if (df_dataset[key] != 0.0):
                    if (key == keys_list[-1]):
                        if (dataset_id == 'Multitask'):
                            sep_char = '\\\ \t'
                        else:
                            sep_char = '&'
                        #end
                    else:
                        sep_char = '~'
                    #end
                    piece1 = '\t\t{0}\scriptsize {1}{2}{3}{4}{5} {6} \n'.format(
                            '{', '$', caption_char, df_dataset[key], '$', '}', sep_char)
                    output_source.write(piece1)
                #end
            #end
            output_source.write('\t\t%end of dataset ' + dataset_id + '\n')
        #end
    #end
    
    output_source.write('\t\end{tabular}\n\end{center}')
    output_source.write('\n\caption{}\n\label{tab:}')
    output_source.write('\n\end{table}')    
    output_source.close()
    
#end


def significance_profiles(all_motifs, path_images,
                          initialisations, datasets,
                          weighted_graph, size, plot,
                          detail, by):
    
    
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
                          'mvg'  : {'color'  : 'b',
                                    'marker' : 'x'}
                          }
    aestetics_initialisations = {'orth'   : {'color'  : 'c',
                                             'marker' : '^'},
                                 'normal' : {'color'  : 'm',
                                             'marker' : 'v'},
                                 'glorot' : {'color'  : 'y',
                                             'marker' : 's'}
                                 }
    
    if (by == 'initialisation'):
        outer_loop       = initialisations
        inner_loop       = datasets
        legend_loop      = initialisations
        legend_dict      = ds_dict
        legend_aestetics = aestetics_datasets
        plot_aestetics   = aestetics_datasets
        title_dict       = is_dict
    elif (by == 'dataset'):
        outer_loop       = datasets
        inner_loop       = initialisations
        legend_loop      = datasets
        legend_dict      = is_dict
        legend_aestetics = aestetics_initialisations
        plot_aestetics   = aestetics_initialisations
        title_dict       = ds_dict
    #end
        
    
    
    for item_outer in outer_loop:
        
        fig,ax = plt.subplots(figsize = (15,4))
        
        for item_inner in inner_loop:
            
            
            if (by == 'initialisation'):
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
            #end
            box = ax.get_position()
            ax.set_position([box.x0, box.y0,
                             box.width * 0.85, box.height]),
            ax.legend(handles = handles,
                      prop = {'size' : 10},
                      loc = 'upper center', bbox_to_anchor = (1.1, 0.9),
                      fancybox = True)
            
            ax.set_ylim((min_of_df-1,max_of_df+1))
            
            if (by == 'initialisation'):
                title = 'Significance profile for fixed {} initialisation scheme'.format(title_dict[item_outer])
            else:
                title = 'Significance profile for fixed {} environment'.format(title_dict[item_outer])
            #end
            
            ax.set_title(title,fontsize = 15)
            ax.set_xticks(xspan)
            ax.set_xticklabels(keys)
            ax.set_xlabel('Motif ID')
            ax.set_ylabel('Significance profile')
            
            if (item_inner == inner_loop[-1] and plot['motifs']):
#                plt.savefig(path_images + r'\s{}_{}_summplotby_{}_{}.png'.format(str(size),
#                            weighted_graph,item_outer,item_inner),
#                            dpi = 300, bbox_inches = 'tight')
                if (weighted_graph == 'u'):
                    title = path_images + r'\noMVG__s{}_u_summplotby_{}_{}.png'.format(str(size),
                            by, item_outer)
                elif (weighted_graph == 'w'):
                    title = path_images + r'\noMVG__s{}_w_summplotby_{}_{}_{}.png'.format(str(size),
                            by, item_outer, detail)
                #end
                plt.savefig(title, dpi = 300, bbox_inches = 'tight')
            #end
        #end
        
    #end
    
    plt.show()
#end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    