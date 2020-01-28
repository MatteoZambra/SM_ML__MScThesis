

"""
Directories managment,
to lighten the main file
"""


import os 
import pickle


def check_create_directory(path_name):

    """
    If the ``path_name'' directory exists, then nothing happens
    Otherwise ``path_name'' is created
    
    Input:
        ~ path_name         string, directory to create, if not present
        
    Returns:
        nothing
    """
    
    if (not os.path.exists(path_name)):
        os.system('mkdir ' + path_name)
    else:
        """
        directory already present
        """
    #end
#end
    


def hierarchy(seed_value,**flow):
    """
    Given the seed value and the flow dictionary, the directories to
    store results are created.
    
    Input:
        ~ seed_value (integer) : self explanatory
        ~ **flow (dictionary)  : keywords and paths, useful to collect the
                                 details of a single run
                                 
    Returns:
        ~ path_results ... (strings) : paths to the directories of interest
    """
    
    path_results = os.getcwd() + r'\Results'               
    check_create_directory(path_results)
    
    path_results += r'\seed_{}'.format(seed_value)
    if flow['mnist']: path_results += r'\mnist'
    else: path_results += r'\synds' #end
    check_create_directory(path_results)
    
    path_summary_plots = path_results + r'\_summary_plots'
    check_create_directory(path_summary_plots)
    
    path_latex_tables = path_results + r'\_LaTeX_tables'
    check_create_directory(path_latex_tables)
    
    path_serialized_dataframes = path_results + r'\_serialized_dataframes'
    check_create_directory(path_serialized_dataframes)
    
    return path_results, path_summary_plots, path_latex_tables, path_serialized_dataframes
#end


def serialize_dataframes(dataframes, **flow):
    """
    Write the dataframes containing the results of the motifs analyses stage
    to serialized objects (binary format)
    
    Input:
        ~ dataframes (dictionary) : dict of lists of pandas.DataFrame instances.
                                    Dataframes of motifs significance variations
                                    and others 
        ~ **flow (dictionary)     : see above
    
    Returns:
        ~ nothing
    """
    
    if flow['weighted_graph'] == 'w':
        _append = r'{}_s{}_{}.pickle'.format(flow['weighted_graph'], flow['motifs_size'], flow['detail'])
    else:
        _append = r'{}_s{}.pickle'.format(flow['weighted_graph'], flow['motifs_size'])
    #end
    
    for name in dataframes.keys():
        handle = open(flow['path_serialize'] + r'\{}_{}'.format(name,_append), 'wb')
        pickle.dump(dataframes[name], handle)
        handle.close()
    #end
#end

    
