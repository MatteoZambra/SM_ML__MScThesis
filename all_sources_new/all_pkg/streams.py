

"""
Directories managment

In practice, only the check_create_directory function is used.
It checks whether the directory in the argument exists, and if
not that is created
"""


import os 


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
    
def init_directories(seeds):

    """
    *** D E P R E C A T E D ***
    
    Directory are rather initialised in the seeds and inits loops
    in the main script
    """
    
    for seed in seeds:
        directory = '\_seeds\seed_'+str(seed)
        path_to_model = os.getcwd() + r'\Model' + directory
        
        check_create_directory(path_to_model)
        
        names = ['\orth','\\normal','\glorot']
        for name in names:
            path_in_directory = path_to_model + name
            check_create_directory(path_in_directory)
        #end
    #end
#end

