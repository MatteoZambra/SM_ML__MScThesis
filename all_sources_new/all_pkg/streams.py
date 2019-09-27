

"""
Directories managment

In practice, only the check_create_directory function is used.
It checks whether the directory in the argument exists, and if
not that is created
"""


import os 


def check_create_directory(path_name):
    if (not os.path.exists(path_name)):
#        print('Folder {} not there'.format(path_name))
        os.system('mkdir ' + path_name)
#        print('Folder {} created'.format(path_name))
    else:
        """
        directory already present
        """
#        print('Folder {} present'.format(path_name))
    #end
#end
    
def init_directories(seeds):
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

