


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomNormal, Orthogonal, glorot_normal, Zeros


import PlotFunctions as pfn
import NetworkPlot as npl
import streams 
import pickle

from keras.callbacks import EarlyStopping
from keras.models import load_model

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")


"""
For the sake of simplicity, the path to figures folder
is set to a global variable. The user is free to specify
whatever dicrectory is thought useful
"""


path_save_figs =  r'../figures'
streams.check_create_directory(path_save_figs)


def model_initialisation(N_input, N_classes, initialiser, 
                         seed_value, path_init):
    """
    Model initialization according to a scheme given as input
    
    Input:
        ~ N_input, N_classes        Integers
        ~ initialiser               string, among 'normal', 'orth', 'glorot'
        ~ seed_value                seed for reproducibility, it specifies the
                                    directory in which to store the results
        ~ path_init                 path to store the initialised model
            
    Returns:
        ~ model                     keras.Models.Sequential instance. A linear stack
                                    of layers. Parameters are initialised according
                                    to the scheme specified
                                    
    Note that the task to be performed by the network are straight-forward. An equally
    overall simple model and algorithmic setup suffices to capture the problem complexity
    """
    
    print("\nModel initialisation. Scheme:")
    
    model = Sequential()
    
    if (initialiser == 'orth'):
        print("Orthogonal weights initialisation")
        weights_initializer = Orthogonal(gain = 1.0, seed = seed_value)
    elif (initialiser == 'normal'):
        print("Normal weights initialisation")
        weights_initializer = RandomNormal(mean = 0.0,
                                          stddev = 0.1,
                                          seed = seed_value)
    elif (initialiser == 'glorot'):
        print("Glorot weights initialisation")
        weights_initializer = glorot_normal(seed = seed_value)
    elif (initialiser == 'zeros'):
        weights_initializer = Zeros()
    else:
        print('NO initialiser match')
    #end
    
    model.add(Dense(input_dim = N_input, units = 20,
                kernel_initializer = weights_initializer,
                bias_initializer = RandomNormal(mean = 0.0, 
                                                stddev = 0.1, 
                                                seed = seed_value),
                activation = 'relu'))
    model.add(Dense(input_dim = 20, units = 10,
                kernel_initializer = weights_initializer,
                bias_initializer = RandomNormal(mean = 0.0, 
                                                stddev = 0.1, 
                                                seed = seed_value),
                activation = 'relu'))
                
    model.add(Dense(units = N_classes,
                kernel_initializer = weights_initializer,
                bias_initializer = RandomNormal(mean = 0.0,
                                                stddev = 0.1,
                                                seed = seed_value),
                activation = 'softmax'))
                
    
    """
    The optimization algorithm details are defined here 
    once for all, the model is returned with these details
    embedded yet. Hereafter, in the actual training stage
    it is ready to use with the
    
        model.fit(args)
        
    method
    """
                
    sgd = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, 
                               momentum = 0.6, nesterov = True)
    
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = sgd, metrics = ['accuracy'])
                
    streams.check_create_directory(path_init + r'\init')
    model.save(path_init + r'\init' + r'\model_init.h5')
    return model

#end
    

def load_data(path_data):

    """
    Data sets are generated with two standalone scripts.
    The desired data set is fetched by giving the proper path
    
    Input:
        ~ path_data                 string
        
    Returns:
        ~ X,Y                       numpy.ndarray. Respectively the design matrix
                                    and the labels matrix, encoded as one-hots
    """
    
    fileID = open(path_data, 'rb')
    data_set = pickle.load(fileID)
    fileID.close()
    
    X = data_set[0]
    Y = data_set[1]
    
    return X,Y
#end


def model_training(path_save_model,
                   dataset_id, 
                   split_fraction,
                   plot, init_scheme):
                   
    """
    Proper training stage.
    
    As usual, proper directories are checked and if not present created,
    in order to store the trained model file in a devoted space.
    
    Input:
        ~ path_save_model           string, where to save the model (.h5)
        ~ dataset_id                string, environment specifier
        ~ split_fraction            float, the percentage/100 of held-out
                                    samples to evaluate the model with once trained
        ~ plot                      dict, containing a bool flag, specifying whether
                                    plot meaning figures or not
        ~ init_scheme               string, specifies the initialisation scheme used
        
    Returns:
        ~ model                     This time the keras.Models.Sequential type, instantiated
                                    in the model_initialisation function, is returned with
                                    the parameters adjusted according to the task learned
        ~ history.history['acc']    list, contains the values of the accuracy for each training
                                    epoch, so that it is possible to plot the learning profiles,
                                    if needed
                                    
    Note again that since the tasks are straight-forward, the only anti-overfit measure
    adopted is the early stopping.
    """
    
    print("\nModel Training with " + dataset_id + " data set.\n")
    
    model = load_model(path_save_model + r'\init\model_init.h5')
    params_pre = model.get_weights()
    
    streams.check_create_directory(path_save_model + r'\images')
    path_save_pic = path_save_figs + r'\{}\{}_'.format(init_scheme,dataset_id)
    if (plot['network']):
        plotNet = npl.plotNet(params_pre, path_save_pic, 
                              trained = False, asGraph = False)
        plotNet.plotNetFunction()
    #end
    
    if (dataset_id == 'tree'):
        X,Y = load_data(r'DataSets/TreeLev2_DS_list.pkl')
    elif (dataset_id == 'clus'):
        X,Y = load_data(r'DataSets/Clusters_DS_list.pkl')
    #end
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
            X, Y, test_size = split_fraction, random_state = 20)
    
    es1 = EarlyStopping(monitor='val_acc', 
                        mode='auto', patience = 30, verbose = 0)
    es2 = EarlyStopping(monitor='val_loss', 
                        mode='auto', patience = 20, verbose = 0)
    
    
    history = model.fit(Xtrain, Ytrain, 
                        validation_split = 0.1, 
                        epochs = 100, verbose = 0, 
                        callbacks = [es1,es2])
    
    if (plot['training']):
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1,2,2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(path_save_pic + "performance.png")
        plt.show()
    #end
    
    params_post = model.get_weights()    
    
    print("Model evaluation on test data: loss and accuracy\n",
        model.evaluate(Xtest,Ytest, verbose = 2))
    
    
   if (plot['distributions']):
       numLayers = len(model.layers)
       pfn.jointPlotter(numLayers, path_save_pic, params_pre,
                        params_post, plot_hist = False)
   #end
   if (plot['network']):
       plotNet = npl.plotNet(params_post, path_save_pic, 
                             trained = True, asGraph = False)
       plotNet.plotNetFunction()
   #end
    
   streams.check_create_directory(path_save_model + r'\{}'.format(dataset_id))
   model.save(path_save_model + r'\\' + dataset_id +  \
              r'\\model_' + dataset_id + ".h5")
   model.save(path_save_model + r'\{}\model_{}.h5'.format(dataset_id,
              dataset_id))
    
   return model, history.history['acc']
#end


def model_train_multitask(path_save_model,
                          dataset_id,
                          split_fraction,
                          plot, init_scheme):
                          
    """
    *** D E P R E C A T E D ***
    
    In the original spirit of the work, a multitask training was thought to
    be an interesting feature to analyse. In the full swing of the work by
    Kashtan and Alon ``Spontaneous evolution of modularity and network motifs'',
    PNAS September 27, 2005 102 (39) 13773-13778; https://doi.org/10.1073/pnas.0503610102,
    training was performed pericodically on both the data set, in order to observe
    an hypothetic topological response encoding commonalities and/or differences informations.
    
    
    In the multitask training, the dataset_id flag is set to 'mvg'. The two data sets
    are fetched and training is performed cyclically on both.
    
    The call signature and return type are the same as in the model_training
    function above, modulo differences in the labelling conventions
    """

    print("\n\nMultitask training.\n\n")
    
    model = load_model(path_save_model + r'\init\model_init.h5')
    
    X_1,Y_1 = load_data(r'DataSets/TreeLev2_DS_list.pkl')
    X_2,Y_2 = load_data(r'DataSets/Clusters_DS_list.pkl')
    
    Xtrain_1, Xtest_1, Ytrain_1, Ytest_1 = train_test_split(
            X_1, Y_1, test_size = split_fraction, random_state = 20)
    
    Xtrain_2, Xtest_2, Ytrain_2, Ytest_2 = train_test_split(
            X_2, Y_2, test_size = split_fraction, random_state = 20)
    
    es1 = EarlyStopping(monitor='val_acc', 
                        mode='auto', patience = 30, verbose = 0)
    es2 = EarlyStopping(monitor='val_loss', 
                        mode='auto', patience = 20, verbose = 0)
   
    
    
    params = model.get_weights()
    params_pre = params
    
    streams.check_create_directory(path_save_model + r'\images')
    path_save_pic = path_save_figs + r'\{}\{}_'.format(init_scheme,dataset_id)
    if (plot):
        plotNet = npl.plotNet(params_pre, path_save_pic, 
                              trained = False, asGraph = False)
        plotNet.plotNetFunction()
    #end
    
    for I in range(10):
        
        
        model.set_weights(params)
        model.fit(Xtrain_1, Ytrain_1,
            validation_split = 0.1, epochs = 100, 
            verbose = 0, callbacks = [es1, es2])
        if (I % 2 == 0):
            print("\nSuperepoch: {}, goal 1\n".format(I))
            print("Model evaluation on test data: loss and accuracy : ",
            model.evaluate(Xtest_1,Ytest_1, verbose = 2))
        params = model.get_weights()
        
        
        model.set_weights(params)
        model.fit(Xtrain_2, Ytrain_2,
            validation_split = 0.1, epochs = 100, 
            verbose = 0, callbacks = [es1, es2])
        if (I % 2 == 0):
            print("\nSuperepoch: {}, goal 2".format(I))
            print("Model evaluation on test data: loss and accuracy : ",
            model.evaluate(Xtest_2,Ytest_2, verbose = 2))
        params_post = model.get_weights()
        params = params_post
    #end
    
   if (plot['training']):
       
       numLayers = len(model.layers)
       pfn.jointPlotter(numLayers, path_save_pic, params_pre,
                        params_post, plot_hist = False)
   #end
   if (plot['network']):
       plotNet = npl.plotNet(params_post, path_save_pic, 
                             trained = True, asGraph = False)
       plotNet.plotNetFunction()
   #end
   
   streams.check_create_directory(path_save_model + r'\{}'.format(dataset_id))
   model.save(path_save_model + r'\{}\model_{}.h5'.format(dataset_id,
              dataset_id))
    
    
    return model
#end
