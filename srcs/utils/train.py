


import graphds
import streams

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.initializers import RandomNormal, Orthogonal, glorot_normal
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")



ds_dict = {'init' : 'Initial',
           'tree' : 'Tree',
           'clus' : 'Clusters',
           'mnist': 'MNIST'}




def model_init(init_scheme, **flow):
    """
    Model initialization
    
    Input:
        ~ init_scheme (string) : specifies the initialization scheme to use
        ~ flow (dictionary) : flow control
        
    Returns:
        model (keras.models.Sequential) : initialized model
    """
    
    if flow['mnist']:
        input_dim = (784,)
        output_dim  = 10
    else:
        input_dim = (31,)
        output_dim  = 4
    #end
        
    print('Model ``{}`` initialization'.format(init_scheme))
    
    model = Sequential()
    
    if init_scheme == 'orth':
        weight_init = Orthogonal(gain = 1.0, seed = flow['seed'])
    elif init_scheme == 'normal':
        weight_init = RandomNormal(mean = 0.0, stddev = 0.1, seed = flow['seed'])
    elif init_scheme == 'glorot':
        weight_init = glorot_normal(seed = flow['seed'])
    #end
    
    bias_init = RandomNormal(mean = 0.0, stddev = 0.1, seed = flow['seed'])
    
    model.add(Dense(flow['network'][flow['seed']][0], activation = 'relu', input_shape = input_dim,
                    kernel_initializer   = weight_init,
                        bias_initializer = bias_init))
    
    for hidden_units in flow['network'][flow['seed']][1:]:
        model.add(Dense(hidden_units, activation = 'relu',
                            kernel_initializer = weight_init,
                            bias_initializer   = bias_init))
    #end
    
    model.add(Dense(output_dim, activation = 'softmax',
                    kernel_initializer = weight_init,
                    bias_initializer = bias_init))
    
    sgd = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.6, nesterov = True)
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    
    path_model_initialized = flow['path_output'] + r'\init'
    streams.check_create_directory(path_model_initialized)
    model.save(path_model_initialized + r'\model_init.h5')
    
    return model
#end



def load_synth_data(dataset):
    """
    Loads data set
    
    Input:
        ~ dataset (string) : data set specifier
        
    Returns:
        ~ (Training set)(Test set) (each of which are numpy.ndarray types)
    """ 
    
    if dataset == 'tree':
        path_data = r'DataSets/TreeLev2_DS_list.pkl'
    elif dataset == 'clus':
        path_data = r'DataSets/Clusters_DS_list.pkl'
    #end
    
    fileID = open(path_data, 'rb')
    data_set = pickle.load(fileID)
    fileID.close()
    
    X = data_set[0]
    Y = data_set[1]
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
            X, Y, test_size = 0.3, random_state = 20)
    
    return Xtrain,Ytrain, Xtest,Ytest
#end


def load_mnist_data():
    """
    Same as above but thought to load the MNIST data set, fetched from 
    the keras class
    """
    
    (Xtrain,Ytrain),(Xtest,Ytest) = mnist.load_data()
    
    Xtrain = Xtrain.reshape(60000, 784)
    Xtest  = Xtest.reshape(10000, 784)
    Xtrain = Xtrain.astype('float32')
    Xtest  = Xtest.astype('float32')
    Xtrain /= 255
    Xtest  /= 255
    
    Ytrain = keras.utils.to_categorical(Ytrain, 10)
    Ytest  = keras.utils.to_categorical(Ytest, 10)
    
    return Xtrain,Ytrain, Xtest,Ytest
#end


def model_train(path_results, dataset, init_scheme, **flow):
    """
    Model training with Stochastic Gradient Descent. Note that the 
    batch size is a fiddleable parameter.
    
    Input:
        ~ path_results (string) : main directory, from which the directory
                                  hierarchy stems
        ~ dataset (string) : see above
        ~ init_scheme (string) : see above
        
    Returns:
        ~ model (keras.models.Sequential) : trained model
        ~ history.history['acc'] : training accuracy. It is needed for the 
                                   efficacy plots
    """
    
    if flow['mnist']:
        Xtrain,Ytrain, Xtest,Ytest = load_mnist_data()
        batch_size = 128
    else:
        Xtrain,Ytrain, Xtest,Ytest = load_synth_data(dataset)
        batch_size = 20
    #end
        
    print('Model train with {} data set'.format(dataset))
    
    if path_results == '':
        model = load_model(flow['path_output'] + r'\init\model_init.h5')
    else:
        model = load_model(path_results + r'\{}\init\model_init.h5'.format(init_scheme))
    #end
    
    es1 = EarlyStopping(monitor='val_acc', 
                        mode='auto', patience = 30, verbose = 0)
    es2 = EarlyStopping(monitor='val_loss', 
                        mode='auto', patience = 20, verbose = 0)
    
    history = model.fit(Xtrain, Ytrain, batch_size = batch_size,
                        validation_split = 0.1, epochs = 20, verbose = 0,
                        callbacks = [es1,es2])
    
    
    path_save_model = flow['path_output'] + r'\{}'.format(dataset)
    streams.check_create_directory(path_save_model)
    score = model.evaluate(Xtest,Ytest, verbose = 2)
    scores_log = 'Test loss and accuracy : {:.6f} ; {:.6f} '.format(score[0], score[1])
    
    with open(path_save_model + r'\train_log.txt','w') as f:
        f.write('Training log\n')
        f.write(scores_log)
        print('Saved on log:\n' + scores_log)
    #end
    f.close()
    
    if flow['plot']['train']:
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
        plt.savefig(flow['path_figures'] + r'\{}_{}_performance.png'.format(init_scheme, dataset),
                    dpi=300, bbox_inches = 'tight')
        plt.show()
    #end
    
    if flow['save_model']:
        model.save(path_save_model + r'\model_{}.h5'.format(dataset))
    #end
    
    return model,history.history['acc']
#end
        


def weights_kdes(datasets, init_scheme, **flow):
    """
    Estimated densities plots.
    Such estimates are performed by means of Kernel Density Estimate, see
    https://en.wikipedia.org/wiki/Kernel_density_estimation
    
    Input:
        ~ datasets (list of strings) : see above
        ~ init_scheme (string) : see above
        ~ flow (dictionary) : see above
    
    Returns:
        ~ nothing
    """
    
    path_figures = flow['path_output'] + r'\_Figures'.format(init_scheme)
    model_init = load_model(flow['path_output'] + r'\init\model_init.h5'.format(init_scheme))
    graph_init = graphds.proGraph(model_init)
    Edges_init = graph_init.GetEdges()
    EdgesDF = pd.DataFrame.from_dict(Edges_init, orient = 'index',
                                     columns = ['edge','src','trg', 'param'])
    
    weights_init = np.sort(EdgesDF['param'].values)
    
    for dataset in datasets[1:]:
        
        model = load_model(flow['path_output'] + r'\{}\model_{}.h5'.format(dataset,dataset))
        fig,ax = plt.subplots(figsize=(7.5,4))
        graph = graphds.proGraph(model)
        Edges = graph.GetEdges()
        EdgesDF = pd.DataFrame.from_dict(Edges, orient = 'index',
                                         columns = ['edge','src','trg','param'])
        weights = np.sort(EdgesDF['param'].values)
        bw = 1.06 * weights_init.std() * weights_init.size ** (-1 / 5.)
        sns.kdeplot(weights_init, bw=bw, shade=True, label='Before training')
        bw = 1.06 * weights.std() * weights.size ** (-1 / 5.)
        sns.kdeplot(weights, bw=bw, shade=True, label='After training')
        _title = r'{} domain'.format(ds_dict[dataset])
        ax.set_title(_title)
        ax.set_xlabel('Weights')
        ax.set_ylabel('Estimated density')
        plt.savefig(path_figures + r'\{}_{}_kdeplot.png'.format(init_scheme,dataset),
                    dpi=300, bbox_inches = 'tight')
        plt.show()
        plt.close('all')
    #end
#end


def efficacy_plots(path_results, datasets, initializations, **flow):
    """
    Efficacy plots
    
    Input:
        ~ path_results (string)  : see above
        ~ datasets (list of strings) : see above
        ~ initializations (list of strings) : see above
        
    Returns:
        ~ nothing
    """
    
    flow['save_model'] = False
    for dataset in datasets[1:]:
        histories = []
        plt.figure(figsize=(7.5,4))
        for init_scheme in initializations:
            model,hist = model_train(path_results, dataset, init_scheme, **flow)
            histories.append(hist)
        #end
        for hist in histories:
            plt.plot(np.arange(0,10),hist[:10],lw = 2, alpha = 0.75)
            plt.xticks(np.arange(0,10))
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training efficacy')
            plt.legend(['Normal','Orthogonal','Glorot'])
            
        #end
        plt.savefig(flow['path_splots'] + r'\efficacy_{}.png'.format(dataset), dpi = 300)
        plt.show()
    #end
    
#end



























    
    