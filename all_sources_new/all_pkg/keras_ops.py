


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


path_save_figs = r'C:\Users\matte\Desktop\MasterThesis\SKRITTURA\immagini\simul'


def model_initialisation(N_input, N_classes, initialiser, 
                         seed_value, path_init):
    
    print("\nModel initialisation. Scheme:")
    
    model = Sequential()
    
    if (initialiser == 'orth'):
        print("Orthogonal weights initialisation")
        weights_initializer = Orthogonal(gain = 1.0, seed = None )#seed_value)
    elif (initialiser == 'normal'):
        print("Normal weights initialisation")
        weights_initializer = RandomNormal(mean = 0.0,
                                          stddev = 0.1,
                                          seed = None )#seed_value)
    elif (initialiser == 'glorot'):
        print("Glorot weights initialisation")
        weights_initializer = glorot_normal(seed = None )#seed_value)
    elif (initialiser == 'zeros'):
        weights_initializer = Zeros()
    else:
        print('NO initialiser match')
    #end
    
    model.add(Dense(input_dim = N_input, units = 20,
#                kernel_initializer = Orthogonal(gain = 1.0, seed = None),
                kernel_initializer = weights_initializer,
                bias_initializer = RandomNormal(mean = 0.0, 
                                                stddev = 0.1, 
                                                seed = None),
                activation = 'relu'))
    model.add(Dense(input_dim = 20, units = 10,
#                kernel_initializer = Orthogonal(gain = 1.0, seed = None),
                kernel_initializer = weights_initializer,
                bias_initializer = RandomNormal(mean = 0.0, 
                                                stddev = 0.1, 
                                                seed = None),
                activation = 'relu'))
                
    model.add(Dense(units = N_classes,
#                kernel_initializer = Orthogonal(gain = 1.0, seed = None),
                kernel_initializer = weights_initializer,
                bias_initializer = RandomNormal(mean = 0.0,
                                                stddev = 0.1,
                                                seed = None),
                activation = 'softmax'))
                
    sgd = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, 
                               momentum = 0.6, nesterov = True)
    
    model.compile(loss = 'categorical_crossentropy', 
                  optimizer = sgd, metrics = ['accuracy'])
                
    streams.check_create_directory(path_init + r'\init')
    model.save(path_init + r'\init' + r'\model_init.h5')
    return model

#end
    


"""
-------------------------------------------------------------------------------
NEW VERSION

More flexibility
-------------------------------------------------------------------------------
"""
def load_data(path_data):
    
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
    
    print("\nModel Training with " + dataset_id + " data set.\n")
    
    model = load_model(path_save_model + r'\init\model_init.h5')
    params_pre = model.get_weights()
    
#    path_save_pic = path_images + r'_' + dataset_id + r'_'
    streams.check_create_directory(path_save_model + r'\images')
#    path_save_pic = path_save_model + r'\images\\' + dataset_id + r'_'
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
    
#    sgd = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, 
#                            momentum = 0.6, nesterov = True)
#    
#    model.compile(loss = 'categorical_crossentropy', 
#                optimizer = sgd, metrics = ['accuracy'])
    
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
    
    
#    if (plot['distributions']):
#        numLayers = len(model.layers)
##        pfn.jointPlotter(numLayers, path_save_pic, params_pre,
##                         params_post, plot_hist = True)
#        pfn.jointPlotter(numLayers, path_save_pic, params_pre,
#                         params_post, plot_hist = False)
#    #end
#    if (plot['network']):
#        plotNet = npl.plotNet(params_post, path_save_pic, 
#                              trained = True, asGraph = False)
#        plotNet.plotNetFunction()
#    #end
    
#    streams.check_create_directory(path_save_model + r'\{}'.format(dataset_id))
#    model.save(path_save_model + r'\\' + dataset_id +  \
#               r'\\model_' + dataset_id + ".h5")
#    model.save(path_save_model + r'\{}\model_{}.h5'.format(dataset_id,
#               dataset_id))
    
#    return model
    return history.history['acc']
#end


def model_train_multitask(path_save_model,
                          dataset_id,
                          split_fraction,
                          plot, init_scheme):

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
    
#    sgd = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, 
#                            momentum = 0.6, nesterov = True)
#    
#    model.compile(loss = 'categorical_crossentropy', 
#                optimizer = sgd, metrics = ['accuracy'])
    
    
    params = model.get_weights()
    params_pre = params
    
#    path_save_pic = path_images + r'_' + dataset_id + r'_'
    streams.check_create_directory(path_save_model + r'\images')
#    path_save_pic = path_save_model + r'\images\\' + dataset_id + r'_'
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
    
#    if (plot['training']):
#        
#        numLayers = len(model.layers)
##        pfn.jointPlotter(numLayers, path_save_pic, params_pre, 
##                         params_post, plot_hist = True)
#        pfn.jointPlotter(numLayers, path_save_pic, params_pre,
#                         params_post, plot_hist = False)
#    #end
#    if (plot['network']):
#        plotNet = npl.plotNet(params_post, path_save_pic, 
#                              trained = True, asGraph = False)
#        plotNet.plotNetFunction()
#    #end
#    
#    streams.check_create_directory(path_save_model + r'\{}'.format(dataset_id))
##    model.save(path_save_model + r'\\' + dataset_id + \
##               r'\\model_' + dataset_id + ".h5")
#    model.save(path_save_model + r'\{}\model_{}.h5'.format(dataset_id,
#               dataset_id))
    
    
    return model
#end
