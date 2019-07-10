
"""
Adesso i pesi init sono stati creati, quindi non li tocchiamo più. 
Qui li carichiamo ed effettivamente trainiamo il modello. Poi magari non
serve nemmeno esportare tutto il model daccapo, basterebbero solo i pesi


REMARK: remember SCOOP1718: this code breaks almost all the SE rules.
To improve it, it could be a good idea to create a class: ModelTrain. Then 
two classes which inherit what could be in common and differ in that a 
first one performs singular goal training whilst the second performs mvg

"""

import PlotFunctions as pfn
import pickle

import keras
from keras.callbacks import EarlyStopping

# note: RandomNormal(mean = 0.0, stddev = 0.05, seed = None)
#       Orthogonal(gain = 1.0, seed = None)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")


def model_train(model, X, Y, split_fraction, plots, mvg):
    
    params_pre = model.get_weights()
    
    if (mvg == False):
    
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(
            X, Y, test_size = split_fraction, random_state = 20)
    
        es1 = EarlyStopping(monitor='val_acc', 
                            mode='auto', patience = 30, verbose = 1)
        es2 = EarlyStopping(monitor='val_loss', 
                            mode='auto', patience = 20, verbose = 1)
        
        sgd = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, 
                                momentum = 0.6, nesterov = True)
        
        model.compile(loss = 'categorical_crossentropy', 
                    optimizer = sgd, metrics = ['accuracy'])
        
        history = model.fit(Xtrain, Ytrain, 
                            validation_split = 0.1, 
                            epochs = 100, verbose = 1, 
                            callbacks = [es1,es2])
        
        
        if (plots == True):
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='lower right')
#            plt.show()
            
#            plt.figure(figsize=(6,4))
            plt.subplot(1,2,2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.show()
        #end
        
        params_post = model.get_weights()    
        
        print("Model evaluation on test data: loss and accuracy\n",
            model.evaluate(Xtest,Ytest, verbose = 1))
        
        strPrint = model.evaluate(Xtest,Ytest, verbose = 1)
        strPrint = str(strPrint)
        f = open("train_report.txt","w+")
        f.write(strPrint)
        f.close()
        
        if (plots == True):
        
            numLayers = len(model.layers)
            pfn.jointPlotter(numLayers, params_pre, params_post, plot_hist = True)
            pfn.jointPlotter(numLayers, params_pre, params_post, plot_hist = False)
        #end
        
        model.save("Model/model_trained_sg.h5")
            
        return params_post

    else:
        
        fileID = open(r'DataSets/TreeLev2_DS_list.pkl', 'rb')
        
        DataSet_1 = pickle.load(fileID)
        fileID.close()
        
        fileID = open(r'DataSets/Clusters_DS_list.pkl', 'rb')
        
        DataSet_2 = pickle.load(fileID)
        fileID.close()
        
        X_1 = DataSet_1[0]
        Y_1 = DataSet_1[1]
        
        X_2 = DataSet_2[0]
        Y_2 = DataSet_2[1]
        
        
        Xtrain_1, Xtest_1, Ytrain_1, Ytest_1 = train_test_split(
                X_1, Y_1, test_size = split_fraction, random_state = 20)
        
        Xtrain_2, Xtest_2, Ytrain_2, Ytest_2 = train_test_split(
                X_2, Y_2, test_size = split_fraction, random_state = 20)
        
        es1 = EarlyStopping(monitor='val_acc', 
                            mode='auto', patience = 30, verbose = 1)
        es2 = EarlyStopping(monitor='val_loss', 
                            mode='auto', patience = 20, verbose = 1)
        
        sgd = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, 
                                momentum = 0.6, nesterov = True)
        
        model.compile(loss = 'categorical_crossentropy', 
                    optimizer = sgd, metrics = ['accuracy'])
        
        
        params = params_pre
        
        for I in range(10):
            
            print("\nSuperepoch: {}, goal 1\n".format(I))
            model.set_weights(params)
            model.fit(Xtrain_1, Ytrain_1,
                validation_split = 0.1, epochs = 100, 
                verbose = 0, callbacks = [es1, es2])
            print("Model evaluation on test data: loss and accuracy\n",
            model.evaluate(Xtest_1,Ytest_1, verbose = 1))
            params = model.get_weights()
            
            print("\nSuperepoch: {}, goal 2\n".format(I))
            model.set_weights(params)
            model.fit(Xtrain_2, Ytrain_2,
                validation_split = 0.1, epochs = 100, 
                verbose = 0, callbacks = [es1, es2])
            print("Model evaluation on test data: loss and accuracy\n",
            model.evaluate(Xtest_2,Ytest_2, verbose = 1))
            params_post = model.get_weights()
            
        #end
            
        if (plots == True):
            
            numLayers = len(model.layers)
            pfn.jointPlotter(numLayers, params_pre, params_post, plot_hist = True)
            pfn.jointPlotter(numLayers, params_pre, params_post, plot_hist = False)
        #end
        
        model.save("Model/model_trained_mvg.h5")

        return params_post

    #endif
#enddef