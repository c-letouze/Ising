#! /usr/bin/env python3
# Coraline Letouzé 
# last revision 1 jan 2021
# CNN.py
# goal : classify the Ising phases with a CNN

################## IMPORTS ################

# numpy for arrays
import numpy as np

# pickle for saving the datasets
import pickle as pkl

# matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm, gray

# machine learning
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping

################### FUNCTIONS ###################""

def train_set(size, path_ordered, path_disordered):
    """
    Build a training set of ordered and disordered samples.
    
    arg size (int): number of samples in the returned training set
    arg path_ordered (str): paths to the ordered, CNN-preprocessed set
    arg path_disordered (str): paths to the disordered, CNN-preprocessed set
    
    return x, y: array of samples, array of features
    """
    # ratio ordered/disordered samples: 7:6
    ordered_size = int(size * 7/13)
    disordered_size = size - ordered_size
    
    with open(path_ordered, 'rb') as file_in:
        x_load, y_load = pkl.load(file_in)
    x_ordered = x_load[:ordered_size]
    y_ordered = y_load[:ordered_size]

    with open(path_disordered, 'rb') as file_in:
        x_load, y_load  = pkl.load(file_in)
    x_disordered = x_load[:disordered_size]
    y_disordered = y_load[:disordered_size]
    
    x = np.concatenate((x_ordered, x_disordered), axis=0)
    y = np.concatenate((y_ordered, y_disordered), axis=0)
    x, y = shuffle(x, y)
    
    return x, y
    
   
def test_set(size, path):
    """
    Build a test set of near-critical samples.
    
    arg size (int): number of samples in the test set
    
    return x, y: array of samples, array of features
    """
    with open(path, 'rb') as file_in:
        x_load, y_load = pkl.load(file_in)
    x = x_load[:size]
    y = y_load[:size]

    return x, y

def plot_metrics(rec, metrics='accuracy', test=False, savepath=''):
    """ 
    Plot the training history of the model.
    
    arg rec: a dictionary returned by model.fit()
    arg metrics: (str) the metrics that will be plotted
    arg test: (bool) if True, also plot the test values
    arg savepath (str): if non-empty, the figure is saved in the 
        specified file
    """
    plt.plot(rec[metrics], 'o--', label='training')
    plt.plot(rec['val_'+metrics], 's--', label='validate')
    if test:
        plt.plot(rec['test_'+metrics], 'x--', label='test')
    plt.title(metrics+" history")
    plt.xlabel("epochs")
    plt.ylabel(metrics)
    plt.grid()
    plt.legend()
    if savepath != '':
        plt.savefig(savepath)
    plt.show()
    plt.clf()
    
def create_CNN(input_shape=(40,40,1), n_Conv2D=[4, 16], kernel=2, 
                learning_rate=1e-4, dropout=0.0, summary=False):
    """ 
    Build and compile the CNN model.
    
    arg input_shape: shape of the input images
    arg n_Conv2D: a list of the number of convolutionnal kernels per layer
    arg learning_rate: learning rate of the optimizer (Adam)
    arg dropout: dropout ratio (float between 0. and 1.0)
    arg summary: (bool) if True, print the model summary
    
    return : the compiled model
    """
    model = Sequential()

    model.add(Conv2D(n_Conv2D[0], kernel_size=(kernel, kernel), activation='relu',
              input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    
    for layer in range(len(n_Conv2D)):
        model.add(Conv2D(n_Conv2D[layer], kernel_size=(kernel, kernel), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))
    
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    loss_fct = keras.losses.BinaryCrossentropy()
    opt = keras.optimizers.Adam(learning_rate = learning_rate)

    model.compile(loss=loss_fct, optimizer='Adam', metrics=['accuracy'])
    
    if summary:
        model.summary()
        print("\n Additional parameters: ")
        print("kernel size: ", kernel)
        print("pooling size: ", 2)
        print("dropout: ", dropout)
        print("optimizer: Adam")
        print("learning rate: ", learning_rate)
        print("loss: BinaryCrossentropy()")
        print()
        
    return model

class Monitor(keras.callbacks.Callback):
    """ Monitor an additional set during training."""
    
    def __init__(self, x, y, name='test'):
        """ 
        Initialization.
        arg dataset: a list [x, y]
        """
        self.x = x
        self.y = y
        self.name = name
        
    def on_epoch_end(self, epoch, logs):
        """ Add the test score to the logs."""
        scores = self.model.evaluate(self.x, self.y, verbose=0) 
        logs[self.name+'_loss'] = scores[0]
        logs[self.name+'_accuracy'] = scores[1]

###################### DATASET ############################

img_row, img_col = 40, 40
input_shape = (img_row, img_col, 1)

build_datasets = True
# if there is no dataset
if build_datasets: 
    print("Building the datasets...")
    # Build train set
    path_ordered = 'ferro_ordered_CNN.pkl'
    path_disordered = 'ferro_disordered_CNN.pkl'
    X, Y = train_set(5000, path_ordered, path_disordered)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
    print("Training samples: ", np.shape(X_train))
    print("Training labels: ", np.shape(Y_train))
    print("Validation samples: ", np.shape(X_val))
    T_train, L_train = Y_train[:, 0], Y_train[:, 1]
    T_val, L_val = Y_val[:, 0], Y_val[:, 1]
    # Build test set
    path_critical = 'ferro_critical_CNN.pkl'
    X_test, Y_test = test_set(1000, path_critical)
    T_test, L_test = Y_test[:, 0], Y_test[:, 1]
    
    # save 
    print("Saving the datasets...")
    dump_train = (X_train, L_train, T_train)  
    with open("datasets_train.pkl", "wb") as file_out:
        pkl.dump(dump_train, file_out)
    dump_val = (X_val, L_val, T_val)
    with open("datasets_val.pkl", "wb") as file_out:
        pkl.dump(dump_val, file_out)
    dump_test = (X_test, L_test, T_test)
    with open("datasets_test.pkl", "wb") as file_out:
        pkl.dump(dump_test, file_out)
        
#if the data sets already exist :          
else:
    # load 
    print("Loading the datasets...")
    with open("datasets_train.pkl", "rb") as file_in:
        load_data = pkl.load(file_in)
    X_train, L_train, T_train = load_data
    with open("datasets_val.pkl", "rb") as file_in:
        load_data = pkl.load(file_in)
    X_val, L_val, T_val = load_data
    with open("datasets_test.pkl", "rb") as file_in:
        load_data = pkl.load(file_in)
    X_test, L_test, T_test = load_data


################### Learning rate ####################
   
study_learning_rate = False
if study_learning_rate:
    
    print("\n Studying the learning rate...")
     
    # Callback
    monitor = Monitor(X_test, L_test)
    stopping = EarlyStopping(monitor='accuracy', patience=2)
    print("Callbacks:")
    print(" - Monitor(X_test, L_test)")
    print(" - EarlyStopping(monitor='accuracy', patience=2)")
                
    # Parameter grid
    lr_list = [1e-5, 1e-4, 1e-3, 1e-2]
    print("Parameters: ", lr_list)
    
    # Runs 
    cv = 5
    print("Number of runs: ", cv)
        
    for lr in lr_list:
        print("\n Learning rate: ", lr)
        acc_val = np.empty((cv,))
        acc_test = np.empty((cv,))
        st_epoch = np.empty((cv,))
        
        # print summary
        model = create_CNN(learning_rate=lr, summary=True)
        
        for i in range(cv):
            model = create_CNN(learning_rate=lr)
            model.fit(X_train, L_train, epochs=15, batch_size=64, 
                        verbose=0, validation_data=(X_val, L_val), 
                        callbacks=[monitor, stopping])
                        
            acc_val[i] = model.evaluate(X_val, L_val, verbose=0)[1]
            acc_test[i] = model.evaluate(X_test, L_test, verbose=0)[1]
            st_epoch[i] = stopping.stopped_epoch
        # statistics
        print("Accuracy on validate: " , np.mean(acc_val), np.std(acc_val))
        print("Accuracy on test: ", np.mean(acc_test), np.std(acc_test))
        print("Stopping epoch: ", np.mean(st_epoch), np.std(st_epoch))
        

###################### GRID SEARCH: ARCHITECTURE ###################
        
search_architecture = False
if search_architecture:
     
    print("\n Search architecture...")
    
    cv = 5
    
    # Filepath 
    filename = 'architecture.out'
    
    #Callback
    monitor = Monitor(X_test, L_test)
    stopping = EarlyStopping(monitor='accuracy', patience=2)
    
    with open(filename, 'w') as file_out:
        file_out.write("Callbacks: \n")
        file_out.write(" - Monitor(X_test, L_test)\n")
        file_out.write("Number of runs: {}\n".format(cv))
        file_out.write(" - EarlyStopping(monitor='accuracy', patience=2)\n")
    
    kernel_list = [2, 4]
    arch_list = [   [1], [2], [4], [6], [8], [16],
                    [1, 8], [4, 4], [8, 1], [8, 16] ]

    for kernel in kernel_list:
        for arch in arch_list:
            print("kernel - arch", kernel, arch)
            with open(filename, 'a') as file_out:
                file_out.write("\n Kernel size: {}\n".format(kernel))
                file_out.write("Architecture: {}\n".format(arch))
            acc_val = np.empty((cv,))
            acc_test = np.empty((cv,))
            st_epoch = np.empty((cv,))
            for i in range(cv):
                model = create_CNN(kernel=kernel, n_Conv2D=arch)
                model.fit(X_train, L_train, epochs=10, batch_size=64, 
                            verbose=0, validation_data=(X_val, L_val), 
                            callbacks=[monitor, stopping])      
                acc_val[i] = model.evaluate(X_val, L_val, verbose=0)[1]
                acc_test[i] = model.evaluate(X_test, L_test, verbose=0)[1]
                st_epoch[i] = stopping.stopped_epoch
            # statistics
            with open(filename, 'a') as file_out:
                file_out.write("Accuracy on validate: {} - {}\n".format(np.mean(acc_val), np.std(acc_val)))
                file_out.write("Accuracy on test: {} - {}\n".format(np.mean(acc_test), np.std(acc_test)))
                file_out.write("Stopping epoch: {} - {}\n".format(np.mean(st_epoch), np.std(st_epoch)))

################## Plot the accuracy history ##################
    
plot_accuracy = False
if plot_accuracy:
    
    print("Plot accuracy...")
    
    # Callback
    monitor = Monitor(X_test, L_test)
    
    # 1 Conv2D
    model = create_CNN(n_Conv2D=[6], kernel=4)
    history = model.fit(X_train, L_train, epochs=5, batch_size=64, 
                    validation_data=(X_val, L_val), callbacks=[monitor])
    rec = history.history
    #plot
    plot_metrics(rec, test=True, savepath='../report/fig/history_accuracy_CNN_2.png')
    # save data
    arr_accuracy = np.column_stack((rec['accuracy'], rec['val_accuracy'], 
                                    rec['test_accuracy']))
    np.savetxt("../report/data/history_accuracy_CNN_2.dat", arr_accuracy, delimiter=' ')
    
############# Plot the fluctuations between models ##########
    
plot_fluctuations = False
if plot_fluctuations:
    
    print("Plot fluctuating models...")
    
    # Callback
    monitor = Monitor(X_test, L_test)
    
    n_epochs = 10
    n_iter = 5
    rec = np.empty((n_epochs, n_iter))
    for i in range(n_iter):
        model = create_CNN(n_Conv2D=[6], kernel=4, learning_rate=1e-5)
        history = model.fit(X_train, L_train, epochs=n_epochs, batch_size=64, 
                    validation_data=(X_val, L_val), callbacks=[monitor])
        rec[:, i] = history.history['test_accuracy']
        plt.plot(history.history['test_accuracy'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Accuracy on the test set for identical models')
    plt.savefig('../report/fig/fluctuations_CNN_3.png')
    plt.show()
    #save data
    np.savetxt('../report/data/fluctuations_CNN_3.dat', rec, delimiter=' ')
