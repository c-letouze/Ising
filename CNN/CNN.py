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
from keras.models import load_model
from keras.callbacks import LearningRateScheduler

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
    
def create_CNN(input_shape=(40,40,1), n_Conv2D=[8, 16, 16], kernel=4, 
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

    model.add(Conv2D(n_Conv2D[0], kernel_size=(kernel, kernel), 
                activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    
    for layer in range(1, len(n_Conv2D)):
        model.add(Conv2D(n_Conv2D[layer], kernel_size=(kernel, kernel), 
                    activation='relu'))
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
        
        
def create_CNN_regressor(input_shape=(40,40,1), n_Conv2D=[8, 16], kernel=4, 
                learning_rate=1e-4, dropout=0.0, summary=False):
    """ 
    Build and compile the CNN model for temperature regression.
    
    arg input_shape: shape of the input images
    arg n_Conv2D: a list of the number of convolutionnal kernels per layer
    arg learning_rate: learning rate of the optimizer (Adam)
    arg dropout: dropout ratio (float between 0. and 1.0)
    arg summary: (bool) if True, print the model summary
    
    return : the compiled model
    """
    model = Sequential()

    model.add(Conv2D(n_Conv2D[0], kernel_size=(kernel, kernel), 
                activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    
    for layer in range(1, len(n_Conv2D)):
        model.add(Conv2D(n_Conv2D[layer], kernel_size=(kernel, kernel), 
                    activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))
    
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    loss_fct = keras.losses.MeanSquaredError()
    opt = keras.optimizers.Adam(learning_rate = learning_rate)

    model.compile(loss=loss_fct, optimizer='Adam', metrics=['mae'])

    return model
    
def scheduler(epoch, lr):
    """ A simple learning_rate Scheduler."""
    if epoch % 2 == 0:
        return lr/10
    else:
        return lr 

###################### DATASET ############################

img_row, img_col = 40, 40
input_shape = (img_row, img_col, 1)

ferromagnetic = True

if ferromagnetic:
    build_datasets = False
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
        
        
# antiferromagnetic ###############################"
else:
    print("Building the ANTI datasets ...")

    path_ordered = './anti_ordered_CNN.pkl'
    with open(path_ordered, 'rb') as file_in:
        x_ordered, y_ordered = pkl.load(file_in)
        
    path_disordered = './anti_disordered_CNN.pkl'
    with open(path_disordered, 'rb') as file_in:
        x_disordered, y_disordered = pkl.load(file_in)
        
    X_train = np.concatenate((x_ordered, x_disordered), axis=0)
    Y_train = np.concatenate((y_ordered, y_disordered), axis=0)
    X_train, Y_train = shuffle(X_train, Y_train)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3)
    L_train, L_val = Y_train[:, 1], Y_val[:, 1]
    
    path_critical = './anti_critical_CNN.pkl'
    with open(path_critical, 'rb') as file_in:
        X_test, Y_test = pkl.load(file_in) 
    L_test = Y_test[:, 1]
        
    print("DONE")
    
    # Callback
    monitor = Monitor(X_test, L_test)
    stopping = EarlyStopping(monitor='val_accuracy', patience=2)
    print("Callbacks:")
    print(" - Monitor(X_test, L_test)")
    print(" - EarlyStopping(monitor='val_accuracy', patience=0)")
    
    #model
    model = create_CNN(n_Conv2D=[8, 16])
    history = model.fit(X_train, L_train, epochs=5, batch_size=64, 
                        verbose=0, validation_data=(X_val, L_val), 
                        callbacks=[monitor, stopping])                    
                
    scores_test = model.evaluate(X_test, L_test, verbose=0)
    scores_val = model.evaluate(X_val, L_val, verbose=0)
    print("Test score:", scores_test)
    print("Val scores:", scores_val)
    
    #plot training history
    plt.plot(history.history['accuracy'], 'o--', label='train')
    plt.plot(history.history['val_accuracy'], 's--', label='val')
    plt.plot(history.history['test_accuracy'], 'x--', label='test')
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.title("Training history on the antiferromagnetic dataset")
    plt.legend()
    plt.savefig("../report/fig/CNN_anti_history.png")
    plt.show()
    
    #plot kernel weights
    model = Sequential()
    layer = Conv2D(6, kernel_size=(3, 3), 
            activation='relu', input_shape=input_shape)
    model.add(layer)
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    loss_fct = keras.losses.BinaryCrossentropy()
    opt = keras.optimizers.Adam()
    model.compile(loss=loss_fct, optimizer='Adam', metrics=['accuracy'])

    model.fit(X_train, L_train, epochs=5, batch_size=64, verbose=1)
    print("Val score: ", model.evaluate(X_val, L_val))
    print("Test score: ", model.evaluate(X_test, L_test))
    
    #https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
    # get filter weights
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters = 6
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # specify subplot and turn of axis
        ax = plt.subplot(2, 3, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(f[:, :, 0], cmap='gray')
    # show the figure
    plt.show()
        
######################### KERNEL WEIGHTS ##################

plot_filters = True
if plot_filters:
    
    model = Sequential()
    layer = Conv2D(6, kernel_size=(3, 3), 
            activation='relu', input_shape=input_shape)
    model.add(layer)
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    loss_fct = keras.losses.BinaryCrossentropy()
    opt = keras.optimizers.Adam()
    model.compile(loss=loss_fct, optimizer='Adam', metrics=['accuracy'])

    model.fit(X_train, L_train, epochs=3, batch_size=64, verbose=1)
    print("Val score: ", model.evaluate(X_val, L_val))
    print("Test score: ", model.evaluate(X_test, L_test))
    
    #https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
    # get filter weights
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters = 6
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # specify subplot and turn of axis
        ax = plt.subplot(2, 3, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(f[:, :, 0], cmap='gray')
    # show the figure
    plt.show()

################## WHERE ARE THE WRONG LABELS ? ########################

wrong_labels = False
if wrong_labels:
    
    # Callback
    monitor = Monitor(X_test, L_test)
    stopping = EarlyStopping(monitor='val_accuracy', patience=0)
    print("Callbacks:")
    print(" - Monitor(X_test, L_test)")
    print(" - EarlyStopping(monitor='val_accuracy', patience=0)")
    
    # model
    model = create_CNN(n_Conv2D=[6])
    history = model.fit(X_train, L_train, epochs=15, batch_size=64, 
                        verbose=0, validation_data=(X_val, L_val), 
                        callbacks=[monitor, stopping])
    scores_test = model.evaluate(X_test, L_test, verbose=0)
    scores_val = model.evaluate(X_val, L_val, verbose=0)
    print("Test score:", scores_test)
    print("Val scores:", scores_val)
    
    L_pred = np.round(model.predict(X_test).reshape((-1, )), 0)
    print("Predicted labels: ", L_pred.shape)
    print(L_pred[:10])
    wrong_pred = np.nonzero(L_test - L_pred)[0]
    print((L_test-L_pred).shape)
    print("Wrong labels: ", wrong_pred.shape)
    tot_wrong = wrong_pred.shape[0]
    print("Total wrong labels: ", tot_wrong)

    wrong_200 = np.nonzero(T_test[wrong_pred] - 2.00)[0]
    print("Wrong labels on T=2.0: ", tot_wrong - wrong_200.shape[0])
    
    wrong_225 = np.nonzero(T_test[wrong_pred] - 2.25)[0]
    print("Wrong labels on T=2.25: ", tot_wrong-wrong_225.shape[0])

    wrong_250 = np.nonzero(T_test[wrong_pred] - 2.50)[0]
    print("Wrong labels on T=2.50: ", tot_wrong - wrong_250.shape[0])

################# Temperature prediction ################
temp_prediction = False
if temp_prediction:
    
    print("Temperature regression with CNN")

    #callback
    schedule = LearningRateScheduler(scheduler)
    stopping = EarlyStopping(monitor='val_mae')
    
    # create and train
    CNNreg = create_CNN_regressor()
    history = CNNreg.fit(X_train, T_train, batch_size=64, epochs=10,
                         callbacks=[schedule, stopping], verbose=0, 
                         validation_data=(X_val, T_val))

    #plot training history
    plt.plot(history.history['loss'], 'o--', label='train')
    plt.plot(history.history['val_loss'], 's--', label='val')
    plt.xlabel("epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training history of the CNN regressor")
    plt.legend()
    plt.savefig("../report/fig/CNNreg_history.png")
    plt.show()
    
    # evaluate on test set
    scores_val = CNNreg.evaluate(X_val, T_val, verbose=0)
    print("Scores on the val set: ", scores_val)
    scores_test = CNNreg.evaluate(X_test, T_test, verbose=0)
    print("Scores on the test set: ", scores_test)
    
    # predict temperature over the whole range
    X_plot = np.concatenate((X_val, X_test), axis=0)
    T_plot = np.concatenate((T_val, T_test), axis=0)
    T_pred = CNNreg.predict(X_plot).reshape(-1, )
    AE_pred = np.abs(T_pred - T_plot)
    #plot 
    plt.scatter(T_plot, T_pred, c=AE_pred)
    plt.colorbar()
    plt.xlabel("Temperature")
    plt.ylabel("Prediction")
    plt.plot(T_plot, T_plot, 'ko--')
    plt.xlim([0, 4])
    plt.ylim([0, 4])
    plt.title("Predictions VS real values")
    plt.savefig("../report/fig/CNN_temp_predictions.png")
    plt.show()

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
    
    n_epochs = 5
    n_iter = 10
    rec = np.empty((n_epochs, n_iter))
    for i in range(n_iter):
        model = create_CNN()
        history = model.fit(X_train, L_train, epochs=n_epochs, batch_size=64, 
                    validation_data=(X_val, L_val), callbacks=[monitor])
        rec[:, i] = history.history['test_accuracy']
        plt.plot(history.history['test_accuracy'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Accuracy on the test set for identical models')
    plt.savefig('../report/fig/fluctuations_CNN_4.png')
    plt.show()
    #save data
    np.savetxt('../report/data/fluctuations_CNN_4.dat', rec, delimiter=' ')
