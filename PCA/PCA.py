#! /usr/bin/env python3
# PCA.py
# Coraline Letouzé
# last revision: 1 jan 2021
# classifying the Ising phases with the first component (magnetization)

################## IMPORTS #################

# array manipulation
import numpy as np
import pickle as pkl

# plot
import matplotlib.pyplot as plt
from matplotlib.cm import coolwarm, viridis, magma

# machine learning
import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping

#################### FUNCTIONS ####################

def load_samples(path, size):
    """ 
    Return a specified number of samples from 'path'. 
    
    arg path: (string), the path to the input file
    arg size: (int) the number of samples to return
    
    return x, y: numpy array of samples, numpy array of features
    """
    with open(path, 'rb') as file_in:
        x_load, y_load = pkl.load(file_in)
    x_load, y_load = shuffle(x_load, y_load)
    x = x_load[:size]
    y = y_load[:size]
    return x, y
    
def build_train_set(size):
    """ 
    Return a set of training samples 
    (from the ferromagnetic Ising model). 
    
    arg size (int): the number of samples in the training set
    return x, y,: numpy array of samples, numpy array of features
    """
    # number of ordered + disordered samples = 130000
    # ratio ordered/disordered samples in Mehta's dataset: 7:6
    # balance the ordered/disordered samples in the training set
    size_ordered = int(size * 7/13)
    size_disordered = size - size_ordered
    #ordered set
    path_ordered = '../DataSets/ordered_set.pkl'
    x_ordered, y_ordered = load_samples(path_ordered, size_ordered)
    #disordered set
    path_disordered = '../DataSets/disordered_set.pkl'
    x_disordered, y_disordered = load_samples(path_disordered, size_disordered)
    # shuffle
    x = np.concatenate((x_ordered, x_disordered), axis=0)
    y = np.concatenate((y_ordered, y_disordered), axis=0)
    x, y = shuffle(x, y)
    return x, y
    
def transform_PCA(x, pca, scaler):
    """ 
    Transform the array x with the fitted PCA and scaler.
    
    arg x: numpy array
    arg pca: a fitted PCA tool
    arg scaler: a fitted scaling tool
    
    return: the transformed array
    """
    return scaler.transform(pca.transform(x))
    
def sigmoid(x, w, i=0):
    """
    Return the sigmoid function evaluated at x
    
    arg x: numpy array or float
    arg w: the logistic regression coefficient
    arg i: the "intercept" coefficient
    
    return: a numpy array or a float, like x
    """
    return 1 / (1 + np.exp( +i +w * x))
    
def create_model(dropout=0.5, learning_rate=1e-4):
	"""
	Build and compile a simple regression DNN."""
	model = Sequential()
	model.add(keras.Input(shape=(3, )))
	
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(dropout))
	
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(dropout))
	
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(dropout))
	
	model.add(Dense(1, activation='linear'))
	
	opt = keras.optimizers.Adam(learning_rate = learning_rate)
	model.compile(loss='mean_squared_error', optimizer=opt,
					metrics=(['accuracy']))
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

######################## MAIN ##############################

# Datasets

ferromagnetic = True

# Ferromagnetic case

if ferromagnetic:
    
    print("-----------------------------------")
    print("\n\n PCA ON THE FERROMAGNETIC SAMPLES \n")
    print("-----------------------------------")

    build_datasets = False
    
    if build_datasets:
        
        print("Building the datasets ...")
    
        # training and validation sets
        training_size = 12000
        X, Y = build_train_set(training_size)
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3)
        T_train, L_train = Y_train[:, 0], Y_train[:, 1]
        T_val, L_val = Y_val[:, 0], Y_val[:, 1]
        print(" - Training samples: ", np.shape(X_train))
        print(" - Training labels: ", np.shape(L_train))
        print(" - Validation samples: ", np.shape(X_val))
        
        #test set
        path_critical = '../DataSets/near_critical_set.pkl'
        test_size = 5000
        X_test, Y_test = load_samples(path_critical, test_size)
        T_test, L_test = Y_test[:, 0], Y_test[:, 1]
        print(" - Test samples: ", np.shape(X_test))
        print(" - Test labels: ", np.shape(L_test))
    
        # save with pickle
        print("Dumping the datasets...")
        with open('PCA_datasets.pkl', 'wb') as file_out:
            dump_datasets = (X_train, X_val, X_test, 
                    T_train, T_val, T_test, 
                    L_train, L_val, L_test)
            pkl.dump(dump_datasets, file_out)
            
    else:
        print("Loading the datasets...")
        with open('PCA_datasets.pkl', 'rb') as file_in:
            load_datasets = pkl.load(file_in)
            X_train, X_val, X_test, T_train, T_val, T_test, \
                L_train, L_val, L_test = load_datasets
    
    # Make a subset for plotting
    make_subset = False
    if make_subset:
        
        print("Building a subset...")
        nb_samples_in_plot = 5000
        # ratio training:test  = 13:3
        nb_samples_from_train = int(nb_samples_in_plot * 13/16)
        nb_samples_from_test = nb_samples_in_plot - nb_samples_from_train
        print("Plotting {} samples from the training set".format(nb_samples_from_train))
        print("Plotting {} samples from the test set".format(nb_samples_from_test))
        X_plot = np.concatenate((X_train[:nb_samples_from_train], 
                                X_test[:nb_samples_from_test]), axis=0)
        T_plot = np.concatenate((T_train[:nb_samples_from_train], 
                                T_test[:nb_samples_from_test]), axis=0)
        X_plot, T_plot = shuffle(X_plot, T_plot)
        
        # save with pickle
        print("Dumping the subset...")
        with open('PCA_subset.pkl', 'wb') as file_out:
            dump_subset = (X_plot, T_plot)
            pkl.dump(dump_subset, file_out)
            
    else:
        
        print("Loading the sub-dataset...")
        with open('PCA_subset.pkl', 'rb') as file_in:
            load_subset = pkl.load(file_in)
            X_plot, T_plot = load_subset

# Antiferromagnetic case 

else:
    
    print("-----------------------------------")
    print("\n\n PCA ON THE ANTI-FERROMAGNETIC SAMPLES \n")
    print("-----------------------------------")
    
    print("Building the datasets ...")

    path_ordered = '../DataSets/anti_ordered2.pkl'
    with open(path_ordered, 'rb') as file_in:
        x_ordered, y_ordered = pkl.load(file_in)
        
    path_disordered = '../DataSets/anti_disordered2.pkl'
    with open(path_disordered, 'rb') as file_in:
        x_disordered, y_disordered = pkl.load(file_in)
        
    X_train = np.concatenate((x_ordered, x_disordered), axis=0)
    Y_train = np.concatenate((y_ordered, y_disordered), axis=0)
    X_train, Y_train = shuffle(X_train, Y_train)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3)
    
    path_critical = '../DataSets/anti_critical2.pkl'
    with open(path_critical, 'rb') as file_in:
        X_test, Y_test = pkl.load(file_in)
    
    L_train = Y_train[:, 1]
    L_val = Y_val[:, 1]
    L_test = Y_test[:, 1]
    
    T_train = Y_train[:, 0]
    T_val = Y_val[:, 0]
    T_test = Y_test[:, 0]
    
    print("Training set (X / L): ", X_train.shape, L_train.shape)
    print("Validation set (X / L): ", X_val.shape, L_val.shape)
    print("Test set (X / L): ", X_test.shape, L_test.shape)
    
    # subset for plot
    T_plot = np.concatenate((T_val, T_test))
    X_plot = np.concatenate((X_val, X_test), axis=0)
    
 

# Principal Component Analysis
print("\n Principal Component Analysis...")

# number of components to keep
components = 3
print(" - Number of components: ", components)

# fit X_train
pca = PCA(n_components=components)
X_train_PCA = pca.fit_transform(X_train)

# Results
print("Explained variance: ", pca.explained_variance_)
print("Explained variance ratio: ", pca.explained_variance_ratio_)
print("Singular values: ", pca.singular_values_)

# plot the component weights
plot_weights = False
if plot_weights:
    
    path_weights_0_plot = "../report/fig/pca_weights_0.png"
    path_weights_1_plot = "../report/fig/pca_weights_1.png"
    path_weights_0_data = "../report/data/pca_weights_0.dat"
    path_weights_1_data = "../report/data/pca_weights_1.dat"
    
    if not ferromagnetic:
        path_weights_0_plot = "../report/fig/pca_weights_0_anti.png"
        path_weights_1_plot = "../report/fig/pca_weights_1_anti.png"
        path_weights_0_data = "../report/data/pca_weights_0_anti.dat"
        path_weights_1_data = "../report/data/pca_weights_1_anti.dat"
        
    # weights of the first component
    plt.imshow(pca.components_[0].reshape(40, 40), cmap=viridis)
    plt.title('First component')
    plt.colorbar(format='%.2e', label='Weights')
    plt.savefig(path_weights_0_plot)
    plt.show()
    plt.clf()

    # weights of the second one
    plt.imshow(pca.components_[1].reshape(40, 40), cmap=magma)
    plt.title('Second component')
    plt.colorbar(format='%.2e',label='Weights')
    plt.savefig(path_weights_1_plot)
    plt.show()
    plt.clf()
    
    # save data
    np.savetxt(path_weights_0_data, pca.components_[0].reshape(40, 40), 
                delimiter=' ')
    np.savetxt(path_weights_1_data, pca.components_[1].reshape(40, 40), 
                delimiter=' ')
    
        
# plot the data along those two components
plot_data_PCA = False
if plot_data_PCA:
    
    path_pca_dataset_plot = "../report/fig/pca_dataset.png"
    path_pca_dataset_data = '../report/data/pca_dataset.dat'
    
    if not ferromagnetic:
        path_pca_dataset_plot = "../report/fig/pca_dataset_anti.png"
        path_pca_dataset_data = '../report/data/pca_dataset_anti.dat'
    
    X_plot_PCA = pca.transform(X_plot)
    plt.figure(figsize=[15, 10])
    plt.scatter(X_plot_PCA[:, 0], X_plot_PCA[:, 1], c=T_plot, s=10)
    plt.colorbar(label='Temperature')
    plt.xlabel("First component")
    plt.ylabel("Second component")
    plt.title("Ferromagnetic samples along the two main components")
    plt.savefig(path_pca_dataset_plot)
    plt.show()
    plt.clf()
    
    #save data 
    arr_pca_dataset = np.column_stack((X_plot_PCA, T_plot))
    np.savetxt(path_pca_dataset_data, arr_pca_dataset, delimiter=' ')

# re-scale the data
scaler = StandardScaler()
X_train_PCA = scaler.fit_transform(X_train_PCA)

# apply to the others sets
X_val_PCA = transform_PCA(X_val, pca, scaler)
X_test_PCA = transform_PCA(X_test, pca, scaler)
X_plot_PCA = transform_PCA(X_plot, pca, scaler)

# the new dataset
plot_PCA_VS_temp = False

if plot_PCA_VS_temp:
    
    path_PCA_VS_temp_plot = "../report/fig/PCA_VS_temp.png"
    path_PCA_VS_temp_data = "../report/data/PCA_VS_temp.dat"
    
    if not ferromagnetic:
        path_PCA_VS_temp_plot = "../report/fig/PCA_VS_temp_anti.png"
        path_PCA_VS_temp_data = "../report/data/PCA_VS_temp_anti.dat"
    
    # plot
    plt.scatter(T_plot, X_plot_PCA[:, 0])
    plt.title("First component w.r.t temperature")
    plt.ylabel("First component")
    plt.xlabel("Temperature")
    plt.savefig(path_PCA_VS_temp_plot)
    plt.show()
    plt.clf()
    
    # save data
    arr_pca = np.column_stack((T_plot, X_plot_PCA[:, 0]))
    np.savetxt(path_PCA_VS_temp_data, arr_pca, delimiter=' ')

# Logistic Regression
do_logreg = False
if do_logreg :
    
    print("\n Logistic Regression...")
    logreg = LogisticRegression()
    logreg.fit(np.abs(X_train_PCA), L_train)
    score_val = logreg.score(np.abs(X_val_PCA), L_val)
    score_test = logreg.score(np.abs(X_test_PCA), L_test)
    print("Mean accuracy of Logistic Regression (before optimization) on: ")
    print(" - Validation set: {:.4f}".format(score_val))
    print(" - Test set: {:.4f}".format(score_test))
    print("with the coefficients: ")
    print(" - coef_: ", logreg.coef_)
    print(" - intercept_:" ,logreg.intercept_)

    # Grid Search
    print("\n Grid Search...")
    parameters = {'C':np.logspace(-1, 2.5)}
    # default C = 1.0
    logReg_estimator = LogisticRegression()
    gs = GridSearchCV(estimator=logReg_estimator, param_grid=parameters)
    gs.fit(np.abs(X_train_PCA), L_train)
    print(" - Best parameter: ", gs.best_params_['C'])

    # plot the GridSearchCV results
    plot_gs = False
    if plot_gs:
        path_gs_plot = "../report/fig/logreg_gridsearch.png"
        if not ferromagnetic:
            path_gs_plot = "../report/fig/logreg_gridsearch_anti.png"
            
        plt.semilogx(parameters['C'], gs.cv_results_['mean_test_score'])
        plt.xlabel("C parameter")
        plt.ylabel("Mean test score")
        plt.title("GridSeachCV results")
        plt.savefig(path_gs_plot)
        plt.show()
        plt.clf()

    #save data
    save_data = False
    if save_data:
        path_gs_results = "../report/data/logreg_gridsearch.dat"
        if not ferromagnetic:
            path_gs_results = "../report/data/logreg_gridsearch_anti.dat"
        arr_gs_results = np.array([parameters['C'], gs.cv_results_['mean_test_score']])
        np.savetxt(path_gs_results, arr_gs_results, delimiter=' ')
            
    # Best estimator
    logreg = LogisticRegression(C=gs.best_params_['C'])
    logreg.fit(np.abs(X_train_PCA), L_train)
    score_val = logreg.score(np.abs(X_val_PCA), L_val)
    score_test = logreg.score(np.abs(X_test_PCA), L_test)
    print("\n Mean accuracy of Logistic Regression (after optimization) on: ")
    print(" - Validation set: {:.4f}".format(score_val))
    print(" - Test set: {:.4f}".format(score_test))
    print("with the coefficients: ")
    print(" - coef_: ", logreg.coef_)
    print(" - intercept_:" ,logreg.intercept_)
    mag = np.linspace(0, np.max(X_train_PCA[:, 0]))
    sig = sigmoid(mag, logreg.coef_[0, 0], logreg.intercept_)
    arr_sigmoid = np.column_stack((mag, sig))

    # plot
    plot = False
    if plot:
        path_logreg_plot = "../report/fig/logistic_reg_PCA.png"
        path_logreg_data = "../report/data/logReg_PCA.dat"
        if not ferromagnetic:
            path_logreg_plot = "../report/fig/logistic_reg_PCA_anti.png"    
            path_logreg_data = "../report/data/logReg_PCA_anti.dat"
        
        fig, ax = plt.subplots()
        ax.scatter(np.abs(X_plot_PCA[:, 0]), T_plot)
        ax.set_xlabel("First component")
        ax.set_ylabel("Temperature")
        ax2 = ax.twinx()
        ax2.plot(mag, sig, 'k')
        ax2.plot([-0.05, np.max(X_train_PCA[:, 0])], [0.5, 0.5], 'k:')
        ax2.set_ylabel("Probability")
        plt.title("Logistic Regression on temperature")
        plt.savefig(path_logreg_plot)
        plt.show()
        plt.clf()
    
        # save data
        np.savetxt(path_logreg_data, arr_sigmoid, delimiter=' ')
    
    
do_temp_regression = True
if do_temp_regression:
    
    
    # Callback   
    monitor = Monitor(np.abs(X_test_PCA), T_test)
    stopping = EarlyStopping(monitor='test_loss', patience=2)
        
    regDNN = create_model()
    history = regDNN.fit(np.abs(X_train_PCA), T_train, epochs=15, batch_size=64,
                    validation_data=(np.abs(X_val_PCA), T_val), 
                    verbose=0, callbacks=[monitor, stopping])
	
	#monitor history
    plt.plot(history.history['loss'], 'o--', label='train')  
    plt.plot(history.history['val_loss'], 's--', label='val')            
    plt.plot(history.history['test_loss'], 'x--', label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Mean squared error history of the model')
    plt.savefig('temp_reg_loss.png')
    plt.show()
    plt.clf()
    
    # plot results
    t_plot_pred = regDNN.predict(np.abs(X_plot_PCA)).reshape((-1, ))
    abs_error = np.abs(np.subtract(T_plot, t_plot_pred))
    plt.scatter(T_plot, t_plot_pred, c=abs_error)
    plt.xlabel("Temperature")
    plt.ylabel("Predicted temperature")
    plt.title("Temperature prediction after a PCA")
    cbar = plt.colorbar()
    cbar.set_label("Absolute error")
    plt.grid()
    plt.savefig('../report/fig/PCA_temp_regression_results.png')
    plt.show()
    
    # plot regression
    plt.scatter(X_plot_PCA[:0], t_plot_pred, c=abs_error)
    plt.xlabel("Firts component (absolute value)")
    plt.ylabel("Predicted temperature")
    plt.title("Temperature regression")
    cbar = plt.colorbar()
    cbar.set_label("Absolute error")
    plt.grid()
    plt.savefig('../report/fig/PCA_temp_regression.png')
    plt.show()
    
