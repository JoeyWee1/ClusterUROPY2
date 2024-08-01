import numpy as np
import tensorflow as tf
from tensorflow import keras
from orbits import Orbit
from datasplit import create_and_split_data, yz_create_and_split_data
from sklearn.preprocessing import StandardScaler, RobustScaler
import datetime
import pandas as pd
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
import subprocess
import torch
import tqdm
import copy
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


#%% Importing orbits
orbits = np.load("Offset NN Cal/cleanedOrbitsArrayV6(FinalRange2DataOnly).npy", allow_pickle = True)

#%% Modify datasets
def modify_datasets(x_in, y_in, time_in):
    '''
    Takes the orbits and modifies the datasets to include the dates, the time since the last orbit, and the derivative of the data. 
    
    Args:
        x_in: The input feature set
        y_in: The input label set
        time_in: The input time set
    
    Returns:
        x_out: The feature array with additional features and removed first orbit
        y_out: The label array with the first orbit removed
        time_out: The time array with the first orbit removed
    '''
    x_out = []
    y_out = []
    time_out = []
    for i in range(1, len(x_in)):
        x_temp = x_in[i]
        # x_temp = [0] * len(x_temp)
        time_diff = time_in[i] - time_in[i - 1]
        time_diff = time_diff.total_seconds()
        # Calculae and append the derivatives
        for j in range(0, len(x_in[i])):
            feature_now = x_in[i][j]
            feature_prior = x_in[i - 1][j]
            derivative = (feature_now - feature_prior) / time_diff
            x_temp.append(derivative)
        #Append the date (day, month, year) and time since last orbit
        x_temp.append(time_in[i].day)
        x_temp.append(time_in[i].month)
        x_temp.append(time_in[i].year)
        x_temp.append(time_diff)
        x_out.append(x_temp)
        # Deal with label and time
        y_out.append(y_in[i])
        time_out.append(time_in[i])
    
    return x_out, y_out, time_out

#%% Scale non-time data
def scale_data(x_train_in, x_test_in, y_train_in, y_test_in):
    '''
    Scales the non-time data using robust scaler

    Args:
        x_train_in: The training feature set
        x_test_in: The testing feature set
        y_train_in: The training label set
        y_test_in: The testing label set

    Returns:
        x_train_out: The scaled training feature set
        x_test_out: The scaled testing feature set
        y_train_out: The training label set
        y_test_out: The testing label set
        x_scaler: The scaler used to scale the x data
        y_scaler: The scaler used to scale the y data
    '''
    x_train_in = np.array(x_train_in)
    x_test_in = np.array(x_test_in)
    train_time_features = x_train_in[:, -4:]
    train_data_features = x_train_in[:, :-4]
    test_time_features = x_test_in[:, -4:]
    test_data_features = x_test_in[:, :-4]

    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    x_scaler.fit(train_data_features)
    y_train_in = np.array(y_train_in).reshape(-1, 1)
    y_test_in = np.array(y_test_in).reshape(-1, 1)
    y_scaler.fit(y_train_in)

    x_train_out = np.concatenate((x_scaler.transform(train_data_features), train_time_features), axis = 1)
    x_test_out = np.concatenate((x_scaler.transform(test_data_features), test_time_features), axis = 1)
    y_train_out = y_scaler.transform(y_train_in)
    y_test_out = y_scaler.transform(y_test_in)

    return x_train_out, x_test_out, y_train_out, y_test_out, x_scaler, y_scaler

#%% Create sequences
def create_sequences(x, y, time_steps = 24):
    '''
    Creates sequences to train model

    Args:
        x: The feature set
        y: The label set
        time_steps: The number of time steps to look back on to predict the next value

    Returns:
        xs: The feature sequences
        ys: The label of the target to predict
    '''
    xs, ys = [], []
    for i in range(len(x) - time_steps - 1):
        x_temp = x[i:(i + time_steps + 1)] #Creates a sequence of time_steps before the target value
        xs.append(x_temp)
        ys.append(y[i + time_steps]) #The target value is the value after the sequence
    return np.array(xs), np.array(ys)

#%%Run things
def train_test_model(sc, training_years, n_epochs, neurons, time_steps):
    #Create the sets using datasplit
    x_train, x_test, y_train, y_test, time_train, time_test, sc, training_years, split_time = yz_create_and_split_data(orbits, sc, training_years) 
    print("Data created")

    #Modify the sets 
    x_train, y_train, time_train = modify_datasets(x_train, y_train, time_train)
    x_test, y_test, time_test = modify_datasets(x_test, y_test, time_test)
    print("Data modified")

    #Scale the data
    x_train, x_test, y_train, y_test, x_scaler, y_scaler = scale_data(x_train, x_test, y_train, y_test) #The scalers will be used for descaling
    print("Data scaled")

    #Creating sequences
    x_train_sequences, y_train_sequences = create_sequences(x_train, y_train, time_steps=time_steps)
    x_test_sequences, y_test_sequences = create_sequences(x_test, y_test, time_steps=time_steps)

    #Creating the model
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional( #Model learns the intricacies of the sequences in both directions
            keras.layers.LSTM( 
                units=neurons, #Number of neurons in the LSTM layer; this can be tuned
                input_shape=(x_train_sequences.shape[1], x_train_sequences.shape[2])
            )
        )
    )
    model.add(keras.layers.Dropout(rate=0.2)) #Dropout layer to prevent overfitting 
    model.add(keras.layers.Dense(units=1)) #Output layer with 1 neuron because this is a regression problem
    model.compile(loss='mean_squared_error', optimizer='adam')
    print("Model created")

    #Training the model
    history = model.fit(
      x_train_sequences, y_train_sequences, 
      epochs= n_epochs, 
      batch_size=20, 
      validation_split=0.1,
      shuffle=False
    )

    #Testing the model
    y_pred = model.predict(x_test_sequences)
    y_train_inv = y_scaler.inverse_transform(y_train_sequences)
    y_test_inv = y_scaler.inverse_transform(y_test_sequences)
    y_pred_inv = y_scaler.inverse_transform(y_pred)

    #Plotting the model and history + saving graphs
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig("/home/joey/Desktop/UROP Y2/ClusterUROPY2-1/Outputs_Z_From_Y_Axis/C{}/LSTM/V1 History LSTM {} neurons {} years {} epochs {} steps.png".format(sc+1, neurons, training_years, n_epochs, time_steps))
    plt.clf()

    plt.figure(figsize=(15, 6))
    plt.scatter(time_train[(time_steps + 1):], y_train_inv.flatten(), label='Training Data', marker='x')
    plt.scatter(time_test[(time_steps + 1):], y_test_inv.flatten(), label='Actual Values', marker='x')
    plt.scatter(time_test[(time_steps + 1):], y_pred_inv.flatten(), label='Test Predictions', marker='x')
    plt.axvline(x=split_time, color='r', linestyle='--', label='Split Time')
    plt.xlabel('Time')
    plt.ylabel('Offset')
    plt.title('Z from Y: Cluster {} LSTM {} neurons {} years {} epochs'.format(sc+1, neurons, training_years, n_epochs))
    plt.legend()
    plt.savefig("/home/joey/Desktop/UROP Y2/ClusterUROPY2-1/Outputs_Z_From_Y_Axis/C{}/LSTM/V1 LSTM {} neurons {} years {} epochs {} steps.png".format(sc+1, neurons, training_years, n_epochs, time_steps))



for i in range(0,4):
    train_test_model(i, 15, 1000, 128, 25)
    train_test_model(i, 15, 1000, 128, 25)