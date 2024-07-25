#This code uses the other two axes and the temperatures to predict the x offsets with an LSTM model

import numpy as np
import tensorflow as tf
from tensorflow import keras
from orbits import Orbit
from datasplit import x_axis_create_and_split_data
from sklearn.preprocessing import StandardScaler, RobustScaler
import datetime
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
import subprocess
import torch.nn as nn
import torch
import torch.optim as optim
import tqdm
import copy

#%% Importing orbits
orbits = np.load("Spin Axis NN Cal/cleanedOrbitsArrayV6(FinalRange2DataOnly).npy", allow_pickle = True)

#%% Function to create sequences
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

def train_test_model(spacecraft=3, training_years=15, neurons=128, time_steps=3, epochs=100, batch_size=32):
    '''
    Runs everything to train and test the model for a spacecraft and axis
    '''
    axis = 0 #This code only runs on the x axis
    x_train_unscaled, x_test_unscaled, y_train_unscaled, y_test_unscaled, time_train, time_test, sc, axis, training_years, split_time, x_interp_test_unscaled, time_interp_test, y_interp_train, time_interp_train = x_axis_create_and_split_data(orbits, spacecraft, training_years)
    '''
    In this line:
        x_train_unscaled: The training feature set with interpolated results removed
        x_test_unscaled: The testing feature set with interpolated data points removed
        y_train_unscaled: The training label set with interpolated results removed
        y_test_unscaled: The testing label set with interpolated data points removed
        time_train: The time values for the non-interpolated training set
        time_test: The time values for the non-interpolated testing set
        x_interp_test_unscaled: The interpolated data points in the testing set
        time_interp_test: The time values for the interpolated testing set
        y_interp_train: The interpolated data points in the training set; DOES NOT NEED TO BE SCALED BC NOT USED FOR TRAINING OR PREDICTION
        time_interp_train: The time values for the interpolated training set
    '''

    #%% Converting Arrays to Pandas Frames
    x_train_unscaled = pd.DataFrame(x_train_unscaled) #The set of non-interpolated training features for model fitting
    x_train_unscaled.rename(columns={0: 'F74'}, inplace=True)
    x_train_unscaled.rename(columns={1: 'F55'}, inplace=True)
    x_train_unscaled.rename(columns={2: 'Y_Offset'}, inplace=True)
    x_train_unscaled.rename(columns={3: 'Z_Offset'}, inplace=True)

    x_test_unscaled = pd.DataFrame(x_test_unscaled) #The set of non-interpolated testing features for model evaluation
    x_test_unscaled.rename(columns={0: 'F74'}, inplace=True)
    x_test_unscaled.rename(columns={1: 'F55'}, inplace=True)
    x_test_unscaled.rename(columns={2: 'Y_Offset'}, inplace=True)
    x_test_unscaled.rename(columns={3: 'Z_Offset'}, inplace=True)

    x_interp_test_unscaled = pd.DataFrame(x_interp_test_unscaled) #Features of the data points that were originally interpolated in the testing region
    x_interp_test_unscaled.rename(columns={0: 'F74'}, inplace=True)
    x_interp_test_unscaled.rename(columns={1: 'F55'}, inplace=True)
    x_interp_test_unscaled.rename(columns={2: 'Y_Offset'}, inplace=True)
    x_interp_test_unscaled.rename(columns={3: 'Z_Offset'}, inplace=True)

    y_train_unscaled = pd.DataFrame(y_train_unscaled) #The set of non-interpolated training labels for model fitting
    y_train_unscaled.rename(columns={0: 'Offset'}, inplace=True)

    y_interp_train = pd.DataFrame(y_interp_train) 
    y_interp_train.rename(columns={0: 'Offset'}, inplace=True)

    y_test_unscaled = pd.DataFrame(y_test_unscaled) #The set of non-interpolated testing labels for model evaluation
    y_test_unscaled.rename(columns={0: 'Offset'}, inplace=True)

    #%% Scaling the data
    feature_columns = ['F74', 'F55', 'Y_Offset', 'Z_Offset']

    feature_transformer = RobustScaler() #First scaler to transform the features
    feature_transformer = feature_transformer.fit(x_train_unscaled[feature_columns].to_numpy()) #Fit the feature scaler to the training features
    x_train = feature_transformer.transform(x_train_unscaled[feature_columns].to_numpy()) 
    x_test = feature_transformer.transform(x_test_unscaled[feature_columns].to_numpy())
    x_interp_test = feature_transformer.transform(x_interp_test_unscaled[feature_columns].to_numpy())

    label_transformer = RobustScaler() #Second scaler to transform the labels
    label_transformer = label_transformer.fit(y_train_unscaled.to_numpy().reshape(-1, 1)) #Fit the label scaler to the training labels
    y_train = label_transformer.transform(y_train_unscaled.to_numpy().reshape(-1, 1))
    y_test = label_transformer.transform(y_test_unscaled.to_numpy().reshape(-1, 1))

    #%% Creating sequences



  