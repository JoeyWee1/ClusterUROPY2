import numpy as np
from datasplit import yz_create_and_split_data
from sklearn.preprocessing import RobustScaler
from orbits import Orbit
import datetime
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy

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
        time_diff = time_in[i] - time_in[i - 1]
        time_diff = time_diff.total_seconds()
        # Calculae and append the derivatives
        for j in range(0, len(x_in[i])):
            feature_now = x_in[i][j]
            feature_prior = x_in[i - 1][j]
            derivative = (feature_now - feature_prior) / time_diff
            x_temp.append(derivative)
        #Append the date (day, month, year) and time since last orbit
        # x_temp.append(time_in[i].day)
        # x_temp.append(time_in[i].month)
        # x_temp.append(time_in[i].year)
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


#%% Run things
def train_test_model(sc, training_years, n_epochs, batch_size):
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

    #Convert data into tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1) #Labels must be in 2D array
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    print("Data converted to tensors")

    #Define the model
    #V2
    model = nn.Sequential( 
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

    #V3 try leaky relu
    # Using LeakyReLU activation functions
    # model = nn.Sequential(
    #     nn.Linear(10, 20),
    #     nn.LeakyReLU(),
    #     nn.Linear(20, 10),
    #     nn.LeakyReLU(),
    #     nn.Linear(10, 5),
    #     nn.LeakyReLU(),
    #     nn.Linear(5, 1)
    # )

    #V4 try dropout
    # Adding dropout layers
    # model = nn.Sequential(
    #     nn.Linear(10, 20),
    #     nn.ReLU(),
    #     nn.Dropout(0.5),  # Dropout with 50% probability
    #     nn.Linear(20, 10),
    #     nn.ReLU(),
    #     nn.Dropout(0.5),
    #     nn.Linear(10, 5),
    #     nn.ReLU(),
    #     nn.Linear(5, 1)
    # )


    print("Model defined")

    #Define the loss function and batch start indices
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    batch_start = torch.arange(0, len(x_train), batch_size)

    #Run training and save best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                x_batch = x_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(x_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
    
    #%% Restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))

    #Plot predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test) #Predicting the Z axis offsets using the data, including the y axis offsets
        y_pred = y_scaler.inverse_transform(y_pred)
        y_test = y_scaler.inverse_transform(y_test)
        y_train = y_scaler.inverse_transform(y_train)
    
    plt.figure(figsize=(30, 6))
    plt.scatter(time_train, y_train, label='Training', marker='x')
    plt.scatter(time_test, y_test, label='Calibrated', marker='x', s=5)
    plt.scatter(time_test, y_pred, label='Model Predicted', marker='x', s=5)
    plt.axvline(x=split_time, color='r', linestyle='--', label='Split Time')
    plt.xlabel('Time')
    plt.ylabel('Offset')
    plt.title('Z from Y: Cluster {} basic NN {} years {} epochs batch {}'.format(sc+1, training_years, n_epochs, batch_size))
    plt.legend()
    plt.savefig("./Outputs_Z_From_Y_Axis/C{}/NN/Basic NN {} years {} epochs v2.png".format(sc+1, training_years, n_epochs))

for i in range(0,4):
    train_test_model(i, 15, 500, 10)
    train_test_model(i, 15, 500, 25)

        
