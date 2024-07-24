#This code uses the other two axes and the temperatures to predict the x offsets

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

orbits = np.load("Spin Axis NN Cal\cleanedOrbitsArrayV6(FinalRange2DataOnly).npy", allow_pickle = True)

#%% Convert to 2D PyTorch tensors
def train_test_model(n_epochs=100, batch_size=10, sc=1, years=15):
    '''
    Runs everything to train and test the model for a spacecraft and axis
    '''
    x_train, x_test, y_train, y_test, time_train, time_test, sc, axis, training_years, split_time, x_interp, time_interp = x_axis_create_and_split_data(orbits, sc, years)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    x_interp = torch.tensor(x_interp, dtype=torch.float32) #The interpolated values to predict

    #%% Define the model
    model = nn.Sequential(
        nn.Linear(4, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

    #%% Loss function and optimiser
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    batch_start = torch.arange(0, len(x_train), batch_size)

    #%% Run training and hold the best model
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

    #%% Print
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    # plt.plot(history)
    # plt.show()

    #%% Plot the model
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        y_interp = model(x_interp)

    plt.figure(figsize=(10, 6))
    plt.scatter(time_train, y_train, label='Training', marker='x')
    plt.scatter(time_interp, y_interp, label='Interpolated', marker='o')
    plt.scatter(time_test, y_test, label='Actual', marker='x')
    plt.scatter(time_test, y_pred, label='Predicted', marker='x')
    plt.axvline(x=split_time, color='r', linestyle='--', label='Split Time')
    plt.xlabel('Time')
    plt.ylabel('Offset')
    plt.title('Cluster {} basic NN {} years {} epochs predictions vs actual'.format(sc+1, years, n_epochs))
    plt.legend()
    plt.savefig("./Outputs_X_Axis/C{}/NN/Basic NN {} years {} epochs.png".format(sc+1, years, n_epochs))


# #%% Loop over all spacecraft
for spacecraft in range(2,4):
    train_test_model(n_epochs=50, batch_size=10, sc=spacecraft, years=10)
    train_test_model(n_epochs=100, batch_size=10, sc=spacecraft, years=10)
    train_test_model(n_epochs=500, batch_size=10, sc=spacecraft, years=10)
    train_test_model(n_epochs=50, batch_size=10, sc=spacecraft, years=15)
    train_test_model(n_epochs=100, batch_size=10, sc=spacecraft, years=15)
    train_test_model(n_epochs=500, batch_size=10, sc=spacecraft, years=15)

#%% Commit to git
# Define the commands to run
# commands = [
#     'git add .',
#     'git commit -m "Auto-commit"',
#     'git push origin main'  # Replace 'main' with your branch name if different
# ]
# def run_command(command):
#     """Run a shell command and print its output."""
#     result = subprocess.run(command, shell=True, capture_output=True, text=True)
#     print(result.stdout)
#     if result.stderr:
#         print(result.stderr)
# def run_commands(commands):
#     for command in commands:
#         run_command(command)
  
# run_commands(commands)
   
