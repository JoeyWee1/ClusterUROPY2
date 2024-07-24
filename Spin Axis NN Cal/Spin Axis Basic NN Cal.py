#This code learns a basic NN model to predict the offset param

#%% Importing libraries
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from orbits import Orbit
import datetime

#%% Importing orbits
orbits = np.load("Spin Axis NN Cal\cleanedOrbitsArrayV6(FinalRange2DataOnly).npy", allow_pickle = True)

#%% Create features
def create_and_split_data(orbits, sc, training_years, axis): #Axis y = 1, z = 2
    """
    Function to split the data set into testing and training data based on a training length (time)

    Args:
        orbits: The array of orbits
        sc: The spacecraft index
        training_years: The number of years to train on
        axis: The axis to train on

    Returns:
        x_train: The training feature set
        x_test: The testing feature set
        y_train: The training label set
        y_test: The testing label set
        time_train: The training start times
        time_test: The testing start times
        sc: The spacecraft index
        axis: The axis index
        training_years: The number of years to train on
        split_time: The time the data is split on
    """
    x_train = []
    y_train = []
    time_train = [] #Start times of the training orbits
    x_test = []
    y_test = []
    time_test = []
    first_orbit_start_time = datetime.datetime(2000, 8, 24, 8, 56, 52)
    split_time = first_orbit_start_time + datetime.timedelta(seconds = (training_years * 31557600)) #Defines the split time based on the number of training years

    for orbit in orbits[sc]: 
        orbit_start_time = orbit.startTime #Orbit start date to check if it goes into the test or train datasets
        training_period_marker = True; #True indicates that the orbit belongs in the training period. 
        data_cleanliness_marker = True; #True indicates the data is clean.
        # feature_vector = [orbit.F074, orbit.F055, orbit.F047, orbit.F034, orbit.F048]
        feature_vector = [orbit.F074, orbit.F055]
        label = orbit.calParams[axis][0][0]; #Gets the label for the range 2 offset data

        #Check for data cleanliness    
        for i in range (0, len(feature_vector)): #Check if any feature is nan
            if (str(feature_vector[i]) == "nan"):
                data_cleanliness_marker = False
        if (str(label) == "nan"): #Checks the label is nan
            data_cleanliness_marker = False 

        #Classify data into test and train periods
        if (orbit_start_time > split_time): #Find orbits that start after the split time
            training_period_marker = False
        else: 
            pass

        #Add the data to the relevant array if it should be added
        if (data_cleanliness_marker == True): 
            if (training_period_marker == True):
                x_train.append(feature_vector)
                y_train.append(label)
                time_train.append(orbit_start_time)
            else: #The training period marker is false and the data is in the testing period
                x_test.append(feature_vector)
                y_test.append(label)
                time_test.append(orbit_start_time);                
        else: #Else, the data is dirty and isn't added to either array
            pass
    return x_train, x_test, y_train, y_test, time_train, time_test, sc, axis, training_years, split_time

#%% Convert to 2D PyTorch tensors
x_train, x_test, y_train, y_test, time_train, time_test, sc, axis, training_years, split_time = create_and_split_data(orbits, 1, 10, 1)
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

#%% Define the model
model = nn.Sequential(
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
# model = nn.Sequential(
#     nn.Linear(5, 20),
#     nn.ReLU(),
#     nn.Linear(20, 10),
#     nn.ReLU(),
#     nn.Linear(10, 5),
#     nn.ReLU(),
#     nn.Linear(5, 1)
# )

print("Model defined successfully")

#%% Loss function and optimiser
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
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
plt.plot(history)
plt.show()

#%% Make the model predict the offset and plot
def plot_predictions(model, x_test, y_test, time_test, split_time):
    """
    Function to plot the predictions of the model and the actual values against time after the split time

    Args:
        model: The trained model
        x_test: The testing feature set
        y_test: The testing label set
        time_test: The testing start times
        split_time: The time the data is split on
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(time_test, y_test, label='Actual', marker='x')
    plt.scatter(time_test, y_pred, label='Predicted', marker='x')
    plt.axvline(x=split_time, color='r', linestyle='--', label='Split Time')
    plt.xlabel('Time')
    plt.ylabel('Offset')
    plt.title('Predictions vs Actual')
    plt.legend()
    plt.show()

plot_predictions(model, x_test, y_test, time_test, split_time)