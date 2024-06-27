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

#%% Importing orbits
orbits = np.load("Spin Axis NN Cal\cleanedOrbitsArrayV6(FinalRange2DataOnly).npy", allow_pickle = True)

#%% Create features
def create_and_split_data(orbits, sc_index, test_size):
    """
    Creates the features and labels from the orbits data and splits it into training and testing sets

    Args:
        orbits: The array of orbits
        sc_index: The spacecraft index
        test_size: The size of the test set as a fraction of the total data

    Returns:
        x_train: The training feature set
        x_test: The testing feature set
        y_train: The training label set
        y_test: The testing label set
        sc_index: The spacecraft index
    """
    feature_vectors = []
    offsets = [] #These two arrays are index matched
    for orbit in orbits[sc_index]: #Loop through every orbit for the spacecraft
        data_cleanliness = True #Stores if the data is clean and whether or not it should be added to the arrays
        feature_vector = [orbit.F074, orbit.F055, orbit.F034, orbit.F047, orbit.F048] #For each orbit, create a feature vector composed of the spacecraft/instrument telems
        offset = orbit.calParams[2][0][0] #Gets the z axis range 2 offset
        for i in range (0, len(feature_vector)): #Looping through the coordinates in the feature vector to check for nan values for removal
            if (str(feature_vector[i]) =="nan"): #Checking for the nan values
                data_cleanliness = False #Setting cleanliness to false, so the data is not appended
        if (str(offset) == "nan"):#Checks the y values for nan
            data_cleanliness = False #Data is not appended
        if (data_cleanliness == True):
            feature_vectors.append(feature_vector)
            offsets.append(offset)
        else:
            pass
    x_train, x_test, y_train, y_test = train_test_split(feature_vectors, offsets, test_size = test_size)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), sc_index

#%% Convert to 2D PyTorch tensors
x_train, x_test, y_train, y_test, sc_index = create_and_split_data(orbits, 0, 0.3)
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

#%% Define the model
model = nn.Sequential(
    nn.Linear(5, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
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