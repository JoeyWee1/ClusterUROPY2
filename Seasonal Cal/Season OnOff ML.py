#This code splits the orbits into seasons based on their datetimes
from datetime import datetime
from orbits import Orbit
import numpy as np
from datasplit import season_create_and_split_data
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import copy


# The times by which to split the orbit
season_dates = [
    datetime(1066, 10, 14, 0, 0, 0),
    datetime(2001, 3, 14, 15, 20, 33),
    datetime(2002, 3, 16, 1, 35, 48),
    datetime(2003, 3, 17, 12, 24, 1),
    datetime(2004, 3, 17, 21, 16, 29),
    datetime(2005, 3, 20, 22, 42, 11),
    datetime(2006, 3, 23, 3, 43, 35),
    datetime(2007, 4, 3, 2, 2, 3),
    datetime(2008, 4, 15, 8, 30, 32),
    datetime(2009, 5, 10, 12, 19, 15),
    datetime(2010, 5, 30, 18, 17, 1),
    datetime(2011, 5, 29, 21, 32, 13),
    datetime(2012, 3, 12, 1, 21, 20),
    datetime(2013, 3, 4, 8, 39, 21),
    datetime(2014, 3, 3, 11, 14, 57),
    datetime(2015, 3, 4, 20, 19, 53),
    datetime(2016, 3, 5, 5, 26, 53),
    datetime(2017, 3, 6, 15, 9, 59),
    datetime(2018, 3, 10, 6, 48, 50),
    datetime(2019, 3, 11, 16, 19, 9),
    datetime(2020, 3, 14, 8, 14, 15),
    datetime(2021, 3, 17, 23, 18, 49),
    datetime(2022, 3, 21, 15, 11, 22),
    datetime(2023, 3, 27, 12, 29, 1),
    datetime.now()
]

#%% Importing orbits
orbits = np.load("Seasonal Cal/cleanedOrbitsArrayV6(FinalRange2DataOnly).npy", allow_pickle = True)
print("Orbits imported")

#%% Splititng orbits into seasons
seasons = [[],[],[],[]] #This will be a 3D array, each subarray of spacecraft will contain subarrays of seasons
for sc in range(0, 4):
    for i in range(1, len(season_dates) - 1):
        season = []
        season_start_date = season_dates[i - 1]
        season_end_date = season_dates[i]
        for j in range(0, len(orbits[sc])):
            orbit_start_date = orbits[sc][j].startTime
            if (orbit_start_date >= season_start_date) and (orbit_start_date < season_end_date):
                season.append(orbits[sc][j])
        seasons[sc].append(season)
print("Orbits split into seasons")

#%% Modifying the seasons using to on-off data
def modify_seasons(seasons):
    '''
    Modifies the data in the seasons to account for the jumps which occur when the instument is restarted

    Args:
        seasons: The seasons to modify

    Returns:
        seasons: The modified seasons
    '''
    #Load the on-off data from a command history of alternating on-off times, taking every other datetime

    differences = [[],[],[],[]]

    command_history = np.load('./Seasonal Cal/command_datetimes.npy', allow_pickle = True)#Gets the command history of the satellites (datetimes of commands alternating starting from an on)
    reset_times = [[],[],[],[]] #Array of every other datetime
    seasonal_reset_times = [[],[],[],[]] #Reset times sorted into seasons

    for sc in range(0, 4):
        for i in range(0,len(command_history[sc])):
            if (i % 2 == 0):
                reset_times[sc].append(command_history[sc][i])
        reset_times[sc].append(datetime.now()) #Adding the datetime of now because we will check for periods with orbits in them to offset them

    #Sort the reset times into seasons for each of the spacecraft
    for sc in range(0, 4):
        for i in range(1, len(season_dates) - 1):
            season_start_date = season_dates[i - 1]
            season_end_date = season_dates[i]
            season = []
            for j in range(0, len(reset_times[sc])):
                if (reset_times[sc][j] >= season_start_date) and (reset_times[sc][j] <= season_end_date):
                    season.append(reset_times[sc][j])
            seasonal_reset_times[sc].append(season) 

    #Split the seasons into periods by reset times
    periodic_seasons = [[],[],[],[]]
    for sc in range(0, 4):
        for season in range(0, len(seasons[sc])):
            periodic_season = []
            for i in range(0, len(seasonal_reset_times[sc][season]) - 1):
                period = []
                period_start = seasonal_reset_times[sc][season][i]
                period_end = seasonal_reset_times[sc][season][i+1]
                for orbit in seasons[sc][season]:
                    if (orbit.startTime >= period_start and orbit.startTime < period_end):
                        period.append(orbit)
                periodic_season.append(period)
            periodic_seasons[sc].append(periodic_season)
   
   #Iterate through the seasons and offset the orbit offsets to account for the jumps after each reset if the following reset period has an orbit in it
    for sc in range(0, 4):
        for season in range(0, len(periodic_seasons[sc])):
            for period in range(0,len(periodic_seasons[sc][season])):
                if (len(periodic_seasons[sc][season][period]) != 0) : #If there are orbits in a period
                    last_orbit = np.nan #If still np.nan later then there is no previous so set the difference to 0
                    for i in range(period, 0): #Get the last orbit of the previous non-empty period
                        if (len(periodic_seasons[sc][season][i]) != 0):
                            last_orbit = periodic_seasons[sc][season][i][-1]
                            break
                    for axis in range(0, 3):
                        if (np.isnan(last_orbit)):
                            offset_difference = 0
                        else:
                            previous_offset = last_orbit.calParams[axis][0][0]
                            current_offset = periodic_seasons[sc][season][period][0].calParams[axis][0][0] #Offset of the first orbit in the period
                            offset_difference = current_offset - previous_offset
                            differences[sc].append(offset_difference) #Add valid differences to array of differences
                            for orbit in periodic_seasons[sc][season][period]:
                                orbit.calParams[axis][0][0] = orbit.calParams[axis][0][0] - offset_difference

    #Remerge the periodic seasons into the original seasons
    remerged_seasons = [[],[],[],[]]
    for sc in range(0, 4):
        for season in range(0, len(seasons[sc])):
            remerged_season = []
            for period in range(0,len(periodic_seasons[sc][season])):
                for orbit in periodic_seasons[sc][season][period]:
                    remerged_season.append(orbit)
            remerged_seasons[sc].append(remerged_season)

    print("Seasons modified")
    return remerged_seasons, differences, seasonal_reset_times
                        
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

    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    x_scaler.fit(x_train_in)
    y_train_in = np.array(y_train_in).reshape(-1, 1)
    y_test_in = np.array(y_test_in).reshape(-1, 1)
    y_scaler.fit(y_train_in)

    x_train_out = x_scaler.transform(x_train_in)
    x_test_out = x_scaler.transform(x_test_in)
    y_train_out = y_scaler.transform(y_train_in)
    y_test_out = y_scaler.transform(y_test_in)

    print("Data scaled")
    return x_train_out, x_test_out, y_train_out, y_test_out, x_scaler, y_scaler

#%% NNCal function
axes = ['X', 'Y', 'Z']
def train_test_model(sc, season, training_years, axis, n_epochs, batch_size):
    if len(remerged_seasons[sc][season]) == 0:
        return
    #Create the sets using datasplit
    x_train_raw, x_test_raw, y_train_raw, y_test_raw, time_train, time_test, sc, axis, training_years, split_time = season_create_and_split_data(remerged_seasons[sc][season], sc, training_years=training_years, axis=axis)
    print(len(x_train_raw))

    if len(x_train_raw) == 0:
        return
    if len(x_test_raw) == 0:
        return
    print("Data created")

    x_train, x_test, y_train, y_test, x_scaler, y_scaler = scale_data(x_train_raw, x_test_raw, y_train_raw, y_test_raw) #The scalers will be used for descaling
    print("Data scaled")

    #Convert data into tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1) #Labels must be in 2D array
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    print("Data converted to tensors")

    #V11
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(64, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(64, 1)
    )
    print("Model defined")

    #Define the loss function and batch start indices
    loss_fn = nn.L1Loss()  # mean absolute error
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    batch_start = torch.arange(0, len(x_train), batch_size)

    #Run training and save best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []

    # for epoch in range(n_epochs):
    #     print(f"Epoch {epoch}")
    #     model.train()
    #     with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
    #         bar.set_description(f"Epoch {epoch}")
    #         for start in bar:
    #             # take a batch
    #             x_batch = x_train[start:start+batch_size]
    #             y_batch = y_train[start:start+batch_size]
    #             # forward pass
    #             y_pred = model(x_batch)
    #             loss = loss_fn(y_pred, y_batch)
    #             # backward pass
    #             optimizer.zero_grad()
    #             loss.backward()
    #             # update weights
    #             optimizer.step()
    #             # print progress
    #             bar.set_postfix(mse=float(loss))

    # Assuming x_train and y_train are your training data tensors
    dataset = TensorDataset(x_train, y_train)
    batch_size = 32  # Ensure batch size is greater than 1
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        model.train()
        with tqdm.tqdm(train_loader, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for x_batch, y_batch in bar:
                # forward pass
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()

        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(x_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    # Restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("MAE: %.2f" % best_mse)
    # print("RMSE: %.2f" % np.sqrt(best_mse))

    #Plot predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test) #Predicting the Z axis offsets using the data, including the y axis offsets
        y_pred = y_scaler.inverse_transform(y_pred)
        y_test = y_scaler.inverse_transform(y_test)
        y_train = y_scaler.inverse_transform(y_train)
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(history)
    # plt.savefig("./Outputs_Z_From_Y_Axis/C{}/NN/History Basic NN {} years {} epochs batch {} v12.2 (Further Reduced Learning Rate).png".format(sc+1, training_years, n_epochs, batch_size))
    # plt.clf()

    plt.figure(figsize=(20, 6))
    plt.scatter(time_train, y_train, label='Training', marker='x')
    plt.scatter(time_test, y_test, label='Calibrated', marker='x', s=5)
    plt.scatter(time_test, y_pred, label='Model Predicted', marker='x', s=5)
    plt.axvline(x=split_time, color='r', linestyle='--', label='Split Time')
    plt.xlabel('Time')
    plt.ylabel('Offset')
    plt.title(f'{axes[axis]}-Axis Basic NN {training_years} years {n_epochs} epochs batch {batch_size}.png')
    plt.legend()
    plt.savefig(f"./Outputs_Season_NN_Cal/C{sc+1}/Flattened Season {season} {axes[axis]}-Axis Basic NN {training_years} years {n_epochs} epochs batch {batch_size}.png")


#%% Plot the offsets of the orbits for each season
remerged_seasons, differences, seasonal_reset_times = modify_seasons(seasons)
seasons = remerged_seasons

for spacecraft in range(0, 4):
    for axis in range (0,3):
        for season in range(0, len(seasons[spacecraft])):
            print(f"Spacecraft {spacecraft+1} Axis {axes[axis]} Season {season+1} Length: {len(seasons[spacecraft][season])}")
            train_test_model(spacecraft, season, 0.5, axis, 1000, 32)
        

            

    



    

