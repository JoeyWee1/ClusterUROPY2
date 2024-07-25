#This code plots the features (x, y axes and temperatures) alongside offsets (See notes July 16)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from orbits import Orbit
from datasplit import create_and_split_data
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

#%% Importing orbits
orbits = np.load("Spin Axis NN Cal\cleanedOrbitsArrayV6(FinalRange2DataOnly).npy", allow_pickle = True)

#%% Getting the first offset value in each step

def find_calibrated_offsets(orbits, spacecraft_index):
    '''
    The offsets for the x-axis are only calibrated occasionally.
    Orbits that are not calibrated are given the offset of the last calibrated orbit.
    This code will get orbits with calibrated x-axis offsets by finding the first orbit with that offset.

    Args:
        orbits: numpy array of orbits
        spacecraft: int, spacecraft index (0-3)
    
    Returns:
        calibrated_offsets: list of orbits with calibrated offsets for the chosen spacecraft
    '''
    calibrated_offsets = []
    spacecraft_orbits = orbits[spacecraft_index]
    last_offset = spacecraft_orbits[0].calParams[0][0][0] #0th axis (x), range 2, offset
    calibrated_offsets.append(spacecraft_orbits[0])
    for i in range(1, len(spacecraft_orbits)):
        if spacecraft_orbits[i].calParams[0][0][0] != last_offset:
            calibrated_offsets.append(spacecraft_orbits[i])
            last_offset = spacecraft_orbits[i].calParams[0][0][0]
    
    return calibrated_offsets

#%% Testing code 
calibrated_orbits = [find_calibrated_offsets(orbits, i) for i in range(4)]
lengths = [len(calibrated_orbits[i]) for i in range(4)]
print(lengths)

for spacecraft in range(0, 4):
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))

    x_train_raw, x_test_raw, y_train_raw, y_test_raw_x, time_train, time_test, sc, axis, training_years, split_time = create_and_split_data(calibrated_orbits, spacecraft, training_years=0, axis=0)
    x_train_raw, x_test_raw, y_train_raw, y_test_raw_y, time_train, time_test, sc, axis, training_years, split_time = create_and_split_data(calibrated_orbits, spacecraft, training_years=0, axis=1)
    x_train_raw, x_test_raw, y_train_raw, y_test_raw_z, time_train, time_test, sc, axis, training_years, split_time = create_and_split_data(calibrated_orbits, spacecraft, training_years=0, axis=2)

    axs[0].scatter(time_test, y_test_raw_x, marker='x', label='x', color='blue')
    axs[1].scatter(time_test, y_test_raw_y, marker='o', label='y', color='green')
    axs[2].scatter(time_test, y_test_raw_z, marker='o', label='z', color='red')
    axs[3].scatter(time_test, [temperature[0] for temperature in x_test_raw], marker='o', label='F074', color='purple') 
    axs[4].scatter(time_test, [temperature[1] for temperature in x_test_raw], marker='o', label='F055', color='orange')
    axs[2].set_xlabel('Time')
    axs[0].set_title(f'Spacecraft {spacecraft+1}')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()    
    axs[3].legend()
    axs[4].legend()
    plt.tight_layout()
    plt.show()

