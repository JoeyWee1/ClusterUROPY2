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


fig, axs = plt.subplots(4, 1, figsize=(10, 20))

for spacecraft in range(0, 4):
    x_train_raw, x_test_raw, y_train_raw, y_test_raw, time_train, time_test, sc, axis, training_years, split_time = create_and_split_data(orbits, spacecraft, training_years=0, axis=0)
    
    axs[spacecraft].scatter(time_test, y_test_raw, marker='x')
    axs[spacecraft].set_xlabel('Time')
    axs[spacecraft].set_ylabel('y_test_raw')
    axs[spacecraft].set_title(f'Spacecraft {spacecraft+1}')
    
plt.tight_layout()
plt.show()

