import numpy as np
import tensorflow as tf
from tensorflow import keras
from orbits import Orbit
from datasplit import create_and_split_data
from sklearn.preprocessing import StandardScaler
import datetime



#%% Importing orbits
orbits = np.load("Spin Axis NN Cal\cleanedOrbitsArrayV6(FinalRange2DataOnly).npy", allow_pickle = True)
x_train_raw, x_test_raw, y_train, y_test, time_train, time_test, sc, axis, training_years, split_time = create_and_split_data(orbits, 1, 10, 1)

#%% Normalizing the data
scaler = StandardScaler() #Scaling the data to a normal distribution
scaler.fit(x_train_raw)
x_train = scaler.transform(x_train_raw)
x_test = scaler.transform(x_test_raw)

#%% Splitting data into sequences
num_samples = 23 #Number of seqeunces to split the data into

def create_sequences(data, seq_length): 
    '''
    Function to split the data into sequences of a given length

    Args:
        data: The data to be split
        seq_length: The number of orbits (samples) to be included in each sequence
    
    Returns:
        sequences: The sequences of data
    '''
    sequences = []
    i = 0
    while i < len(data) - seq_length:
        sequence = data[i : i + seq_length]
        sequences.append(sequence)
        i += seq_length
    return np.array(sequences)

