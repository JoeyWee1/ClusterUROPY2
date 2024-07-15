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

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 10

#%% Importing orbits
orbits = np.load("Spin Axis NN Cal\cleanedOrbitsArrayV6(FinalRange2DataOnly).npy", allow_pickle = True)
print("Orbits loaded")
axes = ['x', 'y', 'z']

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

def train_test_model(spacecraft=3, axis= 2, training_years=15, neurons=128, time_steps=24, epochs=100):
  '''
  Runs everything to train and test the model for a spacecraft and axis
  '''
  #x = 0, y = 1, z = 2
  training_years = 15
  x_train_raw, x_test_raw, y_train_raw, y_test_raw, time_train, time_test, sc, axis, training_years, split_time = create_and_split_data(orbits, spacecraft, training_years, axis)
  print("Data split")

  #%% Converting x arrays to pandas frames
  x_train_raw = pd.DataFrame(x_train_raw)
  x_train_raw.rename(columns={0: 'F74'}, inplace=True)
  x_train_raw.rename(columns={1: 'F55'}, inplace=True)
  x_test_raw = pd.DataFrame(x_test_raw)
  x_test_raw.rename(columns={0: 'F74'}, inplace=True)
  x_test_raw.rename(columns={1: 'F55'}, inplace=True)
  y_train_raw = pd.DataFrame(y_train_raw)
  y_train_raw.rename(columns={0: 'Offset'}, inplace=True)
  y_test_raw = pd.DataFrame(y_test_raw)
  y_test_raw.rename(columns={0: 'Offset'}, inplace=True)
  print("Pandas frames created")

  #%% Scaling the data
  feature_columns = ['F74', 'F55']
  feature_transformer = RobustScaler()
  label_transformer = RobustScaler()
  feature_transformer = feature_transformer.fit(x_train_raw[feature_columns].to_numpy())
  label_transformer = label_transformer.fit(y_train_raw.to_numpy().reshape(-1, 1))
  x_train = feature_transformer.transform(x_train_raw[feature_columns].to_numpy()) #Transform training features
  y_train = label_transformer.transform(y_train_raw.to_numpy().reshape(-1, 1)) #Transform training labels
  x_test = feature_transformer.transform(x_test_raw[feature_columns].to_numpy()) #Transform testing features
  y_test = label_transformer.transform(y_test_raw.to_numpy().reshape(-1, 1)) #Transform testing labels
  print("Data scaled")

  #%% Creating sequences
  x_train_sequences, y_train_sequences = create_sequences(x_train, y_train, time_steps=time_steps)
  x_test_sequences, y_test_sequences = create_sequences(x_test, y_test, time_steps=time_steps)

  print("Sequences created")
  print(x_train_sequences.shape, y_train_sequences.shape)

  #%% Creating the model
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

  #%% Training the model
  history = model.fit(
      x_train_sequences, y_train_sequences, 
      epochs=epochs, 
      batch_size=20, 
      validation_split=0.1,
      shuffle=False
  )
  # plt.plot(history.history['loss'], label='train')
  # plt.plot(history.history['val_loss'], label='test')
  # plt.legend()
  # plt.show()

  #%% Testing the model
  y_pred = model.predict(x_test_sequences)
  y_train_inv = label_transformer.inverse_transform(y_train_sequences.reshape(1, -1))
  y_test_inv = label_transformer.inverse_transform(y_test_sequences.reshape(1, -1))
  y_pred_inv = label_transformer.inverse_transform(y_pred)
  plt.scatter(time_train[(time_steps + 1):], y_train_inv.flatten(), label='Training Data', marker='x')
  plt.scatter(time_test[(time_steps + 1):], y_test_inv.flatten(), label='Actual Values', marker='x')
  plt.scatter(time_test[(time_steps + 1):], y_pred_inv.flatten(), label='Test Predictions', marker='x')
  plt.xlabel('Time')
  plt.ylabel('Offset')
  plt.title('{} Neuron Bi-LSTM Cluster {} {}-axis with {} years of training data'.format(neurons ,spacecraft + 1, axes[axis], training_years))
  plt.legend()
  plt.savefig('./Outputs/C{}/{}/{} Neuron Bi-LSTM {} years {} steps {} epochs.png'.format(spacecraft + 1, axes[axis], neurons, training_years, time_steps, epochs))
  
# Loop over all spacecraft
for spacecraft in range(0,4):
  # Loop over x and y axis
  for axis in range(0,3):
    # Call the train_test_model function
    train_test_model(spacecraft=spacecraft, axis=axis, neurons=10, epochs = 50)






