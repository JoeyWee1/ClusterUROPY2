#This program compares all the models for all the axes of all the spacecrafts
#It split each dataset and then optimises the model hyperparameters for KNN, SVR, NN, and LSTM
#It will output a set of minimum mean squared errors and plot the four models and the actual test data in time series

#%% Import modules and data
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import json
from orbits import Orbit #The orbit class stores all the instrument calibration and telemetry for one orbit
from datasplit import create_and_split_data #This function creates and splits the data for the models
print("Modules imported")

orbits = np.load("Offset NN Cal/cleanedOrbitsArrayV6(FinalRange2DataOnly).npy", allow_pickle=True) 
print("Orbits loaded")

#%% Stuff outside the loop
min_error = np.zeros((4, 3, 4)) #Spacecraft, axis, model (KNN, SVR, NN, LSTM)
min_error.fill(np.inf)
# best_predictions = np.zeros((4, 3, 4)) 
# test_times = [[[[],[],[],[]], [[],[],[],[]], [[],[],[],[]]], [[[],[],[],[]], [[],[],[],[]], [[],[],[],[]]], [[[],[],[],[]], [[],[],[],[]], [[],[],[],[]]], [[[],[],[],[]], [[],[],[],[]], [[],[],[],[]]]]
# test_times = [[[],[]], [[],[]], [[],[]]], [[[],[]], [[],[]], [[],[]]], [[[],[]], [[],[]], [[],[]]]
# train_times = [[[],[]], [[],[]], [[],[]]], [[[],[]], [[],[]], [[],[]]], [[[],[]], [[],[]], [[],[]]]
# knn_best_predictions = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]]
# svr_best_predictions = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]]
# nn_best_predictions = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]]
# lstm_best_predictions = [[[],[],[]], [[],[],[]], [[],[],[]], [[],[],[]]]
training_years = 15
print("Variables initialised")

#%% Loop!

for sc in range(0,4):
    for axis in range(0,3):
        print("Spacecraft: ", sc, "Axis: ", axis)
        lengths = [0,0,0,0]
        knn_best_predictions = []
        svr_best_predictions = []
        nn_best_predictions = []
        lstm_best_predictions = []

        ##Split the data
        #Regular features and labels
        x_train, x_test, y_train, y_test, time_train, time_test, split_time = create_and_split_data(orbits, sc, axis, training_years)
        # test_times[sc][axis][0] = time_test
        # train_times[sc][axis][0] = time_train

        #Create and split sequences
        time_steps = 10

        sequence_scaler = RobustScaler()
        x_train_scaled = sequence_scaler.fit_transform(x_train)
        x_test_scaled = sequence_scaler.transform(x_test)

        label_scaler = RobustScaler()
        y_train_scaled = label_scaler.fit_transform(np.array(y_train).reshape(-1, 1))
        y_test_scaled = label_scaler.transform(np.array(y_test).reshape(-1, 1))

        x_train_sequences, y_train_sequences = [], []
        x_test_sequences, y_test_sequences = [], []

        sequences_time_train = []
        sequences_time_test = []

        for i in range(0, len(x_train_scaled) - time_steps - 1):
            x_train_sequences.append(x_train_scaled[i:i+time_steps+1]) #Creates the sequence of feature vectors
            y_train_sequences.append(y_train_scaled[i+time_steps])
            sequences_time_train.append(time_train[i+time_steps])
        print(f"Training sequences {len(x_train_sequences)}")

        for i in range(0, len(x_test_scaled) - time_steps - 1):
            x_test_sequences.append(x_test_scaled[i:i+time_steps+1])
            y_test_sequences.append(y_test_scaled[i+time_steps])
            sequences_time_test.append(time_test[i+time_steps])
        print(f"Testing sequences {len(x_test_sequences)}")

        # test_times[sc][axis][1] = sequences_time_test
        # train_times[sc][axis][1] = sequences_time_train

        x_train_sequences = np.array(x_train_sequences)
        y_train_sequences = np.array(y_train_sequences)
        x_test_sequences = np.array(x_test_sequences)
        y_test_sequences = np.array(y_test_sequences)
        
      
        print("Data split")

        ##KNN
        for k in range(1, 2): #Test k values from 1 to 10
            print("Spacecraft: ", sc, "Axis: ", axis, "Model: KNN", "k: ", k)
            knn_model = KNeighborsRegressor(n_neighbors=k, weights='distance') 
            knn_model.fit(x_train, y_train)
            y_pred = knn_model.predict(x_test)
            knn_error = np.mean((y_pred - y_test)**2)
            if (knn_error < min_error[sc][axis][0]):
                min_error[sc][axis][0] = knn_error
                knn_best_predictions = y_pred
                lengths[0] = len(y_pred)

        ##SVR
        c_range = np.linspace(0.1, 10, 1)
        for c in c_range:
            print("Spacecraft: ", sc, "Axis: ", axis, "Model: SVR", "C: ", c)
            svr_model = SVR(kernel='rbf', C=c, gamma='auto') #Automatically scale the gamma parameter
            svr_model.fit(x_train, y_train)
            y_pred = svr_model.predict(x_test)
            svr_error = np.mean((y_pred - y_test)**2)
            if (svr_error < min_error[sc][axis][1]):
                min_error[sc][axis][1] = svr_error
                svr_best_predictions = y_pred
                lengths[1] = len(y_pred)

        ##NN
        x_train_tensor_unscaled = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor_unscaled = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        x_test_tensor_unscaled = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor_unscaled = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        # Scale the data
        feature_scaler = RobustScaler()
        x_train_tensor = feature_scaler.fit_transform(x_train_tensor_unscaled)
        x_test_tensor = feature_scaler.transform(x_test_tensor_unscaled)
        label_scaler = RobustScaler()
        y_train_tensor = label_scaler.fit_transform(y_train_tensor_unscaled)
        y_test_tensor = label_scaler.transform(y_test_tensor_unscaled)

        x_train_tensor = torch.tensor(x_train_tensor, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_tensor, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test_tensor, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_tensor, dtype=torch.float32)        

        model = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

        loss_fn = nn.MSELoss()  # mean square error
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        n_epochs = 1   # number of epochs to run
        batch_size = 10  # size of each batch
        batch_start = torch.arange(0, len(x_train), batch_size)

        best_weights = None
        
        for epoch in range(n_epochs):
            print("Spacecraft: ", sc, "Axis: ", axis, "Model: NN", "Epoch: ", epoch)
            model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    #Take a batch
                    x_batch = x_train_tensor[start:start+batch_size]
                    y_batch = y_train_tensor[start:start+batch_size]
                    #Forward pass
                    y_pred = model(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    #Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    #Update weights
                    optimizer.step()
                    #Print progress
                    bar.set_postfix(mse=float(loss))

            #Evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(x_test_tensor)
            y_pred_unscaled = label_scaler.inverse_transform(y_pred.detach().numpy())
            y_pred_unscaled_tensor = torch.tensor(y_pred_unscaled, dtype=torch.float32)
            NN_error = loss_fn(y_pred_unscaled_tensor, y_test_tensor_unscaled)
            NN_error = float(NN_error)
            if NN_error < min_error[sc][axis][2]:
                min_error[sc][axis][2] = NN_error
                nn_best_predictions = y_pred_unscaled
                lengths[2] = len(y_pred_unscaled)

        ##LSTM
        #Scale the data
        
        #Define the model
        # for neurons in (32, 64, 128, 256):
        for neurons in (1,2):
            print("Spacecraft: ", sc, "Axis: ", axis, "Model: LSTM", "Neurons: ", neurons)
            model = keras.Sequential()
            model.add(
                keras.layers.Bidirectional( #Model learns the intricacies of the sequences in both directions
                    keras.layers.LSTM(units=neurons, input_shape=(x_train_sequences.shape[1], x_train_sequences.shape[2]))
                )
            )
            model.add(keras.layers.Dropout(rate=0.2))
            model.add(keras.layers.Dense(units=1)) #Regression output
            model.compile(optimizer='adam', loss='mean_squared_error')

            model.fit(x_train_sequences, y_train_sequences, epochs=1, batch_size=10, validation_split=0.1, shuffle=False)

            y_pred = model.predict(x_test_sequences)
            # print(f"Predictions: {len(y_pred)}")
            y_pred_unscaled = label_scaler.inverse_transform(y_pred.reshape(1, -1))
            y_pred_unscaled = y_pred_unscaled.reshape(-1)
            # print(f"Predictions unscaled: {len(y_pred_unscaled)}")
            LSTM_error = np.mean((y_pred_unscaled - y_test_sequences)**2)
            if LSTM_error < min_error[sc][axis][3]:
                # print("New minimum error; best pred of length: ", len(y_pred_unscaled))
                min_error[sc][axis][3] = LSTM_error
                lstm_best_predictions = y_pred_unscaled
                lengths[3] = len(y_pred_unscaled)

        ##Plot Everything
        plt.figure(figsize=(10, 6), dpi=200)
        # plt.scatter(time_train, y_train, label='Training', marker='o', linewidths = 1, color = 'red')
        
        # plt.scatter(time_test, y_test, label='Actual', marker='x', linewidths = 1, color = 'red')

        # plt.scatter(time_test[::3], knn_best_predictions[::3], label='KNN', marker='x', linewidths = 1, color = 'peru')
        # plt.scatter(time_test[::3], svr_best_predictions[::3], label='SVR', marker='x', linewidths = 1, color = 'springgreen')
        # plt.scatter(time_test[::3], nn_best_predictions[::3], label='NN', marker='x', linewidths = 1, color = 'dodgerblue')
        # plt.scatter(sequences_time_test[::3], lstm_best_predictions[::3], label='LSTM', marker='x', linewidths = 1, color = 'deeppink')

        plt.plot(time_train, y_train, label='Training', color='red')
        plt.plot(time_test, y_test, label='Actual', color='red')
        plt.plot(time_test[::3], knn_best_predictions[::3], label='KNN', color='peru')
        plt.plot(time_test[::3], svr_best_predictions[::3], label='SVR', color='springgreen')
        plt.plot(time_test[::3], nn_best_predictions[::3], label='NN', color='dodgerblue')
        plt.plot(sequences_time_test[::3], lstm_best_predictions[::3], label='LSTM', color='deeppink')
        
        plt.legend()

        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Offset', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        spacecraft = ["Tango", "Salsa", "Rumba", "Samba"]
        axes = ["X", "Y", "Z"]
        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years.png'.format(spacecraft[sc], axes[axis], training_years))


        print("Spacecraft: ", sc, "Axis: ", axis, "Lengths: ", lengths)

#%% Save the results
np.save("Outputs/Outputs_Models_Compared/min_error.npy", min_error)

#%% Plot and print results
print(min_error)

# for sc in range(0,4):
#     for axis in range(0,3):
#         plt.figure()
#         plt.plot(time_train[sc][axis][0], y_train[sc][axis][0], label='Training')
# plt.plot(time_test, y_test, label='Actual')
# plt.plot(time_test, nn_best_predictions[sc][axis], label='NN')
# plt.plot(time_test, knn_best_predictions[sc][axis], label='KNN')
# plt.plot(time_test, svr_best_predictions[sc][axis], label='SVR')
# plt.plot(time_test[0:-1 * batch_size], lstm_best_predictions[sc][axis][0:-1 * batch_size], label='LSTM')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Offset')
# spacecraft = ["Tango", "Salsa", "Rumba", "Samba"]
# axes = ["X", "Y", "Z"]
# plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis.png'.format(spacecraft[sc], axes[axis]))
