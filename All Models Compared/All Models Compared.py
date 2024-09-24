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
        knn_residuals = []
        knn_best_k = np.nan
        svr_best_predictions = []
        svr_residuals = []
        svr_best_c = np.nan
        nn_best_predictions = []
        nn_residuals = []
        lstm_best_predictions = []
        lstm_residuals = []

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
        for k in range(1, 11): #Test k values from 1 to 10
            print("Spacecraft: ", sc, "Axis: ", axis, "Model: KNN", "k: ", k)
            knn_model = KNeighborsRegressor(n_neighbors=k, weights='distance') 
            knn_model.fit(x_train, y_train)
            y_pred = knn_model.predict(x_test)
            # knn_error = np.mean((y_pred - y_test)**2)
            knn_error = np.mean(np.abs(y_pred - y_test))
            if (knn_error < min_error[sc][axis][0]):
                min_error[sc][axis][0] = knn_error
                knn_best_predictions = y_pred
                knn_residuals = np.abs(y_test - y_pred)
                lengths[0] = len(y_pred)
                knn_best_k = k

        ##SVR
        c_range = np.linspace(0.1, 1, 10)
        for c in c_range:
            print("Spacecraft: ", sc, "Axis: ", axis, "Model: SVR", "C: ", c)
            svr_model = SVR(kernel='rbf', C=c, gamma='auto') #Automatically scale the gamma parameter
            svr_model.fit(x_train, y_train)
            y_pred = svr_model.predict(x_test)
            # svr_error = np.mean((y_pred - y_test)**2)
            svr_error = np.mean(np.abs(y_pred - y_test))
            if (svr_error < min_error[sc][axis][1]):
                min_error[sc][axis][1] = svr_error
                svr_best_predictions = y_pred
                svr_residuals = np.abs(y_test - y_pred)
                lengths[1] = len(y_pred)
                svr_best_c = c

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

        # loss_fn = nn.MSELoss()  # mean square error
        loss_fn = nn.L1Loss()  # mean absolute error
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        n_epochs = 100   # number of epochs to run
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
                y_pred_unscaled = np.array(y_pred_unscaled).reshape(-1)
                nn_residuals = np.abs(y_test - y_pred_unscaled)
                lengths[2] = len(y_pred_unscaled)

        ##LSTM
        #Scale the data
        
        #Define the model
        for neurons in (32, 64, 128, 256):
            # for neurons in (1,2):
            print("Spacecraft: ", sc, "Axis: ", axis, "Model: LSTM", "Neurons: ", neurons)
            model = keras.Sequential()
            model.add(
                keras.layers.Bidirectional( #Model learns the intricacies of the sequences in both directions
                    keras.layers.LSTM(units=neurons, input_shape=(x_train_sequences.shape[1], x_train_sequences.shape[2]))
                )
            )
            model.add(keras.layers.Dropout(rate=0.2))
            model.add(keras.layers.Dense(units=1)) #Regression output
            # model.compile(optimizer='adam', loss='mean_squared_error')
            model.compile(optimizer='adam', loss='mean_absolute_error')

            model.fit(x_train_sequences, y_train_sequences, epochs=100, batch_size=10, validation_split=0.1, shuffle=False)

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
                y_test_sequences = np.array(y_test_sequences).reshape(-1)
                lstm_residuals = np.abs(y_test_sequences - y_pred_unscaled)
                lengths[3] = len(y_pred_unscaled)


        ##Save the predictions, testing data, and residuals
        axes = ["X", "Y", "Z"]
        np.save(f"./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_{training_years}_KNN_Predictions.npy", knn_best_predictions)
        np.save(f"./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_{training_years}_KNN_Residuals.npy", knn_residuals)
        np.save(f"./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_{training_years}_SVR_Predictions.npy", svr_best_predictions)
        np.save(f"./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_{training_years}_SVR_Residuals.npy", svr_residuals)
        np.save(f"./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_{training_years}_NN_Predictions.npy", nn_best_predictions)
        np.save(f"./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_{training_years}_NN_Residuals.npy", nn_residuals)
        np.save(f"./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_{training_years}_LSTM_Predictions.npy", lstm_best_predictions)
        np.save(f"./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_{training_years}_LSTM_Residuals.npy", lstm_residuals)
        np.save(f"./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_{training_years}_Test_Y.npy", y_test)
        np.save(f"./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_{training_years}_Test_Y_Sequences.npy", y_test_sequences)
        np.save(f"./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_{training_years}_KNN_Best_K{knn_best_k}.npy", np.array(knn_best_k))
        np.save(f"./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_{training_years}_SVR_Best_C{svr_best_c}.npy", np.array(svr_best_c))

        ##Plot Everything
        time_test = np.array(time_test)
        sequences_time_test = np.array(sequences_time_test)

        #All
        plt.figure(figsize=(10, 6), dpi=200)
        plt.scatter(time_train[::3], y_train[::3], marker='o', linewidths = 0.3, color = '#e60049', s = 28)
        plt.scatter(time_test[::3], y_test[::3], label='Actual', marker='o', linewidths = 1, color = '#e60049', s = 20)
        plt.axvline(x=split_time, color='#e60049', linestyle='--', label='Split Time')
        plt.scatter(time_test[::4], nn_best_predictions[::4], label='NN', marker='x', linewidths = 1, color = 'chartreuse', s = 16)
        plt.scatter(sequences_time_test[::4], lstm_best_predictions[::4], label='LSTM', marker='x', linewidths = 1, color = '#c760ff', s = 16)
        plt.scatter(time_test[::4], knn_best_predictions[::4], label='KNN', marker='x', linewidths = 1, color = '#0bb4ff', s = 16)
        plt.scatter(time_test[::4], svr_best_predictions[::4], label='SVR', marker='x', linewidths = 1, color = '#ffa300', s = 16)
        plt.legend()
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Offset', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        spacecraft = ["Tango", "Salsa", "Rumba", "Samba"]

        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years.png'.format(spacecraft[sc], axes[axis], training_years))

        #NN
        plt.figure(figsize=(10, 6), dpi=200)
        plt.scatter(time_train, y_train, marker='o', linewidths = 0.3, color = '#e60049', s = 28)
        plt.scatter(time_test, y_test, label='Actual', marker='o', linewidths = 1, color = '#e60049', s = 20)
        plt.axvline(x=split_time, color='#e60049', linestyle='--', label='Split Time')
        plt.scatter(time_test, nn_best_predictions, label='NN', marker='x', linewidths = 1, color = 'chartreuse', s = 16)
        plt.legend()
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Offset', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years_NN.png'.format(spacecraft[sc], axes[axis], training_years))

        #NN residuals
        plt.figure(figsize=(10, 6), dpi=200)
        plt.scatter(time_test, np.abs(nn_residuals), label='NN', marker='x', linewidths = 1, color = 'chartreuse', s = 16)
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Residual', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years_NN_Residuals.png'.format(spacecraft[sc], axes[axis], training_years))

        #LSTM
        plt.figure(figsize=(10, 6), dpi=200)
        plt.scatter(time_train, y_train, marker='o', linewidths = 0.3, color = '#e60049', s = 28)
        plt.scatter(time_test, y_test, label='Actual', marker='o', linewidths = 1, color = '#e60049', s = 20)
        plt.axvline(x=split_time, color='#e60049', linestyle='--', label='Split Time')
        plt.scatter(sequences_time_test, lstm_best_predictions, label='LSTM', marker='x', linewidths = 1, color = '#c760ff', s = 16)
        plt.legend()
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Offset', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years_LSTM.png'.format(spacecraft[sc], axes[axis], training_years))

        #LSTM residuals
        plt.figure(figsize=(10, 6), dpi=200)
        plt.scatter(sequences_time_test, np.abs(lstm_residuals), label='LSTM', marker='x', linewidths = 1, color = '#c760ff', s = 16)
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Residual', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years_LSTM_Residuals.png'.format(spacecraft[sc], axes[axis], training_years))

        #KNN
        plt.figure(figsize=(10, 6), dpi=200)
        plt.scatter(time_train, y_train, marker='o', linewidths = 0.3, color = '#e60049', s = 28)
        plt.scatter(time_test, y_test, label='Actual', marker='o', linewidths = 1, color = '#e60049', s = 20)
        plt.axvline(x=split_time, color='#e60049', linestyle='--', label='Split Time')
        plt.scatter(time_test, knn_best_predictions, label='KNN', marker='x', linewidths = 1, color = '#0bb4ff', s = 16)
        plt.legend()
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Offset', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years_KNN.png'.format(spacecraft[sc], axes[axis], training_years))

        #KNN residuals
        plt.figure(figsize=(10, 6), dpi=200)
        plt.scatter(time_test, np.abs(knn_residuals), label='KNN', marker='x', linewidths = 1, color = '#0bb4ff', s = 16)
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Residual', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years_KNN_Residuals.png'.format(spacecraft[sc], axes[axis], training_years))

        #SVR
        plt.figure(figsize=(10, 6), dpi=200)
        plt.scatter(time_train, y_train, marker='o', linewidths = 0.3, color = '#e60049', s = 28)
        plt.scatter(time_test, y_test, label='Actual', marker='o', linewidths = 1, color = '#e60049', s = 20)
        plt.axvline(x=split_time, color='#e60049', linestyle='--', label='Split Time')
        plt.scatter(time_test, svr_best_predictions, label='SVR', marker='x', linewidths = 1, color = '#ffa300', s = 16)
        plt.legend()
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Offset', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years_SVR.png'.format(spacecraft[sc], axes[axis], training_years))

        #SVR residuals
        plt.figure(figsize=(10, 6), dpi=200)
        plt.scatter(time_test, np.abs(svr_residuals), label='SVR', marker='x', linewidths = 1, color = '#ffa300', s = 16)
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Residual', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years_SVR_Residuals.png'.format(spacecraft[sc], axes[axis], training_years))

        #All residuals together on the same plot
        plt.figure(figsize=(10, 6), dpi=200)
        plt.scatter(time_test, np.abs(nn_residuals), label='NN', marker='x', linewidths = 1, color = 'chartreuse', s = 16)
        plt.scatter(sequences_time_test, np.abs(lstm_residuals), label='LSTM', marker='x', linewidths = 1, color = '#c760ff', s = 16)
        plt.scatter(time_test, np.abs(knn_residuals), label='KNN', marker='x', linewidths = 1, color = '#0bb4ff', s = 16)
        plt.scatter(time_test, np.abs(svr_residuals), label='SVR', marker='x', linewidths = 1, color = '#ffa300', s = 16)
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Residual', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()
        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years_Residuals.png'.format(spacecraft[sc], axes[axis], training_years))

        #Residuals as subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 12), dpi=200)
        axs[0, 0].scatter(time_test, np.abs(nn_residuals), label='NN', marker='x', linewidths = 1, color = 'chartreuse', s = 16)
        axs[0, 0].set_ylabel('Residual', fontsize=20)
        axs[0, 0].set_title('NN', fontsize=20)
        axs[0, 1].scatter(sequences_time_test, np.abs(lstm_residuals), label='LSTM', marker='x', linewidths = 1, color = '#c760ff', s = 16)
        axs[0, 1].set_title('LSTM', fontsize=20)
        axs[1, 0].scatter(time_test, np.abs(knn_residuals), label='KNN', marker='x', linewidths = 1, color = '#0bb4ff', s = 16)
        axs[1, 0].set_xlabel('Time', fontsize=20)
        axs[1, 0].set_ylabel('Residual', fontsize=20)
        axs[1, 0].set_title('KNN', fontsize=20)
        axs[1, 1].scatter(time_test, np.abs(svr_residuals), label='SVR', marker='x', linewidths = 1, color = '#ffa300', s = 16)
        axs[1, 1].set_xlabel('Time', fontsize=20)
        axs[1, 1].set_title('SVR', fontsize=20) 
        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years_Subplot_Residuals.png'.format(spacecraft[sc], axes[axis], training_years))

        #Everything as subplots with only the bottom plots having the time label and the left plots having the offset label
        fig, axs = plt.subplots(2, 2, figsize=(20, 12), dpi=200)
        axs[0, 0].scatter(time_train, y_train, marker='o', linewidths = 0.3, color = '#e60049', s = 28)
        axs[0, 0].scatter(time_test, y_test, label='Actual', marker='o', linewidths = 1, color = '#e60049', s = 20)
        axs[0, 0].axvline(x=split_time, color='#e60049', linestyle='--', label='Split Time')
        axs[0, 0].scatter(time_test, nn_best_predictions, label='NN', marker='x', linewidths = 1, color = 'chartreuse', s = 16)
        axs[0, 0].legend()
        axs[0, 0].set_ylabel('Offset', fontsize=20)
        axs[0, 0].set_title('NN', fontsize=20)
        axs[0, 1].scatter(time_train, y_train, marker='o', linewidths = 0.3, color = '#e60049', s = 28)
        axs[0, 1].scatter(time_test, y_test, label='Actual', marker='o', linewidths = 1, color = '#e60049', s = 20)
        axs[0, 1].axvline(x=split_time, color='#e60049', linestyle='--', label='Split Time')
        axs[0, 1].scatter(sequences_time_test, lstm_best_predictions, label='LSTM', marker='x', linewidths = 1, color = '#c760ff', s = 16)
        axs[0, 1].legend()
        axs[0, 1].set_title('LSTM', fontsize=20)
        axs[1, 0].scatter(time_train, y_train, marker='o', linewidths = 0.3, color = '#e60049', s = 28)
        axs[1, 0].scatter(time_test, y_test, label='Actual', marker='o', linewidths = 1, color = '#e60049', s = 20)
        axs[1, 0].axvline(x=split_time, color='#e60049', linestyle='--', label='Split Time')
        axs[1, 0].scatter(time_test, knn_best_predictions, label='KNN', marker='x', linewidths = 1, color = '#0bb4ff', s = 16)
        axs[1, 0].legend()
        axs[1, 0].set_xlabel('Time', fontsize=20)
        axs[1, 0].set_ylabel('Offset', fontsize=20)
        axs[1, 0].set_title('KNN', fontsize=20)
        axs[1, 1].scatter(time_train, y_train, marker='o', linewidths = 0.3, color = '#e60049', s = 28)
        axs[1, 1].scatter(time_test, y_test, label='Actual', marker='o', linewidths = 1, color = '#e60049', s = 20)
        axs[1, 1].axvline(x=split_time, color='#e60049', linestyle='--', label='Split Time')
        axs[1, 1].scatter(time_test, svr_best_predictions, label='SVR', marker='x', linewidths = 1, color = '#ffa300', s = 16)
        axs[1, 1].legend()
        axs[1, 1].set_xlabel('Time', fontsize=20)
        axs[1, 1].set_title('SVR', fontsize=20) 
        plt.savefig('./Outputs/Outputs_Models_Compared/Spacecraft_{}_{}-Axis_{}_Training-Years_All.png'.format(spacecraft[sc], axes[axis], training_years))


#%% Save the results
np.save("Outputs/Outputs_Models_Compared/min_error_15.npy", min_error)

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
