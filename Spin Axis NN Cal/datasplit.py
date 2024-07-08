import datetime
import numpy as np
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