import datetime
import numpy as np
#%% Create features
def create_and_split_data(orbits, sc, axis, training_years): #Axis y = 1, z = 2
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

    return x_train, x_test, y_train, y_test, time_train, time_test, split_time

def yz_create_and_split_data(orbits, sc, training_years):
    """
    Split the dataset based on split time 
    The feature will include information on the y offset and gradient
    The label will be the z offset

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

    for i in range(1, len(orbits[sc])): #! 
        orbit = orbits[sc][i]
        orbit_start_time = orbit.startTime #Orbit start date to check if it goes into the test or train datasets
        training_period_marker = True; #True indicates that the orbit belongs in the training period. 
        data_cleanliness_marker = True; #True indicates the data is clean.
        # feature_vector = [orbit.F074, orbit.F055, orbit.F047, orbit.F034, orbit.F048]
        feature_vector = [orbit.F074, orbit.F055, orbit.calParams[1][0][0],  orbits[sc][i-1].calParams[2][0][0]] #! #Temperatures and y offset and previous z offset
        label = orbit.calParams[2][0][0]; #Gets the label for the range 2 z axis offset data

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

    return x_train, x_test, y_train, y_test, time_train, time_test, sc, training_years, split_time

def x_axis_create_and_split_data(orbits, sc, training_years): 
    """
    Function to split the data set into testing and training data based on a training length (time)
    Only the data taken during non-interpolated periods is used

    Args:
        orbits: The array of orbits
        sc: The spacecraft index
        training_years: The number of years to train on

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
    axis = 0

    x_train = []
    y_train = []
    time_train = [] #Start times of the training orbits
    x_test = []
    y_test = []
    time_test = []
    x_interp_test = []
    time_interp_test = []    
    y_interp_train = []
    time_interp_train = []
    first_orbit_start_time = datetime.datetime(2000, 8, 24, 8, 56, 52)
    split_time = first_orbit_start_time + datetime.timedelta(seconds = (training_years * 31557600)) #Defines the split time based on the number of training years

    for orbit in orbits[sc]: 
        orbit_start_time = orbit.startTime #Orbit start date to check if it goes into the test or train datasets
        training_period_marker = True; #True indicates that the orbit belongs in the training period. 
        data_cleanliness_marker = True; #True indicates the data is clean.
        data_interp_marker = False; #True indicates the data is interpolated
        # feature_vector = [orbit.F074, orbit.F055, orbit.F047, orbit.F034, orbit.F048]
        y_offset = orbit.calParams[1][0][0]
        z_offset = orbit.calParams[2][0][0]
        feature_vector = [orbit.F074, orbit.F055, y_offset, z_offset]
        label = orbit.calParams[axis][0][0]; #Gets the label for the range 2 offset data

        #Check for data cleanliness    
        for i in range (0, len(feature_vector)): #Check if any feature is nan
            if (str(feature_vector[i]) == "nan"):
                data_cleanliness_marker = False
        if (str(label) == "nan"): #Checks the label is nan
            data_cleanliness_marker = False 
        
        month = orbit_start_time.month
        if (month == 1 or month == 2 or month == 3 or month == 4):
            pass
        else: #If the data is interpolated, don't use it
            data_interp_marker = True

        #Classify data into test and train periods
        if (orbit_start_time > split_time): #Find orbits that start after the split time
            training_period_marker = False
        else: 
            pass

        #Add the data to the relevant array if it should be added
        if (data_cleanliness_marker == True): 
            if (training_period_marker == True):
                if(data_interp_marker == False):
                    x_train.append(feature_vector)
                    y_train.append(label)
                    time_train.append(orbit_start_time)
                else:
                    y_interp_train.append(label)
                    time_interp_train.append(orbit_start_time)
            else: #The training period marker is false and the data is in the testing period
                if(data_interp_marker == False):
                    x_test.append(feature_vector)
                    y_test.append(label)
                    time_test.append(orbit_start_time);   
                else:
                    x_interp_test.append(feature_vector)
                    time_interp_test.append(orbit_start_time)             
        else: #Else, the data is dirty and isn't added to either array
            pass
    return x_train, x_test, y_train, y_test, time_train, time_test, sc, axis, training_years, split_time, x_interp_test, time_interp_test, y_interp_train, time_interp_train