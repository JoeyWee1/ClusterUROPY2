#This code learns a SVM model based on the first n years of data and then tests the accuracy of the model for subsequent periods
#%% IMPORTING REQUIRED LIBRARIES
import numpy as np; 
from sklearn.model_selection import train_test_split;
from sklearn import neighbors;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import mean_absolute_error;
import matplotlib.pyplot as plt; 
import math;
import datetime;
from sklearn.svm import SVC;
from sklearn.svm import SVR;

#%% DEFINING REQUIRED OBJECTS
#The only required objects are the Orbits and OnPeriods
class Orbit: #Creating orbit class
    def __init__(self,orbitNumber,startTime,endTime): #Initiation function
        self.orbitNumber = orbitNumber; #Creating attributes to store each bit of required data
        self.startTime = startTime;
        self.endTime = endTime;
        self.calParams = np.empty((3,6,4,));#Calibration stats are set to nan as default; [axes][range][offset,gain,theta,phi]
        self.calParams[:] = np.nan;
    def setF074(self,F074): #Sets the FGM temperature (F074); done like this so that more params can be added later
        self.F074 = F074; #All telems are the averages across an orbit
    def setF034(self,F034): #-12V voltage line
        self.F034 = F034;
    def setF047(self,F047): #+5V voltage line
        self.F047 = F047;
    def setF048(self,F048): #+12V voltage line
        self.F048 = F048;
    def setF055(self,F055): #PSU temp
        self.F055 = F055;
    def setJ236(self,J236): #Instrument current
        self.J236 = J236;        
class OnPeriod: #Object to store all the orbit data beween two switch-on moments; this will include the off period; however, because the data for the off periods will be NaN, it doesn't matter
    def __init__(self,startTime, endTime): #The start of the current on period and the start of the next on period 
        self.startTime = startTime; #Defining the start and ends so that the relevant orbits can be searched for
        self.endTime = endTime;
        self.periodOrbits = [];

#%% IMPORTING THE ORBITS
orbits = np.load("./Inputs/cleanedOrbitsArrayV6(FinalRange2DataOnly).npy", allow_pickle = True);

#%% CREATE FEATURES
def createAndSplitData(orbits, sc, trainingYears, axis): #Axis y = 1, z = 2
    xTrain = []; #Array to store the training feature vectors of the orbits within the training period
    yTrain = []; #Array to store the labels (offsets) of the orbits within the training period
    timeTrain = []; #Array to store the start times of the training orbits
    xTest = []; #Array to store the feature vectors of orbits in the test period (all orbits after training period)
    yTest = []; #Array to store the offsets of the orbits in the test period
    timeTest = []; #Array to store the start times of the test orbits
    firstOrbitStartTime = datetime.datetime(2000, 8, 24, 8, 56, 52);#First orbit start time
    splitTime = firstOrbitStartTime + datetime.timedelta(seconds = (trainingYears * 31557600));#Defines the split time based on the number of training years
    for orbit in orbits[sc]: #Loop through every orbit for each spacecraft
        orbitStartTime = orbit.startTime; #Pull out orbit start date to check if it's before or after the split date to choose if it goes into the test or train datasets
        trainingPeriodMarker = True; #True indicates that the orbit belongs in the training period. Later, this is checked and if the orbit start time is after the cutoff, this is switched to false.
        dataCleanlinessMarker = True; #Stores if the data is clean and whether or not it should be added to the arrays. True indicates the data is clean.
        featureVector = [orbit.F074, orbit.F055]; #For each orbit, create a feature vector composed of the spacecraft/instrument telems
        label = orbit.calParams[axis][0][0]; #Gets the label for the range 2 offset data
        #Check for data cleanliness    
        for i in range (0, len(featureVector)): #Looping through the coordinates in the feature vector to check for nan values for removal
            if (str(featureVector[i]) =="nan"): #Checking for the nan values
                dataCleanlinessMarker = False; #Setting cleanliness to false, so the data will not be appended
        if (str(label) == "nan"):#Checks the y values for nan
            dataCleanlinessMarker = False; #Sets cleanliness to false 
        #Classify data into test and train periods
        if (orbitStartTime > splitTime): #If the orbit start time is after the split time
            trainingPeriodMarker = False; #The orbit is in the testing period, not the training period. So, the marker is set to false.
        else: #The orbit start time is before the split time
            pass; #No change
        #Add the data to the relevant array if it should be added
        if (dataCleanlinessMarker == True): #Data should be added
            if (trainingPeriodMarker == True): #The data is in the selected training period from the first orbit to the split time
                xTrain.append(featureVector);
                yTrain.append(label);
                timeTrain.append(orbitStartTime);
            else: #The training period marker is false and the data is in the testing period
                xTest.append(featureVector);
                yTest.append(label);
                timeTest.append(orbitStartTime);                
        else: #Else, the data is dirty and isn't added to either array
            pass; #Nothing happens
    return xTrain, xTest, yTrain, yTest, timeTrain, timeTest, sc, axis, trainingYears;

#%% CREATE AND TRAIN MODEL ON TRAINING DATASET
def createAndTrainRBFModel(splitData, c, gamma): #The splitData argument is the output of the createAndSplitData function 
    xTrain = splitData[0]; #Pull out the x training data from the splitData
    yTrain = splitData[2]; #Pull out the y training data from the splitData
    model = SVR(kernel="rbf", C=c, gamma=gamma); #Creating the model
    model.fit(xTrain,yTrain); #Training the model on the data
    return model; #The model is an object and can be returned

#%% TEST MODEL ON TESTING DATASET
def testModel(splitData,model): #This function should return two arrays: absolute error of a data point, time since splitTime
    xTest = splitData[1]; #Pull out the x testing data
    yTest = splitData[3]; #Pull out the y testing data
    timeTest = np.array(splitData[5]); #Pulling out the times of the data points 
    trainingYears = splitData[8]; #Pulling out the length of the training time
    axis = splitData[7];
    sc = splitData[6];
    #Dealing with the time data
    timeSinceSplit = timeTest - splitTime; #Creating array of times in timedelta objects
    secondsSinceSplit = [];
    for i in range(0,len(timeSinceSplit)):
        secondsSinceSplit.append(timeSinceSplit[i].total_seconds());
    secondsSinceSplit = np.array(secondsSinceSplit);
    yearsSinceSplit = secondsSinceSplit/31557600;
    #Calculating the error data
    yPredict = model.predict(xTest); #Predicts the y values based on the xTest values
    error = mean_absolute_error(yTest, yPredict);
    errorData = []; #Array for appending error data of each data point
    for i in range(0, len(xTest)): #Loops through test x values to create the array of absolute errors
        currentX = np.array(xTest[i]);
        currentX = currentX.reshape(1,-1);
        predictedY = model.predict(currentX); #Predict the y value for every feature vector
        actualY = yTest[i]; #Pull out the human-calculated y value
        absoluteError = np.abs(predictedY - actualY); #Find the absolute error
        errorData.append(absoluteError[0]); #Adds the absolute error to the error data array
    return yearsSinceSplit, errorData, trainingYears, axis, sc, error; #Returning what will be the x and y values of our plots and the length of the training period

#%% PLOT THE REGRESSION IN 3D SPACE
def plotRegression(splitData, model): #Plots the regression in 3D space
    xTrain = np.array(splitData[0]); #Pulls the data out of the splitData input
    xTest = np.array(splitData[1]);
    yTrain = splitData[2];
    yTest = splitData[3];
    sc = splitData[6];
    yPredict = model.predict(xTest); #Predicts the y values based on the xTest values
    error = mean_absolute_error(yTest, yPredict);
    fig = plt.figure(); #Creates the figure
    plt.rcParams["figure.dpi"] = 200; #Sets the figure dpi
    ax = fig.add_subplot(111, projection='3d'); #Sets it to a 3D plot
    ax.view_init(20, 45); #Changes the angle
    ax.set_ylim(7.5, 20); #Sets the range of the y axis
    ax.set_xlabel('F074');
    ax.set_zlabel('Offset');
    ax.set_ylabel('F055');
    ax.scatter(xTrain[:, 0], xTrain[:, 1],yTrain, color = "orange",s = 0.5, label = "Training");
    ax.scatter(xTest[:, 0],  xTest[:, 1],yTest , color = 'red', s = 0.5, label = "Test");
    ax.scatter(xTest[:, 0],  xTest[:, 1],yPredict , color = 'blue', s = 0.5, label = "Prediction");
    title = "SVM C{}; Training Years {}; Kernal: RBF; MeanAbsError = {:.2f}".format(sc+1,trainingYears,error);
    print(title)
    ax.set_title(title);
    plt.legend(loc = "upper left");
    plt.show();

#%% Plot errors against times since the split
def plotErrorAgainstTime(testModelData):
    axes = ["X", "y", "Z"];
    yearsSinceSplit = testModelData[0];
    errorData = testModelData[1];
    trainingLength = testModelData[2];
    axis = testModelData[3];
    sc = testModelData[4] + 1;
    #print(type(yearsSinceSplit[0]))
    #print(errorData[0])
    fig, ax =plt.subplots(1,1, dpi=200);
    ax.scatter(yearsSinceSplit, errorData);
    plt.ylabel("Absolute Error");
    plt.xlabel("Years Since Mission Start");
    plt.title("Error Against Time for Training Length {} Years; C{}; {} axis".format(trainingLength, sc, axes[axis]));
    plt.show();

#%% GET DATA TO PLOT
splitTime = datetime.datetime(2000, 8, 24, 8, 56, 52);
trainingYears = 15;
cValues = []; #Array to store tested C values
gammaValues = []; #Array to store tested gamma values
errorValues = []; #Index matched error for the error corresponding the the c,gamma coordinates
#c = 0.1;
#gamma = 1;
lowestError = 100000;
lowestC = np.nan;
lowestGamma = np.nan;
lowests = []; #c,gamma for each

for sc in range(0,4):
    for cx10 in range (1,100,5): #Tests c in range 0 to 100 in jumps of 10
        c = cx10 / 10;
        for gammax100 in range (1,100, 5): #tests gamma in range 0 to 0.1 in 2000 steps
            gamma = gammax100 / 100;
            print("Cluster " + str(sc+1) +"; Testing: C"+str(c)+"G"+str(gamma));
            splitData = createAndSplitData(orbits, sc, trainingYears, 2);
            model = createAndTrainRBFModel(splitData, c, gamma);
            testModelData = testModel(splitData, model);
            error = testModelData[5]; #This is the mean absolute error of the mode
            cValues.append(c);
            gammaValues.append(gamma);
            errorValues.append(error);
            #plotErrorAgainstTime(testModelData);
            #print(error)
            if (error < lowestError):
                lowestError = error;
                lowestC = c;
                lowestGamma = gamma;
    lowest = [lowestC, lowestGamma];
    lowests.append(lowest);
                
    #print(errorValues)
    plt.scatter(cValues, gammaValues, c=errorValues, cmap='jet',vmin=0.05, vmax=0.3);
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.family"] = "sans-serif";
    plt.title("Error against C and Gamma Hyperparameters for Cluster {}".format(sc+1), )
    plt.xlabel("C");
    plt.ylabel("Gamma")
    plt.colorbar()
    plt.show()
    print("lowest error for Cluster {} is {} at gamma {} and c {}".format(sc,lowestError,lowestGamma,lowestC))
    print(lowests)


