#This code learns and exports a 6 dimensional KNN model, which takes in 6 telems and returns the predicted offset param
#%% IMPORTING REQUIRED LIBRARIES
import numpy as np; 
from sklearn.model_selection import train_test_split;
from sklearn import neighbors;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import mean_absolute_error;
import matplotlib.pyplot as plt; 
import math;

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
orbits = np.load("./Inputs/cleanedOrbitsArrayV2(MoreData).npy", allow_pickle = True)

#%% CREATE FEATURES
def createAndSplitData(orbits, scIndex, testSize):
    featureVectors = [];
    offsets = []; #These two arrays are index matched
    sc = 3; #Select a spacecraft
    for orbit in orbits[sc]: #Loop through every orbit for each spacecraft
            dataCleanliness = True; #Stores if the data is clean and whether or not it should be added to the arrays
            featureVector = [orbit.F074, orbit.F055]; #For each orbit, create a feature vector composed of the spacecraft/instrument telems
            y = orbit.calParams[2][0][0]; #Gets the z axis range 2 offset
            for i in range (0, len(featureVector)): #Looping through the coordinates in the feature vector to check for nan values for removal
                if (str(featureVector[i]) =="nan"): #Checking for the nan values
                    dataCleanliness = False; #Setting cleanliness to false, so the data is not appended
            if (str(y) == "nan"):#Checks the y values for nan
                dataCleanliness = False;#Replaces it with an out of bounds value
            if (dataCleanliness == True):
                featureVectors.append(featureVector);
                offsets.append(y);
            else:
                #print("Data point rejected");
                pass;
    xTrain, xTest, yTrain, yTest = train_test_split(featureVectors,offsets,test_size = testSize);
    #Convert everything in np arrays
    xTrain = np.array(xTrain);
    xTest = np.array(xTest);
    yTrain = np.array(yTrain);
    yTest = np.array(yTest);
    return xTrain, xTest, yTrain, yTest, scIndex;

#%% CHECK COUNT IN RADIUS

def findDistance(x1,y1,x2,y2):
    dx = x2 - x1;
    dy = y2 - y1;
    return math.sqrt(dx * dx + dy * dy);

def checkDensity(xTrain,yTrain,maxRadius, minCount, x1, y1):
    count = 0;
    for i in range(0,len(xTrain)):
        x2 = xTrain[i];
        y2 = yTrain[i];
        distance = findDistance(x1, y1, x2, y2);
        if (distance <= maxRadius):
            count += 1;
    if (minCount <= count):
        return True, count;
    else:
        return False, count;

#%% MAIN FUNCTION TO TAKE IN F055 AND F074 INPUTS AND RETURN AN OFFSET FOR ONE AXIS
#y=0,z=1

def predictOffset(spacecraftNumber,axis,maxRadius,minCount,minK,maxK,inputF074,inputF055):
    #Train the model
    splitData = createAndSplitData(orbits, spacecraftNumber-1,0.2);
    xTrain, yTrain = splitData[0], splitData[2];
    xTest, yTest = splitData[1], splitData[3];
    weights = "distance"; #Electing to weigh the neighbours by distance
    bestK = 0;
    bestError = 1e4;
    for testK in range (minK,maxK+1):
        model = neighbors.KNeighborsRegressor(n_neighbors=testK,weights=weights);
        model.fit(xTrain,yTrain);
        predictionForError = model.predict(xTest);
        error = mean_absolute_error(yTest, predictionForError);
        if(error<bestError):
            bestK = testK;
            bestError = error;
    print("The best K was ", bestK);
    #Recover the model with the bestK using all data
    splitData = createAndSplitData(orbits, spacecraftNumber-1,0);
    xTrain, yTrain = splitData[0], splitData[2];
    model = neighbors.KNeighborsRegressor(n_neighbors=bestK,weights=weights);
    model.fit(xTrain,yTrain);
    clean, count = checkDensity(xTrain,yTrain,maxRadius,minCount,inputF074,inputF055);#Cleanliness denotes the density is high enough
    if (clean == True):
        print("Sufficient density! Count = ", count);
        predictedOffset = model.predict([inputF074,inputF055]);
        print("Predicted offset is ", predictedOffset);
    else:
        print("WARNING: Insufficient density! Manual calibration required!");





