#%% IMPORTING LIBRARIES
from pandas import read_csv,to_datetime;
from os import system;
from os.path import exists;
from numpy import load,mean,std,array,save;
from matplotlib.pyplot import subplots,plot,xlabel,ylabel,title,grid;
from matplotlib import pyplot;
import numpy as np;

#%% GLOBAL VARIABLES
global orbit_numbers, orbit_start_times, orbit_end_times;

#%% LOAD ORBIT INFO
orbit_numbers = list(load("./orbit_numbers.npy",allow_pickle=True)); 
orbit_start_times = list(load("./orbit_start_times.npy",allow_pickle=True));
orbit_end_times = list(load("./orbit_end_times.npy",allow_pickle=True));

#%% WRITE PARAMETER FILE
def writeParamsFile(start_time,end_time,spacecraft): #Creates a 'param' file containing the orbit specific part of the command
    pstring = '{"date_from": "';
    pstring += start_time.replace(tzinfo=None).isoformat(timespec='minutes')+"Z";
    pstring += '","date_to": "' ;
    pstring += end_time.replace(tzinfo=None).isoformat(timespec='minutes')+"Z";
    pstring += '","sc": ';
    pstring += str(spacecraft);
    pstring += ',"parameters": [';
    pstring += '"F_050", "F_055", "F_074", "F_047", "F_048", "F_034", "J_236", "J_106", "J_213", "J_216"';
    pstring += ']}';
    # write the string into the file params.txt
    file = open("params.txt","w");
    file.truncate(); # erase content
    file.write(pstring);
    file.close();

#%% GENERATE FILE NAME
def generateFilename(spacecraft,orbit_number):
    filename = "C" + str(spacecraft) + "_orbit_" + str(orbit_number) + ".csv";
    return filename;

#%% FUNCTION TO RETRIEVE DATA
#Will retrieve data according to the contents of an (assumed to exist) params.txt file
#Save according to string 'filename', which is to be supplied with a .csv extension
#Will abort if file exists
def getRDM(filename):
    relativepath = "./CSVs/" + filename;
    #Check for existing csv file
    if exists(relativepath):
        print("Aborting: %s exists" % relativepath);
    else:
        relativepath += ".gz";
        parameter_string = 'curl -X GET https://caa.esac.esa.int/rdm-hk/api/data/ -H "Content-Type: application/json" -d @params.txt -o ' + relativepath;
        system(parameter_string);
        parameter_string = 'gunzip ' + relativepath;
        system(parameter_string);
        
#%% FUNCTION TO LOOP THROUGH ORBITS AND TELEMS, DOWNLOADING DATA
def getTelemFiles(spacecraft,startOrbit,endOrbit):
    #telems = ["F_050", "F_055", "F_074", "F_047", "F_048", "F_034", "J_236", "J_106", "J_213", "J_216"];
    for i in range(0,len(orbit_numbers)):
        orbit_number = orbit_numbers[i];
        start_time = orbit_start_times[i];
        end_time = orbit_end_times[i];
        writeParamsFile(start_time,end_time,spacecraft);
        filename = generateFilename(spacecraft,orbit_number);
        print("Getting data for orbit "+str(orbit_number)+" for spacecraft "+str(spacecraft)+" from "+start_time.isoformat()+" to "+end_time.isoformat());
        getRDM(filename);
            
#%% CREATE ORBIT AVERAGES AND SIGMAS
def orbitAverages(spacecraft):
    telems = ["F_055", "F_074", "F_047", "F_048", "F_034", "J_236", "J_106", "J_213", "J_216"]; #The telems for which to process data
    telemValidRanges = [[9,26],[-80,20],[5,6],[12,13],[-14,-12],[70,81],[26.5,30],[3,7.5],[4,11]]; #Anything within these ranges inclusive can be determined to be valid
    #telems = ["F_055", "F_074", "F_047", "F_048", "F_034"];
    spacecraftTelemStatistics = [];
    for i in range(0,len(orbit_numbers)):
        orbitNumber = orbit_numbers[i];
        dataFilename = generateFilename(spacecraft,orbitNumber); #Gets the filename for the range data for that orbit
        dataPathstring = "./CSVs/" + dataFilename;
        dataCSV = read_csv(dataPathstring);
        rangeDictReference = str(spacecraft)+"F_050";
        rangeDataPanda = dataCSV[rangeDictReference];
        rangeData = rangeDataPanda.to_numpy();
        orbitTelemStatistics = [];
        for j in range(0,len(telems)):
            telem = telems[j];
            telemDictReference = str(spacecraft)+telem;
            telemDataPanda = dataCSV[telemDictReference][:];
            telemData = telemDataPanda.to_numpy();
            range2TelemData = [];
            for k in range(0, len(rangeData)):
                if (rangeData[k] == 2):
                    range2TelemData.append(telemData[k]);
            telemValidRange = telemValidRanges[j];
            validTelemData = [x for x in range2TelemData if ((x > telemValidRange[0]) and (x < telemValidRange[1]))]; #Within a reasonable range
            nonZeroTelemData = [x for x in validTelemData if x != 0]; #Are not 0
            if len(range2TelemData) == 0: 
                print("WARNING: Orbit %i file has no range 2 values" % orbitNumber);
            elif len(validTelemData) ==0:
                print("WARNING: Orbit %i contains no valid (within expected range) temperature values" % orbitNumber);
            elif len(nonZeroTelemData) ==0:
                print("WARNING: Orbit %i contains only zero values" % orbitNumber);
            if (len(range2TelemData)==0):
                count = 0;
                average = np.nan;
                sigma = np.nan;
                maximum = np.nan;
                minimum = np.nan;
            else:
                #Minimises the effects of strange 0 values
                if ((len(nonZeroTelemData)<len(validTelemData)) and (mean(nonZeroTelemData)<-10)):
                    validTelemData = nonZeroTelemData;
                    print("WARNING: Orbit %i discarding zero values" % orbitNumber);   
                count = len(validTelemData);
                average = mean(validTelemData);
                sigma = std(validTelemData);
                maximum = max(validTelemData);
                minimum = min(validTelemData);
            telemStatistic = [count,average,sigma,maximum,minimum];
            orbitTelemStatistics.append(telemStatistic);
            print(telem +" loop complete")
        spacecraftTelemStatistics.append(orbitTelemStatistics);
        print("Orbit " + str(orbitNumber) + " loop complete")
    return spacecraftTelemStatistics;
            
#%% RUN THINGS
getTelemFiles(1,0,3650);
spacecraftTelemStatistics = orbitAverages(spacecraft='1'); #In the order ["F_055", "F_074", "F_047", "F_048", "F_034", "J_236", "J_106", "J_213", "J_216"]
save("./C1SpacecraftTelemStatisticsTestV2(incl J)", spacecraftTelemStatistics);












