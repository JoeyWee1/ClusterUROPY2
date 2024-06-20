from pandas import read_csv
from os import system
from os.path import exists
from numpy import load,mean,std,save
import numpy as np

global orbit_numbers, orbit_start_times, orbit_end_times

telems = ["F_055", "F_074"] #The telems for which to process data
telem_valid_ranges = [[9,26],[-80,20]] #The valid ranges for the telems

#telems ["F_055", "F_074", "F_047", "F_048", "F_034", "J_236", "J_106", "J_213", "J_216"]
#valid ranges  [[9,26],[-80,20],[5,6],[12,13],[-14,-12],[70,81],[26.5,30],[3,7.5],[4,11]] #Anything within these ranges inclusive can be determined to be valid

# Loading orbit info
orbit_numbers = list(load("./orbit_numbers.npy",allow_pickle=True))
orbit_start_times = list(load("./orbit_start_times.npy",allow_pickle=True))
orbit_end_times = list(load("./orbit_end_times.npy",allow_pickle=True))

def write_params_file(start_time,end_time,spacecraft): 
    """
    Creates a params.txt readme file with the parameters of the download 
    """
    pstring = '{"date_from": "'
    pstring += start_time.replace(tzinfo=None).isoformat(timespec='minutes')+"Z"
    pstring += '","date_to": "' 
    pstring += end_time.replace(tzinfo=None).isoformat(timespec='minutes')+"Z"
    pstring += '","sc": '
    pstring += str(spacecraft)
    pstring += ',"parameters": ['
    pstring += '", "'.join(telems)
    pstring += ']}'
    # write the string into the file params.txt
    file = open("params.txt","w")
    file.truncate() # erase content
    file.write(pstring)
    file.close()

#%% GENERATE FILE NAME
def generate_filename(spacecraft,orbit_number):
    """
    Generates a filename to save data to, based on the spacecraft and orbit number
    """
    filename = "C" + str(spacecraft) + "_orbit_" + str(orbit_number) + ".csv"
    return filename


def get_RDM(filename):
    """
    Retrieves data according to the contents of a params.txt file
    Saves according to string 'filename', which is to be supplied with a .csv extension
    Aborts if file already exists
    """
    relativepath = "./CSVs/" + filename
    #Check for existing csv file
    if exists(relativepath):
        print("Aborting: %s exists" % relativepath)
    else:
        relativepath += ".gz"
        parameter_string = 'curl -X GET https://caa.esac.esa.int/rdm-hk/api/data/ -H "Content-Type: application/json" -d @params.txt -o ' + relativepath
        system(parameter_string)
        parameter_string = 'gunzip ' + relativepath
        system(parameter_string)
        
def get_telem_files(spacecraft, start_orbit = 0, end_orbit = len(orbit_numbers)):
    """
    Loop through orbits for one spacecraft, downloading data for each orbit using get_RDM
    Saves all data as .csv files in the CSVs folder
    """
    for i in range(start_orbit, end_orbit):
        orbit_number = orbit_numbers[i]
        start_time = orbit_start_times[i]
        end_time = orbit_end_times[i]
        write_params_file(start_time,end_time,spacecraft)
        filename = generate_filename(spacecraft,orbit_number)
        print("Getting data for orbit "+str(orbit_number)+" for spacecraft "+str(spacecraft)+" from "+start_time.isoformat()+" to "+end_time.isoformat());
        get_RDM(filename)
            
def orbitAverages(spacecraft):
    """
    Creates orbital for data ranges for each telem in the spacecraft
    """
    spacecraftTelemStatistics = []
    for i in range(0, len(orbit_numbers)):
        orbit_number = orbit_numbers[i]
        data_filename = generate_filename(spacecraft, orbit_number); #Gets the filename for the range data for that orbit
        data_pathstring = "./CSVs/" + data_filename
        data_CSV = read_csv(data_pathstring) #Uses pandas to read the CSV as pandas dataframe
        range_dict_reference = str(spacecraft) + "F_050" #The range data is stored in the F_050 column
        range_data_panda = data_CSV[range_dict_reference]  #Extracts the range data from the CSV
        range_data = range_data_panda.to_numpy() #Converts the range data to a numpy array
        orbit_telem_statistics = []
        for j in range(0, len(telems)):
            telem = telems[j]
            telem_dict_reference = str(spacecraft) + telem
            telem_data_panda = data_CSV[telem_dict_reference][:]
            telem_data = telem_data_panda.to_numpy()
            range_2_telem_data = []
            for k in range(0, len(range_data)):
                if (range_data[k] == 2):
                    range_2_telem_data.append(telem_data[k]) #Extracts the telem data for range 2 to take a suborbital average
            telem_valid_range = telem_valid_ranges[j]
            valid_telem_data = [x for x in range_2_telem_data if ((x > telem_valid_range[0]) and (x < telem_valid_range[1]))] #Within a reasonable range
            non_zero_telem_data = [x for x in valid_telem_data if x != 0] #Are not 0
            if len(range_2_telem_data) == 0: 
                print("WARNING: Orbit %i file has no range 2 values" % orbit_number)
            elif len(validTelemData) ==0:
                print("WARNING: Orbit %i contains no valid (within expected range) temperature values" % orbit_number)
            elif len(non_zero_telem_data) ==0:
                print("WARNING: Orbit %i contains only zero values" % orbit_number)
            if (len(range_2_telem_data)==0): #If the length of the range 2 data is 0, then the orbit is discounted
                count = 0
                average = np.nan
                sigma = np.nan
                maximum = np.nan
                minimum = np.nan
            else:
                #Minimises the effects of strange 0 values
                if ((len(non_zero_telem_data)<len(validTelemData)) and (mean(non_zero_telem_data)<-10)):
                    validTelemData = non_zero_telem_data
                    print("WARNING: Orbit %i discarding zero values" % orbit_number);   
                count = len(validTelemData)
                average = mean(validTelemData)
                sigma = std(validTelemData)
                maximum = max(validTelemData)
                minimum = min(validTelemData)
            telemStatistic = [count,average,sigma,maximum,minimum]
            orbit_telem_statistics.append(telemStatistic)
            print(telem +" loop complete")
        spacecraftTelemStatistics.append(orbit_telem_statistics);
        print("Orbit " + str(orbit_number) + " loop complete")
    return spacecraftTelemStatistics;
            
#%% RUN THINGS
get_telem_files(1,0,3650);
spacecraftTelemStatistics = orbitAverages(spacecraft='1'); #In the order ["F_055", "F_074", "F_047", "F_048", "F_034", "J_236", "J_106", "J_213", "J_216"]
save("./C1SpacecraftTelemStatisticsTestV2(incl J)", spacecraftTelemStatistics);












