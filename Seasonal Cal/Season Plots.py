#This code splits the orbits into seasons based on their datetimes
import datetime
from orbits import Orbit
import numpy as np
from datasplit import season_create_and_split_data
import matplotlib.pyplot as plt

# The times by which to split the orbit
season_dates = [ #These are the starting times of each season
    datetime.datetime(1066, 10, 14, 0, 0, 0),
    datetime.datetime(2001, 3, 14, 15, 20, 33),
    datetime.datetime(2002, 3, 16, 1, 35, 48),
    datetime.datetime(2003, 3, 17, 12, 24, 1),
    datetime.datetime(2004, 3, 17, 21, 16, 29),
    datetime.datetime(2005, 3, 20, 22, 42, 11),
    datetime.datetime(2006, 3, 23, 3, 43, 35),
    datetime.datetime(2007, 4, 3, 2, 2, 3),
    datetime.datetime(2008, 4, 15, 8, 30, 32),
    datetime.datetime(2009, 5, 10, 12, 19, 15),
    datetime.datetime(2010, 5, 30, 18, 17, 1),
    datetime.datetime(2011, 5, 29, 21, 32, 13),
    datetime.datetime(2012, 3, 12, 1, 21, 20),
    datetime.datetime(2013, 3, 4, 8, 39, 21),
    datetime.datetime(2014, 3, 3, 11, 14, 57),
    datetime.datetime(2015, 3, 4, 20, 19, 53),
    datetime.datetime(2016, 3, 5, 5, 26, 53),
    datetime.datetime(2017, 3, 6, 15, 9, 59),
    datetime.datetime(2018, 3, 10, 6, 48, 50),
    datetime.datetime(2019, 3, 11, 16, 19, 9),
    datetime.datetime(2020, 3, 14, 8, 14, 15),
    datetime.datetime(2021, 3, 17, 23, 18, 49),
    datetime.datetime(2022, 3, 21, 15, 11, 22),
    datetime.datetime(2023, 3, 27, 12, 29, 1),
    datetime.datetime.now()
]

#%% Importing orbits
orbits = np.load("Seasonal Cal/cleanedOrbitsArrayV6(FinalRange2DataOnly).npy", allow_pickle = True)

#%% Splititng orbits into seasons
seasons = [[],[],[],[]] #This will be a 3D array, each subarray of spacecraft will contain subarrays of seasons
for sc in range(0, 4):
    for i in range(1, len(season_dates) - 1):
        season = []
        season_start_date = season_dates[i - 1]
        season_end_date = season_dates[i]
        for j in range(0, len(orbits[sc])):
            orbit_start_date = orbits[sc][j].startTime
            if (orbit_start_date >= season_start_date) and (orbit_start_date < season_end_date):
                season.append(orbits[sc][j])
        seasons[sc].append(season)


#%% Plot the offsets of the orbits for each season

axes = ['X', 'Y', 'Z']
for spacecraft in range(0, 4):
    for axis in range (0,3):
        for season in range(0, len(seasons[spacecraft])):
            print(f"Spacecraft {spacecraft+1} Axis {axis+1} Season {season+1} Length: {len(seasons[spacecraft][season])}")
            x_train_raw, x_test_raw, y_train_raw, y_test_raw, time_train, time_test, sc, axis, training_years, split_time = season_create_and_split_data(seasons[spacecraft][season], spacecraft, training_years=0, axis=axis)
            if(len(x_test_raw) == 0):
                break
            fig, axs = plt.subplots(len(x_test_raw[0]) + 1, 1, figsize=(10, 10))
            x_test_raw = np.array(x_test_raw)
            for i in range(0,len(x_test_raw[0])):
                axs[i].scatter(time_test, x_test_raw[:,i])
                axs[i].set_xlabel('Time')
                axs[i].set_ylabel(f'Feature {i+1}')
                axs[i].set_title(f'Spacecraft {spacecraft+1} Axis {axis+1} Season {season+1} Orbit {i+1}')

            axs[-1].scatter(time_test, y_test_raw)
            plt.tight_layout()
            plt.savefig(f"./Outputs_Season_Plots/C{spacecraft+1}/Time/Features {axes[axis]}-Axis Season {season+1}.png")
            # plt.show()
            plt.clf()

    



    

