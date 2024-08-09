#This code turns the cleanCommandsArray.npy into a series of datetimes
import numpy as np
from datetime import datetime

class Command: #Object to store each command
    def __init__(self,command,datetime):
        self.command = command
        self.datetime = datetime

# Importing the commands
commands = np.load("Seasonal Cal/cleanCommandsArray.npy", allow_pickle = True)

# Converting the commands into a series of datetimes
command_datetimes = [[],[],[],[]]
for i in range(0, len(commands)):
    for j in range(0, len(commands[i])):
        command_datetimes[i].append(commands[i][j].datetime)

# Save the array of datetimes
np.save("Seasonal Cal/command_datetimes.npy", command_datetimes)
