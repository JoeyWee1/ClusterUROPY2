import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt


axes = ["X", "Y", "Z"]
best_ks = []
for sc in range(0,4): #Spacecrafts
    for axis in (1, 2): #Y and Z axes
        residuals = np.load(f'./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_12.6_KNN_Residuals.npy')

        mean_abs_error = np.mean(np.abs(residuals))
        standard_deviation = np.std(np.abs(residuals))

        plt.figure(figsize=(10, 6), dpi=150)
        plt.hist(residuals, bins=15, alpha=0.75, edgecolor='black')
        plt.title(f'Error Distribution for Spacecraft {sc+1} Axis {axes[axis]}')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.axvline(mean_abs_error, color='r', linestyle='dashed', linewidth=1)
        plt.savefig(f'./Outputs/Outputs_Error_Distributions/Spacecraft_{sc+1}_Axis_{axes[axis]}_Training-Years_12.6_KNN_Residuals_Histogram.png')

        # Output the mean absolute error
        print(f'Mean Absolute Error for Spacecraft {sc} Axis {axes[axis]}: {mean_abs_error}')
        print(f'Standard Deviation of Absolute Error for Spacecraft {sc} Axis {axes[axis]}: {standard_deviation}')

sc1y = ufloat(0.37, 0.3)
sc1z = ufloat(0.2, 0.18)
sc2y = ufloat(0.16, 0.13)
sc2z = ufloat(0.21, 0.12)
sc3y = ufloat(0.13, 0.08)
sc3z = ufloat(0.15, 0.12)
sc4y = ufloat(0.20, 0.16)
sc4z = ufloat(0.18, 0.08)

# List of ufloats
ufloats = [sc1y, sc1z, sc2y, sc2z, sc3y, sc3z, sc4y, sc4z]

# Calculate the average of the ufloats
average_ufloat = sum(ufloats) / len(ufloats)

# Print the average
print(f'Average of ufloats: {average_ufloat}')