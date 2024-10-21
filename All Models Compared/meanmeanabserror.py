import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt


axes = ["X", "Y", "Z"]
best_ks = []
ufloats = []
for sc in range(0,4): #Spacecrafts
    for axis in (1, 2): #Y and Z axes
        residuals = np.load(f'./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_15_LSTM_Residuals.npy')

        mean_abs_error = np.mean(np.abs(residuals))
        standard_deviation = np.std(np.abs(residuals))

        plt.figure(figsize=(10, 6), dpi=150)
        plt.hist(residuals, bins=15, alpha=0.75, edgecolor='black')
        plt.title(f'Error Distribution for Spacecraft {sc+1} Axis {axes[axis]}')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.axvline(mean_abs_error, color='r', linestyle='dashed', linewidth=1)
        plt.savefig(f'./Outputs/Outputs_Error_Distributions/Spacecraft_{sc+1}_Axis_{axes[axis]}_Training-Years_15_LSTM_Residuals_Histogram.png')

        # Output the mean absolute error
        print(f'Mean Absolute Error for Spacecraft {sc} Axis {axes[axis]}: {mean_abs_error} +/- {standard_deviation}')

        ufloats.append(ufloat(mean_abs_error, standard_deviation))

# ufloats = [ufloat(0.42,0.24), ufloat(0.12,0.10), ufloat(0.15,0.12), ufloat(0.15,0.11)]
average_ufloat = sum(ufloats) / len(ufloats)

# Print the average
print(f'Average of ufloats: {average_ufloat}')