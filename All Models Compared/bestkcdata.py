import numpy as np

axes = ["X", "Y", "Z"]
best_ks = []
for sc in range(0,4): #Spacecrafts
    for axis in (1, 2): #Y and Z axes
        best_k = np.load(f'./Outputs/Outputs_Models_Data/Spacecraft_{sc}_Axis_{axes[axis]}_Training-Years_12.6_KNN_Best_K.npy')
        best_ks.append(best_k)

print(best_ks)