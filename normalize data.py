import os
import numpy as np



mean = np.zeros((20,5168))
for filename in os.listdir('npFiles/training data'):
    Arr = np.load('npFiles/training data/' + filename)
    mean = np.add(mean, Arr)
    print(filename)
mean = np.divide(mean,6368)

print("mean done")

std = np.zeros((20,5168))
#sum of value - mean squared
for filename in os.listdir('npFiles/training data'):
    Arr = np.load('npFiles/training data/' + filename)
    std = np.add(std,np.square(np.subtract(Arr,mean)))
    print(filename)

print("std step one done")

#divide by population then square root
std = np.divide(std,6368)
std = np.sqrt(std)

print("std done")

np.save('TrainingMeanImportant', mean)
np.save('TrainingStdImportant', std)