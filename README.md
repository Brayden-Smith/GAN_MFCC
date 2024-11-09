# Purpose
This project aims to create a Generate adversial network using data from the bytedance midi piano data set

https://github.com/bytedance/GiantMIDI-Piano

Although the results from this project are not ground breaking many lessons and practices on machine learning were found.

# The Data
The data, once downloaded as midis, would be transformed into Mel-frequency cepstral coefficients. This data type is a 2d array representing varioues discrete time snippets in the data. At each time snippet the y-axis represents the frequency wave and its element represents the strength of that frequency. For example Arr[0][0] reresents the strength of "wave 0" at time zero.

![image](https://github.com/Brayden-Smith/TeamScheduler/blob/master/image_2024-11-09_181107507.png)
This data type was chosen as 2d arrays are easy to work with and can easily be interpreted by the machine model. 
The downsides of this choice was the amount of time and effort taken to transform the data into this graph and this transformation effectively results in a 30% quality in the audio once turned back.

# Repo Setup
In this repo there is the machine learning model as well as the code used to transform the data into the mfcc graphs. 
The files containing the code for the data transformation scripts are adequately named. Included in this Repo is trained weights and biases from the model at epoch 49.  Their is also a file with some output wav files the model created
