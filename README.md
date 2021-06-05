# Anomaly Detecion in ECGs <img src="/assets/ecg_logo.png" alt="ECG" width="30"/>
With every heart beat, an electrical impulse travels through the heart which causes it to pump blood from and to the muscle. An Electrocardiogram or ECG is a test that measures and records this electrical activity. The medical test is used detect cardiac abnormalities in patients such as heart disease, abnormal heart rhythms, or an enlarged heart. 

## Project Overview
The purpose of this project is to use an unsupervised learning technique, specifically autoencoders, to detect anomalies in ECG data. You can find more information about the dataset and the model archiecture on the site linked below. After reading, you can try to detect anomalies yourself and see how you perform compared to the model and the ground truth

👉🏼 [Visit the Site]()

## Autoencoders
An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. It works by compressing input data to create a latent-space representation, and then reconstructing the data from this representation to create the output. It consists of two components - an encoder and a decoder. The encoder first learns a mapping from the data to a lower-dimensional latent space, and the decoder then learns a mapping from the latent space back to the original higher-dimensional space. **It is important to note that the model is only trained on normal data.** This makes it learn in an unsupervised manner by minimizing the error between the original data and the reconstructed data.

<img src="/assets/autoencoder.png" alt="Autoencoder"/>

Reconstruction error is a key calculation that helps distinguish between normal and anomaly data. It is defined as the distance between the original data point and its projection onto a lower-dimensional space. The reconstruction error for normal data is very low since the model has been trained on it and thus can generalize the data very well. However, the reconstruction error for anomaly data is high because the model has not been trained on it and so it cannot generalize the anomalous data. There exists a threshold value at which values above this point can be classified as anomalies and values lower than it can be classified as normal.
