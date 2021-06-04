import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import model

df = pd.read_csv('ECG5000/ecg_data.txt', sep='  ', header=None)
df = df.add_prefix('column_')

train_data, test_data, train_labels, test_labels = train_test_split(df.values, df.values[:,0:1], test_size=0.2, random_state=100)
scaler = MinMaxScaler()
scaled_data = scaler.fit(train_data)

scaled_train_data = scaled_data.transform(train_data)
scaled_test_data = scaled_data.transform(test_data)

normal_train_data = pd.DataFrame(scaled_train_data).add_prefix("column_").query('column_0 == 0').values[:, 1:]
anomaly_train_data = pd.DataFrame(scaled_train_data).add_prefix("column_").query('column_0 > 0').values[:, 1:]

normal_test_data = pd.DataFrame(scaled_test_data).add_prefix("column_").query('column_0 == 0').values[:, 1:]
anomaly_test_data = pd.DataFrame(scaled_test_data).add_prefix("column_").query('column_0 > 0').values[:, 1:]

my_model = model.Autoencoder()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,mode='min')

my_model.compile(optimizer='adam', loss='mae')
history = my_model.fit(normal_train_data, normal_train_data, 
            epochs=50, 
            batch_size=128, 
            validation_data=(scaled_train_data[:,1:], scaled_train_data[:,1:]), 
            shuffle=True, 
            # callbacks=[early_stopping]
            )

my_model.save('autoencoder_model')


                                    