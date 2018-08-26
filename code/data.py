import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

keep_everything = ['Occupancy']
no_time = ['Occupancy', 'time/hour', 'time/minute']
no_time_light = ['Occupancy', 'time/hour', 'time/minute', 'Light']
no_light = ['Occupancy', 'Light']
just_co2_humid = ['Occupancy', 'time/hour', 'time/minute', 'Light', 'Temperature', 'HumidityRatio']
just_diffs = ['Temperature','Humidity','Light','CO2','HumidityRatio','Occupancy','time/hour','time/minute','Temperature_d1','Temperature_d3','Temperature_d5']
features_to_drop = just_diffs

# Loading Train Data
trainDataFrame = pd.read_csv('processed_data/datatraining.csv')
y_train = trainDataFrame['Occupancy'].values.tolist()
y_train = LabelBinarizer().fit_transform(y_train)
y_train = np.hstack((y_train, 1 - y_train))
x_train = trainDataFrame.drop(features_to_drop, axis=1)

# Load Test Data
testDataFrame = pd.read_csv('processed_data/datatest.csv')
y_test = testDataFrame['Occupancy'].values.tolist()
y_test = LabelBinarizer().fit_transform(y_test)
y_test = np.hstack((y_test, 1 - y_test))
x_test = testDataFrame.drop(features_to_drop, axis=1)
