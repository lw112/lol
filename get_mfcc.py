import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import os

import keras
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization


from keras.callbacks import EarlyStopping

# Audio  MFCC Package
import librosa

# Merge the audio and video in the folders
import subprocess


# Keep randomness the same
np.random.seed(0)


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.height', 1000)
pd.set_option('display.width', 1000)


# Downsample all the laughs to 2.5 seconds = (44100 samples/sec * 2.5)   @44KHz
num_samples = 110250  # Each audio file should have length of 110,250 samples
num_MFCC = 20  # Number of MFCC features  (this will result in each audio having 20x216 input shape)

input_data = []


# Save home directory
project_home_dir = os.getcwd()


# Empty Annotations File
df_annotations = pd.DataFrame(index=range(0,0), columns=['Type', 'Start Time (sec)', 'End Time (sec)'])


for directory, subdirs, files in os.walk("./data/Sessions"):
	for subdir in subdirs:
		# Inside the session folder
		os.chdir( os.path.join(project_home_dir, directory, subdir) )
		
		# List the files in current directory
		audio_file = [f for f in os.listdir('.') if (os.path.isfile(f) and '.wav' in f)][0]  #Find the audio file
		video_file = [f for f in os.listdir('.') if (os.path.isfile(f) and '.avi' in f)][0]	 # Find the video file
		
		df_annotation = pd.read_csv('laughterAnnotation.csv', encoding="ISO-8859-1")
		
		# Only include "Laughter" or "PosedLaughter" and only columns [ Type  Start Time (sec)  End Time (sec)]
		df_annotation = df_annotation.loc[(df_annotation['Type'] == 'Laughter') | (df_annotation['Type']=='PosedLaughter')].iloc[:, 1:4]
		# Concatenate it to the main annotations file
		df_annotations = pd.concat([df_annotations,df_annotation])
		
		# Go through the annotations for this audio only
		for row in np.array(df_annotation):
			#row is:  	[Type   start_time	end_time]
			start_time = row[1]
			end_time = row[2]
			# Load "durarion" seconds of a wav file, starting "offset" seconds in
			# y is the audio time series samples
			# sr is the sampling rate at which the audio file was samples  (default sr=22050). 
			y, sr = librosa.load(audio_file, offset=start_time, duration=(end_time-start_time), sr=None)
			# If the audio size is bigger than the desired number of samples, downsample it
			if len(y) > num_samples:
				downsampled_audio = [ y[ int(np.floor(i)) ] for i in np.linspace(0,len(y)-1, num_samples)]
			#Just pad the end with zeros
			else:
				padded_zeros = [0 for i in range(0, num_samples-len(y))]
				downsampled_audio = list(y) + padded_zeros
			downsampled_audio = np.array(downsampled_audio)
			# Find MFCC features
			# y is the audio time series and sr is the sampling rate of y, 
			MFCCs  = librosa.feature.mfcc(y=downsampled_audio, sr=sr, n_mfcc=num_MFCC)
			#20ms window (frame) with 10ms stride (overlap is 10ms)
			# print (librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20,hop_length=int(0.010*sr), n_fft=int(0.020*sr) ).shape)
			input_data.append( MFCCs )
	# Return back to home directory
	os.chdir( project_home_dir )
	break



x_train = np.array(input_data)

# TODO (input the labels)
# y_train = keras.utils.to_categorical(np.array(df_annotations['label']), num_classes)

# This is just random labels
num_classes = 4
y_train = keras.utils.to_categorical(np.array([np.random.randint(0,4) for i in range(0, x_train.shape[0])])  , num_classes)


# LSTM expects timesteps x features  (ie. 216 x 20 MFCC features)
x_train = np.swapaxes(x_train,1,2)

_, timesteps, data_dim = x_train.shape

num_hidden_lstm = 128
num_hidden_units = 256
batch_size = 128
# stateful_batch_size = 
epochs = 14
model_patience = 20;



#LSTM expects 3D data (batch_size, timesteps, features)
model = Sequential()
model.add(LSTM(num_hidden_lstm, input_shape=(timesteps,data_dim), return_sequences=True))  # returns a sequence of vectors of dimension 128
model.add(LSTM(num_hidden_lstm, return_sequences=False))  # return a single vector of dimension 128
model.add(Dense(4, activation='softmax'))


model.summary()

model.compile(
              loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

H = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            # validation_data=(x_valid, y_valid),
            validation_split = 0.2,   #20% of last x_train and y_train taken for CV before shuffling
            callbacks =[EarlyStopping(monitor='val_loss', patience= model_patience)]  #6 epochs patience
            )
