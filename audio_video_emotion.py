import numpy as np
import h5py
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import os
import itertools

import keras
from keras.utils import np_utils

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Bidirectional, Input, concatenate, Flatten
from keras.layers.core import Permute, Reshape
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



# Keep randomness the same
np.random.seed(0)


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.height', 1000)
pd.set_option('display.width', 1000)


# Write the data into a HDF5 File
# h5f = h5py.File('Audio_Features_DL.h5', 'w')
# h5f.create_dataset('dataset_features', data=x_train)
# h5f.close()


# Labels
df_labels = pd.read_csv('merged_labels.csv', encoding="ISO-8859-1")  # Shape is ()

def labelling(x):
	if x == 'Happy':
		return 0
	elif x == 'Schadenfreude':
		return 1
	elif x == 'Embarrassment':
		return 2
	elif x == 'Courtesy':
		return 3


df_labels['Merged Emotion'] = df_labels['Merged Emotion'].apply( lambda x: labelling(x)  )

df_labels['Merged Emotion'].value_counts()
train_labels = np.array( df_labels['Merged Emotion'] )


####################################### Audio Model  ##################################################

# Loading the File
h5f = h5py.File('Audio_Features_DL.h5','r')
group_name = list(h5f.keys())[0]
x_train_audio = h5f[group_name][:]
h5f.close()



# LSTM expects timesteps x features  (ie. 216 x 20 MFCC features)
_, timesteps, data_dim = x_train_audio.shape

x_train, x_cv, y_train_labels, y_cv_labels = train_test_split(x_train_audio, train_labels, test_size=0.20, random_state=42)

y_train = keras.utils.to_categorical( y_train_labels, len(np.unique(train_labels)))
y_cv = keras.utils.to_categorical( y_cv_labels, len(np.unique(train_labels)))


num_hidden_lstm = 128
num_hidden_units = 256
batch_size = 128
epochs = 50
model_patience = 10


#LSTM expects 3D data (batch_size, timesteps, features)
model_audio = Sequential()
model_audio.add(BatchNormalization(input_shape=(timesteps,data_dim)))
# model_audio.add(Bidirectional(LSTM(num_hidden_lstm, return_sequences=True)))
model_audio.add(LSTM(num_hidden_lstm, return_sequences=True))
# model_audio.add(LSTM(num_hidden_lstm, input_shape=(timesteps,data_dim), return_sequences=True))  # returns a sequence of vectors of dimension 128
model_audio.add(BatchNormalization())
model_audio.add(LSTM(num_hidden_lstm, return_sequences=False))  # return a single vector of dimension 128
# model_audio.add(Bidirectional(LSTM(num_hidden_lstm, return_sequences=False)))
model_audio.add(Dense(len(np.unique(train_labels)), activation='softmax'))


model_audio.summary()

model_audio.compile(
              loss=keras.losses.categorical_crossentropy,   # For emotional expression
              metrics=['accuracy'], 						# For emotional expression
              optimizer='adam')

checkpoint = ModelCheckpoint(filepath="audio_weights_emotion.hdf5", monitor='val_loss', verbose=1, save_best_only=True)

model_audio.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=(x_cv, y_cv),
            # validation_split = 0.20,   #20% of last x_train and y_train taken for CV before shuffling
            callbacks = [checkpoint, EarlyStopping(monitor='val_loss', patience= model_patience)]  #9 epochs patience
            )

model_audio.load_weights("audio_weights_emotion.hdf5")

# Predict the scores
predicted_audio_labels_probs = model_audio.predict(x_cv)
predicted_audio_labels = np.argmax(predicted_audio_labels_probs, axis=1)

# Evaluation Metric (Emotion)
print ('Test Accuracy: ', sum(predicted_audio_labels == y_cv_labels)/y_cv_labels.shape[0])
confusion_matrix(y_cv_labels,predicted_audio_labels).T
print (metrics.classification_report(y_cv_labels, predicted_audio_labels))



############################### Audio Multi-Stream (Gender & Laugh Duration) #######################################

gender = pd.read_csv('gender_relational.csv')
gender = gender['Female (y=1)'].values

length = pd.read_csv('subclip_lengths.csv')
length = length.iloc[0:614,1].values

gender_train, gender_cv, _, _ = train_test_split(gender, train_labels, test_size=0.20, random_state=42)
length_train, length_cv, _, _ = train_test_split(length, train_labels, test_size=0.20, random_state=42)


aux_input = Input(shape=(1,),dtype='float32', name='gender')
aux_input_2 = Input(shape=(1,),dtype='float32', name='length')
input = Input(shape=(timesteps, data_dim), name='X_train')

y = BatchNormalization()(input)
y = LSTM(num_hidden_lstm,return_sequences=True)(y)
y = BatchNormalization()(y)
y = LSTM(num_hidden_lstm, return_sequences=False)(y)
y = Dropout(0.5)(y)

# x = Reshape((-1, timesteps*data_dim))(input)
# x = LSTM(num_hidden_lstm, return_sequences=False)(x)  # return a single vector of dimension 128

x = keras.layers.concatenate([y, aux_input, aux_input_2])
output = Dense(len(np.unique(train_labels)), activation='softmax')(x)

model_audio = Model(inputs=[input, aux_input, aux_input_2], outputs=output)
model_audio.summary()

model_audio.compile(
					loss=keras.losses.categorical_crossentropy,
					metrics=['accuracy'],
					optimizer='adam')

checkpoint = ModelCheckpoint(filepath="audio_weights_multistream_emotion.hdf5", monitor='val_loss', verbose=1, save_best_only=True)

model_audio.fit([x_train, gender_train, length_train], y_train,
				batch_size=batch_size,
				epochs=epochs,
				verbose=1,
				shuffle=True,
				validation_data=([x_cv, gender_cv, length_cv], y_cv),
				# validation_split = 0.20,   #20% of last x_train and y_train taken for CV before shuffling
				callbacks = [checkpoint, EarlyStopping(monitor='val_loss', patience= model_patience)]  #9 epochs patience
            )

model_audio.load_weights("audio_weights_multistream_emotion.hdf5")

# Predict the scores
predicted_audio_labels_probs = model_audio.predict([x_cv, gender_cv, length_cv])
predicted_audio_labels = np.argmax(predicted_audio_labels_probs, axis=1)

# Evaluation Metric (Emotion)
print ('Test Accuracy: ', sum(predicted_audio_labels == y_cv_labels)/y_cv_labels.shape[0])
confusion_matrix(y_cv_labels,predicted_audio_labels).T
print (metrics.classification_report(y_cv_labels, predicted_audio_labels))

################################################################################################




####################################### Video Model  ##################################################

# h5f = h5py.File('Video_Features_RGB_DL.h5','r')
h5f = h5py.File('Cropped_Video_Features_RGB_DL.h5','r')
# h5f = h5py.File('Video_Features_Grayscale_DL.h5','r')
group_name = list(h5f.keys())[0]
x_train_video = h5f[group_name][:]
h5f.close()


# x_train_video.shape: (614, 25, 70, 70, 1) 
frames = x_train_video.shape[1]
width = x_train_video.shape[2]
height = x_train_video.shape[3]
channels = x_train_video.shape[4]  # Grayscale (1) or RGB (3)

num_hidden_lstm = 128
num_hidden_units = 256
batch_size = 128
epochs = 50
model_patience = 10


model_vgg16 = VGG16(weights='imagenet', include_top=False)  # Feature extracted is (2, 2, 512) for each input (video frame)

x_train_video_features = []

# Loop through the video training data to extract features for each frame
for video in x_train_video:
	# frame is (70,70,1) or (70,70,3) depending on features
	# model expects (1,70,70,3)  -  The RGB version
	video_features = [model_vgg16.predict(np.expand_dims(frame, axis=0)).reshape(2048,-1) for frame in video]
	x_train_video_features.append(video_features)


x_train_video_features = np.array(x_train_video_features)  # (num_laughters, num_frames (25), features extracted (2048), 1)
x_train_video_features = x_train_video_features.reshape(len(x_train_video_features), 25, 2048)  #2048 features for each of the 25 frames


train_labels = np.array( df_labels['Merged Emotion'] )

x_train, x_cv, y_train_labels, y_cv_labels = train_test_split(x_train_video_features, train_labels, test_size=0.20, random_state=42)

y_train = keras.utils.to_categorical( y_train_labels, len(np.unique(train_labels)))
y_cv = keras.utils.to_categorical( y_cv_labels, len(np.unique(train_labels)))


# LSTM expects timesteps x features  (ie. 25 x 2048 Frame features)
_, timesteps, data_dim = x_train_video_features.shape

model_video = Sequential()
# model_video.add(Flatten(input_shape=(25, 2048)))
# model_video.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model_video.add(LSTM(num_hidden_lstm, input_shape=(timesteps,data_dim), return_sequences=True))  # returns a sequence of vectors of dimension 128
model_video.add(BatchNormalization())
model_video.add(LSTM(num_hidden_lstm, return_sequences=False))  # return a single vector of dimension 128
model_video.add(Dense(len(np.unique(train_labels)), activation='softmax'))

model_video.summary()

model_video.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='Adam',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath="video_weights_noncropped_emotion.hdf5", monitor='val_loss', verbose=1, save_best_only=True)
checkpoint = ModelCheckpoint(filepath="video_weights_cropped_emotion.hdf5", monitor='val_acc', verbose=1, save_best_only=True)  # Val acc for cropped version

model_video.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_cv, y_cv),
            shuffle=True,
            callbacks = [checkpoint, EarlyStopping(monitor='val_loss', patience= model_patience)])  # 10 epochs patience

model_video.load_weights("video_weights_noncropped_emotion.hdf5")
model_video.load_weights("video_weights_cropped_emotion.hdf5")


# Predict the scores
predicted_video_labels = np.argmax(model_video.predict(x_cv), axis=1)
predicted_video_labels_probs = model_video.predict(x_cv)

# Evaluation Metric (Emotion)
print ('Test Accuracy: ', sum(predicted_video_labels == y_cv_labels)/y_cv_labels.shape[0])
confusion_matrix(y_cv_labels,predicted_video_labels).T
print (metrics.classification_report(y_cv_labels, predicted_video_labels))




####################################### Ensemble Model  ##################################################

# Average them out equally and try unequal weighting too
ensemble_predictions_probs = (predicted_video_labels_probs+predicted_audio_labels_probs)/2  # Weighted equally
ensemble_predictions_labels = np.argmax(ensemble_predictions_probs, axis=1)


# Evaluation Metric (Emotion)
print ('Test Accuracy: ', sum(ensemble_predictions_labels == y_cv_labels)/y_cv_labels.shape[0])
confusion_matrix(y_cv_labels,ensemble_predictions_labels).T
print (metrics.classification_report(y_cv_labels, ensemble_predictions_labels))



# Unequally weighted ensemble
ensemble_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

best_ensemble_accuracy = 0.0
best_ensemble_audio_weight = 0.0
best_ensemble_video_weight = 0.0

for i in zip(ensemble_weights, ensemble_weights[::-1]):  
	audio_weight, video_weight = i  # i is a tuple (sum of the weights should be 1)
	prediction_probs = (audio_weight * predicted_audio_labels_probs) + (video_weight * predicted_video_labels_probs)
	prediction_probs_labels = np.argmax(prediction_probs, axis=1)
	ensemble_accuracy = sum(prediction_probs_labels == y_cv_labels)/y_cv_labels.shape[0]
	if ensemble_accuracy > best_ensemble_accuracy:
		best_ensemble_audio_weight = audio_weight
		best_ensemble_video_weight = video_weight
		best_ensemble_accuracy = ensemble_accuracy


ensemble_predictions_probs = (best_ensemble_audio_weight * predicted_audio_labels_probs) + (best_ensemble_video_weight * predicted_video_labels_probs)  # Weighted Unequally
ensemble_predictions_labels = np.argmax(ensemble_predictions_probs, axis=1)

# Evaluation Metric (Emotion)
print ('Test Accuracy: ', sum(ensemble_predictions_labels == y_cv_labels)/y_cv_labels.shape[0])
confusion_matrix(y_cv_labels,ensemble_predictions_labels).T
print (metrics.classification_report(y_cv_labels, ensemble_predictions_labels))
















