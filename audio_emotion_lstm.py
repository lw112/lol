##### HRI Model ######

import h5py
import pandas as pd
import os
import mord
import pickle as p

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import keras
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Input, concatenate
from keras.layers.convolutional import Conv3D
from keras.layers.core import Permute, Reshape
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, Nadam, Adadelta

os.getcwd()
h5f = h5py.File('model_data/Audio_Features_DL.h5','r')
#h5f = h5py.File('model_data/Model_40feat/Audio_Features_DL40.h5','r')
data = h5f['dataset_features'][:]
h5f.close()

gender = pd.read_csv('gender_relational.csv')
gender = gender['Female (y=1)'].values

gender.shape

length = pd.read_csv('model_data/subclip_lengths.csv')
length = length['length'][0:614].values

#  Shape is (num_samples(614) x timesteps(216) x features (20 MFCCs))

### Get labels
y = pd.read_csv('model_data/merged_labels.csv')
y_all = y['Merged Emotion'].values

# integer encode
label_encoder = LabelEncoder()
#integer_encoded = label_encoder.fit_transform(y_all)
#integer_encoded = label_encoder.fit_transform(y_arous)
integer_encoded = label_encoder.fit_transform(y_all)

# binary encode
enc = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = enc.fit_transform(integer_encoded)
print(onehot_encoded)
onehot_encoded.shape


# Sampling on multiple datasets
ns=np.array(range(614))
size_train = int(round(614*0.66,0))
train_indx = p.load(open("train_indx.p", "rb"))
test_index = p.load(open("test_indx.p", "rb"))


# train_indx = random.sample(range(614), size_train)
# test_index = list(ns[np.isin(ns,train_indx)==False])

# p.dump(train_indx, open("train_indx.p", "wb"))
# p.dump(test_index, open("test_indx.p", "wb"))

#check
len(data[test_index])+len(data[train_indx])

# Now sample
y_train = onehot_encoded[train_indx]
y_test = onehot_encoded[test_index]

x_train = data[train_indx]
x_test = data[test_index]

gender_train = gender[train_indx]
gender_test = gender[test_index]

length_train = length[train_indx]
length_test = length[test_index]

length_train.shape
#X_train, X_test, y_train, y_test = train_test_split(x_train, onehot_encoded, test_size=0.33, random_state=42)

#### Model #####
# LSTM expects timesteps x features  (ie. 216 x 20 MFCC features)
_, timesteps, data_dim = x_train.shape

num_hidden_lstm = 128
num_hidden_units = 256
batch_size = 128
epochs = 25
model_patience = 20;
num_feat_map = 128 #32

# Straightforward LSTM 51.20%
#LSTM expects 3D data (batch_size, timesteps, features)
# model = Sequential()
# model.add(LSTM(num_hidden_lstm, input_shape=(timesteps,data_dim), return_sequences=True))  # returns a sequence of vectors of dimension 128
# model.add(LSTM(num_hidden_lstm, return_sequences=False))  # return a single vector of dimension 128
# model.add(Dense(4, activation='softmax'))


# iNCLUDE batch # 49% accuracy
model = Sequential()
model.add(BatchNormalization(input_shape=(timesteps,data_dim)))
model.add(LSTM(num_feat_map,return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(num_feat_map, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(4,activation='softmax'))
model.summary()

# iNCLUDE batch # 49% accuracy
# model = Sequential()
# model.add(BatchNormalization(input_shape=(timesteps,data_dim)))
# model.add(LSTM(num_feat_map,return_sequences=True))
# model.add(BatchNormalization())
# model.add(LSTM(num_feat_map, return_sequences=False))
# #model.add(Dropout(0.5))
# model.add(Dense(4,activation='softmax'))
# model.summary()

# batch_size = 16
# model_patience = 20
# epochs = 100
# number_of_classes = 4

# Multi-stream model - Adding gender
aux_input = Input(shape=(1,),dtype='float32', name='gender')
aux_input_2 = Input(shape=(1,),dtype='float32', name='length')
input = Input(shape=(timesteps, data_dim), name='X_train')

y = BatchNormalization()(input)
y = LSTM(num_feat_map,return_sequences=True)(y)
y = BatchNormalization()(y)
y = LSTM(num_feat_map, return_sequences=False)(y)
y = Dropout(0.5)(y)

# x = Reshape((-1, timesteps*data_dim))(input)
# x = LSTM(num_hidden_lstm, return_sequences=True)(x) # returns a sequence of vectors of dimension 128
# x = LSTM(num_hidden_lstm, return_sequences=False)(x)  # return a single vector of dimension 128

x = keras.layers.concatenate([y,aux_input, aux_input_2])
output = Dense(4, activation='softmax')(x)




model = Model(inputs=[input, aux_input, aux_input_2], outputs=output)

model.summary()


# 50.24%
model.compile(
              loss=keras.losses.categorical_crossentropy,
              #optimizer='adam',
              optimizer= Adam(lr=0.011),
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="weights_emo_multi.hdf5", monitor='val_acc', verbose=1, save_best_only=True)

# H = model.fit(x_train, y_train,
#             batch_size=batch_size,
#             epochs=epochs,
#             verbose=1,
#             shuffle=True,
#             validation_data=(x_test, y_test),
#             callbacks =[checkpointer]
#             )

H = model.fit([x_train, gender_train, length_train], y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=([x_test, gender_test, length_test], y_test)
            ,callbacks=[checkpointer]
            )


model.load_weights("weights_emo_multi.hdf5")
# model.compile(
#               loss='mean_squared_error',
#               #optimizer='adam',
#               optimizer= Adam(lr=0.0011),
#               metrics=['accuracy'])


preds = model.predict([x_test, gender_test, length_test])
preds

preds = np.argmax(preds,axis=1)
y_test_arg = np.argmax(y_test,axis=1)
confusion_matrix(y_test_arg,preds)
print(classification_report(y_test_arg,preds))
np.savetxt('Emotion_audio_multi_preds.txt', preds, delimiter=',')
np.savetxt('Emotion_audio_multi_results.txt', confusion_matrix(y_test_arg,preds), delimiter=',')
np.savetxt('Emotion_audio_multi_results.txt', classification_report(y_test_arg,preds), delimiter=',')
