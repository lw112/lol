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
y_dom = y['Merged Dominance'].values


# integer encode
label_encoder = LabelEncoder()
#integer_encoded = label_encoder.fit_transform(y_all)
#integer_encoded = label_encoder.fit_transform(y_arous)
integer_encoded = label_encoder.fit_transform(y_dom)

# binary encode
enc = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = enc.fit_transform(integer_encoded)
print(onehot_encoded)
onehot_encoded.shape


# Sampling on multiple datasets
ns=np.array(range(614))
size_train = int(round(614*0.66,0))
# train_indx = random.sample(range(614), size_train)
# test_index = list(ns[np.isin(ns,train_indx)==False])

train_indx = p.load(open("train_indx.p", "rb"))
test_index = p.load(open("test_indx.p", "rb"))


#check
len(data[test_index])+len(data[train_indx])

# Now sample
y_train = integer_encoded[train_indx]
y_test = integer_encoded[test_index]

x_train = data[train_indx]
x_test = data[test_index]

gender_train = gender[train_indx]
gender_test = gender[test_index]

length_train = length[train_indx]
length_test = length[test_index]

#X_train, X_test, y_train, y_test = train_test_split(x_train, onehot_encoded, test_size=0.33, random_state=42)

#### Model #####
# LSTM expects timesteps x features  (ie. 216 x 20 MFCC features)
_, timesteps, data_dim = x_train.shape

num_hidden_lstm = 32
num_hidden_units = 256
batch_size = 128
epochs = 100
model_patience = 20;

# # W/O Batch
# model = Sequential()
# model.add(LSTM(num_hidden_lstm, input_shape=(timesteps,data_dim), return_sequences=True))  # returns a sequence of vectors of dimension 128
# model.add(LSTM(num_hidden_lstm, return_sequences=False))  # return a single vector of dimension 128
# model.add(Dense(1, activation='softmax'))

# Add Batch Normalisation
# model = Sequential()
# model.add(BatchNormalization(input_shape=(timesteps,data_dim)))
# model.add(LSTM(num_hidden_lstm, return_sequences=True))  # returns a sequence of vectors of dimension 128
# model.add(BatchNormalization())
# model.add(LSTM(num_hidden_lstm, return_sequences=False))  # return a single vector of dimension 128
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='linear'))
#
# model.summary()

# Multi-stream model - Adding gender
aux_input = Input(shape=(1,),dtype='float32', name='gender')
aux_input_2 = Input(shape=(1,),dtype='float32', name='length')
input = Input(shape=(timesteps, data_dim), name='X_train')

y = BatchNormalization()(input)
y = LSTM(num_hidden_lstm,return_sequences=True)(y)
y = BatchNormalization()(y)
y = LSTM(num_hidden_lstm, return_sequences=False)(y)
y = Dropout(0.5)(y)

x = keras.layers.concatenate([y,aux_input, aux_input_2])
output = Dense(1, activation='linear')(x)

model = Model(inputs=[input, aux_input, aux_input_2], outputs=output)


# x = Reshape((-1, timesteps*data_dim))(input)
# x = LSTM(num_hidden_lstm, return_sequences=True)(x) # returns a sequence of vectors of dimension 128
# x = LSTM(num_hidden_lstm, return_sequences=False)(x)  # return a single vector of dimension 128




# ### Regression for Dominance  # 45.93%
model.compile(
              loss='mean_squared_error',
              #optimizer='adam',
              optimizer= Adam(lr=0.0011),
              metrics=['mse'])

checkpointer = ModelCheckpoint(filepath="weights_dom_simple_ms.hdf5", monitor='val_mean_squared_error', verbose=1, save_best_only=True)

#[x_train, gender_train,length_train]
#[x_test, gender_test, length_test]

# H = model.fit(x_train, y_train,
#             batch_size=batch_size,
#             epochs=epochs,
#             verbose=1,
#             shuffle=True,
#             validation_data=(x_test, y_test),
#             callbacks =[checkpointer, EarlyStopping(monitor='val_mean_squared_error', patience= model_patience)]  #6 epochs patience
#             #callbacks=[checkpointer]
#             )

H = model.fit([x_train, gender_train, length_train], y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=([x_test, gender_test, length_test], y_test)
            ,callbacks=[checkpointer]
            )

#
# model.load_weights("weights_dom.hdf5")
# model.compile(
#               loss='mean_squared_error',
#               #optimizer='adam',
#               optimizer= Adam(lr=0.0011),
#               metrics=['accuracy'])
# predict = np.argmax(model.predict(x_test),1)
# confusion_matrix(np.argmax(y_test,1), predict, labels=[0,1,2,3,4,5])
model.load_weights("weights_dom_simple_ms.hdf5")
preds = model.predict([x_test, gender_test, length_test])
#np.savetxt('Dominance_audio_LSTM.txt', y_test, delimiter=',')
np.savetxt('Dominance_audio_predictions_LSTM_ms.txt', preds, delimiter=',')
preds = preds.reshape(preds.shape[0])

preds

# Prediction Eval
mean_y_pred = np.mean(preds)
print("Mean of predictions is: "+str(mean_y_pred))
mse = np.mean((y_test - preds)**2)
print("Variance (benchmark) is: "+str(mse))

# Benchmark Eval
mean_y = np.mean(y_test)
print("Mean of test set is: "+str(mean_y))

variance = np.mean((y_test - np.mean(y_test))**2)
print("Variance (benchmark) is: "+str(variance))

try:
    f = open('Dominance_audio_multistream_optimised.txt','w+')
    f.write('Mean of predictions is: ')
    f.write(str(mean_y_pred))
    f.write('\nVariance is: ')
    f.write(str(mse))
    f.write('\nMean of test set is: ')
    f.write(str(mean_y))
    f.write('\nVariance (benchmark) is: ')
    f.write(str(variance))
    f.write('%\n')
    f.close()
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except ValueError:
    print("Could not convert data to an integer.")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise
