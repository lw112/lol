##### HRI Model ######

import h5py
import pandas as pd
import os
import sklearn
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
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, Nadam, Adadelta

os.getcwd()
h5f = h5py.File('model_data/Audio_Features_DL.h5','r')
#h5f = h5py.File('model_data/Model_40feat/Audio_Features_DL40.h5','r')
data = h5f['dataset_features'][:]
h5f.close()

gender = pd.read_csv('gender_relational.csv')
gender = gender['Female (y=1)'].values

#  Shape is (num_samples(614) x timesteps(216) x features (20 MFCCs))

### Get labels
y = pd.read_csv('model_data/merged_labels.csv')
y_all = y['Merged Emotion'].values

# integer encode
label_encoder = LabelEncoder()
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
train_indx = random.sample(range(614), size_train)
test_index = list(ns[np.isin(ns,train_indx)==False])

#check
len(data[test_index])+len(data[train_indx])

# Now sample
y_train = onehot_encoded[train_indx]
y_test = onehot_encoded[test_index]

x_train = data[train_indx]
x_test = data[test_index]

gender_train = gender[train_indx]
gender_test = gender[test_index]

#X_train, X_test, y_train, y_test = train_test_split(x_train, onehot_encoded, test_size=0.33, random_state=42)

#### Model #####
# LSTM expects timesteps x features  (ie. 216 x 20 MFCC features)
_, timesteps, data_dim = x_train.shape

num_hidden_lstm = 128
num_hidden_units = 256
batch_size = 128
epochs = 300
model_patience = 20;

## Straightforward LSTM
# LSTM expects 3D data (batch_size, timesteps, features)
model = Sequential()
model.add(LSTM(num_hidden_lstm, input_shape=(timesteps,data_dim), return_sequences=True))  # returns a sequence of vectors of dimension 128
model.add(LSTM(num_hidden_lstm, return_sequences=False))  # return a single vector of dimension 128
model.add(Dense(4, activation='softmax'))


## Multi-stream model - Adding gender
aux_input = Input(shape=(1,),dtype='float32', name='gender')

input = Input(shape=(timesteps, data_dim), name='X_train')
x = Reshape((-1, timesteps*data_dim))(input)
x = LSTM(num_hidden_lstm, return_sequences=True)(x) # returns a sequence of vectors of dimension 128
x = LSTM(num_hidden_lstm, return_sequences=False)(x)  # return a single vector of dimension 128
x = keras.layers.concatenate([x,aux_input])
output = Dense(4, activation='softmax')(x)

model = Model(inputs=[input, aux_input], outputs=output)

model.summary()

model.compile(
              loss=keras.losses.categorical_crossentropy,
              #optimizer='adam',
              optimizer= Adam(lr=0.011),
              metrics=['accuracy'])

H = model.fit([x_train, gender_train], y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=([x_test, gender_test], y_test)
            #validation_split = 0.2,   #20% of last x_train and y_train taken for CV before shuffling
            #callbacks =[EarlyStopping(monitor='val_loss', patience= model_patience)]  #6 epochs patience
            )


# Confusion Matrix

# Happy - 2
# Schan - 3
# Court - 0
# Emb - 1
# predict = np.argmax(model.predict(X_test),1)
# confusion_matrix(np.argmax(y_test,1), predict, labels=[0,1,2,3])
