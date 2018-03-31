#### Fusion ####

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as p

import os
# import mord

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import keras
import numpy as np
import pandas as pd
from keras.models import Sequential,load_model, Model
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, BatchNormalization, Input
from keras.layers.core import Permute, Reshape
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import Regularizer
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.optimizers import Adam, Nadam, Adadelta
#
# %matplotlib inline
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

os.getcwd()
h5f = h5py.File('features_points.h5','r')
#h5f = h5py.File('model_data/Model_40feat/Audio_Features_DL40.h5','r')
fp_data =  h5f['points'][:]
h5f.close()

os.getcwd()
h5f = h5py.File('Audio_Features_DL.h5','r')
#h5f = h5py.File('model_data/Model_40feat/Audio_Features_DL40.h5','r')
audio_data = h5f['dataset_features'][:]
h5f.close()


targets = pd.read_csv('merged_labels.csv')
targets.head()

y = targets['Merged Arousal']
y1 = targets['Merged Dominance']
y2 = targets['Merged Emotion']
targets
y2
# def create_graph_emotions():
#
#     model.add(BatchNormalization(input_shape=(feat_dim,WINDOW_SIZE,1)))
#
#     model.add(Conv2D(num_feat_map, kernel_size=(1, 5),
#                  activation='relu',
#                  padding='same'))
#     #model.add(MaxPooling2D(pool_size=(1, 2)))
#     model.add(Dropout(0.5))
#     model.add(BatchNormalization())
#     model.add(Conv2D(num_feat_map, kernel_size=(1, 5), activation='relu',padding='same'))
#     #model.add(MaxPooling2D(pool_size=(1, 2)))
#     model.add(Dropout(0.5))
#
#     model.add(Permute((2, 1, 3))) # for swap-dimension
#     model.add(Reshape((-1,num_feat_map*feat_dim)))
#
#     model.add(LSTM(num_feat_map, return_sequences=False))
#     model.add(Dropout(0.5))
#     model.add(Dense(4,activation='softmax'))
#
#     model.summary()


WINDOW_SIZE = 25
num_feat_map = 32
feat_dim = 136

num_hidden_lstm = 128
num_hidden_units = 256
batch_size = 128
epochs = 100
model_patience = 20;

model = Sequential()

label_encoder = LabelEncoder()
y2 = label_encoder.fit_transform(y2)
y2
train_indx = p.load(open("train_indx.p", "rb"))
test_index = p.load(open("test_indx.p", "rb"))

x_train_fp = fp_data[train_indx]
y_train = y2[train_indx]

x_train_audio = audio_data[train_indx]
x_test_audio = audio_data[test_index]

x_test_fp = fp_data[test_index]
y_test= y2[test_index]

x_train_fp = np.reshape(x_train_fp, (x_train_fp.shape[0], x_train_fp.shape[1], -1))
x_test_fp = np.reshape(x_test_fp, (x_test_fp.shape[0], x_test_fp.shape[1], -1))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



_, timesteps, data_dim = x_train_audio.shape

epochs = 20
batch_size=32


# model.fit(x_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               verbose=1,
#               shuffle=True,
#               validation_data=(x_test, y_test))


input_1 = Input(shape=(WINDOW_SIZE, feat_dim), name='X_train_fp')
input_2 = Input(shape=(timesteps, data_dim), name='X_train_audio')


x = BatchNormalization()(input_1)
x = LSTM(num_feat_map, return_sequences=False)(x)
x = Dropout(0.5)(x)

# y = Reshape((-1, timesteps*data_dim))(input_2)
# y = LSTM(num_hidden_lstm, return_sequences=True)(y) # returns a sequence of vectors of dimension 128
# y = LSTM(num_hidden_lstm, return_sequences=False)(y)


y = BatchNormalization()(input_2)
y = LSTM(num_feat_map, return_sequences=False)(y)
y = Dropout(0.5)(y)

x.shape
y.shape

z = keras.layers.concatenate([x,y])
z = Reshape((-1, 64))(z)
z = BatchNormalization()(z)
z = LSTM(num_hidden_lstm, return_sequences=True)(z) # returns a sequence of vectors of dimension 128
z = Dropout(0.5)(z)
z = LSTM(num_hidden_lstm, return_sequences=False)(z)

output = Dense(4, activation='softmax')(z)

model = Model(inputs=[input_1, input_2], outputs=output)

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer= 'adam',
                  metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="Efusion_fp_aud_emotion.hdf5", monitor='val_acc', verbose=1, save_best_only=True)

H = model.fit([x_train_fp, x_train_audio], y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=([x_test_fp, x_test_audio], y_test)
            ,callbacks=[checkpointer]
            )


preds = model.predict([x_test_fp, x_test_audio])
preds = np.argmax(preds,axis=1)

y_test_arg = np.argmax(y_test,axis=1)
y_test_arg

confusion_matrix(y_test_arg,preds)

print(classification_report(y_test_arg,preds))
