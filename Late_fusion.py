import h5py
import pandas as pd
import os

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
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Input, concatenate, Conv1D
from keras.layers.convolutional import Conv3D
from keras.layers.core import Permute, Reshape
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, Nadam, Adadelta



images = h5py.File('features_images.h5','r')
data_imgs = images['images'][:]

y = pd.read_csv('merged_labels.csv')
emotion_lables = y['Merged Emotion'].values
label_encoder = LabelEncoder()
emotion_lables_encoded = label_encoder.fit_transform(emotion_lables)
clip_lens = pd.read_csv('subclip_lengths.csv')
clip_lengths = np.zeros((614,1))
clip_lengths[0] = 0.822
clip_lengths[1:] = np.reshape(clip_lens['0.822'].values[:613], (613,1))

train_indexes = pd.read_pickle('train_indx.p')
test_indexes = pd.read_pickle('test_indx.p')
y_train = emotion_lables_encoded[train_indexes]
y_test = emotion_lables_encoded[test_indexes]

X_train = data_imgs[train_indexes]
X_test = data_imgs[test_indexes]

clip_lengths_train = clip_lengths[train_indexes]
clip_lengths_test = clip_lengths[test_indexes]

audio_predictions = np.loadtxt('Results/Audio/Emotion_audio_multi_preds.txt', delimiter=',')
image_predictions = np.loadtxt('Results/Image/Static Feature/Facial_points_Emotion_predictions_LSTM_ms.txt', delimiter=',')

def weighted_predictions(audio, audio_percentage, image, image_precentage):
    res = audio_percentage*audio*0.01 + image_precentage*image*0.01
    return res

def weighted_probs(audio, image):
    res = audio + image
    return res

try:
    f = open('Results/Late fusion/Emotion_weighted_results.txt','w+')
    wp = weighted_probs(audio_predictions, image_predictions)
    pred = np.argmax(wp, axis = -1)
    acc = np.mean(np.equal(pred, y_test))
    f.write('Audio and Image weighted probs resulting accuracy: ')
    f.write(str(acc))
    f.write('%\n')
    f.close()
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except ValueError:
    print("Could not convert data to an integer.")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise


audio_predictions = np.loadtxt('Results/Audio/Dominance_audio_predictions_LSTM_ms.txt', delimiter=',')
image_predictions = np.loadtxt('Results/Image/Static Feature/Facial_points_Dominance_predictions_LSTM_ms.txt', delimiter=',')


try:
    f = open('Results/Late fusion/Dominace_weighted_results.txt','w+')
    for i in range(0, 110, 10):
        wp = weighted_predictions(audio_predictions, i, image_predictions, 100-i)
        mse = np.mean((wp - y_test)**2)
        f.write('Audio weight ')
        f.write(str(i))
        f.write('%, Image weight: ')
        f.write(str(100-i))
        f.write('% - MSE:')
        f.write(str(mse))
        f.write('\n')
    f.close()
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except ValueError:
    print("Could not convert data to an integer.")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise




audio_predictions = np.loadtxt('Results/Audio/Arousal_audio_predictions_LSTM_ms.txt', delimiter=',')
image_predictions = np.loadtxt('Results/Image/Static Feature/Facial_points_Arousal_predictions_LSTM_ms.txt', delimiter=',')

try:
    f = open('Results/Late fusion/Arousal_weighted_results.txt','w+')
    for i in range(0, 110, 10):
        wp = weighted_predictions(audio_predictions, i, image_predictions, 100-i)
        mse = np.mean((wp - y_test)**2)
        f.write('Audio weight ')
        f.write(str(i))
        f.write('%, Image weight: ')
        f.write(str(100-i))
        f.write('% - MSE:')
        f.write(str(mse))
        f.write('\n')
    f.close()
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except ValueError:
    print("Could not convert data to an integer.")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise
