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

audio_predictions = np.loadtxt('Results/Emotion_audio_result_LSTM_BN.txt')
image_predictions = np.loadtxt('Results/Image_points_Emotion_predictions_LSTM_ms.txt')

def weighted_predictions(audio, audio_percentage, image, image_precentage):
    res = np.floor(audio_percentage*audio*0.01 + image_precentage*image*0.01)
    return res

try:
    f = open('Results/Emotion_weighted_results.txt','w+')
    for i in range(0, 110, 10):
        wp = weighted_predictions(audio_predictions, i, image_predictions, 100-i)
        acc = np.equal(wp, y_test)
        acc_ = np.mean(acc)
        f.write('Audio weight ')
        f.write(str(i))
        f.write('%, Image weight: ')
        f.write(str(100-i))
        f.write('% :')
        f.write(str(acc_))
        f.write('%\n')
    f.close()
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except ValueError:
    print("Could not convert data to an integer.")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise


audio_predictions = np.loadtxt('Results/Dominance_audio_predictions_LSTM.txt')
image_predictions = np.loadtxt('Results/Image_points_Dominance_predictions_LSTM_ms.txt')


try:
    f = open('Results/Dominace_weighted_results.txt','w+')
    for i in range(0, 110, 10):
        wp = weighted_predictions(audio_predictions, i, image_predictions, 100-i)
        acc = np.equal(wp, y_test)
        acc_ = np.mean(acc)
        f.write('Audio weight ')
        f.write(str(i))
        f.write('%, Image weight: ')
        f.write(str(100-i))
        f.write('% :')
        f.write(str(acc_))
        f.write('%\n')
    f.close()
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except ValueError:
    print("Could not convert data to an integer.")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise




audio_predictions = np.loadtxt('Results/Arousal_audio_predictions_LSTM.txt')
image_predictions = np.loadtxt('Results/Image_points_Arousal_predictions_LSTM_ms.txt')


try:
    f = open('Results/Arousal_weighted_results.txt','w+')
    for i in range(0, 110, 10):
        wp = weighted_predictions(audio_predictions, i, image_predictions, 100-i)
        acc = np.equal(wp, y_test)
        acc_ = np.mean(acc)
        f.write('Audio weight ')
        f.write(str(i))
        f.write('%, Image weight: ')
        f.write(str(100-i))
        f.write('% :')
        f.write(str(acc_))
        f.write('%\n')
    f.close()
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except ValueError:
    print("Could not convert data to an integer.")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise
