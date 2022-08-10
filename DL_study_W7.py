# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:16:10 2022

@author: dltjdwls
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import mne
import scipy.io
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 
from scipy.io import savemat

%matplotlib qt5

cwd = os.getcwd() # 현재 작업 경로 가져오기
print('$ Print MNE version:   ', mne.__version__)
print('$ Current working dir:   ', cwd)

# Read EEG_mat file
os.chdir('C:/Users/HBFLAB/Desktop/LSJ/DeepLearning study')
cwd = os.getcwd()
print('$ Current working dir:   ', cwd)

data = scipy.io.loadmat('subj6-68GRAD.mat')

type(data)

acc = data['acc']
event_id = data['event_id']
grad = data['grad']
                
# sampling frequency = 600.615Hz
acc.shape # 240 trials x 3 Dimension x 1803 time points
grad.shape # 240 trials x 68 channels x 1803 time points


# Info
n_channels = 68

sampling_freq = 600.615
ch_names = [f'MEG{n:03}' for n in range(1, 69)]
print(ch_names)
ch_types = ['mag'] * 68
meg_info = mne.create_info(ch_names, ch_types = ch_types, sfreq=sampling_freq)
acc_info = mne.create_info(3, sfreq=sampling_freq)



print(meg_info)
print(acc_info)

# MEG -> MNE epoch으로 변환
meg = mne.EpochsArray(grad, meg_info)
acc = mne.EpochsArray(acc, acc_info)

# Check the information
meg.info

# Down-sampling
meg = meg.resample(120)
acc = acc.resample(120)


meg.info

# Check the Filtering
meg.plot_psd(picks = 'all')

# Check the bad-channels
meg.plot(scalings= 'auto' , n_channels = 68, picks='all')

# ICA

ica = mne.preprocessing.ICA(
    n_components = 20,
    method = 'fastica',
    max_iter = 'auto',
    random_state = 99
    )

ica.fit(meg, picks = 'all')

meg.info

#ica.plot_components(sensors=True, colorbar=True, sphere = 'auto')
ica.plot_sources(meg)

"""
ica.plot_overlay(
    meg.copy(),
    exclude=ica.exclude,
    picks = 'all'
    )
"""

ica.exclude

meg.plot(scalings = 'auto', n_channels = 10, title = 'Raw filtered')
meg_ica = ica.apply(meg.copy())
meg_ica.plot(scalings = 'auto', n_channels = 10, title = 'ICA applied')

type(meg)

np_meg = np.array(meg)
np_meg.shape

np_acc = np.array(acc)
np_acc.shape

#

meg_axes = np_meg.swapaxes(0,1)
meg_reshape = meg_axes.reshape(68, 240*360)
meg_reshape.shape

acc.shape
acc_axes = np_acc.swapaxes(0,1)
acc_reshape = acc_axes.reshape(3, 240*360)


X_train= meg_reshape[:, 1: 69120]
X_test = meg_reshape[:, 69121:]

Y_train = acc_reshape[:, 1: 69120]
Y_test = acc_reshape[:, 69121:]

def LSTM() :
    model = Sequential()
    model.add(LSTM(20, input_shape = (240, 68, 360), return_sequences = True))
    model.add(LSTM(20, return_sequences = True))
    model.add(LSTM(20, return_sequences = True))
    model.add(LSTM(20, return_sequences = False))
    model.add(Dense(46))
    model.add(Activation('softmax'))
    
    adam = optimizer.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model


model = KerasClassifier(build_fn = LSTM, epochs = 100, batch_size = 50, verbose = 1)
model.fit(X_train, Y_train)
