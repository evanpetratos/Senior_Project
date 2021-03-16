# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:41:55 2020

@author: evanpetratos
"""
import sys

import librosa
import librosa.display
import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt


audio_data = ['Gregorian_Chant.wav',
              'Mozart1.wav',
              'Mozart2.wav',
              'Beethoven.wav',
              'Bruckner1.wav',
              'Bruckner2.wav',
              'Bruckner3.wav',
              'Chopin1.wav',
              'Chopin2.wav',
              'Rochmaninoff.wav']

s0 = 100
s1 = 83
s2 = 73
s3 = 146
s4 = 136
s5 = 101
s6 = 85
s7 = 117
s8 = 75
s9 = 89
total_secs = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]
    
feature_data = []

for i in range(len(audio_data)):
    
    print("audio from feature", i)
    x, sr = librosa.load(audio_data[i], sr=44100)
    duration = int(len(x)/44100)
    
    
    x_db = librosa.amplitude_to_db(x)
    x_amp = librosa.db_to_amplitude(x_db)
    x_pow = librosa.db_to_power(x_db)
    """
    int_down = resample(x=x_pow,num=int(total_secs[i]*4))
    int_norm = int_down*1000
    feature_data.append(int_norm)
    
    
    from librosa import feature
    
    flatness = feature.spectral_flatness(y=x_db)
    flatness_1Darray = flatness[0]
    
    spect_down = resample(x=flatness_1Darray,num=int(total_secs[i]*4))
    spect_norm = spect_down*1000

    #plt.figure(figsize=(14,5))
    #plt.plot(spect_norm)
    feature_data.append(spect_norm)
    
    
    
    from librosa import onset
    
    o_env = onset.onset_strength(y=x, sr=sr, aggregate=np.median, fmax=8000, n_mels=128)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = onset.onset_detect(onset_envelope=o_env, sr=sr)
    onsets_list = times[onset_frames]*22050
    
    half_second = (len(x)/duration)/2
    half_sec_marks = []
    onsets_per_half_second = []
    
    #calculate rhythmic density
    def rhythmic_density(window, time_steps, onsets_found, ret_array):
        '''
        Parameters
        ----------
        window : INT
            Specifies the length of window to mark onsets.
        time_steps : ARRAY
            Array that stores the boundaries between each window.
        onsets_found : ARRAY
            Array given that contains onset timings.
        ret_array : ARRAY
            Array, initially empty, that stores the number of onsets found for each window.
    
        Returns
        -------
        ret_array : ARRAY
            Array that stores the number of onsets found for each window.
        '''
        #create initial array that specifies each boundary between windows
        for h in range(1, (duration*2)+1):
            time_steps.append(window*h)
            
        g = 0 #g used to increment windows for iteration
        for j in range(len(time_steps)):
            g = time_steps[j] - window #g is always one iteration behind j, to create the window
            g = int(g) #g must be an int to specify range for iteration
            
            aux_array = [] #auxiliary array used to store each onset within the current window
            for k in range(g, int(time_steps[j])):
                
                #append k to auxiliary array if the onsets_found array contains k
                if k in onsets_found:
                    aux_array.append(k)
                    ret_array.append(len(aux_array))
        
        return ret_array
    
    rhythm_new = rhythmic_density(half_second, half_sec_marks, onsets_list, onsets_per_half_second)    
    
    rhythm_down = resample(x=rhythm_new,num=int(total_secs[i]*4))
    rhythm_norm = rhythm_down
    feature_data.append(rhythm_norm)
    plt.figure(figsize=(14,5))
    plt.plot(rhythm_norm)
    plt.show()
    
    
    
    zcr = librosa.zero_crossings(x)
    zcr_sig = [ ]
    for j in zcr:
        if j == True:
            zcr_sig.append(1)
        elif j == False:
            zcr_sig.append(0)
    zcr_norm = resample(x=zcr_sig,num=int(total_secs[i]*4))
    feature_data.append(zcr_norm)
    plt.figure(figsize=(14,5))
    plt.plot(zcr_norm)
    plt.show()
    """
    
    spectc = librosa.feature.spectral_centroid(y=x_db, sr=sr)
    spectc_flat = spectc[0]
    spectc_norm = resample(x=spectc_flat,num=int(total_secs[i]*4))
    feature_data.append((spectc_norm))
    plt.figure(figsize=(14,5))
    plt.plot(spectc_norm)
    plt.show()
    
    #sys.exit()


flat_list = []
for sublist in feature_data:
    for item in sublist:
        flat_list.append(item)


plt.figure(figsize=(14,5))
plt.plot(flat_list)
plt.show()

import csv

with open('spectral_centroid.csv','w') as f1:
    writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
    for i in range(len(feature_data)):
        row = feature_data[i]
        writer.writerow(row)

