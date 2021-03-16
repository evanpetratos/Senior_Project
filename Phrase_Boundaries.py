# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 08:49:11 2020

@author: evanpetratos
"""

import sys
import time

start = time.time()

import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from scipy.signal import resample

#load audio data
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


#alg_data = []
    
for i in range(len(audio_data)):
    print("iteration:", i+1, "out of", len(audio_data))
    
    x, sr = librosa.load(audio_data[i]) #<class 'numpy.ndarray'> <class 'int'>
    librosa.load(audio_data[i], sr=44100)
    duration = int(len(x)/22050) #length of the audio sample, in seconds
    
     
    #https://librosa.org/doc/latest/generated/librosa.onset.onset_detect.html
    #onset detection
    plt.figure(figsize=(14, 5))
    
    from librosa import onset
    
    o_env = onset.onset_strength(y=x, sr=sr, aggregate=np.median, fmax=8000, n_mels=256)
    times = librosa.times_like(o_env, sr=sr)
    onset_frames = onset.onset_detect(onset_envelope=o_env, sr=sr)
    
    plt.plot(times, o_env, label='Onset strength')
    plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
               linestyle='--', label='Onsets')
    plt.title("Acoustic Onsets")
    plt.xlabel("time (seconds)")
    plt.ylabel("Onsets and Onset Strength")
    plt.legend()
    plt.show()
    
    half_second = (len(x)/duration)/2
    half_sec_marks = []
    onsets_list = times[onset_frames]*22050
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
    
    
    #conversions
    x_db = librosa.amplitude_to_db(x)
    x_amp = librosa.db_to_amplitude(x_db)
    x_pow = librosa.db_to_power(x_db)
    assert x_pow.all() == x_amp.all()**2 #intensity = amplitude**2
    x_pow_log = np.log(x_pow)
    x_pow_log_zero = x_pow_log + (np.min(x_pow_log)*-1) #sets min to zero
    x_first_plot = x_pow_log_zero
    
    plt.figure(figsize=(14, 5))
    plt.plot(x_first_plot)
    plt.title("Logarithm of Intensity")
    plt.xlabel("Sample (Hz)")
    plt.ylabel("log(Intensity)")
    plt.show()
    
    
    #https://librosa.org/doc/latest/generated/librosa.feature.spectral_flatness.html
    #spectral flatness (Wiener entropy)
    from librosa import feature
    
    flatness = feature.spectral_flatness(y=x_pow_log)
    flatness_1Darray = flatness[0]
    
    
    #https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    def smooth(x,window_len=11,window='hanning'):
        '''
        smooth the data using a window with requested size.
        
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        
        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
    
        output:
            the smoothed signal
            
        example:
    
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        
        see also: 
        
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
     
        NOTE: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        '''
        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")
    
        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")
    
        if window_len<3:
            return x
    
    
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    
    
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')
    
        y=np.convolve(w/w.sum(),s,mode='valid')
        return y[int(window_len/2-1):-(int(window_len/2))]
    
    
    y_smooth = smooth(x_pow_log, window_len=20001) #changed from y to x_pow_log
    y_zero = y_smooth + (np.min(y_smooth)*-1) #sets min to zero
    
    
    #Below, find_peaks is used to detect troughs in data, which could be phrase boundaries
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    from scipy.signal import find_peaks
    def getPhraseLength():
        '''
        Optimize listening threshold for the style of the piece
        number is determined by finding the min of the two averages
        phrase lengths from participant data, then subtracting 4
        
        The smaller the threshold, the better the recall
        
        Parameters
        ----------
        None
    
        Returns
        -------
        phrase_length_seconds : FLOAT
            Float that indicates how many seconds a phrase
            should be in the style of the current stimulus.
        '''
        phrase_length_seconds = 1 #number should be changed with piece's style
        
        #these numbers were taken from Statistical_Analysis.ipynb
        #phrase length averages for each stimulus.
        if i==0:
            phrase_length_seconds=(min(14.231772058095238, 14.932438454352681) - 4)
        elif i==1:
            phrase_length_seconds=(min(9.888642274373016, 10.096604769037185) - 4)
        elif i==2:
            phrase_length_seconds=(min(10.07121368932032, 10.461472413978626) - 4)
        elif i==3:
            phrase_length_seconds=(min(11.469779613036549, 9.84461392945918) - 4)
        elif i==4:
            phrase_length_seconds=(min(16.790776123764584, 19.444730335709128) - 4)
        elif i==5:
            phrase_length_seconds=(min(12.612926529279965, 13.299143065786344) - 4)
        elif i==6:
            phrase_length_seconds=(min(12.808746410421524, 14.94746326886962) - 4)
        elif i==7:
            phrase_length_seconds=(min(13.689879785857345, 16.14788508138177) - 4)
        elif i==8:
            phrase_length_seconds=(min(14.552260184388889, 13.233587383602144) - 4)
        elif i==9:
            phrase_length_seconds=(min(24.306343323455028, 17.36317984239619) - 4)
            
        return phrase_length_seconds
    
    
    plt.figure(figsize=(14, 5))
    
    x_second_plot = resample(y_zero, int(duration*4))
    phrase_length1 = (len(x_second_plot)/duration)*getPhraseLength()
    
    #find troughs in intensity plot
    peaks, _ = find_peaks(-x_second_plot, distance=phrase_length1)
    peaks = np.asarray(peaks)
    if peaks[0] < getPhraseLength():
        peaks = np.delete(peaks, 0)
    plt.plot(x_second_plot)
    plt.plot(peaks, x_second_plot[peaks], "x")
    plt.title("Intensity Boundaries")
    plt.xlabel("Time (seconds/4)")
    plt.ylabel("Intensity")
    plt.show()
    
    
    #print("Stimulus", i, "Phrase Boundaries:", (peaks/(len(y_zero)/duration)).round(1))
    #print(len(peaks), "boundaries found from intensity analysis")
    #print("\n")
    
    
    #find troughs in spectral flatness (Wiener Entropy)
    plt.figure(figsize=(14, 5))
    
    x_third_plot = resample(flatness_1Darray, int(duration*4))
    phrase_length2 = (len(x_third_plot)/duration)*getPhraseLength()
    
    spectral_peaks, _ = find_peaks(-x_third_plot, distance=phrase_length2)
    if spectral_peaks[0] < getPhraseLength():
        spectral_peaks = np.delete(spectral_peaks, 0)
    plt.plot(x_third_plot)
    plt.plot(spectral_peaks, x_third_plot[spectral_peaks], "x")
    plt.title("Spectral Flatness Boundaries")
    plt.xlabel("Time (seconds/4)")
    plt.ylabel("Spectral Flatness")
    plt.show()
    
    #print("Stimulus", i, "Phrase Boundaries from Wiener Entropy:", (spectral_peaks/(len(flatness_1Darray)/duration)))
    #print(len(spectral_peaks), "boundaries from spectral analysis found")
    #print("\n")
    
    
    #find troughs in rhythmic density
    plt.figure(figsize=(14, 5))
    
    rD = rhythmic_density(half_second, half_sec_marks, onsets_list, onsets_per_half_second)
    
    rD = np.asarray(rD)
    x_fourth_plot = resample(rD, int(duration*4))
    phrase_length3 = (len(x_fourth_plot)/duration)*getPhraseLength()
    
    rhythmic_peaks, _ = find_peaks(-x_fourth_plot, distance=phrase_length3)
    if rhythmic_peaks[0] < getPhraseLength():
        rhythmic_peaks = np.delete(rhythmic_peaks, 0)
    plt.plot(x_fourth_plot)
    plt.plot(rhythmic_peaks, x_fourth_plot[rhythmic_peaks], "x")
    plt.title("Rhythmic Denisty Boundaries")
    plt.xlabel("Time (seconds/4)")
    plt.ylabel("Frequency of Acoustic Onsets")
    plt.show()
    
    #print("Stimulus", i, "Phrase Boundaries from Rhythmic Density:", (rhythmic_peaks/(len(rD)/duration)))
    #print(len(rhythmic_peaks), "boundaries from rhythmic density calculations found")
    #print("\n")
    
    #alg_data.append(rhythmic_peaks/(len(rD)/duration))
    

#used to write algorithm's boundaries to csv file
#import csv

#with open('_marks.csv','w') as f1:
#    writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
#    for i in range(len(alg_data)):
#        row = alg_data[i]
#        writer.writerow(row)


#shows time enlapsed for entire program
time_elapsed = time.time() - start
print("time elapsed:", time_elapsed)
