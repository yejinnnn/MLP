import librosa
import numpy as np
import tensorflow
import tf as tf
from keras.applications.densenet import layers
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from matplotlib.cbook import flatten
from pygments.lexers import go
from scipy.io import wavfile
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math


import os
DATA_PATH = r'C:/Users/user/Desktop/google_speech/data/'

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0,len(labels))
    return labels, label_indices, to_categorical(label_indices)

def wav2mfcc(file_path,max_pad_len = 11):
    wave,sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')

    return mfcc


def save_data_to_array(path=DATA_PATH, max_pad_len=11):
    labels, _, _ = get_labels(path)

    for label in tqdm(labels):
        mfcc_vectors = []
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        np.save(label+'.npy', mfcc_vectors)
    plt.show()

from sklearn.model_selection import train_test_split

def get_train_test(split_ratio = 0.8, random_state = 42):
    labels, indices, _ = get_labels(DATA_PATH)
    X = np.load(labels[0]+'.npy')
    y = np.zeros(X.shape[0])
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X,x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i+1)))
    assert X.shape[0] == len(y)
    return train_test_split(X, y, test_size = (1-split_ratio), random_state=random_state, shuffle=True)


save_data_to_array()
#
#
X_train, X_test, y_train, y_test = get_train_test()

# X_train = X_train.reshape(X_train.shape[0],20,11,1)
# X_test = X_test.reshape(X_test.shape[0],20,11,1)
X_train = X_train.reshape(X_train.shape[0],220,1)
X_test = X_test.reshape(X_test.shape[0],220,1)
#
# import numpy as np
# from scipy.signal import butter,filtfilt
#
# noise_level = 0.3
#
# # Filter requirements.
#          # Sample Period
# fs = 30.0       # sample rate, Hz
# cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
# nyq = 0.5 * fs  # Nyquist Frequency
# order = 2       # sin wave can be approx represented as quadratic
#  # total number of samples
#
# sig= X_train
# # Lets add some noise
# input = layers.Input(shape=(220, 1))
# noise = input + noise_level * tensorflow.random.uniform(tensorflow.shape(input))
# data = sig + noise


# def butter_lowpass_filter(data, cutoff, fs, order):
#     normal_cutoff = cutoff / nyq
#     # Get the filter coefficients
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y
#
# # Filter the data, and plot both the original and filtered signals.
# y = butter_lowpass_filter(data, cutoff, fs, order)
# fig = go.Figure()
# fig.add_trace(go.Scatter(
#             y = data,
#             line =  dict(shape =  'spline' ),
#             name = 'signal with noise'
#             ))
# fig.add_trace(go.Scatter(
#             y = y,
#             line =  dict(shape =  'spline' ),
#             name = 'filtered signal'
#             ))
# fig.show()

import numpy as np

#
# class LowPassFilter:
#     def __init__(self, cutoff_freq, ts):
#         self.ts = ts
#         self.cutoff_freq = cutoff_freq
#         self.pre_out = 0.
#         self.tau = self.calc_filter_coef()
#
#     def calc_filter_coef(self):
#         w_cut = 2 * np.pi * self.cutoff_freq
#         return 1 / w_cut
#
#     def filter(self, data):
#         out = (self.tau * self.pre_out + self.ts * data) / (self.tau + self.ts)
#         self.pre_out = out
#         return out
#
#
# lpf = LowPassFilter(cutoff_freq=0.5, ts=0.01)
#
import librosa
import librosa.display
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm


# audio_path = 'C:/Users/user/Desktop/google_speech/data/bed/00f0204f_nohash_0.wav'
# y, sr = librosa.load(audio_path)
#
# IPython.display.Audio(data=X_train, rate=sr)
#
# D = librosa.amplitude_to_db(librosa.stft(y[:1024]), ref=np.max)
#
# plt.plot(X_train.flatten())
# plt.show()


#
# from scipy import signal
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.io
# import os
#
# audio_path = 'C:/Users/user/Desktop/google_speech/data/bed/00f0204f_nohash_0.wav'
# (file_path, file_id) = librosa.load(audio_path)
# # file path, file name
# fs = 1024 # sample rate
# order = 10 # order
# cut_off_freq = 150 # cut off frequency
# # raw signal
# # t = np.linspace(0, 1, fs, False)
# # 1 second # sig = np.sin(2*np.pi*100*t) + np.sin(2*np.pi*50*t)
# # signal
#
# sig = audio_path[file_id[:-4.0]][0.0]
# freq = np.fft.fftfreq(len(sig), 1/1024)
#
# # filtered signal
# sos = signal.butter(order, [cut_off_freq], 'low', fs=fs, output='sos') # low pass filter
# filtered = signal.sosfilt(sos, sig)
#
# # raw signal fft
# raw_fft = np.fft.fft(sig) / len(sig)
# raw_fft_abs = abs(raw_fft)
#
# # filter signal fft
# filtered_fft = np.fft.fft(filtered) / len(filtered)
# filtered_fft_abs = abs(filtered_fft)
#
# # plot
# fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2)
#
# # raw signal plot : 0 row 0 column
# ax00.plot(D, sig)
# ax00.set_title('Raw Data Time Domain')
# ax00.set_xlabel('Time [seconds]')
# ax00.set_ylabel('Amplitude')
#
# # filtered signal plot : 1 row 0 column
# ax10.plot(D, filtered)
# ax10.set_title('Filtered Data Time Domain')
# ax10.set_xlabel('Time [seconds]')
# ax10.set_ylabel('Amplitude')
#
# # raw signal fft plot : 0 row 1 column
# ax01.stem(freq, raw_fft_abs, use_line_collection=True)
# ax01.set_title('Raw Data Frequency Domain')
# ax01.set_xlabel('Frequency [HZ]')
# ax01.set_ylabel('Amplitude')
#
# # filtered signal fft plot : 1 row column
# ax11.stem(freq,filtered_fft_abs, use_line_collection=True)
# ax11.set_title('Filtered Data Frequency Domain')
# ax11.set_xlabel('Frequency [HZ]')
# ax11.set_ylabel('Amplitude')
#
# # plot
# plt.tight_layout()
# plt.show()
#
# sos = signal.butter(order, [cut_off_freq], 'low', fs=fs, output='sos')
# filtered = signal.sosfilt(sos, sig)

wav = r'C:/Users/user/Desktop/google_speech/train/happy/5e033479_nohash_2.wav' # Original file
(file_dir, file_id) = os.path.split(wav)

print("Path : ", file_dir)
print("Name : ", file_id)


sample_rate, data = wavfile.read(wav) # sr : sampling rate, x : wave data array

print("Sample rate:{0}, data size:{1}, duration:{2} seconds".format(sample_rate,data.shape,len(data)/sample_rate))

time = np.linspace(0, len(data)/sample_rate, len(data))

plt.figure(figsize=(20,10))
plt.plot(time, data)
plt.ylabel("Amplitude")
plt.xlabel("Time[s]")
plt.title("Amplitude - Time[s]")
plt.show()

sample_rate, data

fft = np.fft.fft(data)

magnitude = np.abs(fft)

f = np.linspace(0, sample_rate, len(magnitude))
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]

plt.figure(figsize=(20,10))
plt.plot(left_f, left_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")


#b = signal.firwin(101, cutoff=1500, fs=sample_rate, pass_zero=False)
# b = signal.firwin(101, cutoff=3000, fs=sample_rate, pass_zero=False)=> 하이패스필터
#b = signal.firwin(101, cutoff=500, fs=sample_rate, pass_zero='lowpass')
# b = signal.firwin(101, cutoff=1000, fs=sample_rate, pass_zero='lowpass')=> 로우패스필터

numtaps = 3
f= 0.1
b = signal.firwin(101, cutoff=3000, fs=sample_rate, pass_zero='lowpass')
# Length of the filter : odd number
# cutoff : Cutoff frequency of filter,
# fs : sampling frequdncy
# pass_zero{True, False,'bandpass','lowpass','highpass','bandstop'}, optional

data1 = signal.lfilter(b, [1.0], data)


fft = np.fft.fft(data1)

magnitude = np.abs(fft)

f = np.linspace(0, sample_rate, len(magnitude))
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]

plt.figure(figsize=(20,10))
plt.plot(left_f, left_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")


time = np.linspace(0, len(data1)/sample_rate, len(data1))

plt.figure(figsize=(20,10))
plt.plot(time, data1)
plt.ylabel("Amplitude")
plt.xlabel("Time[s]")
plt.title("Amplitude - Time[s]")
plt.show()

b = signal.firwin(101, cutoff=1000, fs=sample_rate, pass_zero='lowpass')
data2 = signal.lfilter(b, [1.0], data)

fft = np.fft.fft(data2)

magnitude = np.abs(fft)

f = np.linspace(0, sample_rate, len(magnitude))
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]

plt.figure(figsize=(20,10))
plt.plot(left_f, left_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")

time = np.linspace(0, len(data2)/sample_rate, len(data2))

plt.figure(figsize=(20,10))
plt.plot(time, data1)
plt.ylabel("Amplitude")
plt.xlabel("Time[s]")
plt.title("Amplitude - Time[s]")
plt.show()




print("Sample rate:{0}, data size:{1}, duration:{2} seconds".format(sample_rate,data2.shape,len(data2)/sample_rate))

plt.figure(figsize=(20,20))

plot_a = plt.subplot(311)
plot_a.set_title('Amplitude - Time [original]')
plot_a.plot(time, data)
plot_a.set_xlabel('Time[s]')
plot_a.set_ylabel('Amplitude')
plot_a.set_ylim([-10000,10000])

plot_b = plt.subplot(312)
plot_b.set_title('Amplitude - Time [3000hz]')
plot_b.plot(time, data1)
plot_b.set_xlabel('Time[s]')
plot_b.set_ylabel('Amplitude')
plot_b.set_ylim([-10000,10000])

plot_c = plt.subplot(313)
plot_c.set_title('Amplitude - Time [1000hz]')
plot_c.plot(time, data2)
plot_c.set_xlabel('Time[s]')
plot_c.set_ylabel('Amplitude')
plot_c.set_ylim([-10000,10000])

plt.show()

wav_lpf = r'C:/Users/user/Desktop/google_speech/train/happy/5e033479_nohash_2_data2.wav'
wavfile.write(wav_lpf, sample_rate, data2.astype(np.int16))

wav_lpf = r"C:/Users/user/Desktop/google_speech/train/happy/5e033479_nohash_2_data1.wav"
wavfile.write(wav_lpf, sample_rate, data1.astype(np.int16))