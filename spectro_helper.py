from scipy import signal
from scipy.io import wavfile
from scipy.misc import imresize
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import random

# Return a scaled image representing the spectrogram of a certain audio file
# MAKE SURE THE WAV FILE READ IN IS 32-bit float PCM format!!!!!!!!!!!!!! THIS IS CRUCIAL!
def spectrogram(filename):
	#print(filename)
	sample_rate, samples = wavfile.read(filename)

	frequencies, times, spectrogram = signal.spectrogram(x=samples, fs=sample_rate)

	# Since spectrogram is returned back linearly
	spectrogram = 10 * np.log10(spectrogram)
	spectrogram[spectrogram == -np.inf] = 0

	resized = np.flip(imresize(spectrogram, (128, 256), interp='lanczos'), 0)
	resized = resized.astype(float)
	return resized


def prepare_data(aud_dir):

    audio_dirs = np.array([dirpath for (dirpath, dirnames, filenames) in gfile.Walk(os.getcwd()+'/' + aud_dir)])
    file_list = []
    y_= []
	# Ignoring the first directory as it is the base directory
    for audio_dir in audio_dirs[1:]:
        # extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        dir_name = os.path.basename(audio_dir)
        audio_file_list =[]
        tf.logging.info("Looking for wav files in '" + dir_name + "'")
    
        file_glob = os.path.join(audio_dir,'*.' + 'wav')
        #This looks for a file name pattern, helps us ensure that only jpg extensions are choosen
        audio_file_list = gfile.Glob(file_glob)
        file_list.extend(audio_file_list)
        y_.extend([dir_name]*len(audio_file_list))

    return file_list, y_



def read_wav_array(wav_loc_array):
    spectro_array=[]

    for audio_loc in wav_loc_array:
        #print(audio_loc)
        spectro_img = spectrogram(audio_loc)
        spectro_array.append(spectro_img)

    spectro_array = tf.reshape(tf.stack(spectro_array),[len(wav_loc_array), 128, 256, 1])
    return spectro_array

def read_single_wav(path):
	spectro_img = spectrogram(path)
	spectro_img = tf.reshape(spectro_img, [1, 128, 256, 1])
	return spectro_img