import os
import numpy as np
import pickle

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

path = os.getcwd()
train_dataset = list()
train_dataset_output = list()
data2 = []

for subdir, dirs, files in os.walk(path):
	for file in files:
		if file.endswith('.WAV'):
			fname = os.path.join(subdir, file)
			(rate, sig) = wav.read(fname)
			mfcc_feat = mfcc(sig, rate, numcep = 41, nfilt = 41)
			d_mfcc_feat = delta(mfcc_feat, 2)
			dd_mfcc_feat = delta(d_mfcc_feat, 2)

			data = np.concatenate((np.asarray(mfcc_feat, dtype = np.float32), 
				np.asarray(d_mfcc_feat, dtype = np.float32),
				np.asarray(dd_mfcc_feat, dtype = np.float32)), axis = 1)

			for i in range(data.shape[0]):
				data2.append([])
				data2[(len(data2) - 1)] = data[i]

			train_dataset.append(data)
			
			txtname = fname[:-3] + "TXT"

			with open(txtname, "r") as f:
				for line in f:
					line = line.strip()
					line = line.split(' ', 2)[2]
					train_dataset_output.append(line)
					break

data_array = np.asarray(data2)
mean = np.mean(data_array, axis = 0)
std = np.std(data_array, axis = 0)

train_dataset2 = list()
for single_data in train_dataset:
	for i in range(single_data.shape[0]):
		single_data[i] = (single_data[i] - mean) / std
	train_dataset2.append(single_data)
	print(len(train_dataset2))

with open("train_mfcc", "wb") as fp:
	pickle.dump(train_dataset2, fp, protocol = 2)

with open("train_output_words", "wb") as fp2:
	pickle.dump(train_dataset_output, fp2, protocol = 2)