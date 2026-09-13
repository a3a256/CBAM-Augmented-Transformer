import csv
import json
import torchaudio
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import subprocess


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, df, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """

        self.data = df
        self.audio_conf = audio_conf
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        self.mixup = self.audio_conf.get('mixup')
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')

    def split_fbank(self, fbank, target_length):
        """Split a (n_frames, mel_bins) fbank into consecutive (target_length, mel_bins)
        windows, zero-padding the final window if it's shorter than target_length."""
        n_frames = fbank.shape[0]
        windows = []
        for start in range(0, n_frames, target_length):
            window = fbank[start:start + target_length, :]
            pad_amount = target_length - window.shape[0]
            if pad_amount > 0:
                window = torch.nn.functional.pad(window, (0, 0, 0, pad_amount))
            windows.append(window)

        
        return torch.stack(windows)  # (num_windows, target_length, num_mel_bins)

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')

        fbank = self.split_fbank(fbank, target_length)

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """

        audio_file = self.data[index, 0]
        
        start_audio = self.data[index, 1]
        end_audio = self.data[index, 2]

        output_path = "segment_{}".format(audio_file)

        location = "processed_audio/{}".format(audio_file)

        cmd = "ffmpeg -i {} -ss {} -to {}".format(location, start_audio, end_audio)
        cmd += " {} -y".format(output_path)
        subprocess.run(cmd, shell=True, check=True)

        fbanks, mix_lambda = self._wav2fbank(output_path)

        tensors = []

        for fbank in fbanks:
            # SpecAug, not do for eval set
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = torch.transpose(fbank, 0, 1)
            # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
            fbank = fbank.unsqueeze(0)
            if self.freqm != 0:
                fbank = freqm(fbank)
            if self.timem != 0:
                fbank = timem(fbank)
            # squeeze it back, it is just a trick to satisfy new torchaudio version
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)

            # normalize the input for both training and test
            if not self.skip_norm:
                fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
            # skip normalization the input if you are trying to get the normalization stats.
            else:
                pass

            if self.noise == True:
                fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
                fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

            mix_ratio = min(mix_lambda, 1-mix_lambda) / max(mix_lambda, 1-mix_lambda)

            tensors.append(fbank)

        if os.path.exists(output_path):
            os.remove(output_path)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return torch.stack(tensors), self.data[index, -1]

    def __len__(self):
        return len(self.data)