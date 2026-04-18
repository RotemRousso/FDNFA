import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm
import numpy as np
import os
from os.path import join, basename
from boltons.fileutils import iter_find_files
import soundfile as sf
import librosa
import pickle
from multiprocessing import Pool
import random
import torchaudio
import math
from torchaudio.datasets import LIBRISPEECH


def collate_fn_padd(batch):
    """collate_fn_padd
    Padds batch of variable length

    :param batch:
    """
    # get sequence lengths
    spects = [t[0] for t in batch]
    segs = [t[1] for t in batch]
    labels = [t[2] for t in batch]
    lengths = [t[3] for t in batch]
    fnames = [t[4] for t in batch]

    padded_spects = torch.nn.utils.rnn.pad_sequence(spects, batch_first=True)
    lengths = torch.LongTensor(lengths)
    return padded_spects, segs, labels, lengths, fnames


def spectral_size(wav_len):
    layers = [(10,5,0), (8,4,0), (4,2,0), (4,2,0), (4,2,0)]
    for kernel, stride, padding in layers:
        wav_len = math.floor((wav_len + 2*padding - 1*(kernel-1) - 1)/stride + 1)
    return wav_len


def get_subset(dataset, percent):
    A_split = int(len(dataset) * percent)
    B_split = len(dataset) - A_split
    dataset, _ = torch.utils.data.random_split(dataset, [A_split, B_split])
    return dataset


class WavPhnDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = list(iter_find_files(self.path, "*.wav"))

    def process_file(self, wav_path):
        
        phn_path = wav_path[:-4] + ".phn"

        # load audio
        audio, sr = torchaudio.load(wav_path)
        audio = audio[0]
        audio_len = len(audio)
        spectral_len = spectral_size(audio_len)
        len_ratio = (audio_len / spectral_len)

        # load labels -- segmentation and phonemes
        with open(phn_path, "r") as f:
            lines = f.readlines()
            lines = list(map(lambda line: line.split(" "), lines))

            # get segment times
            times = torch.FloatTensor(list(map(lambda line: int(int(line[1]) / len_ratio), lines)))[:-1]  # don't count end time as boundary

            # get phonemes in each segment (for K times there should be K+1 phonemes)
            phonemes = list(map(lambda line: line[2].strip(), lines))

        return audio, times.tolist(), phonemes, wav_path
    
    def __getitem__(self, idx):
        audio, seg, phonemes, fname = self.process_file(self.data[idx])
        audio_len = len(audio)
        spectral_len = spectral_size(audio_len)
        len_ratio = (audio_len / spectral_len)
        return audio, seg, phonemes, audio_len/len_ratio, fname

    def __len__(self):
        return len(self.data) 
    
class TrainTestDataset(WavPhnDataset):
    @staticmethod
    def get_datasets(path, val_ratio=0.1, overlap=False, seed: int = 42):
        """
        If overlap==False (default) split train into disjoint train/val (random_split).
        If overlap==True  create val as a Subset sampled from the train dataset
                           but keep train_dataset as the full set (so val files are also seen in training).
        """
        train_full = TrainTestDataset(join(path, 'train'))
        test_dataset = TrainTestDataset(join(path, 'test'))
        train_len = len(train_full)

        val_size = int(train_len * val_ratio)
        if val_size <= 0:
            # no validation
            return train_full, None, test_dataset

        if overlap:
            rng = random.Random(seed)
            val_indices = rng.sample(range(train_len), val_size)
            val_dataset = torch.utils.data.Subset(train_full, val_indices)
            train_dataset = train_full  # full training set (contains val files)
        else:
            # exclusive split (current behavior)
            gen = torch.Generator()
            gen.manual_seed(seed)
            train_split = train_len - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_full, [train_split, val_size], generator=gen)
            # keep .path attribute for compatibility
            train_dataset.path = join(path, 'train')
            val_dataset.path = join(path, 'train')
            return train_dataset, val_dataset, test_dataset

        # ensure compatibility of .path attribute
        train_dataset.path = join(path, 'train')
        val_dataset.path = join(path, 'train')
        return train_dataset, val_dataset, test_dataset


class TrainValTestDataset(WavPhnDataset):
    @staticmethod
    def get_datasets(path, percent=1.0):
        train_dataset = TrainValTestDataset(join(path, 'train'))
        if percent != 1.0:
            train_dataset = get_subset(train_dataset, percent)
            train_dataset.path = join(path, 'train')
        val_dataset   = TrainValTestDataset(join(path, 'val'))
        test_dataset  = TrainValTestDataset(join(path, 'test'))

        return train_dataset, val_dataset, test_dataset