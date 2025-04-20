import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import random
from musan import Musan
from rir import RIRReverberation

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()
    for line in l_meta:
        _, key, _, _, _, _, _, _, label, _ = line.strip().split(" ")
        file_list.append(key)
        d_meta[key] = 1 if label == "bonafide" else 0
    return d_meta, file_list
    """
    if is_train:
        for line in l_meta:
            _, key, _, _, _, _, _, _,  label, _ = line.strip().split("\t")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list
    elif is_eval:
        for line in l_meta:
            _, key, _, _, _, _ = line.strip().split("\t")
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, _, label = line.strip().split("\t")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list
    """

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class TrainDataset(Dataset):
    def __init__(self, list_IDs, labels, base_dir, add_noise=False):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

        self.DA = {}
        self.DA['MUS'] = Musan(
                    'musan_data'
                )
        self.category = ['noise','speech','music']
        self.DA['RIR'] = RIRReverberation(
                    'RIR_data'
                )
        self.add_noise = add_noise
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"{key}.flac"))
        if self.add_noise:
            if 0.5 > random.random():
                if random.randint(0, 1) == 0:
                    category = random.choice(self.category)
                    X = self.DA['MUS'](X, category)
                else:
                    X = self.DA['RIR'](X)    
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class TestDataset(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
