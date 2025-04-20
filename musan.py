import os
import random
import numpy as np
import soundfile as sf

class Musan:
    Category = ['noise','speech','music']
    
    SNR = {
        'noise': (0, 15),
        'speech': (13, 20),
        'music': (5, 25)
    }
    
    NumFile = {
        'noise': (1, 1),
        'speech': (3, 8),
        'music': (1, 1)
    }
    
    def __init__(self, path):
        # set vars
        self.noise_list = {}
        self.num_noise_file = {}
        for category in self.Category:
            self.noise_list[category] = []
            self.num_noise_file[category] = 0

        # init noise list
        for root, _, files in os.walk(path):
            if self.Category[0] in root:
                category = self.Category[0]
            elif self.Category[1] in root:
                category = self.Category[1]
            elif self.Category[2] in root:
                category = self.Category[2]

            for file in files:
                if '.wav' in file:
                    self.noise_list[category].append(os.path.join(root, file))
                    self.num_noise_file[category] += 1

    def __call__(self, x, category):
        # calculate dB
        x_size = x.shape[-1]
        x_dB = self.calculate_decibel(x)

        # select noise
        snr_min, snr_max = self.SNR[category]
        file_min, file_max = self.NumFile[category]
        files = random.sample(
            self.noise_list[category],
            random.randint(file_min, file_max)
        )

        # init noise
        noises = []
        for f in files:
            # read .wav
            info = sf.info(f)
            wav_size = int(info.samplerate * info.duration)
            if wav_size <= x_size:
                noise, _ = sf.read(f, start=0)
                noise_size = noise.shape[0]
                if noise_size < x_size:
                    shortage = x_size - noise_size
                    noise = np.pad(
                        noise, (0, shortage), 'wrap'
                    )
            else:
                index = random.randint(0, wav_size - x_size - 1)
                noise, _ = sf.read(f, start=index, stop=index + x_size)

            noises.append(noise)
        
        # noise injection
        if len(noises) != 0:
            noise = np.mean(noises, axis=0)
            # set SNR
            snr = random.uniform(snr_min, snr_max)
            # calculate dB
            noise_dB = self.calculate_decibel(noise)
            # append
            p = (x_dB - noise_dB - snr)
            x += np.sqrt(10 ** (p / 10)) * noise
        
        return x

    def calculate_decibel(self, wav):
        assert 0 <= np.mean(wav ** 2) + 1e-4 
        return 10 * np.log10(np.mean(wav ** 2) + 1e-4)