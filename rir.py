import os
import random
import numpy as np
import soundfile as sf
from scipy import signal

class RIRReverberation:
    def __init__(self, path):
        self.files = []
        
        # parse list
        for root, _, files in os.walk(path):
            for file in files:
                if '.wav' in file:
                    self.files.append(os.path.join(root, file))

    def __call__(self, x):
        path = random.sample(self.files, 1)[0]
        
        rir, _ = sf.read(path)
        rir = rir.astype(np.float)
        rir = np.expand_dims(rir, 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        
        x = np.expand_dims(x, 0)
        x = signal.convolve(x, rir, mode='full')[:,:len(x[0])]

        x = np.squeeze(x, 0)

        return x