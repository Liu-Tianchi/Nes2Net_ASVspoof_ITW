import os
import sys
import glob
import subprocess
import soundfile as sf
import scipy.signal as signal

from tqdm import tqdm
from multiprocessing import Pool

def split_list(l_in, nb_split):
    nb_per_split = int(len(l_in) / nb_split)
    l_return = []
    for i in range(nb_split):
        l_return.append(l_in[i * nb_per_split : (i + 1) * nb_per_split])
    l_return[-1].extend(l_in[nb_split * nb_per_split : ])
    
    return l_return

def multiprocessing(
        path, 
        processing,     # 'codec_convert', 'speed_perturbation'
        types='.flac', 
        n_core=10
    ):
    """
    types -> str: orginal type 
        option) [".flac"]
    convert_types -> list: new type to convert
        option) [".mp3", ".m4a", ".ogg", ".aac", ".wma", ".wav"]
    """ 
    
    all_files = glob.glob(path + '/*.flac')
    all_files.sort()
    print(f"Number of {types[1 : ]}: {len(all_files)}")

    # multiprocessing
    l_split_list = split_list(all_files, n_core)
    p = Pool(n_core)
    
    
    if processing == 'codec_convert':
        p.map(codec_convert, l_split_list)
    elif processing == 'speed_perturbation':
        p.map(speed_perturbation, l_split_list)

    p.close()
    
##
##===============================================================
##

def codec_convert(l_in):
    t = '.flac'
    convert_types = [".mp3", ".m4a", ".ogg", ".aac", ".wma", ".wav"]
    for file in tqdm(l_in):
        for new_t in convert_types:
            try:
                org = file
                new = org.replace(t[1 : ], new_t[1 : ])
                new2org = new.replace(new_t, '_from_' + new_t[1 : ] + t)
                if not os.path.exists(new2org):
                    os.system('ffmpeg -loglevel quiet -i %s %s' %(org, new))
                    os.system('ffmpeg -loglevel quiet -i %s %s' %(new, new2org))
                    os.remove(new)
            except Exception as e:
                print(e)
                with open('error.txt', 'a') as f:
                    f.write(f'Conversion failed {e} {file} to convert {new_t[1:]}\n')
                # raise ValueError(f'Conversion failed {file} to convert {new_t[1:]}')

def speed_perturbation(l_in):
    slow_sp = 0.9
    fast_sp = 1.1
    for file in tqdm(l_in):
        try:
            wav, sr = sf.read(file)
            
            slow_sr = round(len(wav) / slow_sp)
            fast_sr = round(len(wav) / fast_sp)
            
            slow_wav = signal.resample(wav, slow_sr)
            fast_wav = signal.resample(wav, fast_sr)
            
            new_dir = file
            slow_fname = new_dir.replace(".flac", "_slow.flac")
            fast_fname = new_dir.replace(".flac", "_fast.flac")
            
            sf.write(slow_fname, slow_wav, sr)
            sf.write(fast_fname, fast_wav, sr)
        except:
            with open('error.txt', 'a') as f:
                f.write(f'Conversion failed: {file}\n')
            # raise ValueError(f'{file}: Failed to speed_perturbation')

def checking_da(path_train, DA_codec, DA_speed):
    # Checking codec directory
    for codec in DA_codec:
        path = path_train.replace('flac', codec[1 : ])
        os.system(f'mkdir {path}')
    multiprocessing(path_train, "codec_convert")
    
    # Checking speed_perturbation directory
    #os.system(f'mkdir {path}')
    #multiprocessing(path_train, "speed_perturbation")

def remove(path):
    for root, _, files in os.walk(path):
        for file in files:
            if '_from' in file:
                path_rm = os.path.join(root, file)
                os.system(f'rm {path_rm}')

if __name__ == '__main__':
    YOUR_ASVspoof2019_PATH = '/home/tuantruong001/deepfake/asvspoof/asvspoof5_baseline/Baseline-AASIST'   # '/home/shin/exps/DB/ASVspoof2019'
    path_train = YOUR_ASVspoof2019_PATH + '/data/flac_T'
    DA_codec = [".mp3", ".m4a", ".ogg", ".aac", ".wma", ".wav"]
    DA_speed = True

    checking_da(path_train, DA_codec, DA_speed)
    
    path_train = YOUR_ASVspoof2019_PATH + '/data/flac_D'
    DA_codec = [".mp3", ".m4a", ".ogg", ".aac", ".wma", ".wav"]
    DA_speed = True

    checking_da(path_train, DA_codec, DA_speed)
