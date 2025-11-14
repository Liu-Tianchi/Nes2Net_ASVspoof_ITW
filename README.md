# ASVspoof5 Experiment


Welcome to the official release of pretrained models and scripts for *'Nes2Net: a lightweight nested architecture designed for foundation model-driven speech anti-spoofing'* [![arXiv](https://img.shields.io/badge/arXiv-2504.05657-b31b1b.svg)](https://arxiv.org/abs/2504.05657)

## 📁 Supported Datasets
- **asvspoof5 Branch (Current Branch)**: For the **ASVspoof 5** dataset  
  
- **main Branch**: For **ASVspoof 2019/2021** and **In-the-Wild** datasets: 👉 [main branch](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW/tree/main)

- **Controlled Singing Voice Deepfake Detection (CtrSVDD)**: 👉 [View here](https://github.com/Liu-Tianchi/Nes2Net)
  
## Pretrained Model
We have uploaded pretrained models of our experiments. You can download pretrained models from [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/truongdu001_e_ntu_edu_sg/EaBasBsecRpErWEVzXdht7cBiWWYFuLTeXt11ABbHX9yBg?e=vxVOOI). 

## Setting up environment
```
conda create --name asvspoof5 python=3.9
conda activate asvspoof5
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Usage
Before running the experiment, replace the data directory of `database_path` in the config file of `./config/AASIST_ASVspoof5.conf`.

To train & evaluate the model:
```
python ./main.py --config ./config/WavLM_Nes2Net_ASVspoof5.conf
```
### Acknowledge
Our work is built upon the [Baseline-AASIST](https://github.com/asvspoof-challenge/asvspoof5/tree/main/Baseline-AASIST) We also follow some parts of the following codebases:

[HM-Conformer](https://github.com/talkingnow/HM-Conformer/tree/main) (for noise augmentation).

[unilm](https://github.com/microsoft/unilm) (for WavLM model).

## Citation
```
@ARTICLE{Nes2Net,
  author={Liu, Tianchi and Truong, Duc-Tuan and Kumar Das, Rohan and Aik Lee, Kong and Li, Haizhou},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-Spoofing}, 
  year={2025},
  volume={20},
  number={},
  pages={12005-12018},
  keywords={Foundation models;Feature extraction;Computational modeling;Computer architecture;Computational efficiency;Dimensionality reduction;Acoustics;Kernel;Robustness;Deepfakes;Deepfake detection;speech anti-spoofing;Res2Net;Nes2Net;SSL;speech foundation model},
  doi={10.1109/TIFS.2025.3626963}}

```
