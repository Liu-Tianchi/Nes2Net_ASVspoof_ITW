# ASVspoof5 Experiment


Welcome to the official release of pretrained models and scripts for *'Nes2Net: a lightweight nested architecture designed for foundation model-driven speech anti-spoofing'* [![arXiv](https://img.shields.io/badge/arXiv-2504.05657-b31b1b.svg)](https://arxiv.org/abs/2504.05657)

Accpeted to: **IEEE Transactions on Information Forensics and Security** (T-IFS), IEEE Link: https://ieeexplore.ieee.org/document/11222612

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

# License and Usage
This project is released under the Apache License 2.0. The authors welcome and encourage academic use of this project and its released models, and are friendly to commercial use, modification, and redistribution, subject to the terms of the Apache License 2.0. If you use our models or code in your research or applications, we kindly ask that you acknowledge this work and cite the associated paper.

This project may rely on or incorporate publicly available models, datasets, and third-party open-source software. Users are solely responsible for reviewing and complying with the licenses and terms applicable to these third-party components. The release of this project does not grant any additional rights to third-party models, datasets, or software, and the authors are not responsible for users' compliance with such third-party terms.

The project, code, and models are provided "AS IS", without warranties of any kind. To the extent permitted by applicable law, the authors shall not be liable for any claims, damages, or other liabilities arising from the use, modification, or redistribution of the project or models.

## Citation
```
@ARTICLE{Nes2Net,
  author={Liu, Tianchi and Truong, Duc-Tuan and Das, Rohan Kumar and Lee, Kong Aik and Li, Haizhou},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-Spoofing}, 
  year={2025},
  volume={20},
  number={},
  pages={12005-12018},
  keywords={Foundation models;Feature extraction;Computational modeling;Computer architecture;Computational efficiency;Dimensionality reduction;Acoustics;Kernel;Robustness;Deepfakes;Deepfake detection;speech anti-spoofing;Res2Net;Nes2Net;SSL;speech foundation model},
  doi={10.1109/TIFS.2025.3626963}}

```
