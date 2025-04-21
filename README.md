# 🔥🔥🔥Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing

Welcome to the official release of pretrained models and scripts for **Nes2Net**, a lightweight nested architecture designed for foundation model-driven speech anti-spoofing, as described in our paper [![arXiv](https://img.shields.io/badge/arXiv-2504.05657-b31b1b.svg)](https://arxiv.org/abs/2504.05657)

## 📁 Supported Datasets

This repository supports the following datasets:

- **ASVspoof 2019 / 2021**
- **ASVspoof 5**
- **In-the-Wild**
- **Controlled Singing Voice Deepfake Detection (CtrSVDD)**: [View here](https://github.com/Liu-Tianchi/Nes2Net)

## 🌿 Branch Overview

- **Main Branch**  
  For **ASVspoof 2019/2021** and **In-the-Wild** datasets:  
  👉 [main](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW/tree/main)

- **asvspoof5 Branch**  
  For the **ASVspoof 5** dataset:  
  👉 [asvspoof5](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW/tree/asvspoof5)


# Update:
🔥[April 20, 2025] ASVspoof 5 pretrained model and codes are now avaiable!!! 👉 [asvspoof5](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW/tree/asvspoof5)

# Pretrained Models
## ASVspoof 21 LA and DF. Format: best (mean)
| Remark  | Front-end    | Back-end Model | Back-end Parameters | CKPT Avg. | ASVspoof 2021 LA | ASVspoof 2021 **DF**                       |
|---------|-------------|----------------|---------------------|-----------|------------------|--------------------------------------------|
| 2022    | wav2vec 2.0 | FIR-NB         | -                   | -         | 3.54             | 6.18                                       |
| 2022    | wav2vec 2.0 | FIR-WB         | -                   | -         | 7.08             | 4.98                                       |
| 2022    | wav2vec 2.0 | LGF            | -                   | -         | 9.66             | 4.75                                       |
| 2023    | wav2vec 2.0 | Conformer      | 2,51k               | -         | 1.38             | 2.27                                       |
| 2024    | wav2vec 2.0 | Ensembling     | -                   | -         | 2.32 (4.48)      | 5.60 (8.74)                                |
| 2024    | WavLM       | ASP+MLP        | 1,051k              | -         | 3.31             | 4.47                                       |
| 2024    | wav2vec 2.0 | SLIM           | -                   | -         | -                | 4.4                                        |
| 2024    | WavLM       | AttM-LSTM      | 936k                | N/A       | 3.50             | 3.19                                       |
| 2024    | wav2vec 2.0 | FTDKD          | -                   | -         | 2.96             | 2.82                                       |
| 2024    | wav2vec 2.0 | AASIST2        | -                   | -         | 1.61             | 2.77                                       |
| 2024    | wav2vec 2.0 | MFA            | -                   | -         | 5.08             | 2.56                                       |
| 2024    | wav2vec 2.0 | MoE            | -                   | -         | 2.96             | 2.54                                       |
| 2024    | wav2vec 2.0 | OCKD           | -                   | -         | 0.90             | 2.27                                       |
| 2024    | wav2vec 2.0 | TCM            | 2,383k              | 5         | 1.03             | 2.06                                       |
| 2024    | wav2vec 2.0 | SLS            | 23,399k             | -         | 2.87 (3.88)      | 1.92 (2.09)                                |
| 2025    | wav2vec 2.0 | LSR+LSA        | -                   | -         | 1.19             | 2.43                                       |
| 2025    | wav2vec 2.0 | LSR+LSA ※      | -                   | -         | 1.05 ※           | 1.86 ※                                     |
| 2025    | wav2vec 2.0 | conf. ensemble | -                   | -         | -                | 2.03                                       |
| 2025    | wav2vec 2.0 | WaveSpec       | -                   | -         | -                | 1.90                                       |
| 2025    | wav2vec 2.0 | Mamba          | 1,937k              | 5         | 0.93             | 1.88                                       |
| 2022    | wav2vec 2.0 | AASIST         | 447k                | N/A       | **0.82 (1.00)**  | 2.85 (3.69)                                |
| reproduce  | wav2vec 2.0 | AASIST (algo4) | 447k             | N/A       | 1.13 (1.36)      | 3.37 (4.09)                                |
| reproduce  | wav2vec 2.0 | AASIST (algo5) | 447k             | N/A       | 0.93 (1.40)      | 3.56 (5.07)                                |
| **Ours** | wav2vec 2.0 | **Nes2Net-X**  | 511k               | N/A       | 1.73 (1.95)      | 1.65 (1.91) [Google Drive for 1.65%: [ckpt](https://drive.google.com/file/d/1tjuSdbzgCnJSfy_eE_P52jRAonHY4YUT/view?usp=sharing), [score file](https://drive.google.com/file/d/1-bgVLjCTCXxsJHuEpipB1A3hwdDaj2c6/view?usp=sharing)]     |
| **Ours** | wav2vec 2.0 | **Nes2Net-X**  | 511k               | 3         | 1.66 (1.87)      | 1.54 (1.98)                                |
| **Ours** | wav2vec 2.0 | **Nes2Net-X**  | 511k               | 5         | 1.88 (2.00)      | **1.49 (1.78)** [Google Drive for 1.49%: [ckpt](https://drive.google.com/file/d/1JFGv_2TONMnTLGbiOIuHFfMvuo4SIIpg/view?usp=sharing), [score file](https://drive.google.com/file/d/1UbQBoddwtMgGee4BYPHNPb5JwaPEcDtw/view?usp=sharing)] |

※: with extra data augmentation.

## In-the-Wild. Format: best (mean)
| Year | Back-end          | EER         |
|------|------------------|------------|
| 2022 | AASIST       | 10.46      |
| 2024 | SLIM         | 12.5       |
| 2024 | MoE          | 9.17       |
| 2024 | Conformer    | 8.42       |
| 2024 | TCM          | 7.79       |
| 2024 | OCKD         | 7.68       |
| 2024 | SLS          | 7.46       |
| 2024 | Pascu et al. | 7.2        |
| 2025 | Mamba        | 6.71       |
| 2025 | WaveSpec     | 6.71       |
| 2025 | LSR+LSA      | 5.92       |
| 2025 | LSR+LSA ※    | 5.54 ※     |
| -    | **Proposed Nes2Net-X w/o Val Aug.** | **5.52 (6.60)** [Google Drive for 5.52%: [ckpt](https://drive.google.com/file/d/1rKuzoo-Trngjae8sPOduxG9jofgcVkRz/view?usp=sharing), [score file](https://drive.google.com/file/d/12vzWxsVUAgmayk2WYfurib-qmNKMHjL2/view?usp=sharing)] |
| -    | **Proposed Nes2Net-X w/ Val Aug.** | **5.15 (6.31)** [Google Drive for 5.15%: [ckpt](https://drive.google.com/file/d/14YD8A54HLkXzlp7PWZ5iTW56WmQTIeEy/view?usp=sharing), [score file](https://drive.google.com/file/d/1Xx43lvq6xjEMgt8_Eq6NrWYQfkeUXsvo/view?usp=sharing)] |

※: with extra data augmentation.

* Only best model checkpoints are provided.


# Prepare:

  1. Git clone this repo.
  2. Build environment:
     ```
     conda env create -f environment.yml
     ```
     or
     ```
     pip install -r requirements.txt
     ```
     * You may need to adjust the library versions according to your CUDA version and GPU spec.
     
  3. Setup fairseq:
     ```
     cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
     (This fairseq folder can also be downloaded from https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1)
     pip install --editable ./
     pip install -r requirements.txt
     ```
  4. Pre-trained wav2vec 2.0 XLSR (300M): Download the XLSR models from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr)

# Dataset:
     
   If you want to train the model: The ASVspoof 2019 dataset can be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3336).
     
   If you want to test on the ASVspoof 2021 database, it is released on the zenodo site.
     
   -- LA [here](https://zenodo.org/records/4837263#.YnDIinYzZhE)
     
   -- DF [here](https://zenodo.org/records/4835108#.YnDIb3YzZhE)
     
   -- keys (labels) and metadata [here](https://www.asvspoof.org/index2021.html)
   
   If you want to test on the In-the-Wild dataset, it can be downloaded from [here](https://deepfake-total.com/in_the_wild)

# Usage:

## If you want to easy inference with pre-ptrained model:
  1. Download the pretrained checkpoints from above table Google Drive links. For example, WavLM_Nes2Net_X.
  2. Run 
     ```
     CUDA_VISIBLE_DEVICES=0 python easy_inference_demo.py \
     --model_path [pretrained_model_path] \
     --file_to_test [the file to test] \
     --model_name xxxx
     ```
     Following is an example:
     ```
     CUDA_VISIBLE_DEVICES=0 python easy_inference_demo.py \
     --model_path "/data/tianchi/Nes2Net_SVDD_ckpts/WavLM_Nes2Net_X_SeLU_e74_seed420_valid0.04245662278274772.pt" \
     --file_to_test "/home/tianchi/Nes2Net_ASVspoof_ITW/database/LA/ASVspoof2019_LA_dev/flac/LA_D_1000265.flac" \
     --model_name WavLM_Nes2Net_X
     ```
     
## If you want to train the model by yourself on ASVspoof19 dataset:
     
  check the command template: 
  ```
  CUDA_VISIBLE_DEVICES=0 python main.py --track=LA --lr=0.00000025 --batch_size=12 \
  --algo 4 --date 0520 --seed 12345 --loss=WCE --model_name wav2vec2_Nes2Net_X \
  --num_epochs 100 --pool_func 'mean' --SE_ratio 1 --Nes_ratio 8 8 \
  --database_path /home/tianchi/SSL_Anti-spoofing/database/LA/
  ```
  * Change the ```--database_path``` to your ASVspoof dataset path. 

     
## If you want to test on the ASVspoof 21 LA or DF dataset using the released pre-trained models or your own trained model:
     
  check the command template in: 
  ```
  test.sh
  ```
  1. For single model or averaged pretrained model:
     For the ASVspoof 21 LA, an example:
     ```
     python main.py --track=LA --batch_size=2 --is_eval --eval --model_name wav2vec2_Nes2Net_X --pool_func 'mean' --SE_ratio 1 --Nes_ratio 8 8 --test_protocol '4sec' \
     --database_path '/home/tianchi/database/LA/' \
     --model_path="./wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_ep95.pth" \
     --eval_output='score_LA_wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_ep95.txt'
     ```
     For the ASVspoof 21 DF, change the above command with:
     ```
     --track=DF
     ```
     For already averaged model, simply change the model path:
     ```
     --model_path="./wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_seed12345_avg_ckpt_ep59_61_69.pth" \
     ```
     * Change the ```--database_path``` to your ASVspoof dataset path. 
     * Change the ```--model_path``` to your path of the checkpoint to test. You may use the checkpoint with the smallest validation EER for testing.
     * if you wish to average checkpoints and then test, pls refer to point 3 below
    
     And then, to get the final result of EER and minDCF:
     ```
     txtpath=score_LA_wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_ep95.txt
     python evaluate_2021_LA.py $txtpath /home/tianchi/database/keys eval
     ```
     and
     ```
     txtpath=score_DF_wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_ep95.txt
     python evaluate_2021_DF.py $txtpath /home/tianchi/database/keys eval
     ``` 
     
 3. For averaging checkpoints and then test, an example:
    ```
    python main.py --track=DF --batch_size=2 --is_eval --eval --model_name wav2vec2_Nes2Net_X --pool_func 'mean' --SE_ratio 1 --Nes_ratio 8 8 --test_protocol '4sec' \
    --num_average_model 5 --model_ID_to_average 56 60 62 76 95 \
    --database_path '/home/tianchi/database/LA/' \
    --model_folder_path="/home/tianchi/SSL_Anti-spoofing/models/wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4" \
    --eval_output='score_DF_wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_avg_ckpt_ep56_60_62_76_95.txt'
    ```
     * If you use checkpoints average function, choose the serveral epochs with smallest validation EERs for testing according to validation EER, and set as ``` --model_ID_to_average```.
     * change the ```--model_folder_path``` to the path of the folder that saving all checkpoints.
     * Change the ``` --pool_func --Nes_ratio --SE_ratio --model_name``` to match your training setting. 
     * If you are using the pretrained model, these configs are the same as default.
 
    And then, similar as that of point 2 above, following the examples to get the EER and minDCF from the score text file.


## If you want to test on the In-the-Wild dataset using the released pre-trained models or your own trained model:
  1. For single model or averaged pretrained model:
     The same as the command above for ASVspoof 21LA and DF, just change the ```--track=in_the_wild``` and ```--in_the_wild_path```, following is anexnample:
     ```
     python main.py --track=in_the_wild --batch_size=2 --is_eval --eval --model_name wav2vec2_Nes2Net_X --pool_func 'mean' --SE_ratio 1 --test_protocol 'full' \
     --in_the_wild_path='/home/tianchi/SSL_Anti-spoofing/database/release_in_the_wild' \
     --model_path="/home/tianchi/SSL_Anti-spoofing/models/wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_seed12345/wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_seed12345_avg_ckpt_ep59_61_69.pth" \
     --eval_output='score_ITW_wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_seed12345_avg_ckpt_ep59_61_69.txt' \
     ```
  3. Similarly, you can also average checkpoints and then test, an example:
     ```
     python main.py --track=in_the_wild --batch_size=2 --is_eval --eval --model_name wav2vec2_Nes2Net_X --pool_func 'mean' --SE_ratio 1 --test_protocol 'full' \
     --num_average_model 3 --model_ID_to_average 59 61 69 \
     --in_the_wild_path='/home/tianchi/SSL_Anti-spoofing/database/release_in_the_wild' \
     --model_folder_path="/home/tianchi/SSL_Anti-spoofing/models/wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_seed12345" \
     --eval_output='score_ITW_wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_seed12345_avg_ckpt_ep59_61_69.txt' \
     ```
  5. scoring:
     ```
     python Cal_EER_in_the_wild.py --path [path to score txt] 
     ```
     For example:
     ```
     python Cal_EER_in_the_wild.py --path score_ITW_wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_seed12345_avg_ckpt_ep59_61_69.txt
     ```
     
     

# Reference Repo
Thanks for following open-source projects:
1. wav2vec2 + AASIST & Rawboost: https://github.com/TakHemlata/SSL_Anti-spoofing Paper: [[model]](https://arxiv.org/abs/2202.12233), [[Rawboost]](https://arxiv.org/abs/2202.12233)
2. SEA aggregation: https://github.com/Anmol2059/SVDD2024 Paper: [[SEA]](https://arxiv.org/abs/2409.02302)
3. AttM aggregation: https://github.com/pandarialTJU/AttM_INTERSPEECH24 Paper: [[AttM]](https://arxiv.org/abs/2406.10283v1)

# Cite
```  
@article{liu2025nes2net,
  title={Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing},
  author={Liu, Tianchi and Truong, Duc-Tuan and Das, Rohan Kumar and Lee, Kong Aik and Li, Haizhou},
  journal={arXiv preprint arXiv:2504.05657},
  year={2025}
}
```
