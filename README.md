# Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing
Official release of pretrained models and scripts for "Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing"

----------------------------------------------------------------------
🚨 **Important:** This repository is still in preparation. We will try to complete it by March 2025.
----------------------------------------------------------------------


**This repo is for the **ASVspoof** and **In-the-Wild** dataset.**

For the Controlled Singing Voice Deepfake Detection (CtrSVDD) dataset: https://github.com/Liu-Tianchi/Nes2Net

For the **PartialSpoof** dataset: Coming soon

arXiv Link: To add

# Update:


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
| 2025    | wav2vec 2.0 | Mamba          | 1,937k              | 5         | 0.93             | 1.88                                       |
| 2022    | wav2vec 2.0 | AASIST         | 447k                | N/A       | **0.82 (1.00)**  | 2.85 (3.69)                                |
| re-imp  | wav2vec 2.0 | AASIST (algo4) | 447k                | N/A       | 1.13 (1.36)      | 3.37 (4.09)                                |
| re-imp  | wav2vec 2.0 | AASIST (algo5) | 447k                | N/A       | 0.93 (1.40)      | 3.56 (5.07)                                |
| **Ours** | wav2vec 2.0 | **Nes2Net-X**  | 511k                | N/A       | 1.73 (1.95)      | 1.65 (1.91) [Google Drive for 1.65%: [ckpt](), [score file]()]     |
| **Ours** | wav2vec 2.0 | **Nes2Net-X**  | 511k                | 3         | 1.66 (1.87)      | 1.54 (1.98)                                |
| **Ours** | wav2vec 2.0 | **Nes2Net-X**  | 511k                | 5         | 1.88 (2.00)      | **1.49 (1.78)** [Google Drive for 1.49%: [ckpt](), [score file]()] |

## In-the-Wild. Format: best (mean)
| Year | Back-end          | EER         |
|------|------------------|------------|
| 2022 | AASIST [15]      | 10.46      |
| 2024 | SLIM [14]        | 12.5       |
| 2024 | MoE [65]         | 9.17       |
| 2024 | Conformer [59]   | 8.42       |
| 2024 | TCM [29]         | 7.79       |
| 2024 | OCKD [66]        | 7.68       |
| 2024 | SLS [32]        | 7.46       |
| 2024 | Pascu et al. [68]| 7.2        |
| 2025 | Mamba [17]       | 6.71       |
| 2025 | LSR+LSA [67]     | 5.92       |
| -    | **Proposed Nes2Net-X w/o Val Aug.** | **5.52 (6.60)** [Google Drive for 5.52%: [ckpt](), [score file](https://drive.google.com/file/d/12vzWxsVUAgmayk2WYfurib-qmNKMHjL2/view?usp=sharing)] |
| -    | **Proposed Nes2Net-X w/ Val Aug.** | **5.15 (6.31)** [Google Drive for 5.15%: [ckpt](), [score file]()] |


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

# Usage （To update):

  1. If you want to easy inference with pre-ptrained model:
     1. Download the pretrained checkpoints from above table Google Drive links. For example, WavLM_Nes2Net_X_SeLU.
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
     
  2. If you want to train the model by yourself on ASVspoof19 dataset:
     check the command template: 
     ```
     CUDA_VISIBLE_DEVICES=0 python main.py --track=LA --lr=0.00000025 --batch_size=12 --loss=WCE \
     --algo 4 --date 0520 --seed 12345 \
     --model_name wav2vec2_Nes2Net_X --num_epochs 100 --pool_func 'mean' --SE_ratio 1 --Nes_ratio 8 8 \
     --database_path /home/tianchi/SSL_Anti-spoofing/database/LA/
     ```
     * Change the ```--database_path``` to your ASVspoof dataset path. 

     
  3. If you want to test on the ASVspoof 21 LA or DF dataset using the released pre-trained models or your own trained model:
     
     check the command template in: 
     ```
     test.sh
     ```

     Following is an example:
     ```
     CUDA_VISIBLE_DEVICES=6 python eval.py --base_dir /home/tianchi/data/SVDD2024/test_set \
     --model_path "/data/tianchi/Nes2Net_SVDD_ckpts/WavLM_Nes2Net_X_e75_seed420_valid0.03192785031473534.pt" \
     --agg SEA --pool_func 'mean' --dilation 1 --Nes_ratio 8 8 --SE_ratio 1 --model_name WavLM_Nes2Net_X \
     --outputname E75_WavLM_SEA_Nes2Net_X_mean_8x8_SEr1_dila1_algo8_Tmax5_bz34_lr1e6_seed420
     ```
     * Change the ```--base_dir``` to your SVDD2024 dataset **testset** path. 
     * Change the ```--model_path``` to your path of the checkpoint to test.
     * You may use the checkpoint with the smallest validation EER for testing.
     * Change the ``` --pool_func --Nes_ratio --SE_ratio --model_name``` to match your training setting. 
     If you are using the pretrained model, these configs are available in  ```test.sh```. 
     To get the final result of EER and minDCF:
     ```
     python EER_minDCF.py --labels_file [path to the CtrSVDD test set label txt] \
     --path [path to the score file generated by above command]
     ``` 
     For example:
     ```
     python EER_minDCF.py --labels_file '/home/tianchi/data/SVDD2024/test.txt' \
     --path scores/E75_WavLM_SEA_Nes2Net_X_mean_8x8_SEr1_dila1_algo8_Tmax5_bz34_lr1e6_seed420.txt
     ```
  5. If you want to test on the In-the-Wild dataset using the released pre-trained models or your own trained model:

     scoring:
     ```
     python Cal_EER_in_the_wild.py --path [path to score txt] 
     ```
     For example:
     ```
     python Cal_EER_in_the_wild.py --path Test_full_ITW_e68_74_76_90_100_1203_eff_wav2vec2_Nes2Net_SE_cat_e120_bz12_lr2_5e_07_algo4_seed12345.txt
     ```
  6. If you want to test with averaged checkpoints:
     
     

# Reference Repo
Thanks for following open-source projects:
1. wav2vec2 + AASIST & Rawboost: https://github.com/TakHemlata/SSL_Anti-spoofing Paper: [[model]](https://arxiv.org/abs/2202.12233), [[Rawboost]](https://arxiv.org/abs/2202.12233)
2. SEA aggregation: https://github.com/Anmol2059/SVDD2024 Paper: [[SEA]](https://arxiv.org/abs/2409.02302)
3. AttM aggregation: https://github.com/pandarialTJU/AttM_INTERSPEECH24 Paper: [[AttM]](https://arxiv.org/abs/2406.10283v1)

# Cite
```  
To add
```
