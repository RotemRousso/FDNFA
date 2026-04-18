import argparse
import dill
from argparse import Namespace
import torch
import torchaudio
# from utils import (detect_peaks, max_min_norm, replicate_first_k_frames)
from predict import main_predict
from next_frame_classifier import NextFrameClassifier

import dutch_preprocess


# from predict import main_predict
from tqdm import tqdm

from dataloader import spectral_size
import matplotlib.pyplot as plt
import numpy as np
import os


def test_predicts(wav_dir, ckpt, prominence, w_phi):
    total_sum = 0
    num_10 = 0
    num_15 = 0
    num_20 = 0
    num_25 = 0
    num_50 = 0
    num_100 = 0
    num_500 = 0
    
    file_count = 0
    
    wavs = [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")]
    for wav_file_name in tqdm(wavs, desc="Proccessing WAV files"):
        file_count = file_count+1
        # try:
        # if file_count<=75:
        # if file_count<=100: #300: #0000000:
        # if file_count<=300: #0000000:
        # if file_count<=100000000000000000:
        if file_count<=100000000000:
            wav_file_path = os.path.join(wav_dir,wav_file_name)
            pred_bound, truth_bound = main_predict(wav_file_path,ckpt,prominence,w_phi, language="english")
            # pred_bound, truth_bound = main_predict(wav_file_path,ckpt,prominence,w_phi, language="dutch")
            
            # ________OCT20 TRY_________________
            pred_bound = pred_bound[1:] 
            # truth_bound = truth_bound[1:] # -----------------------
            
            
            
            # ----- 0RIG ----------
            for p in truth_bound:
                curr_pred_bound = (pred_bound-p).abs()
                pred_p_diff = min(curr_pred_bound)
            # for t,p in zip(truth_bound,pred_bound):
            #     pred_p_diff = (t-p).abs()    
                if pred_p_diff <= 0.1:
                    num_100 = num_100 +1
                    if pred_p_diff <= 0.05:
                        num_50 = num_50 +1
                        if pred_p_diff <= 0.025:
                            num_25 = num_25 +1
                            if pred_p_diff <= 0.02:
                                num_20 = num_20 +1
                                if pred_p_diff <= 0.015:
                                    num_15 = num_15 +1   
                                    if pred_p_diff <= 0.01:
                                        num_10 = num_10 + 1
                total_sum = total_sum+1
                if pred_p_diff <=0.5:
                    num_500 = num_500 +1
            
            # # --------- OCT 20 TRY ------------
            # diffs = np.abs(np.asarray(truth_bound) - np.asarray(pred_bound))
            # diffs = diffs[1:-1]
            # total_sum += diffs.size
            # num_10  += int((diffs <= 0.01).sum())
            # num_15  += int((diffs <= 0.015).sum())
            # num_20  += int((diffs <= 0.02).sum())
            # num_25  += int((diffs <= 0.025).sum())
            # num_50  += int((diffs <= 0.05).sum())
            # num_100 += int((diffs <= 0.1).sum())
            # num_500 += int((diffs <= 0.5).sum())
            
        # except:
        #     print("failed try")
        #     print(f"failed filename: {wav_file_path}")
    # print(f"finished stats for cuur file {wav_file_name}:")
    print( " =================== TOTAL STATS =================== ")
    print(f"thresh = 10 ms:  {num_10*100/total_sum} % ")
    print(f"thresh = 15 ms:  {num_15*100/total_sum} % ")
    print(f"thresh = 20 ms:  {num_20*100/total_sum} % ")
    print(f"thresh = 25 ms:  {num_25*100/total_sum} % ")
    print(f"thresh = 50 ms:  {num_50*100/total_sum} % ")
    print(f"thresh = 100 ms:  {num_100*100/total_sum} % ")
    print(f"thresh = 500 ms:  {num_500*100/total_sum} % ")
    
    curr_precision_10 = num_10*100/total_sum
    curr_precision_15 = num_15*100/total_sum
    curr_precision_20 = num_20*100/total_sum
    curr_precision_25 = num_25*100/total_sum
    curr_precision_50 = num_50*100/total_sum
    curr_precision_100 = num_100*100/total_sum
    # precision_25 = num_25*100/total_sum
    return(curr_precision_10, curr_precision_15,curr_precision_20, curr_precision_25, curr_precision_50, curr_precision_100)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unsupervised segmentation inference script')
    parser.add_argument('--wav', help='path to wav file')
    parser.add_argument('--ckpt', help='path to checkpoint file')
    parser.add_argument('--prominence', type=float, default=None, help='prominence for peak detection (default: 0.05)')
    args = parser.parse_args()
    
    # preds,times_sec = main_predict(args.wav, args.ckpt, args.prominence)
    # just for check rn: 
    # ckpt = "/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/DP_AFTERDEBUG_tmux3Ironman_best_l1dist_val0.9_17MARCH/2025-03-17_16-21-22-default/epoch=23.ckpt"
    # ckpt = "/home/rotem/projects/CFA/DCAF/runs/DCAF_Hulk31MarchTIMITplaygrounds_tmux26_newconf/2025-04-01_17-30-16-default/epoch=17.ckpt"
    # ckpt = "/home/rotem/projects/CFA/DCAF/runs/DCAF_Hulk31MarchTIMITplaygrounds_tmux26_newconf/2025-04-01_11-54-42-default/epoch=17.ckpt"
    # ckpt = "//home/rotem/projects/CFA/DCAF/runs/DCAF_TMUX23Hulk25March/2025-03-25_18-27-05-default/epoch=6.ckpt"
    # ckpt = "/home/rotem/projects/CFA/DCAF/runs/April_16_TIMIT_tmux27/2025-04-16_03-00-32-default/epoch=2.ckpt"
    # ckpt = "/home/rotem/projects/CFA/DCAF/runs/April_21_TIMIT_tmux30/2025-04-21_17-13-06-default/epoch=49.ckpt"
    
    
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/JULY_11_noloss_soft_Align_full_timit_TMUX2/2025-07-11_17-44-16-default/8_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/JULY_15_wandbFULL_timitFULL_TMUX11/2025-07-15_19-53-51-default/2_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/JULY_15_wandbFULL_timitFULL_TMUX11/2025-07-15_19-53-51-default/5_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/JULY_11_noloss_soft_Align_full_timit_TMUX2/2025-07-11_17-44-16-default/15_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/JULY_27_FULLTIMIT_Tmux14/2025-07-27_18-48-03-default/0_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/July30_morew_phloss_tmux16/2025-07-30_14-50-12-default/0_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Sept3_both_preds_and_derivative_nologphi2_tmux30/2025-09-03_12-41-12-default/0_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Sept2_both_preds_and_derivative_tmux32/2025-09-02_15-10-48-default/11_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Sept2_both_preds_and_derivative_tmux32/2025-09-02_15-10-48-default/6_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Sept16_dp_backtrack_pointer_fixed_Tmux42/2025-09-16_23-24-19-default/12_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/OCT20_FullTIMITsmall_gamma_tmux4ironman/2025-10-20_16-40-00-default/45_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/OCT21_fullTIMIT_adaptive_loss_tmux6ironman/2025-10-21_16-14-08-default/40_best_model.pt"
    
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Nov19_fullTIMIT_adaptive_loss_new_phi_calc_logsumexp_tmux7ironman/2025-11-19_19-23-03-default/8_best_model.pt"
    
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/OCT21_fullTIMIT_adaptive_loss_tmux6ironman/2025-10-21_16-14-08-default/55_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Nov19_fullTIMIT_adaptive_loss_new_phi_calc_logsumexp_tmux7ironman/2025-11-19_19-23-03-default/11_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Nov27_fullBuckeye_lstmfix_zerocrossdriv_tmux1Hulk_gpu1/2025-11-27_13-18-36-default/3_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_BuckeyePreprocessed_frame_labelsNotZeroATlossNotForward_HulkTmux1gpu1/2025-12-02_17-03-59-default/5_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_fullTIMIT_frame_labelsNotZeroATlossNotForward_HulkTmux5gpu5/2025-12-02_16-12-30-default/44_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec28_TIMIT_NaivePeakDetection_noBiLSTM_tmux7Hulk_gpu7/2025-12-28_23-01-56-default/1_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec28_TIMIT_regularDP_tmux0Thanos_gpu0/2025-12-28_11-55-49-default/0_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec28_TIMIT_NaivePeakDetection_noBiLSTM_tmux7Hulk_gpu7/2025-12-30_20-32-44-default/199_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_fullTIMIT_frame_labelsNotZeroATlossNotForward_HulkTmux5gpu5/2025-12-02_16-12-30-default/56_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_fullTIMIT_frame_labelsNotZeroATlossNotForward_HulkTmux5gpu5/2025-12-02_16-12-30-default/"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_BuckeyePreprocessed_frame_labelsNotZeroATlossNotForward_HulkTmux1gpu1/2025-12-02_17-03-59-default/"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_fullTIMIT_frame_labelsNotZeroATlossNotForward_HulkTmux5gpu5/2025-12-02_16-12-30-default/4_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_BuckeyePreprocessed_frame_labelsNotZeroATlossNotForward_HulkTmux1gpu1/2025-12-02_17-03-59-default/"
    
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec28_TIMIT_NaivePeakDetection_noBiLSTM_tmux7Hulk_gpu7/2025-12-30_20-32-44-default/"
    
    
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_BuckeyePreprocessed_frame_labelsNotZeroATlossNotForward_HulkTmux1gpu1/2025-12-02_17-03-59-default/"
    ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_fullTIMIT_frame_labelsNotZeroATlossNotForward_HulkTmux5gpu5/2025-12-02_16-12-30-default/"
    
    
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Feb2_TIMIT_InfoNCE_tmux12Hulk_gpu7/2026-02-03_18-48-24-default/"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_fullTIMIT_frame_labelsNotZeroATlossNotForward_HulkTmux5gpu5/2025-12-02_16-12-30-default/"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_BuckeyePreprocessed_frame_labelsNotZeroATlossNotForward_HulkTmux1gpu1/2025-12-02_17-03-59-default/"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Feb2_TIMIT_InfoNCE_tmux12Hulk_gpu7/2026-02-03_18-48-24-default/"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_TIMITplayground_frame_labelsNotZeroATlossNotForward/2025-12-02_15-23-26-default/"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_BuckeyePreprocessed_frame_labelsNotZeroATlossNotForward_HulkTmux1gpu1/2025-12-02_17-03-59-default/"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/DPnoBiLSTM_tmux2Thanos_gpu0/2025-12-28_22-23-33-default/1_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec28_TIMIT_regularDPnoBiLSTM_tmux1Thanos_gpu1/2025-12-28_12-12-04-default/0_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec28_TIMIT_NaivePeakDetection_noBiLSTM_tmux7Hulk_gpu7/2025-12-28_23-01-56-default/0_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_fullTIMIT_frame_labelsNotZeroATlossNotForward_HulkTmux5gpu5/2025-12-02_16-12-30-default/41_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_BuckeyePreprocessed_frame_labelsNotZeroATlossNotForward_HulkTmux1gpu1/2025-12-02_17-03-59-default/2_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Nov27_fullBuckeye_lstmfix_zerocrossdriv_tmux1Hulk_gpu1/2025-11-27_13-18-36-default/0_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Nov30_fullTIMIT_val_on_trainTrue_tmux3Hulk_gpu3/2025-11-30_21-52-13-default/0_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Nov26_fullTIMIT_lstmfix_zerocrossdriv_tmux0Hulk_gpu0/2025-11-27_13-06-44-default/0_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Nov30_fullTIMIT_val_on_trainTrue_tmux3Hulk_gpu3/2025-11-30_21-52-13-default/0_best_model.pt"
    
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/OCT20_FullTIMITsmall_gamma_tmux4ironman/2025-10-20_16-40-00-default/10_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Sept16_dp_backtrack_pointer_fixed_Tmux42/2025-09-16_23-24-19-default/12_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Sept14_ph61_Tmux40/2025-09-14_18-51-41-default/5_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Sept8_truthPreds_tmux35/2025-09-08_15-49-36-default/22_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Sept8_truthPreds_tmux35/2025-09-08_15-49-36-default/7_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Sept8_Sameastmux34_lowerlr_tmux36/2025-09-08_18-46-55-default/6_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Sept3_both_preds_and_derivative_w_phi123_tmux34/2025-09-03_17-37-23-default/60_best_model.pt"
    
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/JULY_11_noloss_soft_Align_full_timit_TMUX2/2025-07-11_17-44-16-default/3_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/JULY_11_noloss_soft_Align_full_timit_TMUX2/2025-07-11_17-44-16-default/1_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/JULY_11_noloss_soft_Align_full_timit_TMUX2/2025-07-11_17-44-16-default/0_best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/JULY_2_timitPlaygroundtmux41/2025-07-02_18-54-26-default/best_model.pt"
    # ckpt = "/home/rotem/projects/CFA/DCAF/runs/April_16_TIMIT_tmux27/2025-04-16_03-00-32-default/epoch=24.ckpt"
    # ckpt = "/home/rotem/projects/CFA/DCAF/runs/April_16_TIMIT_tmux27/2025-04-16_03-00-32-default/epoch=17.ckpt"
    
    
    
    # ckpt = "/home/rotem/projects/CFA/DCAF/runs/April_21_TIMIT_tmux30/2025-04-22_16-37-25-default/epoch=13.ckpt"
    # ckpt = "/home/rotem/projects/CFA/DCAF/runs/April_16_TIMIT_tmux27/2025-04-16_03-00-32-default/epoch=17.ckpt"
    # ckpt = "/home/rotem/projects/CFA/DCAF/runs/April_21_TIMIT_tmux30/2025-04-22_16-37-25-default/epoch=4.ckpt"
    
    
    # ckpt = "/home/rotem/projects/CFA/DCAF/runs/April_21_TIMIT_tmux30/2025-04-22_16-37-25-default/epoch=0.ckpt"
    # ckpt = "/home/rotem/projects/CFA/DCAF/runs/April_16_TIMIT_tmux27/2025-04-16_03-00-32-default/epoch=3.ckpt"
    

    # "--wav", "/home/rotem/projects/CFA/changed_NFC_negative_not_random/overfit_check_data_timit/test/sa2.wav"
    
    
    
    # wav_dir = "/home/rotem/projects/datasets/IFA_dutch_reorder/test"
    # wav_dir = "/home/rotem/projects/datasets/IFA_dutch_ready"
    
    # wav_dir = "/home/rotem/projects/datasets/phondat1_parsed/test"
    # wav_dir = "/home/rotem/projects/datasets/hebrew_wrd/hebrew/test"
    # wav_dir = "/home/rotem/projects/datasets/phondat_roymeidan/phonedat/test"
    # wav_dir = "/home/rotem/projects/datasets/hebrew_wrd/hebrew/raw_date_1"
    # wav_dir = "/home/rotem/projects/datasets/phondat1_parsed/val"
    # wav_dir = "/home/rotem/projects/datasets/IFA_dutch_split/train"
    # wav_dir = "/home/rotem/projects/datasets/IFA_Dutch_words_reordered"
    
    # wav_dir = "/home/rotem/projects/datasets/timit/input"
    
    # wav_dir = "/home/rotem/projects/datasets/buckeye_preprocessed/test/"
    # wav_dir = "/home/rotem/projects/datasets/buckeye/test"
    wav_dir = "/home/rotem/projects/datasets/timit/timit_tixed/test/"
    # wav_dir = "/home/rotem/projects/datasets/timit/timit_tixed/train/"
    
    prominence =  0.1
    best_w_phi = 0.5
    # best_precision_25 = test_predicts(wav_dir, ckpt, prominence, best_w_phi)
    # for w_phi in tqdm([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    best_precision_10_, best_precision_15_, best_precision_20_, best_precision_25_, best_precision_50_, best_precision_100_  = [], [], [], [], [], []
    best_ckpt_10_, best_ckpt_15_, best_ckpt_20_, best_ckpt_25_, best_ckpt_50_, best_ckpt_100_  = [], [], [], [], [], []
    
    best_precision_10, best_precision_15, best_precision_20, best_precision_25, best_precision_50, best_precision_100  = -1, -1, -1, -1, -1, -1
    best_ckpt_10, best_ckpt_15, best_ckpt_20, best_ckpt_25, best_ckpt_50, best_ckpt_100  = -1, -1, -1, -1, -1, -1
    
    
    
    # for idx in tqdm(range(126)):
    # for idx in tqdm([90]):
    # for idx in tqdm([113]):
    # for idx in tqdm(range(25)):
    # for idx in tqdm([79]):
        
        
    # for idx in tqdm([38]):
    # for idx in tqdm([57]):
    # for idx in tqdm(range(135)):
    # for idx in tqdm(range(32)):
    
    # for idx in tqdm([12]):
    # for idx in tqdm([79]):
    # for idx in tqdm([57]):
    # for idx in tqdm([12]):
    # for idx in tqdm([57]):
    # for idx in tqdm([23]):
    # for idx in tqdm([28]):
    for idx in tqdm([12]):
        curr_ckpt = f"{ckpt}{idx}_best_model.pt"
        print(f"Testing: {curr_ckpt}")
        curr_precision_10, curr_precision_15, curr_precision_20, curr_precision_25, curr_precision_50, curr_precision_100 = test_predicts(wav_dir, curr_ckpt, prominence, best_w_phi)
        if curr_precision_10>best_precision_10:
            best_precision_10=curr_precision_10
            best_precision_10_.append(curr_precision_10)
            best_ckpt_10=idx
            best_ckpt_10_.append(idx)
        if curr_precision_15>best_precision_15:
            best_precision_15_.append(curr_precision_15)
            best_precision_15 =curr_precision_15
            best_ckpt_15_.append(idx)
            best_ckpt_15=idx
        if curr_precision_20>best_precision_20:
            best_precision_20=curr_precision_20
            best_precision_20_.append(curr_precision_20)
            best_ckpt_20=idx
            best_ckpt_20_.append(idx)
        if curr_precision_25>best_precision_25:
            best_precision_25=curr_precision_25
            best_precision_25_.append(curr_precision_25)
            best_ckpt_25=idx
            best_ckpt_25_.append(idx)
        if curr_precision_50>best_precision_50:
            best_precision_50=curr_precision_50
            best_precision_50_.append(curr_precision_50)
            best_ckpt_50=idx
            best_ckpt_50_.append(idx)
        if curr_precision_100>best_precision_100:
            best_precision_100 =curr_precision_100
            best_precision_100_.append(curr_precision_100)
            best_ckpt_100 =idx
            best_ckpt_100_.append(idx)

    print("***********************************")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(best_ckpt_10_, best_precision_10_, label='Precision @ 10ms', marker='o')
    plt.plot(best_ckpt_15_, best_precision_15_, label='Precision @ 15ms', marker='o')
    plt.plot(best_ckpt_20_, best_precision_20_, label='Precision @ 20ms', marker='o')
    plt.plot(best_ckpt_25_, best_precision_25_, label='Precision @ 25ms', marker='o')
    plt.plot(best_ckpt_50_, best_precision_50_, label='Precision @ 50ms', marker='o')
    plt.plot(best_ckpt_100_, best_precision_100_, label='Precision @ 100ms', marker='o')
    plt.xlabel('Checkpoint Index')
    plt.ylabel('Precision (%)')
    plt.title('Precision at Different Time Thresholds Across Checkpoints')
    plt.legend()
    # plt.grid()
    plt.show()
    plt.savefig("/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/plot_try/precision_plot_full_Ablations_March8.png")
    plt.savefig("/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/plot_try/precision_plot_full_pdf_Ablations_March8.pdf")
    # plt.savefig("/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/plot_try/precision_plot_full_Buckeye_March7.png")
    # plt.savefig("/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/plot_try/precision_plot_full_pdf_Buckeye_March7.pdf")
    
    
    print(f"FOR 10ms THE CHOSEN ckpt IS {best_ckpt_10} with precision {best_precision_10}")
    print(f"FOR 15ms THE CHOSEN ckpt IS {best_ckpt_15} with precision {best_precision_15}")
    print(f"FOR 20ms THE CHOSEN ckpt IS {best_ckpt_20} with precision {best_precision_20}")
    print(f"FOR 25ms THE CHOSEN ckpt IS {best_ckpt_25} with precision {best_precision_25}")
    print(f"FOR 50ms THE CHOSEN ckpt IS {best_ckpt_50} with precision {best_precision_50}")
    print(f"FOR 100ms THE CHOSEN ckpt IS {best_ckpt_100} with precision {best_precision_100}")
    x=1
    
    
    
    # prominence =  0.1
    # best_w_phi = 0.5
    # best_precision_25 = test_predicts(wav_dir, ckpt, prominence, best_w_phi)
    # # for w_phi in tqdm([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    # for w_phi in tqdm([]):
    #     print(f"curr w_phi = {w_phi}")
    #     curr_precision_25 = test_predicts(wav_dir, ckpt, prominence, w_phi)
    #     if curr_precision_25>best_precision_25:
    #         best_precision_25 = curr_precision_25
    #         best_w_phi = w_phi
    # print("***********************************")
    # print(f"THE CHOSEN W_PHI IS {best_w_phi}")
    # x=1