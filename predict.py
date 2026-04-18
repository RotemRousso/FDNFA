import argparse
from glob import glob
from unittest import case
import dill
from argparse import Namespace
import torch
import torchaudio
import torch.nn.functional as F
from utils import (detect_peaks,  max_min_norm, replicate_first_k_frames,
                   get_timit_61_phoneme_mappings, get_mu_stats, get_sigma_stats, timit_to_leehon,
                   phoneme_alignment, compute_phi_1,compute_phi_2,compute_phi_3,compute_phi_4)
from next_frame_classifier import NextFrameClassifier

from dataloader import spectral_size
import matplotlib.pyplot as plt
import numpy as np
import os

import dutch_preprocess
from utils import timit_to_leehon_map_MACRO, timit_leehon_39_phonemes, timit_61_phonemes

# def main_predict(wav, ckpt, prominence,w_phi, language="english"):
def main_predict(wav, ckpt, prominence,w_phi, language="dutch"):
    print(f"running inference on: {wav}")
    print(f"running inferece using ckpt: {ckpt}")
    print("\n\n", 90 * "-")

    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    hp = ckpt["hparams"]
    # hp = Namespace(**dict(ckpt["hparams"]))

    # load weights and peak detection params
    model = NextFrameClassifier(hp)
    try:
        weights = ckpt["state_dict"]
    except:
        weights = ckpt["model_state_dict"]
    weights = {k.replace("NFC.", ""): v for k,v in weights.items()}
    model.load_state_dict(weights)
    model.eval()
    
    
    peak_detection_params = dill.loads(ckpt['peak_detection_params'])['cpc_1']
    peak_detection_params["prominence"] = prominence
    # load data
    audio, sr = torchaudio.load(wav)
    assert sr == 16000, "model was trained with audio sampled at 16khz, please downsample."
    audio = audio[0]
    # audio = audio.unsqueeze(0)
    
    base_dir = os.path.dirname(wav)
    base_name = os.path.basename(wav).split('.')[0]
    search_pattern = os.path.join(base_dir, f"{base_name}*.phn")
    # search_pattern = os.path.join(base_dir, f"{base_name}*.wrd")
    # search_pattern = os.path.join(base_dir, f"{base_name}*.word")
    matching_files = glob(search_pattern)
    if matching_files:
        phn_path = matching_files[0]
    else:
        print("No matching .phn file found. Using default naming convention.")
        phn_path = wav.replace("wav", "phn")

    # load audio
    audio_len = len(audio)
    spectral_len = spectral_size(audio_len)
    len_ratio = (audio_len / spectral_len)

    # load labels -- segmentation and phonemes
    with open(phn_path, "r") as f:
        lines = f.readlines()
        lines = list(map(lambda line: line.split(" "), lines))

        # get segment times
        times = torch.FloatTensor(list(map(lambda line: int(float(line[1]) / len_ratio), lines)))[:-1]  # don't count end time as boundary
        # times = torch.FloatTensor(list(map(lambda line: int(int(line[1]) / len_ratio), lines)))[:-1]  # don't count end time as boundary
        times_sec = torch.FloatTensor(list(map(lambda line: (float(line[1]) / sr), lines)))[:-1]  # don't count end #sr = 16000 in TIMIT
        # times_sec = torch.FloatTensor(list(map(lambda line: (int(line[1]) / sr), lines)))[:-1]  # don't count end #sr = 16000 in TIMIT
        
        # get phonemes in each segment (for K times there should be K+1 phonemes)
        phonemes = list(map(lambda line: line[2].strip(), lines))
    
    if language == "dutch":
        lh39_ph = []
        for IFA_ph in phonemes:
            print(f"\nINPUT: {IFA_ph}")
            # output = dutch_preprocess.aligner_pipeline(timit_to_leehon_map_MACRO[IFA_ph.lower()])
            output = dutch_preprocess.aligner_pipeline(IFA_ph if IFA_ph.lower() not in timit_61_phonemes else timit_to_leehon_map_MACRO[IFA_ph.lower()])
            # output = dutch_preprocess.aligner_pipeline(IFA_ph)
            
            # # FOR WORDS: #not good
            # lh39_ph.append(output[0]["lh39"])
            
            # FOR PHONEMES:
            lh39_ph.append([x["lh39"] for x in output])
        if not output:
            print("Results: None")
        print(phonemes)
        print(f"Dutch IPA to LH39 mapping: {lh39_ph}")
        
        # try
        phonemes = np.hstack(lh39_ph).tolist()
        
    audio, seg, phonemes, length = audio.unsqueeze(0), [times.tolist()], [phonemes], [audio_len/len_ratio] #[spectral_size(len(audio))]

    
    
    with torch.no_grad():
        model.eval()
        
        # preds,original_lengths, probs, frame_labels = model(audio,None,phonemes,length)
        # ------- Sept 10 - check with truth preds no truth for nce ---------------
        preds,original_lengths, probs, frame_labels, _,preds_peaks, w_phi = model(audio,None,phonemes,length)
        # preds,original_lengths, probs, frame_labels, _,preds_peaks, w_phi = model(audio,seg,phonemes,length)
        # ------- ------------------------------------------------- ---------------
    
    phoneme_to_idx, idx_to_phoneme = get_timit_61_phoneme_mappings()
    
    # # ------ in debug mode only can plot the z (as probs) --------
    # import numpy as np
    # import matplotlib.pyplot as plt
    
    # audio_spec = torchaudio.transforms.Spectrogram(n_fft=512)(audio.unsqueeze(0) if audio.dim() == 1 else audio)
    # audio_spec_db = torchaudio.transforms.AmplitudeToDB()(audio_spec)

    # fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # axes[0].imshow(audio_spec_db[0].detach().cpu().numpy(), aspect='auto', cmap='viridis')
    # axes[0].set_xlabel('Frame Index')
    # axes[0].set_ylabel('Frequency Bin')
    # axes[0].set_title('(a) Original Spectrogram')
    
    # z_data = probs[0].detach().cpu().numpy().T
    # axes[1].imshow(z_data, aspect='auto', cmap='viridis')
    # axes[1].set_xlabel('Frame Index')
    # axes[1].set_ylabel('z latent representation')
    # axes[1].set_title('(b) z trained CNN Output')

    # # plt.figure(figsize=(16, 6))
    # # plt.imshow(probs[0].detach().cpu().numpy().T, aspect='auto', cmap='viridis')
    # # plt.xlabel('z Frame Index')
    # # plt.ylabel('z latent representation')
    # # plt.title('z trained CNN Output')
    # for i, s in enumerate(times):  # times is your list of segment boundaries in frames
    #     axes[1].axvline(x=s, color='red', linestyle='--', linewidth=1, label='Truth boundary' if i == 0 else "")
    # axes[1].legend(loc='upper right')
    
    # truth_signal = np.zeros(int(original_lengths[0]))
    # for time in times:
    #     idx = min(int(time), truth_signal.shape[0] - 1)
    #     truth_signal[idx] = 1.0
    
    # # axes[2].bar(range(len(truth_signal)), truth_signal, width=1.0, color='red', alpha=0.6)
    # # axes[2].set_xlabel('Frame Index')
    # # axes[2].set_ylabel('Truth Boundary')
    # # axes[2].set_title('(c) Truth Segmentation Boundaries')
    # # axes[2].set_ylim([0, 1.2])
    
    # # for i, s in enumerate(times):
    # #     axes[2].axvline(x=s, color='red', linestyle='--', linewidth=1.5)

    
    # plt.tight_layout()  # Adjust layout to prevent overlap
    # plt.savefig("/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/z_cnn_out_and_truth_boundaries_ckpt12.png")
    # print("/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/z_cnn_out_and_truth_boundaries_ckpt12.png")
    # # ------------------------------------------------------------
    
    
    phoneme_labels = [idx_to_phoneme[i] for i in range(39)]
    # phoneme_labels = [idx_to_phoneme[i] for i in range(61)]
    # phoneme_labels = [idx_to_phoneme[i] for i in range(41)]
    # probs_real = probs #F.softmax(probs, dim=-1) 
    probs_real = F.softmax(probs, dim=-1) 
   
    probs_logits = probs.squeeze(0)
    probs_real = probs_real.squeeze(0)
    # Plot the probability map
    probs_real = probs_real.detach().numpy()
    
    plt.figure(figsize=(15, 5))
    # plt.imshow(torch.log(torch.tensor(probs_real.T)), aspect='auto', cmap='viridis')  # Transpose to make rows phonemes and columns frames
    plt.imshow(probs_real.T, aspect='auto', cmap='viridis')  # Transpose to make rows phonemes and columns frames
    plt.colorbar(label='Probability')
    plt.xlabel('Frame Index')
    plt.ylabel('Phoneme')
    
    plt.yticks(ticks=range(39), labels=phoneme_labels)
    # plt.yticks(ticks=range(41), labels=phoneme_labels)
    # plt.yticks(ticks=range(61), labels=phoneme_labels)
    
    plt.title('Frame-wise Label Probability Map')
    
    for i,s in enumerate(times):  # times is your list of segment boundaries in frames
        plt.axvline(x=s, color='red', linestyle='--', linewidth=1, label='Truth boundary' if s == times[0] else "")
        plt.text(s, probs_real.shape[1] + 1, phonemes[0][i], color='red', rotation=90, va='top', ha='center', fontsize=8)
    
    
    plt.show()
    plt.savefig("/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/probs_DP_LOSS.png")
    print("/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/probs_DP_LOSS.png")
    # plt.savefig("/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/probs_try_phoneme_tmux9.png")
    # print("/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/probs_try_phoneme_tmux9.png")
    plt.close()
    
    plt.figure(figsize=(15, 5))
    plt.imshow(probs_logits.T, aspect='auto', cmap='viridis')  # Transpose to make rows phonemes and columns frames
    plt.colorbar(label='Probability')
    plt.xlabel('Frame Index')
    plt.ylabel('Phoneme')
    # plt.yticks(ticks=range(61), labels=phoneme_labels)
    plt.yticks(ticks=range(39), labels=phoneme_labels)
    plt.title('Frame-wise Label Probability Map')
    plt.show()
    plt.savefig("/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/logits_try_DP_loss.png")
    print("/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/logits_try_DP_loss.png")
    plt.close()
    
    # run inference
    preds = preds[1][0]  # get scores of positive pairs
    preds = max_min_norm(preds)  # normalize scores (good for visualizations)
    # preds = 1 - max_min_norm(preds)  # normalize scores (good for visualizations)

    filename = os.path.basename(wav).split('/')[-1]
    preds_np = preds.detach().numpy()
    preds_np = preds_np[0] #*len_ratio/sr

    median_h = np.median(preds_np)
    preds_np = preds_np - median_h
    
    # # ----------- 25 MARCH --------------
    # preds_np[preds_np < 0] = 0
    # # -----------------------------------
    
    signal = np.zeros(int(original_lengths[0])) #np.zeros(int(audio_len/len_ratio))
    for time in times: #times_sec:
        # signal[int(time)] = max(median_h,0.5) #-0.005  # Convert the float time to an integer index
        idx = min(int(time), signal.shape[0] - 1)
        signal[idx] = max(median_h, 0.5)
    
    # Before the plotting loop
    signal_max_idx = signal.shape[0] - 1
    times = [t for t in times if t <= signal_max_idx]
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(signal, marker='*', linestyle='-')
    # # plt.plot(range(len(preds_np)), preds_np, marker='o')
    # derivative_preds_np = torch.cat([torch.tensor([0]), torch.diff(torch.tensor(preds_np), dim=0)])
    # plt.plot(range(len(derivative_preds_np)), -1* derivative_preds_np, marker='o')
    # ALSO ADDED -1* PREDS INSTEAD OF PREDS IN THE PEAK DETECTION !!!

    # num_peaks = len(phonemes) #TODO: FIX THAT
    num_peaks = len(times_sec)
    print(f"num peaks: {num_peaks}")
    # -------- July 1 -------------------
    # print("----- preds with detect peaks DP -----")
    # preds = detect_peaks(x=(-1*preds), w_phi=w_phi,
    #                      original_lengths_all= [original_lengths],
    #                     #  lengths=[preds.shape[1]],#num_peaks=[len(seg[0])], #len(seg[0]),
    #                      phonemes = phonemes,
    #                      len_ratio = len_ratio,
    #                      probs_real_all = [probs_real])  # run peak detection on scores
    
    preds = torch.tensor(preds_peaks[0], dtype=torch.float32)  # transform frame indexes to seconds
    
    # -----------------------------------
    print ("-------- HELLO THIS IS PREDICT.PY -------- ")
    print("truth boundaries (in seconds):")
    print(times_sec)
    
    print("predicted boundaries (in seconds):")
    print(preds)
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(signal, marker='*', linestyle='-')
    # plt.plot(range(len(preds_np)), preds_np, marker='o')
    derivative_preds_np = torch.cat([torch.tensor([0]), torch.diff(torch.tensor(preds_np), dim=0)])
    # plt.plot(range(len(derivative_preds_np)), -1* derivative_preds_np, marker='o')
    plt.plot(range(len(derivative_preds_np)), derivative_preds_np, marker='*')
    
    
    preds_plot = np.zeros(int(original_lengths[0]))
    for pred in preds:
        # TODO: FIX THAT
        if int(pred*sr/len_ratio) <= len(preds_plot)-1:
            preds_plot[int(pred*sr/len_ratio)] = max(median_h,0.5)
    plt.plot(range(len(preds_plot)), preds_plot, marker='^')
    
    # plt.plot(range(len(preds_np)), preds_np, marker='o')
    
    
    derivative_preds_np = np.diff(preds_np)
    derivative_preds_np = np.concatenate([[0], derivative_preds_np]) 
    plt.plot(range(len(derivative_preds_np)), derivative_preds_np, marker='o', label='Derivative of preds_np', color='magenta')
    plt.plot(range(len(preds_np)), preds_np, marker='*', label='preds_np', color='red')
    
    y_top = plt.ylim()[1]
    for i, s in enumerate(times):  # times = list of segment boundaries (frame indices)
        plt.axvline(x=s, color='red', linestyle='--', linewidth=1, label='Truth boundary' if i == 0 else "")
        plt.text(s, y_top, phonemes[0][i], color='red', rotation=90, va='top', ha='center', fontsize=8)
    
    plt.savefig(f'/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/preds_vs_truth_TIMIT_DP_LOSS_DETECT_PEAKS_w_ph_drivatie.png')
    print(f'/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/preds_vs_truth_TIMIT_DP_LOSS_DETECT_PEAKS_w_ph_drivatie.png')
    plt.close()
    
    # plt.savefig(f'/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/preds_vs_truth_{filename[:-4]}_TIMIT_DP_LOSS_DETECT_PEAKS_w_ph_drivatie.png')
    # print(f'/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/preds_vs_truth_{filename[:-4]}_TIMIT_DP_LOSS_DETECT_PEAKS_w_ph_drivatie.png')
    # plt.savefig(f'/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/preds_vs_truth_{filename[:-4]}_TIMIT_TMUX9_DETECT_PEAKS_w_ph_drivatie.png')
    # print(f'/home/rotem/projects/CFA/changed_NFC_negative_not_random/runs/plots/preds_vs_truth_{filename[:-4]}_TIMIT_TMUX9_DETECT_PEAKS_w_ph_drivatie.png')
    # ----------------------------------------------------------------------

    
    # p_seq = phonemes[0]
    # dp_alignment = phoneme_alignment(p_seq,original_lengths,len_ratio,derivative_preds_np,probs_real)

    # print(dp_alignment)
    # print("in sec:")
    # alignment_ms = dp_alignment*len_ratio/sr #*1000/16000
    # print(alignment_ms)

    pred_bound, truth_bound = preds , times_sec
    return pred_bound, truth_bound

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unsupervised segmentation inference script')
    parser.add_argument('--wav', help='path to wav file')
    parser.add_argument('--ckpt', help='path to checkpoint file')
    parser.add_argument('--prominence', type=float, default=None, help='prominence for peak detection (default: 0.05)')
    args = parser.parse_args()
    main_predict(args.wav, args.ckpt, args.prominence,w_phi=0.5, language="english")
    # main_predict(args.wav, args.ckpt, args.prominence,w_phi=0.5, language="dutch")

