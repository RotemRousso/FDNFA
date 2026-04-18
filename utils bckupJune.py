import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
import wandb
from tqdm import tqdm
import concurrent.futures
from typing import List
import time

timit_leehon_39_phonemes = [
    'ao', 'ae', 'ah','aw', 'er', 'ay', 
    'b', 'sil', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'm', 'en', 'ng', 'ey',
    'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'v', 'w', 'y', 'z', 'sh', 't', 'r', 's', 'th','uh', 'uw', 'ux', 'oy', 'ow','p'
]

timit_IPA_phonemes = [
    'ɑ',    # script a
    'æ',    # digraph
    'ʌ',    # turned v (for "but")
    'ɔː',   # open o
    'aʊ',   # how
    'ə',    # schwa
    'ə',  # schwa (reduced)
    'ɚ',   # schwar (schwa + r)
    'aɪ',   # hide
    'b',     # voiced bilabial stop
    'sil', # closure - map to silence
    'tʃ',   # voiceless postalveolar affricate
    'd',     # voiced alveolar stop
    'sil', # closure
    'ð',    # voiced dental fricative
    'ɾ',    # alveolar tap (flap)
    'ɛ',    # epsilon
    'l̩',    # syllabic l
    'm̩',    # syllabic m
    'n̩',    # syllabic n
    'ŋ',   # eng (velar nasal)
    'sil', # epenthetic silence
    'ɝ',    # stressed schwar (bird)
    'eɪ',   # today
    'f',     # voiceless labiodental fricative
    'g',     # voiced velar stop
    'sil', # closure
    'sil',  # initial silence
    'h',    # voiceless glottal fricative
    'h',    # breathy-voiced h
    'ɪ',    # small capital I
    'ɪ',    # reduced vowel (close to IH)
    'iː',   # heed
    'dʒ',   # voiced postalveolar affricate
    'k',     # voiceless velar stop
    'sil', # closure
    'l',     # voiced alveolar lateral approximant
    'm',     # bilabial nasal
    'n',     # alveolar nasal
    'ŋ',    # velar nasal
    'n̩',   # syllabic n (alternative notation)
    'oʊ',   # hoed
    'ɔɪ',   # joy
    'p',     # voiceless bilabial stop
    'sil', # pause
    'sil', # closure
    'ʔ',     # glottal stop
    'sil', # closure
    'ɹ',     # voiced alveolar approximant
    's',     # voiceless alveolar fricative
    'ʃ',    # voiceless postalveolar fricative
    't',     # voiceless alveolar stop
    'sil', # closure
    'θ',    # voiceless dental fricative
    'ʊ',    # hood, book
    'uː',   # boot
    'uː',   # high back rounded vowel (similar to uw)
    'v',     # voiced labiodental fricative
    'w',     # voiced labiovelar approximant
    'j',     # palatal approximant (yes)
    'z',     # voiced alveolar fricative
    'ʒ'     # voiced postalveolar fricative
]
# Create mappings
# phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(timit_61_phonemes)}
phoneme_to_idx_MACRO = {phoneme: idx for idx, phoneme in enumerate(timit_leehon_39_phonemes)}
idx_to_phoneme_MACRO = {idx: phoneme for phoneme, idx in phoneme_to_idx_MACRO.items()}
timit_to_leehon_map_MACRO = {
        'aa': 'ao', 'ae': 'ae', 'ah': 'ah', 'ao': 'ao', 'aw': 'aw', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'ay': 'ay',
        'b': 'b', 'bcl': 'sil', 'ch': 'ch', 'd': 'd', 'dcl': 'sil', 'dh': 'dh', 'dx': 'dx', 'eh': 'eh', 'el': 'el',
        'em': 'm', 'en': 'en', 'eng': 'ng', 'epi': 'sil', 'er': 'er', 'ey': 'ey', 'f': 'f', 'g': 'g', 'gcl': 'sil',
        'h#': 'sil', 'hh': 'hh', 'hv': 'hh', 'ih': 'ih', 'ix': 'ih', 'iy': 'iy', 'jh': 'jh', 'k': 'k', 'kcl': 'sil',
        'l': 'el', 'm': 'm', 'n': 'en', 'ng': 'ng', 'nx': 'en', 'ow': 'ow', 'oy': 'oy', 'p': 'p', 'pau': 'sil', 'pcl': 'sil',
        'q': 't', 'qcl': 'sil', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't', 'tcl': 'sil', 'th': 'th', 'uh': 'uh', 'uw': 'uw',
        'ux': 'uw', 'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'sh', '@':'sil', 'sil':'sil', 'u':'uw', 'h':'hh', 'x':'sil', 'e:':'eh', 'o:':'ow','i':'iy','j':'y','eh':'eh',
    # ---- leehon to leehon -----
        'ao':'ao', 'ae':'ae', 'ah':'ah','aw':'aw', 'er':'er', 'ay':'ay', 
        'b':'b', 'sil':'sil', 'ch':'ch', 'd':'d', 'dh':'dh', 'dx':'dx', 'eh':'eh', 'el':'el', 'm':'m', 'en':'en', 'ng':'ng', 'ey':'ey',
        'f':'f', 'g':'g', 'hh':'hh', 'ih':'ih', 'iy':'iy', 'jh':'jh', 'k':'k', 'l':'l', 'v':'v', 'w':'w', 'y':'y', 'z':'z', 'sh':'sh', 't':'t', 'r':'r', 's':'s', 'th':'th','uh':'uh', 'uw':'uw', 'ux':'ux', 'oy':'oy', 'ow':'ow','p':'p'
    }

timit_to_IPA_map_MACRO = {
    'aa': 'ɑ',    # script a
    'ae': 'æ',    # digraph
    'ah': 'ʌ',    # turned v (for "but")
    'ao': 'ɔː',   # open o
    'aw': 'aʊ',   # how
    'ax': 'ə',    # schwa
    'ax-h': 'ə',  # schwa (reduced)
    'axr': 'ɚ',   # schwar (schwa + r)
    'ay': 'aɪ',   # hide
    'b': 'b',     # voiced bilabial stop
    'bcl': 'sil', # closure - map to silence
    'ch': 'tʃ',   # voiceless postalveolar affricate
    'd': 'd',     # voiced alveolar stop
    'dcl': 'sil', # closure
    'dh': 'ð',    # voiced dental fricative
    'dx': 'ɾ',    # alveolar tap (flap)
    'eh': 'ɛ',    # epsilon
    'el': 'l̩',    # syllabic l
    'em': 'm̩',    # syllabic m
    'en': 'n̩',    # syllabic n
    'eng': 'ŋ',   # eng (velar nasal)
    'epi': 'sil', # epenthetic silence
    'er': 'ɝ',    # stressed schwar (bird)
    'ey': 'eɪ',   # today
    'f': 'f',     # voiceless labiodental fricative
    'g': 'g',     # voiced velar stop
    'gcl': 'sil', # closure
    'h#': 'sil',  # initial silence
    'hh': 'h',    # voiceless glottal fricative
    'hv': 'h',    # breathy-voiced h
    'ih': 'ɪ',    # small capital I
    'ix': 'ɪ',    # reduced vowel (close to IH)
    'iy': 'iː',   # heed
    'jh': 'dʒ',   # voiced postalveolar affricate
    'k': 'k',     # voiceless velar stop
    'kcl': 'sil', # closure
    'l': 'l',     # voiced alveolar lateral approximant
    'm': 'm',     # bilabial nasal
    'n': 'n',     # alveolar nasal
    'ng': 'ŋ',    # velar nasal
    'nx': 'n̩',   # syllabic n (alternative notation)
    'ow': 'oʊ',   # hoed
    'oy': 'ɔɪ',   # joy
    'p': 'p',     # voiceless bilabial stop
    'pau': 'sil', # pause
    'pcl': 'sil', # closure
    'q': 'ʔ',     # glottal stop
    'qcl': 'sil', # closure
    'r': 'ɹ',     # voiced alveolar approximant
    's': 's',     # voiceless alveolar fricative
    'sh': 'ʃ',    # voiceless postalveolar fricative
    't': 't',     # voiceless alveolar stop
    'tcl': 'sil', # closure
    'th': 'θ',    # voiceless dental fricative
    'uh': 'ʊ',    # hood, book
    'uw': 'uː',   # boot
    'ux': 'uː',   # high back rounded vowel (similar to uw)
    'v': 'v',     # voiced labiodental fricative
    'w': 'w',     # voiced labiovelar approximant
    'y': 'j',     # palatal approximant (yes)
    'z': 'z',     # voiced alveolar fricative
    'zh': 'ʒ'     # voiced postalveolar fricative
}


def phoneme_alignment(p_seq,w_phi,original_lengths,len_ratio,derivative_preds_np,probs_real):
    """
    Performs phoneme alignment using dynamic programming.
    Parameters:
        x (np.array): Acoustic feature sequence, shape (T, D), where T is the number of frames.
        p_seq (list): Sequence of phonemes to align.
        w (np.array): Weight vector for the base functions.
    Returns:
        list: Optimal alignment (start times for each phoneme).
    """
    # T = x.shape[0]  # Number of frames
    T = int(original_lengths[0]) #x.shape[0]  # Number of frames
    n = len(p_seq)+1 #-1  # Number of phonemes
    # IMPORTANT NOTE --> THE FIRST PHONEME IS h# WE IGNORE IT
    # print(f'before: {derivative_preds_np.device}')
    # derivative_preds_np.cpu()
    # print(f'after: {derivative_preds_np.device}')
    # probs_real.cpu()
    device = derivative_preds_np.device
    print(f'hola: {device}')

    # Convert probs_real to tensor if not already
    if isinstance(probs_real, np.ndarray):
        probs_real = torch.tensor(probs_real, device=device)
    cumsum_probs = torch.cumsum(probs_real, dim=0)
    
    # ------------------- April 28 ----------------------------
    phoneme_mappings = {p.lower(): timit_to_leehon_map_MACRO[p.lower()] for p in p_seq}
    # phoneme_mappings = {p.lower(): timit_to_IPA_map_MACRO[p.lower()] for p in p_seq}
    # ---------------------------------------------------------
    
    V = torch.full((n, T), -float('inf'), device=device) #After change
    
    # V[0, 0] = 0  # Base case
    # V[0,1] = 0
    V[0, 0] = (1-w_phi)*compute_phi_1(derivative_preds_np, 0, 0) + w_phi*compute_phi_2(cumsum_probs, phoneme_to_idx_MACRO[phoneme_mappings[p_seq[0].lower()]], 0, 0)
    V[0, 1] = (1-w_phi)*compute_phi_1(derivative_preds_np, 0, 1) + w_phi*compute_phi_2(cumsum_probs, phoneme_to_idx_MACRO[phoneme_mappings[p_seq[0].lower()]], 0, 1)
    V[1, 0] = (1-w_phi)*compute_phi_1(derivative_preds_np, 0, 0) + w_phi*compute_phi_2(cumsum_probs, phoneme_to_idx_MACRO[phoneme_mappings[p_seq[1].lower()]], 0, 0)
    V[1, 1] = (1-w_phi)*compute_phi_1(derivative_preds_np, 0, 1) + w_phi*compute_phi_2(cumsum_probs, phoneme_to_idx_MACRO[phoneme_mappings[p_seq[1].lower()]], 0, 1)  # Base case
    # V[1, 1] = compute_phi_1(derivative_preds_np, 1, 1) +compute_phi_2(cumsum_probs, phoneme_to_idx_MACRO[phoneme_mappings[p_seq[1]]], 0, 1)
    # TODO: CHANGE THE 10 TO LAMBDA
    backtrack = torch.zeros((n, T), dtype=torch.long, device=device) #After change
    
    print("starting")    # Backtracking to recover the alignment 
    # for i in range(1,n): #we ignore the first phoneme 'h#'
    for i in tqdm(range(1,n)): #we ignore the first phoneme 'h#'
        p_idx = phoneme_to_idx_MACRO[phoneme_mappings[p_seq[i-1].lower()]]
        # p_idx = phoneme_to_idx_MACRO[phoneme_mappings[p_seq[i]]]
        # for t in range(1, T):#-1):
        for t in range(10,T):#-1): 
            # t_prev_range = range(0, t)
            t_prev_range = range(max(0, t - 50), t)
            for t_prev in t_prev_range:
                start_time = time.time()
                calc1 = V[i - 1, t_prev]
                end_time = time.time()
                print(f"Time taken for V[i - 1, t_prev]: {end_time - start_time:.4f} seconds")
                start_time = time.time()
                calc2 = (1-w_phi)*compute_phi_1(derivative_preds_np, t_prev, t)
                end_time = time.time()
                print(f"Time taken for compute_phi_1: {end_time - start_time:.4f} seconds")
                start_time = time.time()
                calc3 = w_phi*compute_phi_2(cumsum_probs, p_idx, t_prev, t)
                end_time = time.time()
                print(f"Time taken for compute_phi_2: {end_time - start_time:.4f} seconds")
                scores = calc1 + calc2 + calc3
                if scores > V[i,t]:
                    V[i,t] = scores
                    backtrack[i,t] = t_prev
                # V[i, t], backtrack[i, t] = max((V[i, t], backtrack[i, t]), (scores, t_prev))

#     # --------------- JUNE 10 ------------------
#     # ----------- this is a try for softalign --------------
    
#     # --------- TODO: CHECK THE CHANGE FROM soft min to max!!!
#     # Backtracking to recover the alignment 
#     for i in tqdm(range(1,n)): #we ignore the first phoneme 'h#'
#         p_idx = phoneme_to_idx_MACRO[phoneme_mappings[p_seq[i-1].lower()]]
#         # p_idx = phoneme_to_idx_MACRO[phoneme_mappings[p_seq[i]]]
#         # for t in range(1, T):#-1):
        
#         gamma = 0.5 # need to set the best value maybe need to optimize it
#         soft_gamma_max_sum = 0
        
#         for t in range(10,T):#-1): 
#             # t_prev_range = range(0, t)
#             t_prev_range = range(max(0, t - 50), t)
#             for t_prev in t_prev_range:
#                 scores = (V[i - 1, t_prev] +
#                         (1-w_phi)*compute_phi_1(derivative_preds_np, t_prev, t) +
#                         w_phi*compute_phi_2(cumsum_probs, p_idx, t_prev, t))
                
#                 # original version of soft MIN: (minus instead of +, minus also in the gamma outside of the sum)
#                 # soft_gamma_max_sum = soft_gamma_max_sum + np.exp(-scores/gamma)
#                 # changed from - to + for changing min to max maybe this is wrong need to check!!
                
#                 soft_gamma_max_sum = soft_gamma_max_sum + np.exp(scores/gamma)
                
#                 # if scores > V[i,t]:
#                 #     V[i,t] = scores
#                 #     backtrack[i,t] = t_prev
#                 # # V[i, t], backtrack[i, t] = max((V[i, t], backtrack[i, t]), (scores, t_prev))
#             # soft_gamma_max_sum = -gamma*np.log(soft_gamma_max_sum) # the minus becomes + for max instead of min
#             soft_gamma_max_sum = gamma*np.log(soft_gamma_max_sum)
#    # -------------- END OF TRY ----------------------


    alignment = torch.zeros(n, device=device, dtype=torch.long)
    t = T-1 #2 #1
    # for i in range(n-1, -1, -1): #we need to ignore the first phoneme '#h'
    for i in range(n-1, 0, -1):
        alignment[i] = backtrack[i, t]
        t = backtrack[i, t]
    
    # --------------------------- April 8 -------------------------------
    # NOTE!!! THERE IS LOG INSIDE THE COMPURE INSTEAD F THE LOG(V)
    # V_log = torch.clamp(V, min=1e-10)
    # V_log = torch.log(torch.clamp(V, min=1e-10))

    # ------------- plot April 10 ------------------
    # V_np = V.cpu().numpy()
    # plt.figure(figsize = (10,6) )
    # plt.imshow(V_np, aspect='auto', cmap='Blues')
    # plt.colorbar()
    # plt.title('Log of Trellis V with Backtrack Path')
    # plt.xlabel('Time (frames)')
    # plt.ylabel('Phoneme Index')
    
    # alignment_path = []
    # t = V.shape[1] - 1  # Starting from the last frame
    # for i in range(V.shape[0] - 1, 0, -1):  # Backtrack through phonemes
    #     alignment_path.append((i, t))  # Append (phoneme_index, time_frame)
    #     t = backtrack[i, t].item() 
    # alignment_path = np.array(alignment_path)
    # for (i, t) in alignment_path:
    #     plt.plot(t, i, 'ro', markersize=3)
    # plt.show()
    # plt.savefig("/home/rotem/projects/CFA/DCAF/runs/debug/V_viterbi.png")
    # print("/home/rotem/projects/CFA/DCAF/runs/debug/V_viterbi.png")
    
    # -------------------------------------------------------------------
    
    return alignment.cpu().numpy()  
    
    # return alignment

# THIS IS A TRY FOR THE DYNAMIC PROGRAMMING PROCESS
def compute_phi_1(derivative_preds_np, t_start, t_end): # t is in frames #p is in index    
    # score = derivative_preds_np[max(0,t_start+1)]+derivative_preds_np[max(0,t_end+1)]
    if t_end>=len(derivative_preds_np) or t_start>=len(derivative_preds_np):
        return 0
    if t_end<=0 or t_start<=0:
        return 0
    score = derivative_preds_np[max(0,t_start)]+derivative_preds_np[max(0,t_end)]
    # score = derivative_preds_np[max(0,t_start-1)]+derivative_preds_np[max(0,t_end-1)]
    # TODO: CHECK WHICH SHIFT FROM THE DERIVATIVE IS NECECARRY AND IF NOT DELETE THE +-1 OPTIONS
    if score > 0:
        return torch.log(score+1e-6)
    else:
        return 0
    # return score

def compute_phi_2(cumsum_probs,p, t_start, t_end): # t is in frames #p is in index
    p_idx = p
    # return cumsum_probs[max(0,t_end-1), p_idx] - (cumsum_probs[max(0,t_start-1), p_idx] if t_start > 0 else 0)
    # probs is always positive
    probs_score = cumsum_probs[max(0,t_end), p_idx] - (cumsum_probs[max(0,t_start), p_idx] if t_start > 0 else 0)
    return torch.log(probs_score+1e-6)
    # return torch.log(sourceTensor.clone(probs_score+1e-6).detach())
    # return cumsum_probs[max(0,t_end), p_idx] - (cumsum_probs[max(0,t_start), p_idx] if t_start > 0 else 0)

def compute_phi_3(len_ratio,p, t_start, t_end): # t is in frames #p is in index
    # TODO: WRITE A FUNCTION THAT GETS THE STATISTICS:
    p = timit_to_leehon_map_MACRO[p.lower()]
    #we assume this is the mu_len given in ms:
    mu_len_ms = get_mu_stats(p) # convert the stats to frames!
    #convert it to timit samples unit:
    mu_len_timit_samps = mu_len_ms*16000/1000
    #convert it to spectral_len in our z domain samples:
    mu_len = mu_len_timit_samps/len_ratio

    
    #we assume this is the sigma_len given in ms:
    sigma_len_ms = get_sigma_stats(p) # convert the stats to frames!
    #convert it to timit samples unit:
    sigma_len_timit_samps = sigma_len_ms*16000/1000
    #convert it to spectral_len in our z domain samples:
    sigma_len = sigma_len_timit_samps/len_ratio

    z_score = ((t_end-t_start)-mu_len)/sigma_len
    return z_score

def compute_phi_4(len_ratio,p1,p2, t_start1, t_end1, t_start2, t_end2): # t is in frames #p is in index, p1 and p2 are adjacent
    # TODO: WRITE A FUNCTION THAT GETS THE STATISTICS:
    p = timit_to_leehon(p)
    #we assume this is the mu_len given in ms:
    mu_len_ms_1 = get_mu_stats(p1) # convert the stats to frames!
    #convert it to timit samples unit:
    mu_len_timit_samps_1 = mu_len_ms_1*16000/1000
    #convert it to spectral_len in our z domain samples:
    mu_len_1 = mu_len_timit_samps_1/len_ratio
    
    #we assume this is the mu_len given in ms:
    mu_len_ms_2 = get_mu_stats(p2) # convert the stats to frames!
    #convert it to timit samples unit:
    mu_len_timit_samps_2 = mu_len_ms_2*16000/1000
    #convert it to spectral_len in our z domain samples:
    mu_len_2 = mu_len_timit_samps_2/len_ratio
    
    
    r1 = (t_end1-t_start1)/mu_len_1
    r2 = (t_end2-t_start2)/mu_len_2
    rate_score = (r2-r1)**2
    return rate_score

def get_timit_61_phoneme_mappings():
    """
    Returns the TIMIT 61 phoneme-to-index mapping and the reverse index-to-phoneme mapping.

    Returns:
        phoneme_to_idx (dict): Dictionary mapping phonemes to unique indices.
        idx_to_phoneme (dict): Dictionary mapping indices to their corresponding phonemes.
    """
    # Full TIMIT 61 phoneme set
    # this is actually leehon 39 phonemes!!!!!
    # timit_61_phonemes = [
    #     'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
    #     'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey',
    #     'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng',
    #     'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux',
    #     'v', 'w', 'y', 'z', 'zh'
    # ]
    timit_leehon_39_phonemes = [
        'ao', 'ae', 'ah','aw', 'er', 'ay', 
        'b', 'sil', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'm', 'en', 'ng', 'ey',
        'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'v', 'w', 'y', 'z', 'sh', 't', 'r', 's', 'th','uh', 'uw', 'ux', 'oy', 'ow','p'
    ]
    # Create mappings
    # phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(timit_61_phonemes)}
    phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(timit_leehon_39_phonemes)}
    idx_to_phoneme = {idx: phoneme for phoneme, idx in phoneme_to_idx.items()}

    return phoneme_to_idx, idx_to_phoneme

# --------------------------------


def timit_to_leehon(timit_label):
    # Mapping of TIMIT 61 phonemes to Leehon 39 phonemes
    timit_to_leehon_map = {
        'aa': 'ao', 'ae': 'ae', 'ah': 'ah', 'ao': 'ao', 'aw': 'aw', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'ay': 'ay',
        'b': 'b', 'bcl': 'sil', 'ch': 'ch', 'd': 'd', 'dcl': 'sil', 'dh': 'dh', 'dx': 'dx', 'eh': 'eh', 'el': 'el',
        'em': 'm', 'en': 'en', 'eng': 'ng', 'epi': 'sil', 'er': 'er', 'ey': 'ey', 'f': 'f', 'g': 'g', 'gcl': 'sil',
        'h#': 'sil', 'hh': 'hh', 'hv': 'hh', 'ih': 'ih', 'ix': 'ih', 'iy': 'iy', 'jh': 'jh', 'k': 'k', 'kcl': 'sil',
        'l': 'el', 'm': 'm', 'n': 'en', 'ng': 'ng', 'nx': 'en', 'ow': 'ow', 'oy': 'oy', 'p': 'p', 'pau': 'sil', 'pcl': 'sil',
        'q': 't', 'qcl': 'sil', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't', 'tcl': 'sil', 'th': 'th', 'uh': 'uh', 'uw': 'uw',
        'ux': 'uw', 'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'sh', '':'sil'
    }
    
    # Return the corresponding Leehon 39 label, or None if the label is not found
    return timit_to_leehon_map.get(timit_label.lower(), None)

def load_phoneme_stats():
    phonemes_path = "/home/rotem/projects/CFA/changed_NFC_negative_not_random/phonemes_stats_timit/phonemes_39"
    stats_path = "/home/rotem/projects/CFA/changed_NFC_negative_not_random/phonemes_stats_timit/phoneme_stats_39.out"
    
    # Load phoneme names
    with open(phonemes_path, "r") as f:
        phonemes = [line.strip() for line in f]

    # Load mu values (second row of stats file)
    with open(stats_path, "r") as f:
        lines = f.readlines()
        mu_values = list(map(float, lines[1].strip().split()))  # Convert to float
        sigma_values = list(map(float, lines[2].strip().split()))  # Convert to float
    
    # Create phoneme-to-mu dictionary
    phoneme_mu_dict = dict(zip(phonemes, mu_values))
    phoneme_sigma_dict = dict(zip(phonemes, sigma_values))
    
    return phoneme_mu_dict, phoneme_sigma_dict

# Load phoneme stats once


def get_mu_stats(p):
    phoneme_mu_dict, _ = load_phoneme_stats()
    """Return the mu value for the given phoneme p."""
    return phoneme_mu_dict.get(p, None)  # Return None if phoneme is not found

def get_sigma_stats(p):
    _, phoneme_sigma_dict = load_phoneme_stats()
    """Return the mu value for the given phoneme p."""
    return phoneme_sigma_dict.get(p, None)  # Return None if phoneme is not found
# ----------------------------------------------------------------------------






def replicate_first_k_frames(x, k, dim):
    return torch.cat([x.index_select(dim=dim, index=torch.LongTensor([0] * k).to(x.device)), x], dim=dim)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class PrintShapeLayer(nn.Module):
    def __init__(self):
        super(PrintShapeLayer, self).__init__()
    def forward(self, x):
        print(x.shape)
        return x


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

def detect_peaks_worker(xi,w_phi, p_seq, original_lengths, probs_real, len_ratio, prominence, width, distance):
    print(f"num peaks = {len(p_seq)}")
    # if isinstance(xi, torch.Tensor):
        # xi = xi.detach() # June 30 ------

    
    preds_np = xi
    median_h = preds_np.median()
    preds_np = preds_np - median_h
    # # ----------- 25 MARCH --------------
    # preds_np[preds_np < 0] = 0
    # # -----------------------------------
    derivative_preds_np = torch.cat([torch.tensor([0], device=preds_np.device), torch.diff(preds_np, dim=0)])
    
    xmin, xmax = xi.min(), xi.max()
    xi = (xi - xmin) / (xmax - xmin)
    xi = xi.flatten()
    
    peaks = phoneme_alignment(p_seq,w_phi, original_lengths, len_ratio, derivative_preds_np, probs_real)
    # peaks = phoneme_alignment(p_seq, original_lengths, len_ratio, preds_np, probs_real)
    
    if len(peaks) == 0:
        peaks = torch.tensor([xi.shape[0] - 1], device=xi.device)
    
    return peaks

def detect_peaks(x,w_phi, original_lengths_all, phonemes, len_ratio, probs_real_all):
    """Detect peaks of next_frame_classifier using multithreading."""
    
    out = []
    # TODO: MULTI THREAD --> MULTI PROCESS
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        # futures = [executor.submit(detect_peaks_worker, xi,w_phi, li, p_seq, original_lengths, probs_real, len_ratio, prominence=0.1, width=None, distance=None)
        #            for xi, li, p_seq, original_lengths, probs_real in zip(x, lengths, phonemes, original_lengths_all, probs_real_all)]
        
        # for future in concurrent.futures.as_completed(futures):
        #     print("finished!!! -------------------------")
        #     out.append(future.result())
    
        futures = []
        # -------------------- JUNE 9 -----------
        # for xi, p_seq, original_lengths, probs_real in zip(x, phonemes, original_lengths_all, probs_real_all):
        xi=x
        p_seq = phonemes
        original_lengths = original_lengths_all
        probs_real = probs_real_all
        # ------------------------------------
        futures.append(executor.submit(detect_peaks_worker, xi,w_phi, p_seq, [original_lengths], probs_real, len_ratio, prominence=0.1, width=None, distance=None))
        
        for future in concurrent.futures.as_completed(futures):
            out.append(future.result())
    
    return out

class PrecisionRecallMetric:
    def __init__(self):
        self.precision_counter = 0
        self.recall_counter = 0
        self.pred_counter = 0
        self.gt_counter = 0
        self.eps = 1e-5
        self.data = []
        self.tolerance = 2
        self.prominence_range = [1] # np.arange(0, 0.15, 0.01)
        self.width_range = [1] #[None, 1]
        self.distance_range = [1] #[None, 1]

    def get_metrics(self, precision_counter, recall_counter, pred_counter, gt_counter):
        EPS = 1e-7
        
        precision = precision_counter / (pred_counter + self.eps)
        recall = recall_counter / (gt_counter + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        
        os = recall / (precision + EPS) - 1
        r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
        r2 = (-os + recall - 1) / (np.sqrt(2))
        rval = 1 - (np.abs(r1) + np.abs(r2)) / 2

        return precision, recall, f1, rval

    def zero(self):
        self.data = []
        
    def update(self, seg, pos_pred, length,original_lengths_all, probs_all,phonemes_all):
        for seg_i, pos_pred_i, length_i , original_length, probs,phonemes in zip(seg, pos_pred, length,original_lengths_all,probs_all,phonemes_all):
            # self.data.append((seg_i, pos_pred_i, length_i.item(),[original_length.item()], probs,phonemes))
            self.data.append((seg_i, pos_pred_i.detach(), length_i.item(),[original_length.item()], probs.detach(),phonemes)) # June 30 --------


    def get_stats(self, width=None, prominence=None, distance=None):
        print(f"calculating metrics using {len(self.data)} entries")
        max_rval = -float("inf")
        min_l1_dist = float("inf")
        best_params = None
        segs = list(map(lambda x: x[0], self.data))
        length = list(map(lambda x: x[2], self.data))
        yhats = list(map(lambda x: x[1], self.data))
        original_lengths_all = list(map(lambda x: x[3], self.data))
        probs = list(map(lambda x: x[4], self.data))
        phonemes = list(map(lambda x: x[5], self.data))

        width_range = self.width_range
        distance_range = self.distance_range
        prominence_range = self.prominence_range

        # when testing, we would override the search with specific values from validation
        if prominence is not None:
            width_range = [width]
            distance_range = [distance]
            prominence_range = [prominence]
        sr = 16000
        len_ratio = 161.34011627906978

        for width in width_range:
            for prominence in prominence_range:
                for distance in distance_range:
                    precision_counter = 0
                    recall_counter = 0
                    pred_counter = 0
                    gt_counter = 0
                    
                    peaks = detect_peaks(x=yhats,w_phi=0.5,
                                         original_lengths_all = original_lengths_all,
                                         phonemes = phonemes,
                                         len_ratio = 161.34011627906978 , 
                                         probs_real_all = probs)
                    #TODO: CHANGE THE 160 to the accurate len_ratio!

                    for (y, yhat) in zip(segs, peaks):
                        yhat = yhat[2:]
                        try:
                            l1_dist = np.mean(np.abs(y - yhat))
                            l2_dist = np.mean((y - yhat)**2)
                            if l1_dist<min_l1_dist:
                                min_l1_dist = l1_dist
                                out = (l1_dist,l2_dist)
                                best_params = width, prominence, distance #TODO: NEED TO UPDATE
                        except:
                            print("failed try")
        self.zero()
        print(f"best peak detection params: {best_params} (width, prominence, distance)")
        print(f"best peak detection L1_DIST: {l1_dist}")
        print(f"best peak detection L2_DIST: {l2_dist}")
        return out, best_params


class StatsMeter:
    def __init__(self):
        self.data = []

    def update(self, item):
        if type(item) == list:
            self.data.extend(item)
        else:
            self.data.append(item)

    def get_stats(self):
        data = np.array(self.data)
        if len(data)==0:
            return float('nan')
        mean = data.mean()
        # self.zero()
        return mean

    def zero(self):
        self.data.clear()
        assert len(self.data) == 0, "StatsMeter didn't clear"


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"{self.msg} -- started")

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f"{self.msg} -- done in {(time.time() - self.start_time)} secs")


def max_min_norm(x):
    x -= x.min(-1, keepdim=True)[0]
    x /= x.max(-1, keepdim=True)[0]
    return x


def line():
    print(90 * "-")