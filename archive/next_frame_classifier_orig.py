
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from utils import LambdaLayer, PrintShapeLayer, length_to_mask, get_timit_61_phoneme_mappings, timit_to_leehon
from dataloader import TrainTestDataset
from collections import defaultdict

import random

# ------- JUNE 6 ADD ------
from utils import max_min_norm, detect_peaks
# -------------------------


class NextFrameClassifier(nn.Module):
    def __init__(self, hp):
        super(NextFrameClassifier, self).__init__()
        
        # self.w_phi = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        # self.w_pos_neg = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        
        self.hp = hp

        Z_DIM = hp.z_dim
        LS = hp.latent_dim if hp.latent_dim != 0 else Z_DIM

        self.phoneme_to_index, self.idx_to_phoneme = get_timit_61_phoneme_mappings()

        self.enc = nn.Sequential(
            nn.Conv1d(1, LS, kernel_size=10, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, LS, kernel_size=8, stride=4, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, LS, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, LS, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, Z_DIM, kernel_size=4, stride=2, padding=0, bias=False),
            LambdaLayer(lambda x: x.transpose(1,2)),
        )
        print("learning features from raw wav")
        
        if self.hp.z_proj != 0:
            if self.hp.z_proj_linear:
                self.enc.add_module(
                    "z_proj",
                    nn.Sequential(
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, self.hp.z_proj),
                    )
                )
            else:
                self.enc.add_module(
                    "z_proj",
                    nn.Sequential(
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, Z_DIM), nn.LeakyReLU(),
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, self.hp.z_proj),
                    )
                )
                
        # # similarity estimation projections
        self.pred_steps = list(range(1 + self.hp.pred_offset, 1 + self.hp.pred_offset + self.hp.pred_steps))
        print(f"prediction steps: {self.pred_steps}")
        
        # Define the Bi-LSTM layer
        self.bi_lstm = nn.LSTM(input_size=self.hp.z_proj,  # Automatically set input_size from encoder output
                       hidden_size=512, #256,        # Hidden size of LSTM (adjust as needed)
                       num_layers=5, 
                       bidirectional=True, 
                       batch_first=True)
        # Bi-LSTM
        def init_weights(m):
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
        self.bi_lstm.apply(init_weights)

        # Fully connected layer for class probabilities
        self.fc = nn.Linear(512 * 2, hp.num_classes)  # Bi-LSTM output has 2*Z_DIM features


    def score(self, f, b):
        return F.cosine_similarity(f, b, dim=-1) * self.hp.cosine_coef
    
    # def forward(self, spect):
    def forward(self, spect,seg,phonemes,length):
        device = spect.device
        
        # wav => latent z
        # z = self.enc(spect.unsqueeze(1))

        device = next(self.enc.parameters()).device
        z = self.enc(spect.unsqueeze(1).to(device))

        
        
        # Bi-LSTM for temporal modeling
        z = F.normalize(z, dim=-1)
        z_bilstm, _ = self.bi_lstm(z)  # (batch, time, Z_DIM * 2)
        # Frame-wise class probabilities
        logits = self.fc(z_bilstm)  # (batch, time, num_classes)
        probs = F.softmax(logits, dim=-1)  # (batch, time, num_classes)
        # probs=logits
        
        # Prepare ground truth phoneme labels
        frame_labels = torch.zeros_like(logits, device=device)  # Placeholder for one-hot labels
        
        # Convert segments to frame indices and assign phoneme labels
        if seg is not None:
            for i in range(len(seg)):  # Iterate over batches
                segments = seg[i]  # Segments for batch i
                for j in range(len(segments) - 1):  # Iterate over segment pairs (start, end)
                    start = int(segments[j])  # Start frame index
                    end = int(segments[j + 1])  # End frame index

                    # Get the corresponding phoneme label
                    label = timit_to_leehon(phonemes[i][j])  # Assumes phonemes[i] corresponds to seg[i]
                    if label is None:
                        label = 'sil'
                    label_index = self.phoneme_to_index[label]  # Convert phoneme to integer index
                    
                    # Assign phoneme index to the range of frames (frame_labels should have size [batch, time, num_classes])
                    frame_labels[i, start:end, label_index] = 1.0  # One-hot assignment
        
        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):  # predict for steps 1...t
            b_z_seg_positive_list = [] # -> torch.tensor([False .... False]) / torch.zeros(8,472)/zeros.bool()
            b_z_seg_negative_list = []

            positive_b_scores_list = []
            negative_b_scores_list = []
            if seg is None:
                for b in range(z.shape[0]):
                    z_seg_positive_list = [z[b,:]]
                    z_seg_negative_list = [z[b,:]] #
                    
                    b_z_seg_positive_list.append(z_seg_positive_list)
                    b_z_seg_negative_list.append(z_seg_negative_list)
            else:
                for b in range(z.shape[0]): #size of batch
                    seg_b = seg[b]
                    # seg_b.insert(0,0) #adding 0 to the start for the first segment
                    # seg_b.append(z[b].shape[0]) #size of the longest file in the batch - this is for the last segment - maybe need here to be lengths?
                    seg_b = [0] + seg[b][:] + [z[b].shape[0]]
                    z_seg_positive_list = []
                    z_seg_negative_list = []
                    positive_score_list = []
                    negative_score_list = []
                    for ki,kii in zip(seg_b[1:-2],seg_b[2:-1]):
                        num_repeats = 5 #was best with 20 #100_took_too_long #20 #5 #how many samples we want from each segment/phoneme exmpl
                        for iter in range(num_repeats):
                            mid = ki +(kii-ki)//2
                            sample_len = 1 #2 #4 #8 #4 #kii-ki
                            start_idx = int(ki + 0.2* (kii-ki))
                            end_idx = int(ki+ 0.8*(kii-ki))
                            
                            num_idx = end_idx-start_idx
                            if num_idx >= sample_len:
                                rnd_idx = random.sample(range(start_idx,end_idx),int(sample_len))
                                z_seg_positive = z[b,rnd_idx]
                            else:
                                z_seg_positive = z[b,int(mid- (int(sample_len//2))):int(mid+ (int(sample_len//2)))]
                            positive_score_list.append(self.score(z_seg_positive[:, :-t], z_seg_positive[:, t:]))
                            
                            z_seg_positive_list.append(z_seg_positive)
                            if ki- (int(sample_len//2))<=0 or kii+ (int(sample_len//2))>=z[b].shape[0]:
                                z_seg_negative = z[b,int(mid- (int(sample_len//2))):int(mid+ (int(sample_len//2)))] 
                            else:
                                z_seg_negative = z[b,int(kii- (int(sample_len//2))):int(kii+ (int(sample_len//2)))]
                            negative_score_list.append(self.score(z_seg_negative[:, :-t], z_seg_negative[:, t:]))
                            z_seg_negative_list.append(z_seg_negative)
                    b_z_seg_positive_list.append(z_seg_positive_list)
                    b_z_seg_negative_list.append(z_seg_negative_list)
                
                    # ----------------- JUNE 4 - WAS INDENTED INSIDE ------------------
                    positive_score_list = torch.cat(positive_score_list) # 1x472, was: 1 x P_seg -> TODO 1xlength?
                    negative_score_list = torch.cat(negative_score_list) # 1x[original len(seg)*2], was: 1 x N_seg -> TODO 1xlength?
                    positive_b_scores_list.append(positive_score_list)
                    negative_b_scores_list.append(negative_score_list)
                    # -----------------------------------------------------------------
                    
            if seg is None:
                positive_b_scores_list = []
                negative_b_scores_list = []
                for z_seg_positive_list,z_seg_negative_list in zip(b_z_seg_positive_list,b_z_seg_negative_list):
                    positive_score_list = []
                    negative_score_list = [] 
                    for z_seg_positive,z_seg_negative in zip(z_seg_positive_list,z_seg_negative_list):
                        positive_score_list.append(self.score(z_seg_positive[:, :-t], z_seg_positive[:, t:]))
                        negative_score_list.append(self.score(z_seg_negative[:, :-t], z_seg_negative[:, t:]))
                        
                    positive_score_list = torch.cat(positive_score_list) # 1x472, was: 1 x P_seg -> TODO 1xlength?
                    negative_score_list = torch.cat(negative_score_list) # 1x[original len(seg)*2], was: 1 x N_seg -> TODO 1xlength?
                    positive_b_scores_list.append(positive_score_list)
                    negative_b_scores_list.append(negative_score_list)
            original_lengths =torch.cat([torch.tensor(positive_b_scores_list[i].shape[0], 
                                    dtype=torch.int64, device=device).unsqueeze(0) for i in range(len(positive_b_scores_list))],dim=0)
            max_length = original_lengths.max().item()          
            padded_tensors = []
            for score_tensor in positive_b_scores_list:
                padded_tensor = torch.full((max_length,), float(-1e9), device=device)
                padded_tensor[:score_tensor.shape[0]] = score_tensor
                padded_tensors.append(padded_tensor)
            positive_b_scores = torch.stack(padded_tensors)
            
            padded_tensors = []
            for score_tensor in negative_b_scores_list:
                padded_tensor = torch.full((max_length,), float(-1e9), device=device)
                padded_tensor[:score_tensor.shape[0]] = score_tensor
                padded_tensors.append(padded_tensor)
            negative_b_scores = torch.stack(padded_tensors)
            
            preds[t].append(positive_b_scores)
            preds[t].append(negative_b_scores)           
            
            
        # return preds,original_lengths, probs, frame_labels
        # !! this was where the original return was !! 
    
        # --------------- JUNE 4 INSERTING DP TO TRAIN -------------------
        # maybe this is not one file but a batch - need to check!
        # for now this is implemented only for one file format just like in predict.py
        
        # if seg is not None: # only in Train # --------- JUNE 29 ------------
        total_peaks = []
        # for i in range(len(seg)):  # Iterate over batches # --------- JUNE 29 ------------
        for i in range(len(phonemes)):  # Iterate over batches
            
            # --------- JUNE 29 ------------
            if seg is not None:
                segments = seg[i]  # Segments for batch i
            else:
                segments = []         
            # ------------------------------
            # phoneme_to_idx, idx_to_phoneme = get_timit_61_phoneme_mappings()
            # phoneme_labels = [idx_to_phoneme[i] for i in range(41)]
            probs_real = F.softmax(probs[i], dim=-1) 
            # probs_logits = probs.squeeze(0)
            probs_real = probs_real.squeeze(0)
            # probs_real = probs_real.detach().numpy()
            
            # i is for the right file in batch just like b above in the train
            # --------------- June 9 ---------------------
            # preds = preds[1][0]#[i]  # get scores of positive pairs -
            
            # preds[1=t][0=positives][i=curr_wav] 
            cur_preds = preds[1][0][i] #[i]#[i]  # get scores of positive pairs - 
            cur_preds = 1 - max_min_norm(cur_preds)  # normalize scores (good for visualizations)
            # ------- JUNE 9 --------------------
            # preds_np = preds.detach().numpy()
            preds_np = cur_preds
            
            # preds_np = preds_np[0] #*len_ratio/sr
            median_h = cur_preds.median() #np.median(preds_np)
            preds_np = preds_np - median_h
            # ---------------------------
            
            signal = np.zeros(int(original_lengths[i])) #np.zeros(int(audio_len/len_ratio))
            
            # TODO: TIMES IS THE TRUTH BOUNDRIES - HOW TO GET THAT WITHIN THE BATCH? IS IT SEGS?
            # TODO: also add times_sec this is the same as truth times
            

            
            # TODO: SET len_ratio if needed - for now it's None cause there's not really a use in it within the detect_peaks_worker
            
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # for time in times: #times_sec:
            #     signal[int(time)] = median_h #-0.005  # Convert the float time to an integer index
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            # TODO: set the correct w_phi right now it's 0.5 !!!!!
            w_phi = [0.5, 0.5]  # need to be torch requires_grad=True!!!
            
            # TODO: UPDATE LEN_RATIO TO BE NOT CONST!
            sr = 16000
            len_ratio = 161.34011627906978
            # num_peaks = len(times_sec) #this is the original
            # num_peaks = len(segments) # --------- JUNE 29 ------------
            num_peaks = len(phonemes) 
            print(f"num peaks: {int(num_peaks)}")
            print("----- preds with detect peaks DP -----")
            preds_peaks = detect_peaks(x=(-1*cur_preds), w_phi=w_phi,
                                original_lengths_all= original_lengths[i], #[i],
                                phonemes = phonemes[i],
                                len_ratio = len_ratio, #None,
                                probs_real_all = probs_real)  # run peak detection on scores
            
            # preds peaks is the same as "preds" output that we have in predict.py after the peak detection!
            # here we renames it so it'll be a different variable
            
            # TODO: THIS IS THE REAL PREDS_PEAKS - WE DON'T HEVE LEN_RATIO AND SR SO FOR NOW IT'S NOT IN SEC
            # preds_peaks = torch.tensor(preds[0], dtype=torch.float32) * len_ratio / sr  # transform frame indexes to seconds
            # print("truth boundaries (in seconds):")
            # print(times_sec)
            
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            # preds_peaks = torch.tensor(preds[0], dtype=torch.float32) #* len_ratio / sr  # transform frame indexes to seconds
            preds_peaks = preds_peaks[0]* len_ratio/sr
            if seg is not None:
                segments = np.array(segments)*len_ratio/sr
            print("truth boundaries ('segments'):")
            print(segments)
            
            print("predicted boundaries (TODO - CONVERT TO SECS!! in seconds):")
            print(preds_peaks) 
            
            total_peaks.append(preds_peaks) 
        
        
        print("TOTAL truth boundaries ('seg'):")
        print(seg)
                
        print("TOTAL predicted boundaries (in seconds):")
        print(total_peaks) 
        # ----------------------------------------------------------------
        return preds,original_lengths, probs, frame_labels, seg,total_peaks, w_phi
        # return preds,original_lengths, probs, frame_labels
        # !! this is the original return !! 
    

    def loss(self, preds, original_lengths, probs, frame_labels, seg, total_peaks, w_phi):
        # --------------- July 11 -------------------
        total_loss = torch.tensor(0.0, device=probs.device, requires_grad=True)
        # total_loss = torch.tensor(0.0, device=probs.device)
        # -------------------------------------------
        probs = probs.view(-1, probs.size(-1))  # [batch * time, num_classes]
        frame_labels = frame_labels.view(-1, frame_labels.size(-1))  # [batch * time, num_classes] 
        ph_loss = F.cross_entropy(probs, frame_labels.argmax(dim=-1))   
        loss = 0
        for t, t_preds in preds.items():
            mask = length_to_mask(original_lengths - t+1)
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            pos_loss = out[...,0] * mask
            neg_loss = out[...,1] * mask
            # loss += -(0.7*pos_loss.mean()+0.3*neg_loss.mean()) # maybe wighted loss
            # loss += -(0.5*pos_loss.mean()+0.5*neg_loss.mean()) # maybe wighted loss
            
            # TODO: CHANGE W TO LEARNED
            w_pos_neg = [0.5, 0.5]
            # l_pos_neg = torch.stack([pos_loss.mean(), neg_loss.mean()]) # [pos_loss.mean(), neg_loss.mean()]
            loss+= -(w_pos_neg[0] * l_pos_neg[0] + w_pos_neg[1] * l_pos_neg[1])
            # loss+= -torch.dot(w_pos_neg, l_pos_neg)  # weighted loss
            
            # loss += -(0.5*pos_loss.mean()+0.5*neg_loss.mean())
        # w_ph = loss/(ph_loss+loss)
        # w_nce = ph_loss/(ph_loss+loss)
        # ---------- JULY 12 ----------------
        # total_loss = loss+ph_loss
        # total_loss = ph_loss
        
        
        total_loss = (1e-8*loss)+ph_loss
        # ---------------------------------
        
        # ----------- July 12 we vectorized sum_mse -----------
        if seg is not None:
            seg_tensors = [torch.tensor(s, dtype=torch.float32, device=total_peaks[i].device) for i, s in enumerate(seg)]
            peaks_tensors = [tp[1:] for tp in total_peaks]
            
            max_len = max(seg_tensor.shape[0] for seg_tensor in seg_tensors)
            seg_padded = torch.stack([F.pad(seg_tensor, (0, max_len - seg_tensor.shape[0]), mode='constant', value=0.0) for seg_tensor in seg_tensors])
            peaks_padded = torch.stack([F.pad(peaks_tensor, (0, max_len - peaks_tensor.shape[0])) for peaks_tensor in peaks_tensors])
            
            # l2_losses = torch.mean((seg_padded - peaks_padded) ** 2, dim=1)
            l2_losses = F.mse_loss(seg_padded, peaks_padded, reduction='none').mean(dim=1)
            
            # weights = (1e-5) ** torch.arange(len(l2_losses), device=l2_losses.device)
            # sum_mse = torch.sum(l2_losses * weights.flip(0))
            sum_mse = l2_losses.mean()  # Just take the mean, no weights
            total_loss = total_loss + 1e-4 * sum_mse
            
        # ------------------------------------------
        # # ----------- July 11 we put that on comment -----------
        # if seg is not None:
        #     sum_mse =0
        #     for i in range(len(total_peaks)):
        #         # ---------------- JULY 1 -----------------
        #         seg_tensor = torch.tensor(seg[i], dtype=torch.float32, device=total_peaks[i].device)
        #         l2_loss = torch.mean((seg_tensor - total_peaks[i][1:])**2)
        #         # l2_loss = np.mean((seg[i] - total_peaks[i][2:])**2 )
        #         # -----------------------------------------
        #         sum_mse = (1e-5*sum_mse) + l2_loss
                
        #     # ----------- July 11 we put that on comment -----------
        #     total_loss = total_loss + sum_mse
        # # ------------------------------------------
        # --------- July 15 ----------------
        # return total_loss
        return total_loss, ph_loss, loss, sum_mse, w_pos_neg, w_phi

@hydra.main(config_path='conf/config.yaml', strict=False)
def main(cfg):
    ds, _, _ = TrainTestDataset.get_datasets(cfg.timit_path)
    spect, seg, phonemes, length, fname = ds[0]
    spect = spect.unsqueeze(0)

    model = NextFrameClassifier(cfg)
    out = model(spect, seg, phonemes, length)


if __name__ == "__main__":
    main()