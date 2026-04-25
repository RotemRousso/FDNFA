import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from utils import LambdaLayer, PrintShapeLayer, length_to_mask, get_timit_61_phoneme_mappings, timit_to_leehon
from dataloader import TrainTestDataset
from collections import defaultdict
import random
from utils import max_min_norm, detect_peaks, create_truth_probs_real
import torch.nn.utils.rnn as rnn_utils
from memory_profiler import profile

class NextFrameClassifier(nn.Module):
    def __init__(self, hp):
        super(NextFrameClassifier, self).__init__()
        # self.w_phi = torch.tensor([0.5, 0.5])
        # self.w_phi = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        
        # self.w_phi = torch.tensor([0.33, 0.33, 0.33])
        
        # self.w_phi = nn.Parameter(torch.tensor([0.33, 0.33, 0.33], dtype=torch.float32))
        self.w_phi = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        
        # self.w_pos_neg = nn.Parameter(torch.tensor([0.99, 0.01], dtype=torch.float32))
        self.w_pos_neg = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
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
                        # nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Dropout1d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, self.hp.z_proj),
                    )
                )
            else:
                self.enc.add_module(
                    "z_proj",
                    nn.Sequential(
                        # nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Dropout1d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, Z_DIM), nn.LeakyReLU(),
                        # nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Dropout1d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, self.hp.z_proj),
                    )
                )
        self.pred_steps = list(range(1 + self.hp.pred_offset, 1 + self.hp.pred_offset + self.hp.pred_steps))
        print(f"prediction steps: {self.pred_steps}")
        
        self.bi_lstm = nn.LSTM(
            input_size=self.hp.z_proj,
            hidden_size=512,
            num_layers=5,#3,
            bidirectional=True,
            batch_first=True
        )
        def init_weights(m):
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
        self.bi_lstm.apply(init_weights)
        self.fc = nn.Linear(512 * 2, hp.num_classes)

    def score(self, f, b):
        return F.cosine_similarity(f, b, dim=-1) * self.hp.cosine_coef
    # @profile
    def forward(self, spect, seg, phonemes, length):
        device = next(self.parameters()).device
        spect = spect.to(device)
        if length is not None and isinstance(length, torch.Tensor):
            length = length.to(device)
        z = self.enc(spect.unsqueeze(1))
        
        del spect
        torch.cuda.empty_cache()
        
        z = F.normalize(z, dim=-1)
        z_bilstm, _ = self.bi_lstm(z)
        logits = self.fc(z_bilstm)
        probs = F.softmax(logits, dim=-1)
        
        probs = logits
        
        
        # # ------------- Sept 9 --------------
        # if seg is not None:
        #     probs_real_batch = []
        #     batch_size = len(seg) if seg is not None else logits.shape[0]
        #     num_phonemes = logits.shape[-1]
        #     max_len = logits.shape[1]
        #     for i in range(batch_size):
        #         segments = seg[i] if seg is not None else [0,max_len]
        #         phs = phonemes[i] if phonemes is not None else ['sil']
        #         num_frames = segments[-1] if segments else max_len
        #         probs_real = create_truth_probs_real(
        #             segments=segments,
        #             phonemes=phs,
        #             phoneme_to_index=self.phoneme_to_index,
        #             num_frames=max_len
        #         )
        #         probs_real_batch.append(probs_real.to(device))
        #     probs_real_batch = torch.stack(probs_real_batch, dim=0)
        #     probs = probs_real_batch
        # # -----------------------------------
        
        frame_labels = torch.zeros_like(logits, device=device)

        # ------- Sept 10 - check with truth preds no truth for nce ---------------
        # seg = None
        # -------------------------------------------------------------------------
        
        # Vectorized phoneme label assignment
        
        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):
            positive_b_scores_list = []
            negative_b_scores_list = []
            if seg is None:
                z_seg = z
                score = self.score(z_seg[:, :-t], z_seg[:, t:])
                for b in range(z.shape[0]):
                    positive_b_scores_list.append(score[b])
                    negative_b_scores_list.append(score[b])
            else:
                for b in range(z.shape[0]):
                    seg_b = [0] + seg[b][:] + [z[b].shape[0]]
                    pos_scores = []
                    neg_scores = []
                    # segment_pairs = list(zip(seg_b[1:-2], seg_b[2:-1]))
                    segment_pairs = list(zip(seg_b[:-1], seg_b[1:-1]))
                    num_repeats = 5
                    for ki, kii in segment_pairs:
                        mid = ki + (kii - ki) // 2
                        sample_len = 1
                        start_idx = int(ki + 0.2 * (kii - ki))
                        end_idx = int(ki + 0.8 * (kii - ki))
                        num_idx = end_idx - start_idx
                        # Batch positive sampling
                        if num_idx >= sample_len:
                            idxs = torch.tensor(
                                random.choices(range(start_idx, end_idx), k=num_repeats),
                                device=device
                            )
                            z_pos = z[b, idxs]
                        else:
                            idx_start = int(mid - (sample_len // 2))
                            idx_end = int(mid + (sample_len // 2))
                            z_pos = z[b, idx_start:idx_end].repeat(num_repeats, 1)
                        pos_scores.append(self.score(z_pos[:, :-t], z_pos[:, t:]))
                        # Batch negative sampling
                        if ki - (sample_len // 2) <= 0 or kii + (sample_len // 2) >= z[b].shape[0]:
                            idx_start = int(mid - (sample_len // 2))
                            idx_end = int(mid + (sample_len // 2))
                            z_neg = z[b, idx_start:idx_end].repeat(num_repeats, 1)
                        else:
                            idx_start = int(kii - (sample_len // 2))
                            idx_end = int(kii + (sample_len // 2))
                            z_neg = z[b, idx_start:idx_end].repeat(num_repeats, 1)
                        neg_scores.append(self.score(z_neg[:, :-t], z_neg[:, t:]))
                    if pos_scores:
                        positive_b_scores_list.append(torch.cat(pos_scores))
                        negative_b_scores_list.append(torch.cat(neg_scores))
            # Padding and stacking
            if positive_b_scores_list:
                original_lengths = torch.tensor([x.shape[0] for x in positive_b_scores_list], dtype=torch.int64, device=device)
                max_length = original_lengths.max().item()
                def pad_scores(scores):
                    # padded = torch.full((max_length,), float(-1e9), device=device)
                    padded = torch.full((max_length,), float(0), device=device)
                    padded[:scores.shape[0]] = scores
                    return padded
                positive_b_scores = torch.stack([pad_scores(x) for x in positive_b_scores_list])
                negative_b_scores = torch.stack([pad_scores(x) for x in negative_b_scores_list])
            else:
                batch_size = z.shape[0]
                max_length = 1
                positive_b_scores = torch.full((batch_size, max_length), float(0), device=device)
                negative_b_scores = torch.full((batch_size, max_length), float(0), device=device)
                # positive_b_scores = torch.full((batch_size, max_length), float(-1e9), device=device)
                # negative_b_scores = torch.full((batch_size, max_length), float(-1e9), device=device)
                original_lengths = torch.ones(batch_size, dtype=torch.int64, device=device)
            preds[t].append(positive_b_scores)
            preds[t].append(negative_b_scores)

        # Peak detection and post-processing (not differentiable, but kept for completeness)
        total_peaks = []
        for i in range(len(phonemes)):
            if seg is not None:
                segments = seg[i]
            else:
                segments = []
                
                
            probs_real = F.softmax(probs[i], dim=-1).squeeze(0)
            # TODO: CHECK ON THAT - SEPT 17 -----------------
            # probs_real = F.softmax(probs[i], dim=-1).squeeze(0).T
            
            # ----------- Aug 27 TRY -----------
            # out_preds = torch.stack(preds[1], dim=-1)
            # out_preds = F.log_softmax(out_preds, dim=-1)
            # # mask = length_to_mask(original_lengths[i] - 1 + 1)
            # mask = length_to_mask(torch.tensor([original_lengths[i]], device=original_lengths.device))
            # mask = mask[0]  # get the mask for this sample
            # cur_preds = out_preds[i,:,0] * mask[i]          
            
            cur_preds = preds[1][0][i]
            cur_preds = max_min_norm(cur_preds)
            # cur_preds = 1 - max_min_norm(cur_preds)
            # preds_np = cur_preds
            
            median_h = cur_preds.median()
            preds_np = cur_preds - median_h
            w_phi = torch.softmax(self.w_phi, dim=0)
            

            sr = 16000
            len_ratio = 161.34011627906978
            preds_peaks = detect_peaks(
                x= (cur_preds), #(-1 * cur_preds),
                w_phi=w_phi,
                original_lengths_all=original_lengths[i],
                phonemes=phonemes[i],
                len_ratio=len_ratio,
                probs_real_all=probs_real
            )
            preds_peaks = preds_peaks[0] * len_ratio / sr
            if seg is not None:
                segments = np.array(segments) * len_ratio / sr

            # print("truth boundaries ('segments'):")
            # print(segments)
            # print("predicted boundaries (in seconds):")
            # print(preds_peaks)

            total_peaks.append(preds_peaks)

        return preds, original_lengths, probs, frame_labels, seg, total_peaks, w_phi


    # ------------------- oct 21 original ----------------------------------
    # def loss(self, preds, original_lengths, probs, frame_labels, seg, total_peaks, w_phi):
    #     # total_loss = torch.tensor(0.0, device=probs.device, requires_grad=True)
    #     total_loss = 0.0
    #     probs = probs.view(-1, probs.size(-1))
    #     frame_labels = frame_labels.view(-1, frame_labels.size(-1))
    #     ph_loss = F.cross_entropy(probs, frame_labels.argmax(dim=-1))
    #     loss = 0
        
    #     # ----------- Aug 27 TRY -----------
    #     # for i in range(len(original_lengths)):
    #     #     cur_pos_preds = preds[1][0][i]
    #     #     cur_neg_preds = preds[1][1][i]
    #     #     cur_pos_preds = max_min_norm(cur_pos_preds)
    #     #     cur_pos_preds = 1 - max_min_norm(cur_pos_preds)
    #     #     cur_neg_preds = 1 - max_min_norm(cur_neg_preds)
    #     #     cur_neg_preds = max_min_norm(cur_neg_preds)
    #     #     # mask = length_to_mask(torch.tensor([original_lengths[i]],device=original_lengths.device))[0]
    #     #     mask = torch.arange(cur_pos_preds.shape[0], device=cur_pos_preds.device) < original_lengths[i]
    #     #     pos_valid = cur_pos_preds[mask.bool()]
    #     #     neg_valid = cur_neg_preds[mask.bool()]
            
    #     #     pos_valid = pos_valid[~torch.isnan(pos_valid)]
    #     #     neg_valid = neg_valid[~torch.isnan(neg_valid)]
            
    #     #     pos_loss_mean = pos_valid.mean() if pos_valid.numel() > 0 else torch.tensor(0.0, device=cur_pos_preds.device)
    #     #     neg_loss_mean = neg_valid.mean() if neg_valid.numel() > 0 else torch.tensor(0.0, device=cur_neg_preds.device)
            
    #     #     l_pos_neg = torch.stack([pos_loss_mean, neg_loss_mean])
    #     #     w_pos_neg = torch.softmax(self.w_pos_neg, dim=0)
    #     #     loss += torch.dot(w_pos_neg, l_pos_neg)
        
    #     for t, t_preds in preds.items():
    #         mask = length_to_mask(original_lengths - t + 1)
    #         out = torch.stack(t_preds, dim=-1)
    #         out = F.log_softmax(out, dim=-1)
    #         pos_loss = out[..., 0] * mask
    #         neg_loss = out[..., 1] * mask
    #         # valid = mask > 0
    #         # pos_loss = out[..., 0][valid]
    #         # neg_loss = out[..., 1][valid]
            
            
    #         # # valid = mask > 0
            
    #         pos_count = mask.sum()
    #         neg_count = mask.sum()
            
    #         pos_loss_mean = pos_loss.sum() / pos_count if pos_count > 0 else torch.tensor(0.0, device=pos_loss.device)
    #         neg_loss_mean = neg_loss.sum() / neg_count if neg_count > 0 else torch.tensor(0.0, device=neg_loss.device)
    #         # # For neg_loss, only consider entries where the original (unpadded) data exists
    #         # neg_valid = (out[..., 1] != -1e9) & (mask > 0)
    #         # neg_loss_mean = out[..., 1][neg_valid].mean() if neg_valid.sum() > 0 else torch.tensor(0.0, device=neg_loss.device)
            
    #         # l_pos_neg = torch.stack([pos_loss_mean, neg_loss_mean])
            
            
    #         # # pos_loss_mean = pos_loss.sum() / mask.sum()
    #         # # neg_loss_mean = neg_loss.sum() / mask.sum()
    #         # w_pos_neg = torch.softmax(self.w_pos_neg, dim=0)
    #         # # l_pos_neg = torch.stack([pos_loss_mean, neg_loss_mean])
    #         # loss += -torch.dot(w_pos_neg, l_pos_neg)
    #         w_pos_neg = torch.softmax(self.w_pos_neg, dim=0)
    #         # l_pos_neg = torch.stack([pos_loss_norm.mean(), neg_loss_norm.mean()])
    #         # l_pos_neg = torch.stack([pos_loss.mean(), neg_loss.mean()])
    #         l_pos_neg = torch.stack([pos_loss_mean, neg_loss_mean])
    #         loss += -torch.dot(w_pos_neg, l_pos_neg)
            
    #     # ------------ Sept 9 ---------------    
    #     # total_loss = (loss)
    #     total_loss = (loss) + ph_loss
    #     # -----------------------------------        
        
    #     # total_loss = (loss) + ph_loss
        
    #     # total_loss = ph_loss

    #     # vectorized sum_mse calculation for peak alignment loss
    #     sum_mse = torch.tensor(0.0, device=probs.device)
    #     # plateau_penalty = torch.tensor(0.0, device=probs.device)
    #     if seg is not None and len(total_peaks) > 0:
    #         seg_tensors = [torch.tensor(s, dtype=torch.float32, device=probs.device) for s in seg]
    #         for i in range(len(seg_tensors)):
    #             seg_tensors[i] = seg_tensors[i]*161.34011627906978/16000  # convert to seconds
    #         peaks_tensors = [torch.tensor(tp[1:], dtype=torch.float32, device=probs.device) for tp in total_peaks]
    #         # seg_padded = rnn_utils.pad_sequence(seg_tensors, batch_first=True, padding_value=0.0)
    #         # peaks_padded = rnn_utils.pad_sequence(peaks_tensors, batch_first=True, padding_value=0.0)
    #         # lengths = torch.tensor([len(s) for s in seg_tensors], device=probs.device)
    #         # mask = torch.arange(seg_padded.shape[1], device=probs.device)[None, :] < lengths[:, None]
    #         # mask = mask.float()
    #         # mse = F.mse_loss(seg_padded, peaks_padded, reduction='none')
            
    #         # for peaks in peaks_tensors:
    #         #     plateau_penalty = plateau_penalty + torch.relu(peaks[:-1] - peaks[1:]).sum()
    #         # plateau_penalty = plateau_penalty / len(peaks_tensors)
    #         mse = 0.0
    #         for i in range(len(seg_tensors)):
    #             mse = mse + F.mse_loss(seg_tensors[i], peaks_tensors[i])
    #         mse = mse / len(seg_tensors)
    #         # masked_mse = mse * mask
    #         # sum_mse = masked_mse.sum() / mask.sum()
    #         sum_mse = mse
    #         total_loss = total_loss + sum_mse #+ 1e-2*plateau_penalty #1e-10* 1e-4 * sum_mse  # Lower weight for stability
    #         # total_loss = total_loss + 1e-2* sum_mse #+ 1e-2*plateau_penalty #1e-10* 1e-4 * sum_mse  # Lower weight for stability
    #         # total_loss = total_loss + 1e-4 * sum_mse #1e-10* 1e-4 * sum_mse  # Lower weight for stability
    #         # total_loss = total_loss + sum_mse #1e-10* 1e-4 * sum_mse  # Lower weight for stability

    #     # return total_loss, ph_loss, loss, sum_mse, w_pos_neg, w_phi, plateau_penalty
    #     return total_loss, ph_loss, loss, sum_mse, w_pos_neg, w_phi #, plateau_penalty
    
    
    
    
    
    
    # ------------------- oct 21 changes: ----------------------------------
    # @ profile
    def loss_ph(self, preds, original_lengths, probs, frame_labels, seg, total_peaks, w_phi, phonemes):
        
        # as it was in the forward but now in every mode there supposed to be seg = not none
        if seg is not None:
            for i, (segments, phs) in enumerate(zip(seg, phonemes)):
                # starts = np.array(segments[:-1], dtype=int)
                starts = np.array([0] + list(segments[:-1]), dtype=int)
                ends = np.array(segments[:], dtype=int)
                # we put 'sil' wherever we don't have the mapping - TODO: make sure there's not such case after we debug everything
                labels = [timit_to_leehon(p) or 'sil' for p in phs] 
                # why don't we use - {p.lower(): timit_to_leehon_map_MACRO[p.lower()] for p in phs} like in ustils - phoneme_alignment?
                # TODO - AFTER DEBUGS - might collapse when we will have dutch till we use the MACRO mapping
                # labels = [timit_to_leehon(p) for p in phs]  # where do we use the timit to leehon macro that includes also Dutch?
                
                
                # labels = [(p) or 'sil' for p in phs]
                label_indices = [self.phoneme_to_index[l] for l in labels]
                for start, end, label_index in zip(starts, ends, label_indices):
                    frame_labels[i, start:end, label_index] = 1.0
        
        
        
        
        total_loss = 0.0
        probs = probs.view(-1, probs.size(-1))
        frame_labels = frame_labels.view(-1, frame_labels.size(-1))
        ph_loss = F.cross_entropy(probs, frame_labels.argmax(dim=-1))
        loss = 0
        
        for t, t_preds in preds.items():
            mask = length_to_mask(original_lengths - t + 1)
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            pos_loss = out[..., 0] * mask
            neg_loss = out[..., 1] * mask

            pos_count = mask.sum()
            neg_count = mask.sum()
            
            pos_loss_mean = pos_loss.sum() / pos_count if pos_count > 0 else torch.tensor(0.0, device=pos_loss.device)
            neg_loss_mean = neg_loss.sum() / neg_count if neg_count > 0 else torch.tensor(0.0, device=neg_loss.device)

            w_pos_neg = torch.softmax(self.w_pos_neg, dim=0)

            l_pos_neg = torch.stack([pos_loss_mean, neg_loss_mean])
            loss += -torch.dot(w_pos_neg, l_pos_neg)

        # total_loss = (loss) + ph_loss
        total_loss = (loss) + 0.2*ph_loss

        sum_mse = torch.tensor(0.0, device=probs.device)

        if seg is not None and len(total_peaks) > 0:
            seg_tensors = [torch.tensor(s, dtype=torch.float32, device=probs.device) for s in seg]
            for i in range(len(seg_tensors)):
                seg_tensors[i] = seg_tensors[i]*161.34011627906978/16000  # convert to seconds
            peaks_tensors = [torch.tensor(tp[1:], dtype=torch.float32, device=probs.device) for tp in total_peaks]

            mse = 0.0
            for i in range(len(seg_tensors)):
                mse = mse + F.mse_loss(seg_tensors[i], peaks_tensors[i])
            mse = mse / len(seg_tensors)

            sum_mse = mse
            # total_loss = total_loss + sum_mse 
            total_loss = ph_loss

        return total_loss, ph_loss, loss, sum_mse, w_pos_neg, w_phi
    
    # ------------------------------ Feb 02, 2026 ---------------------------------
    def loss_InfoNCE_classic(self, preds, original_lengths, probs, frame_labels, seg, total_peaks, w_phi, phonemes):
        """
        Classic InfoNCE implementation.
        Standard log-softmax over one positive and N negatives.
        """
        device = probs.device
        total_loss = 0.0
        
        # Optional: Keep the ph_loss calculation if you want to track it, 
        # but in pure InfoNCE ablations, we usually ignore it for gradients.
        probs_flat = probs.view(-1, probs.size(-1))
        frame_labels_flat = frame_labels.view(-1, frame_labels.size(-1))
        ph_loss = F.cross_entropy(probs_flat, frame_labels_flat.argmax(dim=-1))

        nce_loss_accum = 0.0
        
        for t, t_preds in preds.items():
            # t_preds[0] is positive scores, t_preds[1] is negative scores
            # out shape: [batch, length, 2] (pos score and neg score)
            logits = torch.stack(t_preds, dim=-1) 
            
            # Classic InfoNCE: Cross Entropy where index 0 is always the 'correct' class
            # target is a tensor of zeros (index 0) with shape [batch, length]
            batch_size, seq_len, _ = logits.shape
            target = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            
            # We use a mask to ignore padded frames and t-offset frames
            mask = length_to_mask(original_lengths - t + 1)
            
            # Flatten for CrossEntropy
            # logits: [batch * length, 2], target: [batch * length]
            logits_flat = logits.view(-1, 2)
            target_flat = target.view(-1)
            
            # Apply standard CrossEntropy (Log-Softmax + NLLLoss)
            # reduction='none' so we can apply the mask manually
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            raw_loss = loss_fn(logits_flat, target_flat)
            
            # Mask and average
            masked_loss = raw_loss * mask.view(-1)
            nce_loss_accum += masked_loss.sum() / mask.sum()

        # In classic NCE ablation, total_loss is just the contrastive part
        total_loss = nce_loss_accum / len(preds)
        
        # Dummies for the return signature to match your training loop
        sum_mse = torch.tensor(0.0, device=device)
        w_pos_neg = torch.softmax(self.w_pos_neg, dim=0) 

        return total_loss, ph_loss, nce_loss_accum, sum_mse, w_pos_neg, w_phi
    # -----------------------------------------------------------------------------
    
    def loss_nce(self, preds, original_lengths, probs, frame_labels, seg, total_peaks, w_phi, phonemes):
        
                
        # as it was in the forward but now in every mode there supposed to be seg = not none
        if seg is not None:
            for i, (segments, phs) in enumerate(zip(seg, phonemes)):
                # starts = np.array(segments[:-1], dtype=int)
                starts = np.array([0] + list(segments[:-1]), dtype=int)
                ends = np.array(segments[:], dtype=int)
                labels = [timit_to_leehon(p) or 'sil' for p in phs]
                # labels = [(p) or 'sil' for p in phs]
                label_indices = [self.phoneme_to_index[l] for l in labels]
                for start, end, label_index in zip(starts, ends, label_indices):
                    frame_labels[i, start:end, label_index] = 1.0
        
        total_loss = 0.0
        probs = probs.view(-1, probs.size(-1))
        frame_labels = frame_labels.view(-1, frame_labels.size(-1))
        ph_loss = F.cross_entropy(probs, frame_labels.argmax(dim=-1))
        loss = 0
        
        for t, t_preds in preds.items():
            mask = length_to_mask(original_lengths - t + 1)
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            pos_loss = out[..., 0] * mask
            neg_loss = out[..., 1] * mask

            pos_count = mask.sum()
            neg_count = mask.sum()
            
            pos_loss_mean = pos_loss.sum() / pos_count if pos_count > 0 else torch.tensor(0.0, device=pos_loss.device)
            neg_loss_mean = neg_loss.sum() / neg_count if neg_count > 0 else torch.tensor(0.0, device=neg_loss.device)

            w_pos_neg = torch.softmax(self.w_pos_neg, dim=0)

            l_pos_neg = torch.stack([pos_loss_mean, neg_loss_mean])
            loss += -torch.dot(w_pos_neg, l_pos_neg)

        # total_loss = (loss) + ph_loss
        # main version
        # total_loss = (loss) + 0.2*ph_loss       
        # for ablations - no phoneme loss
        total_loss = (loss) # + 0.2*ph_loss

        sum_mse = torch.tensor(0.0, device=probs.device)

        if seg is not None and len(total_peaks) > 0:
            seg_tensors = [torch.tensor(s, dtype=torch.float32, device=probs.device) for s in seg]
            for i in range(len(seg_tensors)):
                seg_tensors[i] = seg_tensors[i]*161.34011627906978/16000  # convert to seconds
            peaks_tensors = [torch.tensor(tp[1:], dtype=torch.float32, device=probs.device) for tp in total_peaks]

            mse = 0.0
            for i in range(len(seg_tensors)):
                mse = mse + F.mse_loss(seg_tensors[i], peaks_tensors[i])
            mse = mse / len(seg_tensors)

            sum_mse = mse
            # total_loss = total_loss + sum_mse 
            total_loss = loss

        return total_loss, ph_loss, loss, sum_mse, w_pos_neg, w_phi
    
    def loss_mse(self, preds, original_lengths, probs, frame_labels, seg, total_peaks, w_phi, phonemes):
        
                
        # as it was in the forward but now in every mode there supposed to be seg = not none
        if seg is not None:
            for i, (segments, phs) in enumerate(zip(seg, phonemes)):
                # starts = np.array(segments[:-1], dtype=int)
                starts = np.array([0] + list(segments[:-1]), dtype=int)
                ends = np.array(segments[:], dtype=int)
                labels = [timit_to_leehon(p) or 'sil' for p in phs]
                # labels = [(p) or 'sil' for p in phs]
                label_indices = [self.phoneme_to_index[l] for l in labels]
                for start, end, label_index in zip(starts, ends, label_indices):
                    frame_labels[i, start:end, label_index] = 1.0
        
        total_loss = 0.0
        probs = probs.view(-1, probs.size(-1))
        frame_labels = frame_labels.view(-1, frame_labels.size(-1))
        ph_loss = F.cross_entropy(probs, frame_labels.argmax(dim=-1))
        loss = 0
        
        for t, t_preds in preds.items():
            mask = length_to_mask(original_lengths - t + 1)
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            pos_loss = out[..., 0] * mask
            neg_loss = out[..., 1] * mask

            pos_count = mask.sum()
            neg_count = mask.sum()
            
            pos_loss_mean = pos_loss.sum() / pos_count if pos_count > 0 else torch.tensor(0.0, device=pos_loss.device)
            neg_loss_mean = neg_loss.sum() / neg_count if neg_count > 0 else torch.tensor(0.0, device=neg_loss.device)

            w_pos_neg = torch.softmax(self.w_pos_neg, dim=0)

            l_pos_neg = torch.stack([pos_loss_mean, neg_loss_mean])
            loss += -torch.dot(w_pos_neg, l_pos_neg)

        # total_loss = (loss) + ph_loss
        # main version
        total_loss = (loss) + 0.2*ph_loss
        # for ablations - no phoneme loss
        # total_loss = (loss) #+ 0.2*ph_loss

        sum_mse = torch.tensor(0.0, device=probs.device)

        if seg is not None and len(total_peaks) > 0:
            seg_tensors = [torch.tensor(s, dtype=torch.float32, device=probs.device) for s in seg]
            for i in range(len(seg_tensors)):
                seg_tensors[i] = seg_tensors[i]*161.34011627906978/16000  # convert to seconds
            peaks_tensors = [torch.tensor(tp[1:], dtype=torch.float32, device=probs.device) for tp in total_peaks]

            mse = 0.0
            for i in range(len(seg_tensors)):
                mse = mse + F.mse_loss(seg_tensors[i], peaks_tensors[i])
            mse = mse / len(seg_tensors)

            sum_mse = mse
            # total_loss = total_loss + sum_mse 
            total_loss = sum_mse

        return total_loss, ph_loss, loss, sum_mse, w_pos_neg, w_phi

    def total_loss(self, preds, original_lengths, probs, frame_labels, seg, total_peaks, w_phi,phonemes):
        
                
        # as it was in the forward but now in every mode there supposed to be seg = not none
        if seg is not None:
            for i, (segments, phs) in enumerate(zip(seg, phonemes)):
                # starts = np.array(segments[:-1], dtype=int)
                starts = np.array([0] + list(segments[:-1]), dtype=int)
                ends = np.array(segments[:], dtype=int)
                labels = [timit_to_leehon(p) or 'sil' for p in phs]
                # labels = [(p) or 'sil' for p in phs]
                label_indices = [self.phoneme_to_index[l] for l in labels]
                for start, end, label_index in zip(starts, ends, label_indices):
                    frame_labels[i, start:end, label_index] = 1.0
        
        total_loss = 0.0
        probs = probs.view(-1, probs.size(-1))
        frame_labels = frame_labels.view(-1, frame_labels.size(-1))
        ph_loss = F.cross_entropy(probs, frame_labels.argmax(dim=-1))
        loss = 0
        
        for t, t_preds in preds.items():
            mask = length_to_mask(original_lengths - t + 1)
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            pos_loss = out[..., 0] * mask
            neg_loss = out[..., 1] * mask

            pos_count = mask.sum()
            neg_count = mask.sum()
            
            pos_loss_mean = pos_loss.sum() / pos_count if pos_count > 0 else torch.tensor(0.0, device=pos_loss.device)
            neg_loss_mean = neg_loss.sum() / neg_count if neg_count > 0 else torch.tensor(0.0, device=neg_loss.device)

            w_pos_neg = torch.softmax(self.w_pos_neg, dim=0)

            l_pos_neg = torch.stack([pos_loss_mean, neg_loss_mean])
            loss += -torch.dot(w_pos_neg, l_pos_neg)

        # total_loss = (loss) + ph_loss
        # main version
        total_loss = (loss) + 0.2*ph_loss
        # for ablations - no phoneme loss
        # total_loss = (loss) # + 0.2*ph_loss

        sum_mse = torch.tensor(0.0, device=probs.device)

        if seg is not None and len(total_peaks) > 0:
            seg_tensors = [torch.tensor(s, dtype=torch.float32, device=probs.device) for s in seg]
            for i in range(len(seg_tensors)):
                seg_tensors[i] = seg_tensors[i]*161.34011627906978/16000  # convert to seconds
            peaks_tensors = [torch.tensor(tp[1:], dtype=torch.float32, device=probs.device) for tp in total_peaks]

            mse = 0.0
            for i in range(len(seg_tensors)):
                mse = mse + F.mse_loss(seg_tensors[i], peaks_tensors[i])
            mse = mse / len(seg_tensors)

            sum_mse = mse
            total_loss = total_loss + sum_mse 
            # total_loss = total_loss + sum_mse 
            

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