import glob
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import timit_to_leehon_map_MACRO, timit_leehon_39_phonemes

# phn_files = glob.glob("/home/rotem/projects/datasets/IFA_dutch_reorder/train/*_ipa.phn")
phn_files = glob.glob("datasets/IFA_dutch_reorder/val/*_ipa.phn")
missing_phonemes = set()

for fname in phn_files:
    with open(fname, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) ==3:
                _,_,ph=parts
                key = ph.lower()
                if key not in timit_to_leehon_map_MACRO:
                    missing_phonemes.add(key)
                    print(f"Missing phoneme: {key} in file {fname}")
                    
print(f"Total missing phonemes: {len(missing_phonemes)}")
for ph in sorted(missing_phonemes):
    print(ph)
    
# ---------------------------------
missing_keys_leehon = extra_keys = set(timit_to_leehon_map_MACRO.values()) - set(timit_leehon_39_phonemes)
print(f"Unique keys in timit_to_leehon_map_MACRO not in timit_leehon_39_phonemes: {len(missing_keys_leehon)}")
for k in sorted(extra_keys):
    print(k)
# ---------------------------------