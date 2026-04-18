import os

# Import your mapping dictionary
from utils import timit_to_leehon_map_MACRO

def find_keyerror_phonemes_in_file(file_path):
    with open(file_path, "r") as f:
        phonemes = [line.strip() for line in f]
    keyerrors = []
    for p in phonemes:
        if p.lower() not in timit_to_leehon_map_MACRO:
            keyerrors.append(p)
    return keyerrors

def scan_dataset(dataset_dir):
    error_report = {}
    for root, dirs, files in os.walk(dataset_dir):
        for fname in files:
            if fname.endswith(".phn"):  # or whatever your phoneme file extension is
                fpath = os.path.join(root, fname)
                keyerrors = find_keyerror_phonemes_in_file(fpath)
                if keyerrors:
                    error_report[fpath] = keyerrors
    return error_report

if __name__ == "__main__":
    dataset_dir = "/home/rotem/projects/datasets/IFA_dutch_reorder/train"  # <-- CHANGE THIS
    report = scan_dataset(dataset_dir)
    with open("/home/rotem/projects/datasets/IFA_dutch_reorder/keyerror_phoneme_files.log", "w") as logf:
        for fpath, keyerrors in report.items():
            logf.write(f"{fpath}: {keyerrors}\n")
    print(f"Done. Found {len(report)} files with keyerrors. See /home/rotem/projects/datasets/IFA_dutch_reorder/keyerror_phoneme_files.log .")