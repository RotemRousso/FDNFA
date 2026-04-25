import os
import glob
import shutil

# --- Configuration ---
ROOT_DIR = "datasets/IFA_dutch"
OUTPUT_DIR = "datasets/IFA_Dutch_words_reordered"
SAMPLE_RATE = 16000 # Standard for TIMIT; change to 44100 if your wavs are high-res
# ---------------------

def clean_word(text):
    """Cleans the WORD label from IFA metadata and handles silences."""
    text = text.strip('"')
    if text.startswith("#"):
        return "sil"
    # Extracts 'hud' from 'hud__F20N1FPA1HVDA6'
    return text.split("__")[0]

def parse_short_textgrid(filepath):
    """Parses the WORD tier from a short-format TextGrid with Latin-1 encoding."""
    with open(filepath, 'r', encoding='latin-1') as f:
        lines = [l.strip() for l in f.readlines()]
    
    words_data = []
    idx = 0
    while idx < len(lines):
        # We look specifically for the WORD interval tier
        if lines[idx] == '"WORDS"':
            idx += 4 # Skip tier header lines
            num_intervals = int(lines[idx-1])
            for _ in range(num_intervals):
                t_start = float(lines[idx])
                t_end = float(lines[idx+1])
                label = clean_word(lines[idx+2])
                
                # Convert seconds to samples for TIMIT format
                start_sample = int(t_start * SAMPLE_RATE)
                end_sample = int(t_end * SAMPLE_RATE)
                
                words_data.append((start_sample, end_sample, label))
                idx += 3
            break
        idx += 1
    return words_data

def process_all_files():
    # Create the single output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Find all .phoneme files regardless of speaker folder
    phoneme_files = glob.glob(os.path.join(ROOT_DIR, "original_labels/sentences/*/phoneme/*.phoneme"))
    print(f"Found {len(phoneme_files)} files to process.")

    success_count = 0

    for pf in phoneme_files:
        # Get speaker ID (e.g., F20N) from path
        path_parts = pf.split('/')
        speaker_id = path_parts[-3] 
        
        base_name = os.path.basename(pf).replace(".phoneme", "")
        # Match wav: F20N1FPA1HVDA_JB.phoneme -> F20N1FPA1HVDA.wav
        wav_name = base_name.split('_')[0] + ".wav"
        wav_path = os.path.join(ROOT_DIR, "speech", speaker_id, wav_name)

        if not os.path.exists(wav_path):
            # Sometimes files might be in a different sub-folder; this is a safety check
            continue

        try:
            intervals = parse_short_textgrid(pf)
            if not intervals:
                continue

            # 1. Create .wrd content (StartSample EndSample Word)
            wrd_lines = [f"{s} {e} {w}" for s, e, w in intervals]
            
            # 2. Create .txt content (Transcript with sil at start and end)
            # Filter out existing silences to clean the middle, then wrap
            words_only = [i[2] for i in intervals if i[2] != "sil"]
            full_txt = f"sil {' '.join(words_only)} sil"

            # Define output paths in the flat folder
            out_wav = os.path.join(OUTPUT_DIR, base_name + ".wav")
            out_wrd = os.path.join(OUTPUT_DIR, base_name + ".wrd")
            out_txt = os.path.join(OUTPUT_DIR, base_name + ".txt")

            # Write files
            shutil.copy(wav_path, out_wav)
            with open(out_wrd, 'w', encoding='utf-8') as f:
                f.write("\n".join(wrd_lines))
            with open(out_txt, 'w', encoding='utf-8') as f:
                f.write(full_txt)
            
            success_count += 1

        except Exception as e:
            print(f"Error processing {pf}: {e}")

    print(f"\nFinished! Successfully processed {success_count} files into {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all_files()