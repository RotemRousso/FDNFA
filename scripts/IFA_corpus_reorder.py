import os
import glob

def convert_timit_to_phn(wav_dir, label_dir, output_dir, sample_rate=16000):
    """
    Scans for .wav and .timit files, converts times to samples, 
    and saves them as paired .wav and .phn files in a new directory.
    """
    # Create the root output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all wav files in the speech folder (recursive to catch speaker subfolders)
    wav_files = glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True)

    for wav_path in wav_files:
        # Extract basename (e.g., F20N1FPA1HVDA)
        basename = os.path.basename(wav_path).replace(".wav", "")
        
        # Determine speaker subfolder (e.g., F20N)
        speaker_id = os.path.basename(os.path.dirname(wav_path))
        
        # Look for the corresponding .timit file in the labels directory
        # Using a wildcard because of suffixes like _JB.timit
        timit_search = os.path.join(label_dir, speaker_id, f"{basename}*.timit")
        timit_matches = glob.glob(timit_search)

        if not timit_matches:
            print(f"Skipping: No .timit file found for {basename}")
            continue

        timit_path = timit_matches[0]
        phn_path = os.path.join(output_dir, f"{basename}.phn")
        new_wav_path = os.path.join(output_dir, f"{basename}.wav")

        # 1. Convert .timit to .phn (Seconds -> Samples)
        try:
            with open(timit_path, 'r') as f_in, open(phn_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    
                    # Convert start and end times to samples
                    start_samples = int(float(parts[0]) * sample_rate)
                    end_samples = int(float(parts[1]) * sample_rate)
                    label = parts[2]
                    
                    f_out.write(f"{start_samples} {end_samples} {label}\n")
            
            # 2. Copy the .wav file to the new location (or symlink it)
            import shutil
            shutil.copy2(wav_path, new_wav_path)
            
            print(f"Processed: {basename}")

        except Exception as e:
            print(f"Error processing {basename}: {e}")

# --- CONFIGURATION ---
# Update these paths based on your environment
WAV_ROOT = "datasets/IFA_dutch/speech"
LABEL_ROOT = "datasets/IFA_dutch/labels"
FINAL_OUTPUT = "datasets/IFA_dutch_ready"

convert_timit_to_phn(WAV_ROOT, LABEL_ROOT, FINAL_OUTPUT)