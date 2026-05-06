import gradio as gr
import os
import shutil
import textgrid
import torchaudio
import re

# Import FDNFA functions without modifying them
from predict import main_predict
import tempfile
import threading

# A lock is necessary because the underlying FDNFA code (utils.py) hardcodes 
# saving to 'debug_plots/dp_matrix_with_path.png'. This lock ensures two users 
# don't overwrite the plot at the exact same millisecond.
inference_lock = threading.Lock()

def run_alignment(audio_file, annotation_file, ckpt_path, mode, lang, w_phi, progress=gr.Progress()):
    if not audio_file or not annotation_file:
        return "Error: Please upload both audio and annotation files.", None, None, None, None, None, None, None
    DEFAULT_CKPT = "/home/rotem/projects/CFA/DCAF2/DCAF2.0/runs/Dec2_fullTIMIT_frame_labelsNotZeroATlossNotForward_HulkTmux5gpu5/2025-12-02_16-12-30-default/23_best_model.pt"
    
    ckpt_to_use = ckpt_path if ckpt_path is not None else DEFAULT_CKPT

    if not ckpt_to_use or not os.path.exists(ckpt_to_use):
        return f"Error: Please provide a valid checkpoint path. Tried: {ckpt_to_use}", None, None, None, None, None, None, None

    progress(0.1, desc="Setting up workspace...")
    # Create a unique temporary workspace for this user's request
    workspace = tempfile.mkdtemp(prefix="fdnfa_demo_")
    
    base_name = "demo_audio"
    wav_path = os.path.join(workspace, f"{base_name}.wav")
    
    ann_ext = os.path.basename(annotation_file).split('.')[-1]
    ann_path = os.path.join(workspace, f"{base_name}.{ann_ext}")
    
    # 1. Resample to 16kHz
    try:
        audio, sr = torchaudio.load(audio_file)
        if sr != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)
        torchaudio.save(wav_path, audio, 16000)
    except Exception as e:
        return f"Error processing audio file: {str(e)}", None, None, None, None, None, None, None
        
    shutil.copy(annotation_file, ann_path)
    
    is_txt_input = ann_ext.lower() == "txt"
    if is_txt_input:
        with open(ann_path, "r") as f:
            raw_text = f.read().strip()
            # 2. Punctuation stripping
            clean_text = re.sub(r'[^\w\s]', '', raw_text)
            labels = clean_text.split()
            
        audio, sr = torchaudio.load(wav_path)
        audio_len = audio.shape[1]
        interval = audio_len / len(labels) if len(labels) > 0 else 0
        
        ann_ext = "dummy_phn"
        dummy_path = os.path.join(workspace, f"{base_name}.{ann_ext}")
        with open(dummy_path, "w") as f:
            for i, label in enumerate(labels):
                end_frame = int((i + 1) * interval)
                f.write(f"0 {end_frame} {label}\n")
        ann_path = dummy_path
    
    language = "dutch" if (mode == "word" or lang == "multilingual") else "english"
    
    dp_matrix_temp_path = os.path.join(workspace, "dp_matrix_with_path.png")
    
    progress(0.3, desc="Running FDNFA Inference (this may take a moment)...")
    try:
        # Acquire lock to prevent race conditions on hardcoded cwd directory
        with inference_lock:
            pred_bound, truth_bound, mapped_ph = main_predict(
                wav=wav_path, 
                ckpt=ckpt_to_use, 
                w_phi=w_phi, 
                language=language, 
                annotation=ann_ext
            )
            
            if mapped_ph is not None:
                labels_to_plot = [ph for seq in mapped_ph for ph in seq]
            else:
                with open(ann_path, "r") as f:
                    lines = f.readlines()
                words_temp = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        words_temp.append(parts[2])
                labels_to_plot = words_temp

            # If input was txt, the plots now have dummy truth boundaries. 
            # We can re-run predict.py with the predicted boundaries acting as the truth boundaries
            # so the plots will show the prediction.
            if is_txt_input and len(pred_bound) > 0:
                with open(dummy_path, "w") as f:
                    for i, pred_sec in enumerate(pred_bound):
                        end_frame = int(pred_sec * 16000)
                        label = labels_to_plot[i] if i < len(labels_to_plot) else "unk"
                        f.write(f"0 {end_frame} {label}\n")
                        
                # Re-run inference to regenerate plots correctly
                main_predict(
                    wav=wav_path, 
                    ckpt=ckpt_to_use, 
                    w_phi=w_phi, 
                    language=language, 
                    annotation=ann_ext
                )
            
            progress(0.7, desc="Gathering matrix visualizations...")
            # Immediately copy the hardcoded matrix plot to our unique temp workspace
            # utils.py saves it in the CURRENT WORKING DIRECTORY, not debug_plots!
            hardcoded_dp_path = "dp_matrix_with_path.png"
            if os.path.exists(hardcoded_dp_path):
                shutil.copy(hardcoded_dp_path, dp_matrix_temp_path)
                
    except Exception as e:
        return f"Error during inference: {str(e)}", None, None, None, None, None, None, None

    progress(0.8, desc="Generating TextGrid and Tables...")
    # Boundaries plot is saved relative to the input wav file, so it's safely in our unique workspace
    boundaries_plot_path = os.path.join(workspace, f"{base_name}_boundaries.png")
    probs_plot_path = os.path.join(workspace, f"{base_name}_probs.png")
    logits_plot_path = os.path.join(workspace, f"{base_name}_logits.png")
    
    if not os.path.exists(boundaries_plot_path):
        boundaries_plot_path = None
    if not os.path.exists(dp_matrix_temp_path):
        dp_matrix_temp_path = None
    if not os.path.exists(probs_plot_path):
        probs_plot_path = None
    if not os.path.exists(logits_plot_path):
        logits_plot_path = None

    # Generate the TextGrid using the exact logic from FDNFA but respecting the w_phi parameter
    with open(ann_path, "r") as f:
        lines = f.readlines()
        
    words = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            words.append(parts[2])
    
    max_time = pred_bound[-1] if len(pred_bound) > 0 else 0
    tg = textgrid.TextGrid(minTime=0, maxTime=max_time)
    tier = textgrid.IntervalTier(name=mode, minTime=0, maxTime=max_time)
    
    start_time = 0
    table_data = []
    
    if mapped_ph is not None:
        labels_to_use = [ph for seq in mapped_ph for ph in seq]
    else:
        labels_to_use = words
    
    for i, end_time in enumerate(pred_bound):
        label = labels_to_use[i] if i < len(labels_to_use) else ""
        if end_time > start_time:
            end_time_f = float(end_time)
            start_time_f = float(start_time)
            tier.add(minTime=start_time_f, maxTime=end_time_f, mark=label)
            table_data.append([round(start_time_f, 3), round(end_time_f, 3), label])
        start_time = end_time
        
    tg.append(tier)
    tg_out_path = os.path.join(workspace, f"{base_name}.TextGrid")
    tg.write(tg_out_path)
    
    return "Success! View results below.", audio_file, boundaries_plot_path, dp_matrix_temp_path, probs_plot_path, logits_plot_path, tg_out_path, table_data

# CSS to make the app feel extremely premium and professional
custom_css = """
body {
    background-color: #0b0f19;
    color: #e2e8f0;
}
.gradio-container {
    font-family: 'Inter', sans-serif;
}
h1 {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
.primary {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    transition: all 0.3s ease;
}
.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
}
"""

with gr.Blocks(title="FDNFA Web Demo", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 FDNFA: Fully Differentiable Neural Forced Aligner")
    gr.Markdown("Upload an audio file and its corresponding transcript to predict phoneme or word boundaries with sub-frame precision using Soft Dynamic Programming.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Configuration")
            audio_in = gr.Audio(label="Audio File (.wav)", type="filepath")
            ann_in = gr.File(label="Annotation File (.phn, .wrd, or .txt)")
            ckpt_in = gr.File(label="Upload Checkpoint (.pt) [Optional]", file_types=[".pt"], type="filepath")
            mode_in = gr.Radio(choices=["phoneme", "word"], value="phoneme", label="Mode")
            lang_in = gr.Radio(choices=["english", "multilingual"], value="english", label="Language")
            w_phi_in = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="w_phi (Acoustic vs Linguistic feature weight)")
            btn = gr.Button("Run Alignment", variant="primary", size="lg")
            status_out = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("### Outputs & Visualizations")
            with gr.Tabs():
                with gr.Tab("Visualizations"):
                    bounds_img = gr.Image(label="Latent Features & Predicted Boundaries", show_download_button=True)
                    dp_img = gr.Image(label="Soft-DP Matrix Path", show_download_button=True)
                    probs_img = gr.Image(label="BiLSTM Probabilities", show_download_button=True)
                    logits_img = gr.Image(label="BiLSTM Logits", show_download_button=True)
                with gr.Tab("Alignment Data"):
                    audio_out = gr.Audio(label="Playback Audio", interactive=False)
                    tg_file_out = gr.File(label="Download TextGrid")
                    tg_table_out = gr.Dataframe(headers=["Start Time (s)", "End Time (s)", "Label"], label="Predicted Boundaries")
            
    btn.click(
        fn=run_alignment,
        inputs=[audio_in, ann_in, ckpt_in, mode_in, lang_in, w_phi_in],
        outputs=[status_out, audio_out, bounds_img, dp_img, probs_img, logits_img, tg_file_out, tg_table_out]
    )

    # Hidden shutdown button
    shutdown_btn = gr.Button("Shutdown", visible=False, elem_id="shutdown-btn")
    def shutdown():
        os._exit(0)
    shutdown_btn.click(fn=shutdown, inputs=[], outputs=[])

    # Inject JavaScript to click the hidden button when the window closes
    js_close = """
    function() {
        window.addEventListener('beforeunload', function (e) {
            const btn = document.querySelector('#shutdown-btn');
            if (btn) {
                btn.click();
            }
        });
    }
    """
    demo.load(None, None, None, js=js_close)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
