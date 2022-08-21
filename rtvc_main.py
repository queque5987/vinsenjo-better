import os
import torch
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

import waveglow_inference as waveglow
class rtvc_args():
    def __init__(self):
        self.enc_model_fpath = Path("saved_models/default/encoder.pt")
        self.syn_model_fpath = Path("saved_models/default/synthesizer.pt")
        self.voc_model_fpath = Path("saved_models/default/vocoder.pt")
        self.cpu = True
        self.seed = None
    def pop(self, idx):
        if idx == "cpu":
            return self.cpu

if __name__ == "__main__":
    args = rtvc_args()
    if args.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    # ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)
    in_fpath = Path("26-495-0000.wav")
    preprocessed_wav = encoder.preprocess_wav(in_fpath)
    # - If the wav is already loaded:
    original_wav, sampling_rate = librosa.load(str(in_fpath))
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    print("Loaded file succesfully")
    embed = encoder.embed_utterance(preprocessed_wav)
    print("Created the embedding")
    text = "this project is gone wrong we should consider abandon the project"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        synthesizer = Synthesizer(args.syn_model_fpath)

    # The synthesizer works in batch, so you need to put your data in a list or numpy array
    texts = [text]
    embeds = [embed]
    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print("Created the mel spectrogram")
    ## Generating the waveform
    print("Synthesizing the waveform:")

    # If seed is specified, reset torch seed and reload vocoder
    if args.seed is not None:
        torch.manual_seed(args.seed)
        vocoder.load_model(args.voc_model_fpath)

    """
    waveglow testing
    """
    # waveglow.write_wav(spec, "powered spec.wav")

    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # spectrogram, the more time-efficient the vocoder.
    
    generated_wav = vocoder.infer_waveform(spec)
    # generated_wav = synthesizer.griffin_lim(spec) #좃구림


    ## Post-generation
    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    generated_wav = encoder.preprocess_wav(generated_wav)
    
    filename = "demo_output_%02d.wav" % 2828
    print(generated_wav.dtype)
    sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    print("\nSaved output as %s\n\n" % filename)