import argparse
import torch
# from taco_inference import get_mel_outputs
import numpy as np
from scipy.io.wavfile import write
from glow import WaveGlow
import json
import librosa

class waveglow_conf():
    def __init__(self):
        self.n_mel_channels = 80
        self.n_flows = 12
        self.n_group = 8
        self.n_early_every = 4
        self.n_early_size = 2
        # self.WN_config = WN_conf
    def get_conf():
        with open("./waveglow_master/config.json") as f:
            data = f.read()
        config = json.loads(data)
        return config["waveglow_config"]

def write_wav(mel, file_name):
    # mel_outputs = get_mel_outputs(texts) # grasping mel_outputs
    mel = librosa.db_to_power(mel)
    mel_outputs = torch.tensor([mel])
    # print(mel_outputs.shape)
    checkpoint_path = "C:/test/models/waveglow_256channels_universal_v5.pt" # grasping waveglow model
    waveglow = torch.load(checkpoint_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.eval()
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs) # inferencing mel_outputs w/ waveglow
    audio_numpy = audio[0].data.cpu().numpy()
    rate = 22050
    write(file_name, rate, audio_numpy) # writing .wav file to directory
    print("saved at {}".format(file_name))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('texts')
    args = parser.parse_args()
    texts = args.texts
    mel_outputs = get_mel_outputs(texts)
    checkpoint_path = "C:/test/models/waveglow_256channels_universal_v5.pt"
    waveglow = torch.load(checkpoint_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.eval()
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs)
    audio_numpy = audio[0].data.cpu().numpy()
    rate = 22050
    write("audio.wav", rate, audio_numpy)
    print("done.")
