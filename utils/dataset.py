import yaml
import os
import glob

import torchaudio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


from .encoder import *
from .transformer import *


### data partition에 따른 xml path list를 load함.
def xml_path_loader(xml_folder, partition):
    xml_path = []
    for folder in os.listdir(xml_folder):
        for x in partition:
            if (x == folder):
                path = os.path.join(xml_folder, folder)
                xml_path.extend(glob.glob('{}/*.xml'.format(path)))
                xml_path.sort()

    return xml_path #list


### xml path를 이용하여 audio file 출력
def audio_path_loader(xml_path):
    filename = os.path.splitext(os.path.basename(xml_path))[0]
    audio_path = os.path.join(os.path.dirname(xml_path), 'v2', filename + '_1.wav')
    return audio_path

# audio 처리 관련 함수
# 함수 세 개 모두 dcase 참고
def pad_audio(audio, target_len, fs):
    
    if len(audio) < target_len:
        audio = torch.nn.functional.pad(
            audio, (0, target_len - audio.shape[-1]), mode="constant"
        )
        padded_indx = [target_len / len(audio)]
        onset_s = 0.000
    
    elif len(audio) > target_len:        # 여기를 수정함
        audio = audio[:target_len]
        onset_s = 0.000

        padded_indx = [target_len / len(audio)] 
    else:

        onset_s = 0.000
        padded_indx = [1.0]

    offset_s = round(onset_s + (target_len / fs), 3)
    return audio, onset_s, offset_s, padded_indx

def to_mono(mixture, random_ch=False):

    if mixture.ndim > 1:  # multi channel
        if not random_ch:
            mixture = torch.mean(mixture, 0)
        else:  # randomly select one channel
            indx = np.random.randint(0, mixture.shape[0] - 1)
            mixture = mixture[indx]
    return mixture

def read_audio(audio_path, pad_to, resample_rate):

    audio, sample_rate = torchaudio.load(audio_path)

    # downsampling(sample_rate : 32000 -> fs : 16000)
    if sample_rate != resample_rate:
        audio, sample_rate = resampler(audio, sample_rate, resample_rate) # audio_data = [channel, signal], channel = 1(mono)

    # multi src -> mono
    audio = to_mono(audio)

    audio_pad, _, _, _ = pad_audio(audio, pad_to, sample_rate)

    return audio_pad, sample_rate    

# Dataset configuration

class mivia(Dataset):
    def __init__(self, audio_folder, encoder, transform = None, target_transform = None, partition = ['A'], pad_to = 60, fs = 16000):
        self.audio_folder = audio_folder # xml_folder: audio_folder

        self.resample_rate = fs # resampling rate
        self.pad_to = pad_to * self.resample_rate # unit of pad_to : [second]

        self.encoder = encoder
        self.transform = transform
        self.target_transform = target_transform
        self.partition = partition

        self.xml_path = xml_path_loader(self.audio_folder, self.partition) # 모든 xml file의 directory를 load

    def __len__(self):
        return len(self.xml_path)  #len(xml_path) = 57

    def __getitem__(self, idx):

        target = self.encoder.encoder_strong_label(self.xml_path[idx]) # label load(xml file -> numpy)

        audio_path = audio_path_loader(self.xml_path[idx])
        audio_data, _ = read_audio(audio_path, self.pad_to, self.resample_rate) # data load, fs = 32000(default)

        if self.transform: #data transform
            # transform. use this function to extract features in dataloader
            spectrogram = self.transform
            audio_data = spectrogram(audio_data) # [freq, frame]

        return audio_data, target