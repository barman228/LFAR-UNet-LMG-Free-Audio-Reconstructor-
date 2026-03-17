import os
import torch
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, root, target_sr=44100, n_fft=2048, hop=512):
        self.low_dir, self.high_dir = os.path.join(root, "low"), os.path.join(root, "high")
        self.target_sr = target_sr
        # power=None заставляет Torchaudio возвращать КОМПЛЕКСНЫЙ тензор (с фазой)
        self.stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop, power=None)
        self.files = sorted([f for f in os.listdir(self.low_dir) if f.lower().endswith(".wav")])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        l_path, h_path = os.path.join(self.low_dir, name), os.path.join(self.high_dir, name)
        
        info = sf.info(l_path)
        chunk = int(info.samplerate * 2.0) # Берем куски по 2 секунды для лучшего контекста
        start = torch.randint(0, max(1, info.frames - chunk), (1,)).item()

        l_data, _ = sf.read(l_path, start=start, frames=chunk)
        h_data, _ = sf.read(h_path, start=start, frames=chunk)

        low = torch.from_numpy(l_data).float()
        high = torch.from_numpy(h_data).float()
        
        # Делаем моно
        if low.ndim > 1: low = low.mean(dim=1)
        if high.ndim > 1: high = high.mean(dim=1)

        if info.samplerate != self.target_sr:
            res = torchaudio.transforms.Resample(info.samplerate, self.target_sr)
            low, high = res(low), res(high)

        # Нормализуем саму волну ДО перевода в спектр, чтобы не сломать фазу
        low /= (low.abs().max() + 1e-8)
        high /= (high.abs().max() + 1e-8)

        spec_l = self.stft(low) # Комплексный тензор
        spec_h = self.stft(high)
        
        # Разбиваем комплексное число на Реальную и Мнимую части (2 канала)
        low_ri = torch.stack([spec_l.real, spec_l.imag], dim=0)
        high_ri = torch.stack([spec_h.real, spec_h.imag], dim=0)

        return low_ri, high_ri