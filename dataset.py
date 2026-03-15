import os
import torch
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, root, target_sr=44100, n_fft=1024, hop=256):
        self.low_dir = os.path.join(root, "low")
        self.high_dir = os.path.join(root, "high")
        self.target_sr = target_sr
        
        # Длина куска для обучения (1 секунда)
        self.chunk_len_sec = 1.0 
        
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, 
            hop_length=hop, 
            power=None 
        )

        # Список файлов
        self.files = sorted([
            f for f in os.listdir(self.low_dir)
            if f.lower().endswith(".wav") and os.path.isfile(os.path.join(self.high_dir, f))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        low_p = os.path.join(self.low_dir, filename)
        high_p = os.path.join(self.high_dir, filename)
        
        info = sf.info(low_p)
        sr = info.samplerate
        chunk_samples = int(self.chunk_len_sec * sr)
        
        # чтобы экономить RAM
        if info.frames > chunk_samples:
            start_frame = torch.randint(0, info.frames - chunk_samples, (1,)).item()
            low_data, _ = sf.read(low_p, start=start_frame, frames=chunk_samples)
            high_data, _ = sf.read(high_p, start=start_frame, frames=chunk_samples)
        else:
            low_data, _ = sf.read(low_p)
            high_data, _ = sf.read(high_p)

        # тензоры [Channels, Samples]
        low = torch.from_numpy(low_data).float()
        if low.ndim == 1: low = low.unsqueeze(0)
        else: low = low.T
            
        high = torch.from_numpy(high_data).float()
        if high.ndim == 1: high = high.unsqueeze(0)
        else: high = high.T

        # Ресэмпл до 44100
        if sr != self.target_sr:
            resample = torchaudio.transforms.Resample(sr, self.target_sr)
            low = resample(low)
            high = resample(high)

        # Выравниваем длину, чтобы UNet (с 4 пулингами) не ругался (кратность 16)
        min_len = min(low.size(-1), high.size(-1))
        min_len = (min_len // 16) * 16 
        low, high = low[..., :min_len], high[..., :min_len]

        low_spec = self.stft(low).abs()
        high_spec = self.stft(high).abs()

        return low_spec, high_spec