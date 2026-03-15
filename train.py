import os
import torch
import gc
import soundfile as sf
import torchaudio
import librosa
from torch.utils.data import DataLoader
from dataset import AudioDataset
from model_unet import UNet
from tqdm import tqdm

# --- НАСТРОЙКИ ---
DEVICE = "cpu"
CHANNELS = 32
LR = 1e-4
BACKUP_EVERY = 50
START_EPOCH_NUM = 0 # Пока 0 потому что билд пон да

def log_loss(pred, target):
    return torch.log(torch.abs(pred - target) + 1e-5).mean()

def save_validation_audio(model, device, epoch):
    model.eval()
    test_path = "test_sample.wav"
    if not os.path.exists(test_path):
        return
    with torch.no_grad():
        try:
            waveform, sr = sf.read(test_path)
            waveform = torch.from_numpy(waveform).float()
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
            else: waveform = waveform.T
            if sr != 44100:
                waveform = torchaudio.transforms.Resample(sr, 44100)(waveform)
            n_fft, hop = 1024, 256
            window = torch.hann_window(n_fft).to(device)
            spec = torch.stft(waveform.to(device), n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
            mag, phase = spec.abs(), spec.angle()
            input_mag = mag.unsqueeze(1) if mag.dim() == 3 else mag.unsqueeze(0).unsqueeze(1)
            w = (input_mag.shape[-1] // 16) * 16
            input_mag = input_mag[..., :w]
            pred_mag = model(input_mag)
            current_phase = phase[..., :w]
            res_spec = torch.polar(pred_mag.squeeze(0).squeeze(0), current_phase)
            res_wav = torch.istft(res_spec, n_fft=n_fft, hop_length=hop, window=window)
            out_path = f"output/val_epoch_{epoch+1}.wav"
            sf.write(out_path, res_wav.squeeze().cpu().numpy(), 44100)
            print(f"\n [+] Валидация сохранена: {out_path}")
        except Exception as e:
            print(f"\n [!] Ошибка валидации: {e}")

if __name__ == "__main__":
    os.makedirs("output/backups", exist_ok=True)
    
    dataset = AudioDataset("data", target_sr=44100)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = UNet(ch=CHANNELS).to(DEVICE)
    
    weights_path = "output/model_latest.pth"
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            print(f" -> Веса успешно загружены из {weights_path}")
        except Exception as e:
            print(f" [!] Ошибка загрузки: {e}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Старт. Архитектура ch={CHANNELS}")

    for epoch in range(500):
        model.train()
        total_epoch_idx = START_EPOCH_NUM + epoch + 1
        pbar = tqdm(loader, desc=f"Epoch {total_epoch_idx}")
        
        for low_spec, high_spec in pbar:
            if low_spec.dim() == 3: low_spec = low_spec.unsqueeze(1)
            if high_spec.dim() == 3: high_spec = high_spec.unsqueeze(1)
            
            low_spec = low_spec.to(DEVICE)
            high_spec = high_spec.to(DEVICE)

            pred_spec = model(low_spec)
            
            loss_mse = torch.nn.functional.mse_loss(pred_spec, high_spec)
            loss_l = log_loss(pred_spec, high_spec)
            
            loss_total = loss_mse + 0.2 * loss_l 

            opt.zero_grad()
            loss_total.backward()
            opt.step()
            
            pbar.set_description(f"Epoch {total_epoch_idx} Loss: {loss_total.item():.6f}")

        # Сохранения
        torch.save(model.state_dict(), "output/model_latest.pth")
        
        # бэкап
        if total_epoch_idx % BACKUP_EVERY == 0:
            b_path = f"output/backups/model_epoch_{total_epoch_idx}.pth"
            torch.save(model.state_dict(), b_path)
            print(f" [!] Бэкап создан: {b_path}")

        # Каждые 5 эпох тест
        if (epoch + 1) % 5 == 0:
            save_validation_audio(model, DEVICE, total_epoch_idx)
        
        gc.collect()