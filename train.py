import os, torch, gc, soundfile as sf, torchaudio, shutil
import numpy as np
from torch.utils.data import DataLoader
from dataset import AudioDataset
from model_unet import UNet
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
DEVICE = "cpu"
CHANNELS = 64
LR = 1e-5
N_FFT, HOP = 2048, 512
SAVE_EVERY = 5 # Эпох

def extreme_vch_loss(pred, target, device):
    p_mag = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2 + 1e-8)
    t_mag = torch.sqrt(target[:, 0]**2 + target[:, 1]**2 + 1e-8)
    
    freq_size = p_mag.shape[1]
    
    # Маска: 0 на НЧ, экспоненциальный взлет к ВЧ (x500 на верхах)
    weights = torch.linspace(0, 1, steps=freq_size).to(device)
    weights = torch.pow(weights, 4) * 500
    
    # Почти полный игнор всего, что ниже ~10кГц (bin 512 при 2048 FFT)
    weights[:freq_size // 2] *= 0.05
    weights = weights.view(1, -1, 1)
    
    # Спектральный лосс + текстурный (разница между соседними кадрами)
    mag_loss = (torch.abs(p_mag - t_mag) * weights).mean()
    
    p_diff = p_mag[:, :, 1:] - p_mag[:, :, :-1]
    t_diff = t_mag[:, :, 1:] - t_mag[:, :, :-1]
    str_loss = (torch.abs(p_diff - t_diff) * weights).mean()
    
    # Ослабленный L1 на фазу
    l1_phase = torch.abs(pred - target).mean() * 0.1
    
    return l1_phase + mag_loss + str_loss

def save_val(model, epoch):
    model.eval()
    test_path = "test_sample.wav"
    if not os.path.exists(test_path): return
    try:
        data, sr = sf.read(test_path)
        wav = torch.from_numpy(data).float()
        if wav.ndim > 1: wav = wav.mean(dim=-1)
        wav = wav.unsqueeze(0)
        if sr != 44100: wav = torchaudio.transforms.Resample(sr, 44100)(wav)
        
        # Dithering для теста
        wav_n = wav + torch.randn_like(wav) * 1e-3
        
        window = torch.hann_window(N_FFT).to(DEVICE)
        spec = torch.stft(wav_n.to(DEVICE), n_fft=N_FFT, hop_length=HOP, window=window, return_complex=True)
        spec_ri = torch.stack([spec.real, spec.imag], dim=1)
        
        with torch.no_grad():
            pred_ri = model(spec_ri)
            
        res_complex = torch.complex(pred_ri[0, 0], pred_ri[0, 1])
        audio_out = torch.istft(res_complex, n_fft=N_FFT, hop_length=HOP, window=window)
        
        out_np = audio_out.cpu().numpy()
        out_np = out_np / (np.max(np.abs(out_np)) + 1e-6)
        sf.write(f"output/val_epoch_{epoch}.wav", out_np, 44100)
        print(f"\n[+] Тест сохранен: val_epoch_{epoch}.wav")
    except Exception as e: print(f"Err in val: {e}")

if __name__ == "__main__":
    os.makedirs("output/backups", exist_ok=True)
    dataset = AudioDataset("data", n_fft=N_FFT, hop=HOP)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = UNet(ch=CHANNELS).to(DEVICE)
    
    # подсказка: Если модель "тупила", удали старый файл или начни заново
    model_path = "output/model_latest.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(">>> Продолжаем обучение...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    
    for epoch in range(1, 2001):
        model.train()
        pbar = tqdm(loader, desc=f"Ep {epoch}")
        for low, high in pbar:
            # шумовой холст
            low_n = low + torch.randn_like(low) * 2e-3
            
            pred = model(low_n.to(DEVICE))
            loss = extreme_vch_loss(pred, high.to(DEVICE), DEVICE)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            pbar.set_description(f"Loss: {loss.item():.4f}")

        if epoch % SAVE_EVERY == 0:
            # Сохранение основного файла
            torch.save(model.state_dict(), model_path)
            # Бекап (храним копию эпохи)
            shutil.copy(model_path, f"output/backups/model_ep_{epoch}.pth")
            # Генерация теста
            save_val(model, epoch)
            
        gc.collect()