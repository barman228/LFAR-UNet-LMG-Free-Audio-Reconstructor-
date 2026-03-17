import torch
import torchaudio
import soundfile as sf
import numpy as np
from model_unet import UNet
from tqdm import tqdm

# --- НАСТРОЙКИ (Должны совпадать с train) ---
DEVICE = "cpu"
CHANNELS = 64
N_FFT = 2048
HOP = 512
CHUNK_SIZE = 44100 * 1  # стандарт - 5 секунд
OVERLAP = 44100 // 2     # стандарт - 1 секунда

def upscale_long_audio(model_path, input_wav, output_wav):
    model = UNet(ch=CHANNELS).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f">>> Модель {model_path} загружена.")
    except Exception as e:
        print(f"Ошибка загрузки весов: {e}")
        return

    waveform, sr = sf.read(input_wav)
    if waveform.ndim > 1: waveform = waveform.mean(axis=-1)
    
    if sr != 44100:
        print(f">>> Ресемплинг {sr} -> 44100...")
        wav_t = torch.from_numpy(waveform).float().unsqueeze(0)
        wav_t = torchaudio.transforms.Resample(sr, 44100)(wav_t)
        waveform = wav_t.squeeze(0).numpy()

    total_samples = len(waveform)
    output_audio = np.zeros(total_samples)
    weight_mask = np.zeros(total_samples)

    # маска кроссфейда
    fade_len = OVERLAP
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    chunk_mask = np.ones(CHUNK_SIZE)
    chunk_mask[:fade_len] = fade_in
    chunk_mask[-fade_len:] = fade_out

    print(f">>> Обработка {total_samples} сэмплов сегментами по {CHUNK_SIZE}...")
    window = torch.hann_window(N_FFT).to(DEVICE)

    # Размер куска минус нахлест
    step = CHUNK_SIZE - OVERLAP
    
    for start in tqdm(range(0, total_samples, step)):
        end = start + CHUNK_SIZE
        
        current_chunk = waveform[start:min(end, total_samples)]
        actual_input_len = len(current_chunk)
        
        # Паддинг до полного CHUNK_SIZE, чтобы UNet не ругалась на размер
        if actual_input_len < CHUNK_SIZE:
            current_chunk = np.pad(current_chunk, (0, CHUNK_SIZE - actual_input_len))

        x = torch.from_numpy(current_chunk).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Преобразование в спектрограмму
            spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, window=window, 
                              return_complex=True, center=True)
            spec_ri = torch.stack([spec.real, spec.imag], dim=1)
            
            # Прогон через нейронку
            pred_ri = model(spec_ri)
            
            # Обратное преобразование
            res_complex = torch.complex(pred_ri[0, 0], pred_ri[0, 1])
            res_wav = torch.istft(res_complex, n_fft=N_FFT, hop_length=HOP, 
                                  window=window, center=True)
            res_np = res_wav.cpu().numpy()

        # --- КРИТИЧЕСКАЯ СИНХРОНИЗАЦИЯ РАЗМЕРОВ ---
        # Вычисляем, сколько реально места осталось до конца файла
        space_left = total_samples - start
        
        safe_len = min(len(res_np), CHUNK_SIZE, space_left, actual_input_len)
        
        # Обрезаем кусок результата и маску под этот безопасный размер
        final_chunk = res_np[:safe_len]
        final_mask = chunk_mask[:safe_len]
        
        # Безопасное сложение
        output_audio[start : start + safe_len] += final_chunk * final_mask
        weight_mask[start : start + safe_len] += final_mask

    print(">>> Финализация и сохранение...")
    output_audio /= (weight_mask + 1e-8)
    
    # Нормализация громкости (peak normalize)
    max_val = np.max(np.abs(output_audio))
    if max_val > 0:
        output_audio = output_audio / (max_val + 1e-6)
    
    sf.write(output_wav, output_audio, 44100)
    print(f"[+] Готово! Файл: {output_wav}")

if __name__ == "__main__":
    upscale_long_audio("output/model_latest.pth", "test_input.wav", "restored_full.wav")