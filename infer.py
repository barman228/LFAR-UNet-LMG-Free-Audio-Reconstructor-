import torch
import torchaudio
from model_unet import UNet

def upscale_audio(model_path, input_wav, output_wav, device="cpu"):

    model = UNet(ch=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    waveform, sr = torchaudio.load(input_wav)
    if sr != 44100:
        waveform = torchaudio.transforms.Resample(sr, 44100)(waveform)
    
    n_fft = 1024
    hop = 256
    window = torch.hann_window(n_fft).to(device)
    
    spec = torch.stft(
        waveform, 
        n_fft=n_fft, 
        hop_length=hop, 
        window=window, 
        return_complex=True
    )
    
    mag = spec.abs()    # Амплитуда
    phase = spec.angle() # Фаза 

    with torch.no_grad():
        # Добавляем размерность батча [1, 1, Freq, Time]
        input_tensor = mag.unsqueeze(0)
        w = (input_tensor.shape[-1] // 16) * 16
        input_tensor = input_tensor[..., :w]
        
        enhanced_mag = model(input_tensor)
    
    phase = phase[..., :w]
    enhanced_spec = torch.polar(enhanced_mag.squeeze(0), phase)
    
    resited_waveform = torch.istft(
        enhanced_spec, 
        n_fft=n_fft, 
        hop_length=hop, 
        window=window
    )

    # Сохраняем 
    torchaudio.save(output_wav, resited_waveform.cpu(), 44100)
    print(f"Готово! Результат сохранен в {output_wav}")

if __name__ == "__main__":

    upscale_audio("output/upscaler_ep20.pth", "test_input.wav", "restored_high_res.wav")