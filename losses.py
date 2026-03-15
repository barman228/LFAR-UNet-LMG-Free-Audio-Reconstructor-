import torch
import torch.nn as nn

class STFTLoss(nn.Module):
    def __init__(self, n_fft=1024, hop=256, win=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.window = torch.hann_window(win)

    def forward(self, pred, target):
        # pred и target: (batch, samples)
        if pred.dim() == 2:
            pred = pred
            target = target
        else:
            pred = pred.squeeze(1)
            target = target.squeeze(1)

        window = self.window.to(pred.device)

        pred_stft = torch.stft(
            pred,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=window,
            return_complex=True
        ).abs()

        targ_stft = torch.stft(
            target,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=window,
            return_complex=True
        ).abs()

        return torch.mean(torch.abs(pred_stft - targ_stft))