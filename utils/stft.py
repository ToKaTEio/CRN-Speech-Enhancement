import torch
from typing import Tuple

class STFTWrapper:
    def __init__(self, n_fft=512, hop_length=128, win_length=None, window='hann'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = torch.hann_window(self.win_length)

    def stft(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # wav: (N, T)
        window = self.window.to(wav.device)
        spec = torch.stft(wav, self.n_fft, self.hop_length, self.win_length, window=window, return_complex=True)
        mag = spec.abs()
        phase = torch.angle(spec)
        return mag, phase

    def istft(self, mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # mag, phase: complex representation split
        complex_spec = torch.polar(mag, phase)
        window = self.window.to(mag.device)
        wav = torch.istft(complex_spec, self.n_fft, self.hop_length, self.win_length, window=window, length=None)
        return wav