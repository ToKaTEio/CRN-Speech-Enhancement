import torch
import numpy as np
from train import Config
from utils.stft import STFTWrapper
from utils.utils import load_audio, mix_at_snr



def validate(model, cfg, val_loader, device):
    model.eval()
    stft = STFTWrapper(n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length)
    sdr_list = []
    stoi_list = []
    with torch.no_grad():
        for mag_n, mag_c, phase, clean_wav, noisy_wav in val_loader:
            mag_n = mag_n.to(device)
            phase = phase.to(device)
            clean_wav = clean_wav.to(device)

            mask = model(mag_n)
            mag_n_lin = torch.expm1(mag_n)
            est_mag_lin = (mask.squeeze(1) * mag_n_lin)
            est_wave = stft.istft(est_mag_lin, phase)
            # compare per example
            B = est_wave.shape[0]
            for b in range(B):
                est = est_wave[b].cpu().numpy()
                ref = clean_wav[b].cpu().numpy()
                # trim/pad
                L = min(len(est), len(ref))
                est = est[:L]
                ref = ref[:L]
                sdr_list.append(si_sdr(est, ref))

    mean_sdr = float(np.mean(sdr_list)) if len(sdr_list)>0 else 0.0
    mean_stoi = float(np.mean(stoi_list)) if len(stoi_list)>0 else 0.0
    print(f"Validation SI-SDR: {mean_sdr:.3f} dB, STOI: {mean_stoi:.3f}")
    return mean_sdr, mean_stoi