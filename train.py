import os
import sys
import argparse
import random
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import soundfile as sf
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

# Optional metrics
try:
    from pystoi import stoi
except Exception:
    stoi = None

# -------------------------------
# Config (simple dataclass)
# -------------------------------
@dataclass
class Config:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    batch_size: int = 16
    epochs: int = 60
    lr: float = 1e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_csv: str = 'data/train_pairs.csv'
    test_csv: str = 'data/test_pairs.csv'
    chkpt_dir: str = 'checkpoints'
    save_every: int = 1
    seed: int = 42
    snr_low: int = -5
    snr_high: int = 15
    max_audio_seconds: float = 4.0

cfg = Config()

# -------------------------------
# Utilities
# -------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)


# -------------------------------
# STFT wrapper
# -------------------------------
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


# -------------------------------
# Audio helpers
# -------------------------------

def load_audio(path, sr=16000, mono=True, max_seconds=None):
    wav, r = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if r != sr:
        wav = librosa.resample(wav, r, sr)
    if max_seconds is not None:
        max_len = int(sr * max_seconds)
        if len(wav) > max_len:
            wav = wav[:max_len]
    return wav.astype(np.float32)


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-9)


def mix_at_snr(clean, noise, snr_db):
    # both numpy arrays
    if len(noise) < len(clean):
        # tile noise
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(clean)]
    rms_c = rms(clean)
    rms_n = rms(noise)
    target_rms_n = rms_c / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (target_rms_n / (rms_n + 1e-9))
    return clean + scaled_noise


# -------------------------------
# Dataset (online mixing) VoiceBank-style
# -------------------------------
class OnlineSpeechEnhancementDataset(Dataset):
    def __init__(self, pairs_csv: str, sr=16000, max_seconds=4.0, snr_low=-5, snr_high=15, rir_list: List[str]=None):
        self.pairs = []
        # pairs_csv expected format: clean_path \t noise_path (or multiple noise paths?) \t optional snr
        with open(pairs_csv, 'r', encoding='utf8') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                parts=line.split(',')
                clean_path = parts[0]
                noise_path = parts[1] if len(parts) > 1 else None
                if len(parts) > 2:
                    try:
                        snr = float(parts[2])
                    except:
                        snr = None
                else:
                    snr = None
                self.pairs.append((clean_path, noise_path, snr))
        self.sr = sr
        self.max_seconds = max_seconds
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.rir_list = rir_list or []
        self.stft = STFTWrapper(n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_path, noise_path, fixed_snr = self.pairs[idx]
        clean = load_audio(clean_path, sr=self.sr, max_seconds=self.max_seconds)
        if noise_path is None or not os.path.exists(noise_path):
            # sample a random noise from dataset list
            raise ValueError('Noise path required per row in pairs CSV')
        noise = load_audio(noise_path, sr=self.sr, max_seconds=self.max_seconds)

        # choose snr
        snr = fixed_snr if fixed_snr is not None else random.uniform(self.snr_low, self.snr_high)
        noisy = mix_at_snr(clean, noise, snr)

        # pad/crop to same length
        L = min(len(clean), len(noisy))
        clean = clean[:L]
        noisy = noisy[:L]

        # to torch
        clean_t = torch.from_numpy(clean).float()
        noisy_t = torch.from_numpy(noisy).float()

        # stft
        mag_noisy, phase_noisy = self.stft.stft(noisy_t.unsqueeze(0))  # (1, F, T) complex handled
        mag_clean, _ = self.stft.stft(clean_t.unsqueeze(0))

        # squeeze batch dim
        mag_noisy = mag_noisy.squeeze(0)
        mag_clean = mag_clean.squeeze(0)
        phase_noisy = phase_noisy.squeeze(0)

        # convert to log1p to stabilize
        mag_noisy = torch.log1p(mag_noisy)
        mag_clean = torch.log1p(mag_clean)

        return mag_noisy, mag_clean, phase_noisy, clean_t, noisy_t


# -------------------------------
# Model: A compact CRN
# -------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), stride=(2,2), padding=(1,1)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class CRN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # encoder
        self.enc1 = ConvBlock(in_channels, 16)  # -> C=16
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        # bottleneck conv to reduce channel dim
        self.bottleneck = nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1))
        # GRU (we will flatten freq*channel as feature dimension)
        # freq after conv: n_fft/2+1 reduced by stride factors (2,2),(2,2),(2,2) on freq dim approx
        # We'll compute runtime
        # decoder
        self.dec3 = DeconvBlock(64, 32)
        self.dec2 = DeconvBlock(32, 16)
        self.dec1 = DeconvBlock(16, 8)
        self.out_conv = nn.Conv2d(8, 1, kernel_size=(1,1))
        self.sigmoid = nn.Sigmoid()

        self.gru = None  # will init lazily when input size known

    def forward(self, x):
        # x: (B, 1, F, T)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        # b: (B, C, F', T')
        B, C, Fp, Tp = b.shape
        # prepare for GRU: permute to (B, T', C*F')
        b_perm = b.permute(0, 3, 1, 2).contiguous()  # (B, T', C, F')
        feat = b_perm.view(B, Tp, C * Fp)
        # lazy init of GRU
        if self.gru is None:
            self.gru = nn.GRU(input_size=C*Fp, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True).to(x.device)
        g_out, _ = self.gru(feat)  # (B, T', hidden*2)
        # project back to shape
        proj = g_out.view(B, Tp, C, Fp).permute(0, 2, 3, 1).contiguous()  # (B, C, F', T')
        d3 = self.dec3(proj)
        d2 = self.dec2(d3 + e2)
        d1 = self.dec1(d2 + e1)
        out = self.out_conv(d1)
        mask = self.sigmoid(out)
        # output shape (B,1,F,T) - mask in linear domain to multiply with magnitude
        return mask

# -------------------------------
# Metrics: SI-SDR and helpers
# -------------------------------

def si_sdr(est, ref, eps=1e-8):
    # est, ref: numpy 1D arrays
    # compute SI-SDR in dB
    if isinstance(est, torch.Tensor):
        est = est.detach().cpu().numpy()
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    s_target = np.sum(ref * est) * ref / (np.sum(ref ** 2) + eps)
    e_noise = est - s_target
    ratio = np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + eps)
    return 10 * np.log10(ratio + eps)

# -------------------------------
# Training & Validation
# -------------------------------

def collate_fn(batch):
    # batch is list of tuples: mag_noisy, mag_clean, phase, clean_wav, noisy_wav
    # pad along time axis to max T in batch
    mags_noisy = [b[0] for b in batch]
    mags_clean = [b[1] for b in batch]
    phases = [b[2] for b in batch]
    clean_wavs = [b[3] for b in batch]
    noisy_wavs = [b[4] for b in batch]
    # find max T
    Ts = [m.shape[-1] for m in mags_noisy]
    T_max = max(Ts)
    F = mags_noisy[0].shape[-2]
    B = len(batch)
    mag_n_padded = torch.zeros(B, 1, F, T_max)
    mag_c_padded = torch.zeros(B, 1, F, T_max)
    phase_padded = torch.zeros(B, F, T_max)
    wav_clean_padded = torch.zeros(B, int(cfg.sample_rate * cfg.max_audio_seconds))
    wav_noisy_padded = torch.zeros_like(wav_clean_padded)
    for i,(mn,mc,ph,cw,nw) in enumerate(batch):
        T = mn.shape[-1]
        mag_n_padded[i,0,:,:T] = mn
        mag_c_padded[i,0,:,:T] = mc
        phase_padded[i,:,:T] = ph
        L = min(len(cw), wav_clean_padded.shape[-1])
        wav_clean_padded[i,:L] = cw[:L]
        wav_noisy_padded[i,:L] = nw[:L]
    return mag_n_padded, mag_c_padded, phase_padded, wav_clean_padded, wav_noisy_padded


def train_loop(model, optim, train_loader, epoch, device):
    model.train()
    running_loss = 0.0
    stft = STFTWrapper(n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length)
    for i,(mag_n, mag_c, phase, clean_wav, noisy_wav) in enumerate(train_loader):
        mag_n = mag_n.to(device)
        mag_c = mag_c.to(device)
        phase = phase.to(device)
        clean_wav = clean_wav.to(device)
        noisy_wav = noisy_wav.to(device)

        mask = model(mag_n)
        # pred magnitude in log domain: we used input as log1p(mag) so mask applies in linear mag: convert back
        mag_n_lin = torch.expm1(mag_n)
        est_mag_lin = (mask.squeeze(1) * mag_n_lin)
        est_mag = torch.log1p(est_mag_lin)

        loss_spec = F.l1_loss(est_mag, mag_c)

        # waveform loss (SI-SDR like) - compute istft, then SI-SDR
        est_wave = stft.istft(est_mag_lin, phase)
        est_wave = est_wave[:, :clean_wav.shape[1]] if est_wave.shape[1] >= clean_wav.shape[1] else F.pad(est_wave, (0, clean_wav.shape[1]-est_wave.shape[1]))
        # SI-SDR (batch)
        loss_wave = 0.0
        for b in range(est_wave.shape[0]):
            loss_wave += -si_sdr(est_wave[b].cpu().numpy(), clean_wav[b].cpu().numpy())
        loss_wave = loss_wave / est_wave.shape[0]
        loss = loss_spec + 0.01 * loss_wave

        optim.zero_grad()
        # loss_spec is torch scalar, loss_wave is numpy float; combine carefully
        total_loss = loss_spec + 0.01 * torch.tensor(loss_wave, dtype=loss_spec.dtype, device=loss_spec.device)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

        running_loss += total_loss.item()
        if (i+1) % 10 == 0:
            print(f"Epoch {epoch} Step {i+1}/{len(train_loader)} Loss {running_loss/(i+1):.4f}")
    return running_loss / len(train_loader)


def validate(model, val_loader, device):
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
                if stoi is not None:
                    try:
                        stoi_list.append(stoi(ref, est, cfg.sample_rate, extended=False))
                    except Exception:
                        pass
    mean_sdr = float(np.mean(sdr_list)) if len(sdr_list)>0 else 0.0
    mean_stoi = float(np.mean(stoi_list)) if len(stoi_list)>0 else 0.0
    print(f"Validation SI-SDR: {mean_sdr:.3f} dB, STOI: {mean_stoi:.3f}")
    return mean_sdr, mean_stoi

# -------------------------------
# CLI: train / validate / infer
# -------------------------------

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def train_command(args):
    device = cfg.device
    model = CRN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    train_ds = OnlineSpeechEnhancementDataset(cfg.train_csv, sr=cfg.sample_rate, max_seconds=cfg.max_audio_seconds, snr_low=cfg.snr_low, snr_high=cfg.snr_high)
    val_ds = OnlineSpeechEnhancementDataset(cfg.test_csv, sr=cfg.sample_rate, max_seconds=cfg.max_audio_seconds, snr_low=cfg.snr_low, snr_high=cfg.snr_high)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    ensure_dir(cfg.chkpt_dir)
    best_sdr = -1e9
    for epoch in range(1, cfg.epochs+1):
        train_loss = train_loop(model, optimizer, train_loader, epoch, device)
        print(f"Epoch {epoch} finished. Train loss {train_loss:.4f}")
        if epoch % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.chkpt_dir, f"crn_epoch{epoch}.pt")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, ckpt_path)
            print(f"Saved checkpoint {ckpt_path}")
        # validate
        mean_sdr, mean_stoi = validate(model, val_loader, device)
        if mean_sdr > best_sdr:
            best_sdr = mean_sdr
            best_path = os.path.join(cfg.chkpt_dir, f"crn_best.pt")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, best_path)
            print(f"Saved best checkpoint {best_path} (SI-SDR {best_sdr:.3f})")


def validate_command(args):
    device = cfg.device
    model = CRN().to(device)
    assert os.path.exists(args.ckpt), 'ckpt not found'
    d = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(d['model_state'])
    val_ds = OnlineSpeechEnhancementDataset(args.test_csv or cfg.test_csv, sr=cfg.sample_rate, max_seconds=cfg.max_audio_seconds, snr_low=cfg.snr_low, snr_high=cfg.snr_high)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    validate(model, val_loader, device)


def infer_command(args):
    device = cfg.device
    model = CRN().to(device)
    assert os.path.exists(args.ckpt), 'ckpt not found'
    d = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(d['model_state'])
    model.eval()
    stft = STFTWrapper(n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length)
    wav, sr = sf.read(args.input)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != cfg.sample_rate:
        wav = librosa.resample(wav, sr, cfg.sample_rate)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    mag, phase = stft.stft(wav_t)
    mag_log = torch.log1p(mag).unsqueeze(0).to(device)  # (1,1,F,T)
    with torch.no_grad():
        mask = model(mag_log)
        mag_lin = torch.expm1(mag_log.squeeze(0))
        est_mag_lin = (mask.squeeze(1).cpu() * mag_lin)
        est_wav = stft.istft(est_mag_lin, phase.squeeze(0))
        est_wav = est_wav.cpu().numpy()
        sf.write(args.output, est_wav, cfg.sample_rate)
    print(f"Wrote enhanced audio to {args.output}")


# -------------------------------
# Entry point
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRN Speech Enhancement Single-file Project')
    sub = parser.add_subparsers(dest='cmd')

    p_train = sub.add_parser('train')
    p_train.add_argument('--cfg', type=str, default=None)

    p_val = sub.add_parser('validate')
    p_val.add_argument('--ckpt', type=str, required=True)
    p_val.add_argument('--test_csv', type=str, default=None)

    p_inf = sub.add_parser('infer')
    p_inf.add_argument('--ckpt', type=str, required=True)
    p_inf.add_argument('--input', type=str, required=True)
    p_inf.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    if args.cmd == 'train':
        train_command(args)
    elif args.cmd == 'validate':
        validate_command(args)
    elif args.cmd == 'infer':
        infer_command(args)
    else:
        parser.print_help()
