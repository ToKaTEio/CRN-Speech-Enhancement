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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-9)


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