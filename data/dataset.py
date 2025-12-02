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
from train import Config
from utils.stft import STFTWrapper
from utils.utils import load_audio, mix_at_snr


class OnlineSpeechEnhancementDataset(Dataset):
    def __init__(self, pairs_csv: str, cfg: Config, sr=16000, max_seconds=4.0, snr_low=-5, snr_high=15, rir_list: List[str]=None):
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