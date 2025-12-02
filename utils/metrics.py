import torch
import numpy as np

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