import os
import json
import numpy as np
import cv2
import sys

def detrend(signal):
    # subtract moving average to remove low-frequency mfg variation
    window = max(3, len(signal) // 4)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(signal, (pad, pad), mode='edge')
    mov_avg = np.convolve(padded, np.ones(window)/window, mode='valid')
    return signal - mov_avg

def naive_fft_period(image):
    signal = np.median(image, axis=0)
    signal = signal - np.mean(signal)
    fft_mag = np.abs(np.fft.rfft(signal))
    fft_mag[0] = 0 # ignore DC
    peak_idx = np.argmax(fft_mag)
    if peak_idx == 0:
        return 0
    period = len(signal) / peak_idx
    return period

def robust_autocorr_period(image):
    signal = np.median(image, axis=0)
    # Detrend using a simple linear fit to avoid messing up short signals
    x = np.arange(len(signal))
    p = np.polyfit(x, signal, 1)
    signal = signal - np.polyval(p, x)
    
    if np.var(signal) < 1e-6:
        return 0
        
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]
    
    # Normalize by overlap length
    overlap = np.arange(len(signal), 0, -1)
    corr = corr / overlap
    
    # Find peaks
    min_lag = max(3, int(len(signal) * 0.1))
    peaks = []
    for i in range(min_lag, len(corr) - 1):
        if corr[i] > corr[i-1] and corr[i] > corr[i+1]:
            peaks.append((i, corr[i]))
            
    if not peaks:
        return 0
        
    # The fundamental is the first prominent peak
    max_peak_val = max(p[1] for p in peaks)
    for idx, val in peaks:
        if val > 0.3 * max_peak_val:
            return float(idx)
    return 0

def run_verification(dataset_dir="phase_b_test"):
    samples_dir = os.path.join(dataset_dir, "eval")
    if not os.path.exists(samples_dir):
        print(f"Directory {samples_dir} not found.")
        return

    sample_dirs = sorted([d for d in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, d))])
    
    success_count = 0
    total_count = 0
    
    print(f"{'Sample':<12} | {'Naive (Search)':<15} | {'Robust (Search)':<15} | {'Naive (Ref)':<15} | {'Robust (Ref)':<15} | {'Ratio':<10} | {'Expected':<10} | {'Ref StdDev':<10}")
    print("-" * 115)
    
    for sample in sample_dirs:
        s_dir = os.path.join(samples_dir, sample)
        with open(os.path.join(s_dir, "ground_truth.json")) as f:
            gt = json.load(f)
            
        ref = cv2.imread(os.path.join(s_dir, "reference.png"), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(s_dir, "search.png"), cv2.IMREAD_GRAYSCALE)
        
        tx = gt["gt_top_left_x"]
        ty = gt["gt_top_left_y"]
        tw = gt["scaled_width"]
        th = gt["scaled_height"]
        
        footprint = search[ty:ty+th, tx:tx+tw]
        
        naive_s = naive_fft_period(footprint)
        robust_s = robust_autocorr_period(footprint)
        
        naive_r = naive_fft_period(ref)
        robust_r = robust_autocorr_period(ref)
        
        ratio = robust_s / robust_r if robust_r > 0 else 0
        expected = 1.0 / gt["scale"]
        ref_std = np.std(ref)
        
        print(f"{sample:<12} | {naive_s:<15.2f} | {robust_s:<15.2f} | {naive_r:<15.2f} | {robust_r:<15.2f} | {ratio:<10.4f} | {expected:<10.4f} | {ref_std:<10.2f}")
        
        if abs(ratio - expected) < 0.04:
            success_count += 1
        total_count += 1
        
    print(f"\nRatio check passed: {success_count}/{total_count}")

if __name__ == "__main__":
    run_verification()
