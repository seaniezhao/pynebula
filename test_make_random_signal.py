"""
Test script for make_random_signal function.

This script tests whether the make_random_signal function correctly generates
signals with the specified fundamental frequencies by analyzing the frequency spectrum.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from make_random_dataset import make_random_signal
import pyworld as pw

def test_fundamental_frequency(normalized_f0, fs=16000, duration=1.0, noise_level=0.1):
    """
    Test if make_random_signal correctly generates a signal with the specified F0.
    
    Args:
        normalized_f0 (float): Normalized fundamental frequency (F0/fs)
        fs (int): Sampling rate in Hz
        duration (float): Duration of signal in seconds
        noise_level (float): Noise amplitude factor
        
    Returns:
        dict: Results including detected F0s and evaluation metrics
    """
    # Calculate actual F0 in Hz
    actual_f0_hz = normalized_f0 * fs
    print(f"Testing with normalized F0: {normalized_f0:.6f} (= {actual_f0_hz:.1f} Hz)")
    
    # Generate signal
    n_samples = int(duration * fs)
    signal, naxis = make_random_signal(normalized_f0, n_samples, noise_level)
    
    # Time domain analysis
    t = np.arange(n_samples) / fs
    
    # Frequency domain analysis
    n_fft = min(8192, n_samples)  # Use reasonably sized FFT
    freqs = np.fft.fftfreq(n_fft, 1/fs)[:n_fft//2]
    fft_spectrum = np.abs(fft(signal[:n_fft]))[:n_fft//2]
    
    # Find peaks in FFT spectrum
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(fft_spectrum, height=np.max(fft_spectrum)/10)
    peak_freqs = freqs[peaks]
    peak_mags = fft_spectrum[peaks]
    
    # Sort peaks by magnitude
    sorted_indices = np.argsort(peak_mags)[::-1]
    peak_freqs = peak_freqs[sorted_indices]
    peak_mags = peak_mags[sorted_indices]
    
    # Estimate F0 using PyWorld for comparison
    signal_double = signal.astype(np.float64)
    _f0_pw, t_pw = pw.dio(signal_double, fs, frame_period=5.0)
    f0_pw = pw.stonemask(signal_double, _f0_pw, t_pw, fs)
    f0_pw_mean = np.mean(f0_pw[f0_pw > 0]) if np.any(f0_pw > 0) else 0
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot signal in time domain
    plt.subplot(3, 1, 1)
    plt.plot(t, signal)
    plt.title(f"Random Signal (F0 = {actual_f0_hz:.1f} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Plot FFT spectrum
    plt.subplot(3, 1, 2)
    plt.plot(freqs, fft_spectrum)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    
    # Mark expected harmonics
    harmonics = np.arange(1, 6) * actual_f0_hz
    for i, h in enumerate(harmonics):
        plt.axvline(x=h, color='r', linestyle='--', 
                   label=f"Expected Harmonic {i+1}: {h:.1f} Hz" if i == 0 else None)
    
    # Mark detected peaks
    for i, (f, m) in enumerate(zip(peak_freqs[:5], peak_mags[:5])):
        plt.plot(f, m, 'go', markersize=8)
        plt.text(f, m, f"{f:.1f} Hz", fontsize=9)
    
    plt.legend()
    
    # Zoom in to show fundamental frequency region
    max_freq = min(1000, fs/2)
    plt.xlim(0, max_freq)
    
    # Plot PyWorld F0 estimation
    plt.subplot(3, 1, 3)
    t_pw_full = np.linspace(0, duration, len(f0_pw))
    plt.plot(t_pw_full, f0_pw, 'g-', label='PyWorld F0')
    plt.axhline(y=actual_f0_hz, color='r', linestyle='--', label=f"Expected F0: {actual_f0_hz:.1f} Hz")
    plt.title("PyWorld F0 Estimation")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"random_signal_test_f0_{actual_f0_hz:.1f}Hz.png")
    
    # Print results
    print(f"Expected F0: {actual_f0_hz:.1f} Hz")
    print(f"PyWorld estimated F0: {f0_pw_mean:.1f} Hz")
    print(f"Top 5 detected peaks in spectrum: {peak_freqs[:5]}")
    
    # Calculate error metrics
    if f0_pw_mean > 0:
        f0_error = np.abs(f0_pw_mean - actual_f0_hz)
        f0_error_percent = (f0_error / actual_f0_hz) * 100
        print(f"F0 absolute error: {f0_error:.1f} Hz")
        print(f"F0 percent error: {f0_error_percent:.2f}%")
    
    # Return results
    results = {
        "normalized_f0": normalized_f0,
        "actual_f0_hz": actual_f0_hz,
        "estimated_f0_pw": f0_pw_mean,
        "peak_frequencies": peak_freqs[:5].tolist(),
        "signal": signal,
        "spectrum": fft_spectrum,
        "frequencies": freqs
    }
    
    return results

def main():
    """
    Test the make_random_signal function with various fundamental frequencies.
    """
    fs = 16000  # Sampling rate in Hz
    
    # Test with different normalized F0 values
    normalized_f0_values = [
        0.005,    # 80 Hz at 16kHz sampling rate
        0.0075,   # 120 Hz at 16kHz sampling rate 
        0.01,     # 160 Hz at 16kHz sampling rate
        0.015,    # 240 Hz at 16kHz sampling rate
        0.02      # 320 Hz at 16kHz sampling rate
    ]
    
    results = {}
    
    for norm_f0 in normalized_f0_values:
        print("\n" + "="*50)
        results[norm_f0] = test_fundamental_frequency(norm_f0, fs=fs, duration=1.0, noise_level=0.1)
        plt.close()  # Close the figure to avoid memory issues
    
    # Summary
    print("\n" + "="*50)
    print("Summary of F0 Testing Results:")
    print("="*50)
    print(f"{'Normalized F0':<15} {'Expected F0 (Hz)':<20} {'PyWorld F0 (Hz)':<20} {'Error (%)':<10}")
    print("-"*65)
    
    for norm_f0, res in results.items():
        expected_f0 = res["actual_f0_hz"]
        estimated_f0 = res["estimated_f0_pw"]
        error_percent = abs(estimated_f0 - expected_f0) / expected_f0 * 100 if expected_f0 > 0 else float('nan')
        print(f"{norm_f0:<15.6f} {expected_f0:<20.1f} {estimated_f0:<20.1f} {error_percent:<10.2f}")

if __name__ == "__main__":
    main()
