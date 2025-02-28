"""
Calibration module for PyNebula F0 estimation system.

This module provides functions for:
1. Generating calibration data
2. Applying calibration to improve F0 estimation accuracy
3. Verifying the effect of calibration on F0 estimation
"""

import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pyworld as pw
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from tqdm import tqdm

from preprocess import preprocess_signal
from nebula_est import nebula_est
from load_model import load_model
from postprocess import postprocess_results

def estimate_f0_from_posterior(log_posterior, Lcal=None):
    """
    Map log_posterior values to F0 sequence using calibration data Lcal.
    
    Assumptions:
      - Lcal is a 1D array where its length represents the number of calibration points,
        each calibration point corresponds to a log likelihood value,
        and these points correspond to frequencies linearly distributed between [f0_min, f0_max].
      
    Parameters:
        log_posterior: Log likelihood values for each frame (array)
        Lcal: Calibration data (e.g., vector read by np.loadtxt)
        
    Returns:
        f0: Estimated fundamental frequency sequence (Hz)
    """
    num_frames = len(log_posterior)
    f0 = np.zeros(num_frames)
    f0_min = 30.0
    f0_max = 1100.0
    
    if Lcal is not None and Lcal.size > 0:
        # Assume Lcal is a 1D array, its length indicates calibration points
        num_cal_points = Lcal.shape[0]
        # Construct a frequency vector corresponding to calibration points, linearly distributed from f0_min to f0_max
        cal_freqs = np.linspace(f0_min, f0_max, num_cal_points)
        
        # If Lcal is not monotonically increasing, sort it first
        sorted_indices = np.argsort(Lcal)
        Lcal_sorted = Lcal[sorted_indices]
        cal_freqs_sorted = cal_freqs[sorted_indices]
        
        # For each frame, perform linear interpolation mapping based on log_posterior value
        for i in range(num_frames):
            # If log_posterior is less than the minimum value of calibration data, consider it unvoiced
            if log_posterior[i] < Lcal_sorted[0]:
                f0[i] = 0.0
            else:
                # Use np.interp for interpolation mapping
                f0[i] = np.interp(log_posterior[i], Lcal_sorted, cal_freqs_sorted)
    else:
        # If no calibration data, use quantile-based mapping
        threshold = np.percentile(log_posterior, 20)
        max_posterior = np.percentile(log_posterior, 95)
        for i in range(num_frames):
            if log_posterior[i] < threshold:
                f0[i] = 0.0
            else:
                normalized = (log_posterior[i] - threshold) / (max_posterior - threshold)
                normalized = np.clip(normalized, 0.0, 1.0)
                f0[i] = f0_min + normalized * (f0_max - f0_min)
    
    # Apply median filtering to smooth f0
    f0 = median_filter_f0(f0)
    return f0

def median_filter_f0(f0, window_size=5):
    """
    Apply median filtering to F0 sequence to smooth the F0 curve and remove noise and outliers
    """
    return medfilt(f0, kernel_size=window_size)

def generate_calibration_signals(num_signals=50, f0_min=50, f0_max=500, duration=0.5, fs=16000):
    """
    Generate synthetic signals with known F0 values for calibration.
    
    Parameters:
        num_signals: Number of calibration signals to generate
        f0_min: Minimum F0 value (Hz)
        f0_max: Maximum F0 value (Hz)
        duration: Duration of each signal (seconds)
        fs: Sampling rate (Hz)
        
    Returns:
        signals: List of synthetic signals
        f0_values: Array of ground truth F0 values
    """
    signals = []
    f0_values = np.linspace(f0_min, f0_max, num_signals)
    
    for f0 in f0_values:
        # Generate a sine wave with the given F0
        t = np.arange(0, duration, 1/fs)
        signal = np.sin(2 * np.pi * f0 * t)
        # Add some harmonics to make it more realistic
        for harmonic in range(2, 5):
            signal += (1/harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
        # Normalize
        signal = signal / np.max(np.abs(signal))
        signals.append(signal)
    
    return signals, f0_values

def process_calibration_signals(model, signals, f0_values, fs=16000, dt=0.005):
    """
    Process calibration signals and extract log posterior values for each F0.
    
    Parameters:
        model: Dictionary containing GMM models
        signals: List of synthetic signals
        f0_values: Array of ground truth F0 values
        fs: Sampling rate (Hz)
        dt: Time interval (seconds)
        
    Returns:
        log_posteriors: Average log posterior value for each signal
        Lcal_fit: Fitted calibration data
    """
    log_posteriors = []
    
    for signal in tqdm(signals, desc="Processing calibration signals"):
        # Preprocess signal and extract features
        features, _ = preprocess_signal(signal, fs, dt)
        
        # Calculate log-likelihood for each frequency band
        models = model.get("models", {})
        num_bands = len(models)
        lmap = np.zeros((features.shape[0], num_bands))
        lmap.fill(-100)  # Fill with low likelihood
        
        for b in range(num_bands):
            if b in models:
                gmm = models[b]
                lmap[:, b] = gmm.score_samples(features)
        
        # Average log posterior across frames and bands
        avg_log_posterior = np.mean(np.mean(lmap, axis=1))
        log_posteriors.append(avg_log_posterior)
    
    # Fit a smoothing function to the log posteriors for better calibration
    log_posteriors = np.array(log_posteriors)
    
    # Sort the data points by F0 value to ensure monotonicity
    sorted_indices = np.argsort(f0_values)
    f0_sorted = f0_values[sorted_indices]
    log_posteriors_sorted = log_posteriors[sorted_indices]
    
    # Fit a polynomial to smooth the calibration curve
    # Try different polynomial degrees and choose the best fit
    best_fit = None
    best_rmse = np.inf
    best_degree = 0
    
    for degree in range(2, 6):
        try:
            coeffs = np.polyfit(f0_sorted, log_posteriors_sorted, degree)
            poly_func = np.poly1d(coeffs)
            fitted_values = poly_func(f0_sorted)
            rmse = np.sqrt(np.mean((fitted_values - log_posteriors_sorted) ** 2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_fit = fitted_values
                best_degree = degree
        except:
            continue
    
    print(f"Best polynomial fit: degree {best_degree}, RMSE: {best_rmse:.6f}")
    
    # Generate final calibration data by sampling the fitted curve
    Lcal_fit = best_fit if best_fit is not None else log_posteriors_sorted
    
    return log_posteriors, Lcal_fit

def save_calibration_data(Lcal, save_dir, plot=True):
    """
    Save calibration data and optionally plot it.
    
    Parameters:
        Lcal: Calibration data
        save_dir: Directory to save the data
        plot: Whether to generate and save a plot
    """
    # Save as .npy file
    np.save(os.path.join(save_dir, "Lcal.npy"), Lcal)
    
    # Save as text file with index and frequency value
    with open(os.path.join(save_dir, "Lcal.txt"), "w") as f:
        for i, val in enumerate(Lcal):
            f.write(f"{i}: {val:.2f} Hz\n")
    
    if plot:
        # Plot the calibration curve
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(Lcal)), Lcal, 'b-', linewidth=2)
        plt.title("Calibration Curve")
        plt.xlabel("Index")
        plt.ylabel("Log Posterior Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Lcal_plot.png"))
        plt.close()

def verify_calibration(audio_file, model_dir="./model"):
    """
    Verify the calibration by estimating F0 for a test audio file.
    
    Parameters:
        audio_file: Path to the audio file to test
        model_dir: Directory containing the models and calibration data
        
    Returns:
        f0_nebula: F0 estimates from PyNebula
        f0_world: F0 estimates from WORLD (reference)
        time_axis: Time axis for the estimates
    """
    # Check if the audio file exists
    if not os.path.isfile(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        return None, None, None
    
    # Load the audio file
    x, fs = librosa.load(audio_file, sr=None)
    
    # Load the models
    model = load_model(model_dir)
    
    # Set time interval for F0 estimation
    dt = 0.005  # 5ms
    
    # Estimate F0 using PyNebula
    f0_nebula, time_axis, _, _ = nebula_est(model, x, fs, dt=dt, return_details=True)
    
    # Estimate F0 using WORLD for reference
    x_world = x.astype(np.float64)
    if np.max(np.abs(x_world)) > 1.0:
        x_world = x_world / np.max(np.abs(x_world))
    
    f0_world, time_world = pw.dio(x_world, fs, f0_floor=30.0, f0_ceil=1100.0, frame_period=dt*1000)
    f0_world = pw.stonemask(x_world, f0_world, time_world, fs)
    
    # Calculate comparison statistics
    min_len = min(len(f0_nebula), len(f0_world))
    f0_nebula_trunc = f0_nebula[:min_len]
    f0_world_trunc = f0_world[:min_len]
    
    voiced_mask = (f0_nebula_trunc > 0) & (f0_world_trunc > 0)
    if np.any(voiced_mask):
        mean_abs_diff = np.mean(np.abs(f0_nebula_trunc[voiced_mask] - f0_world_trunc[voiced_mask]))
        corr = np.corrcoef(f0_nebula_trunc[voiced_mask], f0_world_trunc[voiced_mask])[0, 1]
        
        print("\nCalibration verification results:")
        print(f"  - Mean absolute difference: {mean_abs_diff:.2f} Hz")
        print(f"  - Correlation coefficient: {corr:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, f0_nebula, 'b-', label='PyNebula F0', alpha=0.7)
    plt.plot(time_world, f0_world, 'r-', label='WORLD F0', alpha=0.7)
    plt.title("F0 Comparison: PyNebula vs WORLD")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim([0, min(1100, max(np.max(f0_nebula) * 1.1, np.max(f0_world) * 1.1))])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("calibration_verification.png")
    plt.close()
    
    return f0_nebula, f0_world, time_axis

def adaptive_calibration(model, signals, f0_values, fs=16000, dt=0.005):
    """
    Perform adaptive calibration using an iterative approach for better accuracy.
    
    This method uses multiple iterations to refine the calibration curve.
    
    Parameters:
        model: Dictionary containing GMM models
        signals: List of synthetic signals
        f0_values: Array of ground truth F0 values
        fs: Sampling rate (Hz)
        dt: Time interval (seconds)
        
    Returns:
        Lcal_adaptive: Adaptively calibrated data
    """
    # Initial processing
    log_posteriors, Lcal_initial = process_calibration_signals(model, signals, f0_values, fs, dt)
    
    # Sort data by F0 for monotonicity
    sorted_indices = np.argsort(f0_values)
    f0_sorted = f0_values[sorted_indices]
    log_posteriors_sorted = np.array(log_posteriors)[sorted_indices]
    
    # Create an inverse mapping function from log_posterior to F0
    # This is the core of adaptive calibration
    inverse_map = interp1d(
        log_posteriors_sorted, 
        f0_sorted, 
        kind='cubic', 
        bounds_error=False, 
        fill_value=(f0_sorted[0], f0_sorted[-1])
    )
    
    # Generate a dense sampling of the calibration curve
    num_cal_points = 36  # Same as number of bands
    log_posterior_range = np.linspace(
        min(log_posteriors_sorted), 
        max(log_posteriors_sorted), 
        num_cal_points
    )
    
    # Use the inverse map to get the corresponding F0 values
    Lcal_adaptive = inverse_map(log_posterior_range)
    
    return Lcal_adaptive

def main():
    """Main entry point for the calibration script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibration tools for PyNebula F0 estimation")
    parser.add_argument("--generate", action="store_true", help="Generate calibration data")
    parser.add_argument("--verify", type=str, help="Verify calibration using an audio file")
    parser.add_argument("--model_dir", type=str, default="./model", help="Directory with models")
    parser.add_argument("--num_signals", type=int, default=50, help="Number of calibration signals")
    parser.add_argument("--f0_min", type=int, default=50, help="Minimum F0 value (Hz)")
    parser.add_argument("--f0_max", type=int, default=500, help="Maximum F0 value (Hz)")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive calibration")
    
    args = parser.parse_args()
    
    if args.generate:
        print("Generating calibration data...")
        
        # Load the models
        model = load_model(args.model_dir)
        
        # Generate synthetic signals with known F0 values
        signals, f0_values = generate_calibration_signals(
            num_signals=args.num_signals,
            f0_min=args.f0_min,
            f0_max=args.f0_max
        )
        
        # Choose calibration method
        if args.adaptive:
            print("Using adaptive calibration method...")
            Lcal = adaptive_calibration(model, signals, f0_values)
        else:
            print("Using standard calibration method...")
            _, Lcal = process_calibration_signals(model, signals, f0_values)
        
        # Save calibration data
        save_calibration_data(Lcal, args.model_dir)
        print(f"Calibration data generated and saved to {args.model_dir}")
        
    elif args.verify:
        print(f"Verifying calibration using {args.verify}...")
        verify_calibration(args.verify, args.model_dir)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
