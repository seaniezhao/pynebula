"""
Demo script for the PyNebula F0 estimation algorithm.
Shows how to load a model, estimate F0 from an audio file, and visualize the results.
"""
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import argparse
from nebula_est import nebula_est
from load_model import load_model
from make_random_dataset import make_random_signal

def main(audio_file=None, model_dir='./model', output_file=None):
    """
    Run a demonstration of the PyNebula F0 estimation algorithm.
    
    If an audio file is provided, it will be used for F0 estimation.
    Otherwise, a synthetic signal will be generated.
    
    Parameters:
        audio_file: Path to an audio file (WAV, MP3, etc.)
        model_dir: Directory containing pre-trained models
        output_file: If provided, the estimated F0 will be saved to this file
    """
    # Check if models exist, if not, generate example models
    if not os.path.isdir(model_dir) or len(os.listdir(model_dir)) == 0:
        print("No pre-trained models found. Generating example models...")
        from train_gmm import train_and_save_models
        train_and_save_models(model_dir=model_dir)
    
    # Load the pre-trained models
    print(f"Loading models from {model_dir}...")
    model = load_model(model_dir)
    
    if audio_file and os.path.isfile(audio_file):
        # Load the audio file
        print(f"Loading audio file: {audio_file}")
        x, fs = librosa.load(audio_file, sr=None)
        
        # No ground truth F0 available for real audio files
        f0_truth = None
        
    else:
        # Generate a synthetic signal with known F0
        print("No audio file provided. Generating a synthetic signal...")
        duration = 2.0  # seconds
        fs = 16000  # sampling rate (Hz)
        x, f0_truth, time_axis_truth = make_random_signal(duration=duration, fs=fs)
        
        # Export the synthetic signal for reference
        synthetic_file = "synthetic_signal.wav"
        librosa.output.write_wav(synthetic_file, x, fs)
        print(f"Synthetic signal saved to {synthetic_file}")
    
    # Set time interval for F0 estimation (frame length in seconds)
    dt = 0.005  # 5ms
    
    # Estimate F0 using the Nebula algorithm
    print("Estimating F0...")
    f0, time_axis, log_posterior, lmap = nebula_est(model, x, fs, dt=dt, return_details=True)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot the waveform
    plt.subplot(3, 1, 1)
    t = np.arange(len(x)) / fs
    plt.plot(t, x)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Plot the F0 estimate
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, f0, 'b-', label='Estimated F0')
    
    # If ground truth is available, plot it as well
    if f0_truth is not None:
        plt.plot(time_axis_truth, f0_truth, 'r--', label='Ground Truth F0')
    
    plt.title("Fundamental Frequency (F0)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.legend()
    
    # Plot the log posterior (or other details)
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, log_posterior)
    plt.title("Log Posterior")
    plt.xlabel("Time (s)")
    plt.ylabel("Log Posterior")
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure if requested
    if output_file:
        plt.savefig(output_file)
        print(f"Results saved to {output_file}")
    
    # Show the plots
    plt.show()
    
    return f0, time_axis

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PyNebula F0 Estimation Demo')
    parser.add_argument('--audio', type=str, help='Path to an audio file')
    parser.add_argument('--model', type=str, default='./model', help='Directory containing pre-trained models')
    parser.add_argument('--output', type=str, help='Output file for saving the plot')
    args = parser.parse_args()
    
    # Run the demo
    main(audio_file=args.audio, model_dir=args.model, output_file=args.output)
