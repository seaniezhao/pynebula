"""
Test script for PyNebula F0 estimation using a provided audio file.
"""
import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pyworld as pw
from nebula_est import nebula_est
from load_model import load_model
from train_gmm import train_and_save_models

def test_with_audio_file(audio_file='test.wav', model_dir='./model'):
    """
    Test the PyNebula F0 estimation on a real audio file.
    
    Parameters:
        audio_file: Path to the audio file to test
        model_dir: Directory containing the models (will be created if it doesn't exist)
    """
    # Check if the audio file exists
    if not os.path.isfile(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        return
    
    # Check if models exist, if not, generate example models
    if not os.path.isdir(model_dir) or len(os.listdir(model_dir)) == 0:
        print("No pre-trained models found. Generating example models...")
        train_and_save_models(model_dir=model_dir)
    
    # Load the pre-trained models
    print(f"Loading models from {model_dir}...")
    model = load_model(model_dir)
    
    # Load the audio file
    print(f"Loading audio file: {audio_file}")
    x, fs = librosa.load(audio_file, sr=None)
    
    print(f"Audio file details:")
    print(f"  - Duration: {len(x) / fs:.2f} seconds")
    print(f"  - Sampling rate: {fs} Hz")
    print(f"  - Number of samples: {len(x)}")
    
    # Set time interval for F0 estimation (frame length in seconds)
    dt = 0.005  # 5ms
    
    # Estimate F0 using the Nebula algorithm
    print("\nEstimating F0 using PyNebula...")
    try:
        f0_nebula, time_axis, log_posterior, lmap = nebula_est(model, x, fs, dt=dt, return_details=True)
        
        print(f"PyNebula F0 estimation completed successfully!")
        print(f"  - Number of frames: {len(f0_nebula)}")
        
        # Handle the case where there might be no voiced frames
        voiced_f0 = f0_nebula[f0_nebula > 0]
        if len(voiced_f0) > 0:
            print(f"  - F0 range (PyNebula): {np.min(voiced_f0):.1f} - {np.max(voiced_f0):.1f} Hz")
        else:
            print(f"  - No voiced frames detected by PyNebula")
        
        # Estimate F0 using WORLD (pyworld)
        print("\nEstimating F0 using WORLD (pyworld)...")
        # Convert to float64 and ensure samples are in [-1, 1] if not already
        x_world = x.astype(np.float64)
        if np.max(np.abs(x_world)) > 1.0:
            x_world = x_world / np.max(np.abs(x_world))
            
        # Extract F0, spectral envelope, and aperiodicity using WORLD
        f0_world, time_world = pw.dio(x_world, fs, f0_floor=30.0, f0_ceil=1100.0, frame_period=dt*1000)  # dio provides a coarse F0 estimation
        f0_world = pw.stonemask(x_world, f0_world, time_world, fs)  # stonemask refines the F0 estimation
        
        print(f"WORLD F0 estimation completed successfully!")
        print(f"  - Number of frames: {len(f0_world)}")
        
        # Handle the case where there might be no voiced frames
        voiced_f0_world = f0_world[f0_world > 0]
        if len(voiced_f0_world) > 0:
            print(f"  - F0 range (WORLD): {np.min(voiced_f0_world):.1f} - {np.max(voiced_f0_world):.1f} Hz")
        else:
            print(f"  - No voiced frames detected by WORLD")
        
        # Plot the results
        plt.figure(figsize=(12, 9))
        
        # Plot the waveform
        plt.subplot(3, 1, 1)
        t = np.arange(len(x)) / fs
        plt.plot(t, x)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        # Plot both F0 estimates in the same panel for comparison
        plt.subplot(3, 1, 2)
        plt.plot(time_axis, f0_nebula, 'b-', label='PyNebula F0', alpha=0.7)
        plt.plot(time_world, f0_world, 'r-', label='WORLD F0', alpha=0.7)
        plt.title("Fundamental Frequency (F0) Comparison")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        
        # Set y-limit to include all F0 values but cap at 1100 Hz to avoid extreme outliers
        max_f0 = min(1100, max(np.max(f0_nebula) * 1.1 if len(voiced_f0) > 0 else 500, 
                               np.max(f0_world) * 1.1 if len(voiced_f0_world) > 0 else 500))
        plt.ylim([0, max_f0])
        plt.grid(True)
        plt.legend()
        
        # Plot the log posterior
        plt.subplot(3, 1, 3)
        plt.plot(time_axis, log_posterior)
        plt.title("Log Posterior (PyNebula)")
        plt.xlabel("Time (s)")
        plt.ylabel("Log Posterior")
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        output_file = "f0_comparison_same_panel.png"
        plt.savefig(output_file)
        print(f"Comparison results saved to {output_file}")
        
        # Also save raw F0 values for further analysis
        np.savez('f0_comparison.npz', 
                 f0_nebula=f0_nebula, 
                 time_nebula=time_axis, 
                 f0_world=f0_world, 
                 time_world=time_world)
        print("Raw F0 values saved to f0_comparison.npz")
        
        # Calculate statistics for comparison
        if len(voiced_f0) > 0 and len(voiced_f0_world) > 0:
            # Only compare frames where both methods found voiced speech
            # Need to interpolate as frame counts might be different
            min_len = min(len(f0_nebula), len(f0_world))
            
            # Simple comparison by truncating both arrays
            f0_nebula_trunc = f0_nebula[:min_len]
            f0_world_trunc = f0_world[:min_len]
            
            # Calculate mean absolute difference for voiced frames
            voiced_mask = (f0_nebula_trunc > 0) & (f0_world_trunc > 0)
            if np.any(voiced_mask):
                mean_abs_diff = np.mean(np.abs(f0_nebula_trunc[voiced_mask] - f0_world_trunc[voiced_mask]))
                print(f"\nComparison statistics:")
                print(f"  - Mean absolute difference: {mean_abs_diff:.2f} Hz")
                
                # Calculate correlation
                corr = np.corrcoef(f0_nebula_trunc[voiced_mask], f0_world_trunc[voiced_mask])[0, 1]
                print(f"  - Correlation coefficient: {corr:.4f}")
        
        plt.show()
        
        return f0_nebula, f0_world, time_axis, time_world
        
    except Exception as e:
        print(f"Error during F0 estimation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Get audio file from command line arguments if provided
    audio_file = sys.argv[1] if len(sys.argv) > 1 else 'test.wav'
    
    # Run the test
    test_with_audio_file(audio_file)
