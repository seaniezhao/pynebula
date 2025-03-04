"""
Test script for PyNebula F0 estimation using a provided audio file.
"""
import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pyworld as pw
import pickle
from config import NUM_BANDS
from nebula_est import nebula_est

def test_with_audio_file(audio_file='test.wav', model_dir='./model'):
    """
    Test the nebula_est F0 estimation against pyworld.
    
    Args:
        audio_file (str): Path to the audio file for testing
        model_dir (str): Directory containing the nebula models
    
    Returns:
        dict: Dictionary containing comparison metrics and results
    """
    # Print test information
    print(f"Testing nebula_est with audio file: {audio_file}")
    print(f"Using models from: {model_dir}")
    
    # Load audio file
    print("Loading audio file...")
    x, fs = librosa.load(audio_file, sr=None, mono=True)
    print(f"Loaded audio: {len(x)/fs:.2f}s at {fs}Hz sampling rate")
    
    # Extract middle portion of the audio, which typically has more stable voiced content
    start_sec = len(x) // fs // 4  # Start at 1/4 of the file
    duration_sec = 2  # Use 4 seconds of audio
    start_sample = start_sec * fs
    end_sample = start_sample + duration_sec * fs
    
    if end_sample > len(x):
        end_sample = len(x)
    
    x = x[start_sample:end_sample]
    print(f"Using audio segment: {start_sec}s to {start_sec + duration_sec}s")
    
    # Load models for nebula_est
    print("Loading nebula models...")
    models = []
    for i in range(NUM_BANDS): 
        model_path = os.path.join(model_dir, f"gmm_band_{i}.pkl")
        with open(model_path, 'rb') as f:
            models.append(pickle.load(f))
    print(f"Loaded {len(models)} models")
    
    # Run nebula_est pitch estimation
    print("Running nebula_est pitch estimation...")
    thop = 0.005  # 5ms hop time
    f0_nebula, v_nebula, pv_nebula, lmap = nebula_est(models, x, fs, thop)
    
    # Run pyworld pitch estimation
    print("Running pyworld pitch estimation...")
    frame_period = thop * 1000  # convert to ms for pyworld
    # Convert the audio to float64 for PyWorld
    x_double = x.astype(np.float64)
    _f0_pw, t_pw = pw.dio(x_double, fs, frame_period=frame_period)
    f0_pw = pw.stonemask(x_double, _f0_pw, t_pw, fs)
    
    # Make sure both estimates have the same length
    min_len = min(len(f0_nebula), len(f0_pw))
    f0_nebula = f0_nebula[:min_len]
    f0_pw = f0_pw[:min_len]
    
    # Create voiced mask for pyworld (similar to nebula)
    v_pw = (f0_pw > 0).astype(int) * 2  # Convert to 0/2 format like nebula
    
    # Time axis for plotting
    t = np.arange(min_len) * thop
    
    # Compute metrics for voiced regions only
    voiced_mask_nebula = (v_nebula == 2)
    voiced_mask_pw = (v_pw == 2)
    voiced_mask_both = voiced_mask_nebula & voiced_mask_pw
    
    print(f"Frames with voiced detection - Nebula: {np.sum(voiced_mask_nebula)}, PyWorld: {np.sum(voiced_mask_pw)}, Both: {np.sum(voiced_mask_both)}")
    
    # Compute statistics only for frames where both methods detect voice
    if np.sum(voiced_mask_both) > 0:
        f0_nebula_voiced = f0_nebula[voiced_mask_both]
        f0_pw_voiced = f0_pw[voiced_mask_both]
        
        # Compute mean absolute error and correlation
        mae = np.mean(np.abs(f0_nebula_voiced - f0_pw_voiced))
        correlation = np.corrcoef(f0_nebula_voiced, f0_pw_voiced)[0, 1]
        
        print(f"Mean Absolute Error: {mae:.2f} Hz")
        print(f"Correlation: {correlation:.4f}")
    else:
        print("No common voiced regions detected for comparison")
    
    # Compute voicing agreement
    voicing_agreement = np.mean(v_nebula == v_pw)
    print(f"Voicing Agreement: {voicing_agreement:.4f}")
    
    # Plot Likelihood Map from nebula_est
    plt.figure(figsize=(12, 8))
    plt.subplot(111)
    
    # Generate frequency axis (assuming lmap shape is [time, frequency])
    # For better visualization, use log scale for frequency
    freq_min = 40  # Minimum F0 in Hz (common for speech)
    freq_max = 1000  # Maximum F0 in Hz
    n_freq = lmap.shape[1]
    freq_axis = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freq)
    
    # Create time-frequency heatmap of likelihood values
    plt.pcolormesh(t, freq_axis, lmap.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Likelihood')
    
    # Plot the F0 estimates on top of the likelihood map
    plt.plot(t, f0_nebula * (v_nebula == 2), 'r.', markersize=2, label='Nebula F0')
    plt.plot(t, f0_pw * (v_pw == 2), 'w.', markersize=2, label='PyWorld F0')
    
    plt.yscale('log')  # Use log scale for frequency axis
    plt.ylim([freq_min, freq_max])
    plt.title('Likelihood Map with F0 Estimates')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    
    # Save and show the likelihood map plot
    lmap_plot_path = os.path.join("test_results", os.path.basename(audio_file).replace('.wav', '_lmap.png'))
    plt.tight_layout()
    plt.savefig(lmap_plot_path)
    print(f"Likelihood map plot saved to: {lmap_plot_path}")
    plt.show()
    
    # Plot results (original comparison plots)
    plt.figure(figsize=(12, 10))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    t_audio = np.arange(len(x)) / fs
    plt.plot(t_audio, x)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot F0 contours
    plt.subplot(3, 1, 2)
    # Plot F0 only for voiced frames
    plt.plot(t, f0_nebula * (v_nebula == 2), 'b.', markersize=4, label='Nebula F0')
    plt.plot(t, f0_pw * (v_pw == 2), 'r.', markersize=4, label='PyWorld F0')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('F0 Comparison: Nebula vs PyWorld')
    plt.legend()
    plt.grid(True)
    
    # Plot voicing decisions
    plt.subplot(3, 1, 3)
    plt.plot(t, v_nebula, 'b-', label='Nebula Voicing')
    plt.plot(t, v_pw, 'r-', label='PyWorld Voicing')
    plt.xlabel('Time (s)')
    plt.ylabel('Voicing Status')
    plt.title('Voicing Decision Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(audio_file).replace('.wav', '_comparison.png'))
    plt.savefig(output_file)
    print(f"Plot saved to: {output_file}")
    
    # Display plot
    plt.show()
    
    # Return comparison results
    results = {
        'f0_nebula': f0_nebula,
        'f0_pyworld': f0_pw,
        'v_nebula': v_nebula,
        'v_pyworld': v_pw,
        'lmap': lmap,
        'metrics': {
            'mae': mae if 'mae' in locals() else None,
            'correlation': correlation if 'correlation' in locals() else None,
            'voicing_agreement': voicing_agreement
        }
    }
    
    return results

if __name__ == "__main__":
    # Get audio file from command line arguments if provided
    audio_file = sys.argv[1] if len(sys.argv) > 1 else 'test.wav'
    
    # Run the test
    test_with_audio_file(audio_file)
