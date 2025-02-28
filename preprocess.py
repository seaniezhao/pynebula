"""
Signal preprocessing functions for the Nebula F0 estimation algorithm.
Includes DC removal, downsampling, and feature extraction.
"""
import numpy as np
import librosa
from scipy import signal
import soundfile as sf


def preprocess_signal(x, fs, dither_level=0.05, dc_cutoff=50/4000):
    """
    Preprocess audio signal:
    
    1. DC removal
    2. Dithering
    3. Downsampling
    4. Feature extraction
     
    """
    # Ensure input is a numpy array and is 1D
    x = np.asarray(x)
    if x.ndim > 1:
        # Take the first channel if multichannel
        x = x[:, 0] if x.shape[1] < x.shape[0] else x[0, :]

    xsqr_intg = np.cumsum(x**2)
    xrms = np.sqrt((xsqr_intg[256:] - xsqr_intg[:-256]) / 256)
    # 如果需要与原信号对齐，可以进行适当的填充或插值
    # print(len(xrms), len(x))
    xrms = np.pad(xrms, (128, 128), mode='edge')
    peak = np.max(xrms)
    thrd = peak * dither_level
    
    # Remove DC component
    x = dcnotch(x, dc_cutoff)
    
    # Add dithering noise where signal RMS is below threshold
    # Generate condition mask where RMS is below threshold
    mask = xrms < thrd
    # Generate random noise scaled by difference between threshold and RMS
    noise = np.random.randn(len(x)) * (thrd - xrms)
    # Apply noise only where mask is True (RMS < threshold)
    y = x + mask * noise
    
    # Downsample if sampling rate is greater than 16000 Hz
    target_fs = 16000
    if fs > target_fs:
        x = librosa.resample(x, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return x, fs

def dcnotch(x, cutoff=50/4000):
    """
    Remove DC component from signal using zero-phase digital filtering.
    
    This is a Python implementation of the provided MATLAB code:
    function y = dcnotch(x, cutoff)
      a1 = - 2.0 * cos(pi * cutoff);
      a0 = 8.0 * cos(pi * cutoff) - 7.0;
      r = (-a1 - sqrt(a1 ^ 2 - 4.0 * a0)) / 2.0;
      a = [1.0, -r];
      b = [1.0, -1.0];
      y = filtfilt(b, a, x);
    end
    
    Parameters:
        x: Input signal (numpy array)
        cutoff: Normalized cutoff frequency (default: 50/4000)
    
    Returns:
        y: Signal with DC component removed
    """
    # Calculate filter coefficients
    a1 = -2.0 * np.cos(np.pi * cutoff)
    a0 = 8.0 * np.cos(np.pi * cutoff) - 7.0
    r = (-a1 - np.sqrt(a1 ** 2 - 4.0 * a0)) / 2.0
    
    # Filter coefficients
    a = np.array([1.0, -r])
    b = np.array([1.0, -1.0])
    
    # Apply zero-phase digital filtering
    y = signal.filtfilt(b, a, x)
    
    return y


if __name__ == "__main__":
    x, fs = librosa.load("test.wav", sr=None)
    y, fs = preprocess_signal(x, fs)
     # write wav
    # sf.write('test_preprocessd.wav', y, fs)
    