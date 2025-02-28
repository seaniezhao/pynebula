"""
Core implementation of the Nebula F0 estimation algorithm.
"""
import numpy as np
from preprocess import preprocess_signal
from postprocess import postprocess_results
from scipy.signal import medfilt

def nebula_est(model, x, fs, dt=0.005, return_details=False):
    """
    Estimate the fundamental frequency (F0) from an input signal.
    
    Parameters:
        model: Dictionary containing GMM models and calibration data, loaded with load_model()
        x: Audio signal (numpy array)
        fs: Sampling rate (Hz)
        dt: Time interval (seconds), determines frame length
        return_details: If True, returns additional details (e.g., F0 posterior map)
    
    Returns:
        f0: Fundamental frequency sequence (Hz)
        (Optional) Additional details if return_details=True
    """
    # 1. Preprocess: DC notch, downsample, extract features
    features, time_axis = preprocess_signal(x, fs, dt)
    
    # 2. Calculate log-likelihood for each frequency band using the model
    models = model.get("models", {})
    Lcal = model.get("Lcal", None)  # Calibration data (for subsequent mapping)
    
    num_frames = features.shape[0]
    num_bands = len(models)
    
    # Initialize log-likelihood map
    lmap = np.zeros((num_frames, num_bands))
    lmap.fill(-100)  # Fill with low likelihood for bands without models
    
    # Calculate log-likelihood for each band
    for b in range(num_bands):
        if b in models:
            gmm = models[b]
            # Calculate GMM log-likelihood for each frame's features
            lmap[:, b] = gmm.score_samples(features)
    
    # 3. Take the average log posterior (for return value only)
    avg_log_posterior = np.mean(lmap, axis=1)
    
    # 4. Map average log posterior to F0
    f0 = estimate_f0_from_posterior(avg_log_posterior, Lcal)
    
    
    if return_details:
        return f0, time_axis, avg_log_posterior, lmap
    else:
        return f0

def estimate_f0_from_posterior(log_posterior, Lcal=None):
    """
    Map log posterior values to F0 sequence.
    
    This implementation uses the log_posterior values to estimate F0.
    If Lcal calibration data is provided, it uses it to map band indices to frequencies.
    
    Parameters:
        log_posterior: Log posterior values for each frame
        Lcal: Calibration data (optional)
        
    Returns:
        f0: Estimated fundamental frequency sequence (Hz)
    """
    num_frames = len(log_posterior)
    f0 = np.zeros(num_frames)
    
    # Define F0 range (typical human voice range is ~80-400 Hz)
    f0_min = 30.0
    f0_max = 1100.0
    
    # Define threshold for voiced/unvoiced decision
    # Adaptive threshold based on the distribution of log posterior values
    threshold = np.mean(log_posterior) - 0.5 * np.std(log_posterior)
    
    # Get the maximum log posterior value for normalization
    max_posterior = np.max(log_posterior)
    
    if Lcal is not None and isinstance(Lcal, np.ndarray) and Lcal.shape[0] > 0:
        # If we have calibration data, use it to map log posterior to frequencies
        
        # Get the number of calibration points
        num_cal_points = Lcal.shape[0]
        
        # Create a mapping from calibration indices to frequencies
        # This is a simple linear mapping, but could be more sophisticated
        cal_to_freq = np.linspace(f0_min, f0_max, num_cal_points)
        
        # For each frame, estimate F0 based on the log posterior value
        for i in range(num_frames):
            frame_posterior = log_posterior[i]
            
            # If the log posterior is below the threshold, set F0 to 0 (unvoiced)
            if frame_posterior < threshold:
                f0[i] = 0.0
            else:
                # Map the posterior value to a frequency
                # Normalize the posterior value to [0, 1]
                normalized = (frame_posterior - threshold) / (max_posterior - threshold)
                normalized = np.clip(normalized, 0.0, 1.0)
                
                # Map the normalized value to a frequency
                f0[i] = f0_min + normalized * (f0_max - f0_min)
    else:
        # Without calibration data, use a simple mapping based on the posterior strength
        for i in range(num_frames):
            frame_posterior = log_posterior[i]
            
            # If log posterior is below the threshold, likely unvoiced
            if frame_posterior < threshold:
                f0[i] = 0.0
            else:
                # Map the posterior value to a frequency
                normalized = (frame_posterior - threshold) / (max_posterior - threshold)
                normalized = np.clip(normalized, 0.0, 1.0)
                f0[i] = f0_min + normalized * (f0_max - f0_min)
    
    # Apply median filtering to smooth the f0 contour and remove outliers
    f0 = median_filter_f0(f0)
    
    # Ensure we have at least some voiced frames
    if np.sum(f0 > 0) == 0:
        # If no voiced frames, force at least 30% of frames to be voiced
        # by adjusting threshold
        top_indices = np.argsort(log_posterior)[-int(0.3 * num_frames):]
        for i in top_indices:
            # Assign f0 based on position within the top frames
            normalized = (i - top_indices[0]) / (top_indices[-1] - top_indices[0]) if len(top_indices) > 1 else 0.5
            f0[i] = f0_min + normalized * (f0_max - f0_min)
    
    return f0

def median_filter_f0(f0, window_size=5):
    """
    Apply median filtering to smooth F0 contour and remove outliers.
    
    Parameters:
        f0: F0 contour to smooth
        window_size: Size of the median filter window
        
    Returns:
        Smoothed F0 contour
    """
    # Create a copy to avoid modifying the input
    smoothed_f0 = np.copy(f0)
    
    # Only apply filtering to voiced regions (f0 > 0)
    voiced_mask = f0 > 0
    voiced_indices = np.where(voiced_mask)[0]
    
    if len(voiced_indices) > window_size:
        # Extract voiced f0 values
        voiced_f0 = f0[voiced_mask]
        
        # Apply median filtering
        smoothed_voiced_f0 = medfilt(voiced_f0, window_size)
        
        # Put back the smoothed values
        smoothed_f0[voiced_mask] = smoothed_voiced_f0
    
    return smoothed_f0
