import numpy as np
from scipy.interpolate import interp1d

def postprocess(L, navg):
    """
    Smooths likelihood maps using running average interpolation.
    
    Parameters:
        L (np.ndarray): Input likelihood matrix, shape (n_frames, n_frequencies)
        navg (np.ndarray): Window size for each frequency, shape (n_frequencies,)
    
    Returns:
        Lsmooth (np.ndarray): Smoothed likelihood matrix, same shape as L
    """
    Lsmooth = L.copy()
    
    # Process each column (frequency channel)
    for i in range(L.shape[1]):
        # Calculate normalized cumulative sum
        cs = np.cumsum(L[:, i]) / navg[i] / 2
        
        # Create indices
        idx = np.arange(L.shape[0])
        idx_h = idx + navg[i]
        idx_l = idx - navg[i]
        
        # Create interpolation function
        interpolator = interp1d(idx, cs, bounds_error=False, fill_value="extrapolate")
        
        # Apply interpolation and difference
        Lsmooth[:, i] = interpolator(idx_h) - interpolator(idx_l)
    
    return Lsmooth

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a test likelihood matrix
    n_frames = 100
    n_frequencies = 5
    
    # Create noisy data
    np.random.seed(42)  # For reproducibility
    L = np.zeros((n_frames, n_frequencies))
    
    # Add some patterns and noise to different frequency channels
    x = np.linspace(0, 2*np.pi, n_frames)
    
    # Create different patterns for each frequency
    L[:, 0] = np.sin(x) + 0.3 * np.random.randn(n_frames)  # Sine wave with noise
    L[:, 1] = np.sin(2*x) + 0.3 * np.random.randn(n_frames)  # Higher frequency sine
    L[:, 2] = (x > np.pi).astype(float) + 0.3 * np.random.randn(n_frames)  # Step function
    L[:, 3] = np.exp(-0.05 * (x - np.pi)**2) + 0.3 * np.random.randn(n_frames)  # Gaussian
    L[:, 4] = 0.5 * np.random.randn(n_frames)  # Pure noise
    
    # Define different window sizes for each frequency channel
    navg = np.array([3, 5, 7, 10, 15])
    
    # Apply the postprocessing
    L_smooth = postprocess(L, navg)
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    for i in range(n_frequencies):
        # Original signal
        plt.subplot(n_frequencies, 2, 2*i+1)
        plt.plot(x, L[:, i])
        plt.title(f"Original Signal - Channel {i+1}")
        plt.xlabel("Time")
        plt.ylabel("Likelihood")
        
        # Smoothed signal
        plt.subplot(n_frequencies, 2, 2*i+2)
        plt.plot(x, L_smooth[:, i])
        plt.title(f"Smoothed Signal - Channel {i+1} (Window Size = {navg[i]})")
        plt.xlabel("Time")
        plt.ylabel("Likelihood")
    
    plt.tight_layout()
    plt.savefig("postprocess_test.png")
    plt.show()
    
    print("Postprocessing test completed!")
    print("The smoothed signals were saved to 'postprocess_test.png'")
    
    # Calculate and print signal improvement metrics
    original_variance = np.var(L, axis=0)
    smoothed_variance = np.var(L_smooth, axis=0)
    
    print("\nSignal Improvement Metrics:")
    print("==========================")
    print("Channel | Window Size | Original Variance | Smoothed Variance | Variance Reduction (%)")
    print("-----------------------------------------------------------------------")
    
    for i in range(n_frequencies):
        variance_reduction = 100 * (1 - smoothed_variance[i] / original_variance[i])
        print(f"{i+1:7d} | {navg[i]:11d} | {original_variance[i]:17.4f} | {smoothed_variance[i]:17.4f} | {variance_reduction:20.2f}")
