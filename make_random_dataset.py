"""
Functions for generating random datasets for training GMM models.
In a real implementation, these would be replaced with functions to extract
features from real audio signals.
"""
import numpy as np

def make_random_dataset(num_samples=10000, feature_dim=20, num_clusters=2):
    """
    Generate a random dataset for GMM training.
    
    In this example, we generate data from multiple Gaussian distributions
    to simulate clustered features.
    
    Parameters:
        num_samples: Number of samples to generate
        feature_dim: Dimension of each feature vector
        num_clusters: Number of clusters to generate
    
    Returns:
        dataset: Generated dataset, shape (num_samples, feature_dim)
    """
    np.random.seed(0)  # For reproducibility
    
    # Allocate samples evenly across clusters
    samples_per_cluster = num_samples // num_clusters
    
    # Generate data for each cluster
    dataset = []
    for i in range(num_clusters):
        # Random mean vector for this cluster
        mean = np.random.randn(feature_dim) * 5
        
        # Generate samples from a Gaussian distribution
        cluster_data = np.random.randn(samples_per_cluster, feature_dim) + mean
        
        dataset.append(cluster_data)
    
    # Combine all clusters and shuffle
    dataset = np.vstack(dataset)
    np.random.shuffle(dataset)
    
    return dataset

def make_random_signal(duration=2.0, fs=16000, f0_min=80, f0_max=400):
    """
    Generate a random synthetic signal with a known F0 contour.
    
    This can be useful for testing and evaluating the F0 estimation algorithm.
    
    Parameters:
        duration: Signal duration in seconds
        fs: Sampling rate in Hz
        f0_min: Minimum F0 value in Hz
        f0_max: Maximum F0 value in Hz
    
    Returns:
        signal: Generated audio signal
        f0_truth: Ground truth F0 contour
        time_axis: Time axis for the F0 contour
    """
    # Number of samples
    num_samples = int(duration * fs)
    
    # Time axis
    t = np.arange(num_samples) / fs
    
    # Generate a random F0 contour (smooth changes in F0)
    num_f0_points = 10
    f0_points = np.random.uniform(f0_min, f0_max, num_f0_points)
    
    # Interpolate to get a smooth F0 contour
    f0_times = np.linspace(0, duration, num_f0_points)
    f0_contour = np.interp(t, f0_times, f0_points)
    
    # Generate a signal with this F0 contour (simple sine wave for demonstration)
    # In practice, a more realistic signal model would be used
    phase = 2 * np.pi * np.cumsum(f0_contour) / fs
    signal = np.sin(phase)
    
    # Add some noise
    signal = signal + 0.1 * np.random.randn(num_samples)
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # Create time axis for the F0 contour (e.g., every 10ms)
    frame_interval = 0.01  # 10ms
    num_frames = int(duration / frame_interval)
    time_axis = np.arange(num_frames) * frame_interval
    
    # Resample F0 contour to match the frame rate
    f0_truth = np.interp(time_axis, t, f0_contour)
    
    return signal, f0_truth, time_axis

def save_dataset(dataset, filename):
    """
    Save a dataset to disk.
    
    Parameters:
        dataset: Dataset to save
        filename: Output filename
    """
    np.save(filename, dataset)
    print(f"Dataset saved to {filename}")

if __name__ == '__main__':
    # Example: Generate and save a random dataset
    dataset = make_random_dataset(num_samples=10000, feature_dim=20)
    save_dataset(dataset, "random_dataset.npy")
    
    # Example: Generate a random signal with known F0
    signal, f0_truth, time_axis = make_random_signal(duration=2.0, fs=16000)
    
    # Print some statistics
    print(f"Generated signal with {len(signal)} samples")
    print(f"F0 range: {np.min(f0_truth):.1f}-{np.max(f0_truth):.1f} Hz")
