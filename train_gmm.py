import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
import os
from config import NUM_BANDS
from make_random_dataset import make_random_dataset


def train_gmm(features, n_components=16):
    """
    Python 版 train_gmm.m
    ---------------------
    根据给定的特征矩阵训练一个高斯混合模型（GMM）。
    
    参数：
      每个样本 6 维 (SNR0, SNR1, SNR2, IF1, IF2, F0)
      n_components (int): 高斯混合成分个数，默认为 16
      
    返回：
      gmm (GaussianMixture): 训练好的 GMM 模型对象
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', reg_covar=0.001)
    gmm.fit(features)
    return gmm


# # Define observed features (e.g., SNR0, SNR1, SNR2, IF1, IF2)
# x_values = np.array([-10.2, 5.1, 12.4, 1200.0, 1800.0])
# # Define candidate F0 values
# f_values = np.linspace(40, 1000, 128)  # 128 frequency candidates
# # Compute log-likelihoods P(f | x)
# log_L = conditional_gmm(gmm, f_values, x_values)
# return value is same size with f_values
def conditional_gmm(gmm, f_values, x_values, stdfloor=0.0):
    """
    Computes the conditional log-likelihood P(f | x) using a trained Gaussian Mixture Model (GMM).

    Args:
        gmm (GaussianMixture): Trained GMM model.
        f_values (np.ndarray): (nf,) Possible target variable values (e.g., F0 candidates).
        x_values (np.ndarray): (nx,) Observed feature vector (SNR0, SNR1, SNR2, IF1, IF2).
        stdfloor (float, optional): Floor value for standard deviation to prevent division errors.

    Returns:
        log_likelihoods (np.ndarray): (nf,) Log-likelihoods of each f_value given x.
    """
    means = gmm.means_
    covs = gmm.covariances_
    weights = gmm.weights_
    n_components = gmm.n_components

    x_dim = len(x_values)
    
    log_likelihoods = np.zeros(len(f_values))

    # Precompute log-sum-exp normalization
    log_weights = np.log(weights)
    log_lse_sum = -1e10  # Initialize log-sum-exp accumulator
    
    # Store conditional means and variances
    f0_mu = np.zeros(n_components)
    f0_sigma = np.zeros(n_components)
    f0_log_likelihood = np.zeros(n_components)
    f0_w = np.zeros(n_components)

    for k in range(n_components):
        # Extract mean and covariance for component k
        mu_x = means[k, :x_dim]
        mu_f = means[k, x_dim]
        cov_xx = covs[k, :x_dim, :x_dim]
        cov_ff = covs[k, x_dim, x_dim]
        cov_fx = covs[k, x_dim, :x_dim]

        # Compute conditional mean and variance
        inv_cov_xx = np.linalg.inv(cov_xx)
        mu_f_given_x = mu_f + cov_fx @ inv_cov_xx @ (x_values - mu_x)
        sigma_f_given_x = max(stdfloor ** 2, cov_ff - cov_fx @ inv_cov_xx @ cov_fx.T)

        # Compute log-likelihood of x under this component
        dx = x_values - mu_x
        log_likelihood_x = -0.5 * (dx @ inv_cov_xx @ dx) - 0.5 * np.log(np.linalg.det(cov_xx)) + log_weights[k]

        # Update log-sum-exp accumulator
        log_lse_sum = np.logaddexp(log_lse_sum, log_likelihood_x)

        # Store computed values
        f0_mu[k] = mu_f_given_x
        f0_sigma[k] = sigma_f_given_x
        f0_log_likelihood[k] = log_likelihood_x

    # Normalize mixture weights using log-sum-exp trick
    for k in range(n_components):
        f0_w[k] = np.exp(f0_log_likelihood[k] - log_lse_sum)

    # Compute final log likelihoods over all f_values
    for i, f in enumerate(f_values):
        p = 0.0
        for k in range(n_components):
            d = (f - f0_mu[k]) ** 2 / f0_sigma[k]
            p += f0_w[k] * np.exp(-0.5 * d) / np.sqrt(2.0 * np.pi * f0_sigma[k])
        log_likelihoods[i] = np.log(p + 1e-10)  # Avoid log(0)

    return log_likelihoods
  
  
def compute_lcal(gmm_models, f_values, SNR0, SNR1, SNR2, IF1, IF2):
    """
    Computes Lcal, a per-band likelihood baseline, by averaging conditional GMM likelihoods across frames.

    Args:
        gmm_models (list): (nch, ) List of trained GMM models (one per band).
        f_values (np.ndarray): (nf,) Candidate F0 values (Hz).
        SNR0 (np.ndarray): (nch, num_frames) SNR at half central frequency.
        SNR1 (np.ndarray): (nch, num_frames) SNR at central frequency.
        SNR2 (np.ndarray): (nch, num_frames) SNR at double central frequency.
        IF1 (np.ndarray): (nch, num_frames) Instantaneous frequency.
        IF2 (np.ndarray): (nch, num_frames) Instantaneous frequency at double frequency.

    Returns:
        Lcal (np.ndarray): (nch, nf) Average likelihood map per band.
    """
    nch, num_frames = SNR0.shape
    nf = len(f_values)
    Lcal = np.zeros((nch, nf))

    for i in range(nch):
        gmm = gmm_models[i]  # Get GMM for band j
        Lj = np.zeros((num_frames, nf))  # Store likelihoods per frame
        
        for j in range(num_frames):
            # Create feature vector for current frame and band
            x = np.array([SNR0[i, j], SNR1[i, j], SNR2[i, j], IF1[i, j], IF2[i, j]])

            # Compute conditional log-likelihoods P(f | x)
            L = conditional_gmm(gmm, f_values, x)

            # Normalize by subtracting mean
            L -= np.mean(L)
            Lj[j, :] = L
        
        # Compute mean likelihood over all frames
        Lcal[i, :] = np.mean(Lj, axis=0)
        
    return Lcal


def train_and_save_models(num_samples=10, n_components=16, model_dir='./model'):
    """
    Train GMM models for each frequency band and save them to disk.
    
    Parameters:
        num_samples: Number of samples to generate for training
        n_components: Number of components in each GMM
        model_dir: Directory to save the trained models
    """

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate a small dataset for training
    print(f"Generating {num_samples} training samples...")
    make_random_dataset(num_samples, nch=NUM_BANDS)
    
    # Train a GMM for each frequency band
    print(f"Training GMM models with {n_components} components...")
    
    gmm_models = []
    

    SNR0 = []
    SNR1 = []
    SNR2 = []
    IF1 = []
    IF2 = []
    
    for band in range(NUM_BANDS):
        # Load samples for the current band
        filename = os.path.join('data', f'samples-{band+1}.npy')
        if not os.path.isfile(filename):
            print(f"Error: Sample file '{filename}' not found")
            continue
            
        # Read data
        # shape of data will be (num_samples*naxis, 6)
        # Load data with the following structure in order:
        # {
        #     'SNR0': SNR0 in dB,
        #     'SNR1': SNR1 in dB,
        #     'SNR2': SNR2 in dB,
        #     'IF1': IF1 scaled to Hz,
        #     'IF2': IF2 scaled to Hz,
        #     'f0': fundamental frequency in Hz
        # }
        data = np.load(filename)
        
        SNR0.append(data[:, 0])
        SNR1.append(data[:, 1])
        SNR2.append(data[:, 2])
        IF1.append(data[:, 3])
        IF2.append(data[:, 4])
        
        # Train GMM model
        gmm = train_gmm(data, n_components=n_components)
        gmm_models.append(gmm)
        
    # Save the models
    models = {}
    for i in range(NUM_BANDS):
        model_file = os.path.join(model_dir, f"gmm_band_{i}.pkl")
        with open(model_file, 'wb') as fd:
            pickle.dump(gmm_models[i], fd)
        models[i] = gmm_models[i]
    
    # Create a real calibration file based on the trained GMM models
    nf = 128
    f_values = np.logspace(np.log10(40), np.log10(1000), nf)
    SNR0 = np.stack(SNR0)
    SNR1 = np.stack(SNR1)
    SNR2 = np.stack(SNR2)
    IF1 = np.stack(IF1)
    IF2 = np.stack(IF2)
    
    Lcal = compute_lcal(gmm_models, f_values, SNR0, SNR1, SNR2, IF1, IF2)
    
    # Save calibration file
    Lcal_file = os.path.join(model_dir, "Lcal.npy")
    np.save(Lcal_file, Lcal)
    
    print(f"Trained and saved {len(gmm_models)} GMM models to {model_dir}")
    print(f"Calibration data saved to {Lcal_file}")
    
    return True


if __name__ == "__main__":
    train_and_save_models(100)