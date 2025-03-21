import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from preprocess import preprocess_signal
from estimate_ifsnr import estimate_ifsnr
import librosa
from config import NUM_BANDS
from train_gmm import conditional_gmm
from postprocess import postprocess
from vitfilt import viterbi_filter
from binvitsearch import binvitsearch
import soundfile as sf


def mag2db(magnitude):
    """
    Converts magnitude to decibels (dB).

    Args:
        magnitude (float or np.ndarray): Input magnitude value(s).

    Returns:
        np.ndarray: Converted decibel (dB) values.
    """
    eps = 1e-10
    magnitude = np.maximum(magnitude, eps)
    return 20 * np.log10(magnitude)


def nebula_est(models, x, x_fs, thop=0.005):
    """
    Estimate fundamental frequency (F0) using GMM-based likelihood mapping.

    Args:
        model (dict): Pretrained model containing GMMs and Lcal.
        x (np.ndarray): Input signal (1D array).
        fs (int): Sampling rate.
        thop (float, optional): Time step for analysis. Defaults to 0.005s.

    Returns:
        f0 (np.ndarray): Estimated F0 contour.
        v (np.ndarray): Voicing decision (2 = voiced, 1 = unvoiced).
        pv (np.ndarray): Probability of being voiced.
        lmap (np.ndarray): Final likelihood map.
    """
    # Resample if needed
    fs = 8000
    if x_fs != fs:
        x = librosa.resample(x, orig_sr=x_fs, target_sr=fs)
    x = preprocess_signal(x)  # Assume preprocessing function exists

    # Define frequency bands
    nch = len(models)
    taxis = np.arange(0, len(x) / fs, thop)
    naxis = np.ceil(taxis * fs).astype(int)
    fc = np.logspace(np.log10(40), np.log10(1000), nch) / fs
    fres = fc

    # Compute SNR and Instantaneous Frequency (IF) features
    print("Computing SNR and Instantaneous Frequency features...")
    SNR1, IF1, SNR2, IF2, SNR0 = estimate_ifsnr(x, naxis, fc, fres, "nuttall98", "hanning")

    # Convert SNR features to dB
    SNR0 = mag2db(SNR0)
    SNR1 = mag2db(SNR1)
    SNR2 = mag2db(SNR2)
    IF1 *= fs
    IF2 *= fs

    # Compute likelihood map
    print("Computing likelihood map...")
    Lmap, f = make_likelihood_map(models, SNR1, IF1, SNR2, IF2, SNR0)
    
    # Apply postprocessing
    print("Applying postprocessing...")
    lmap = postprocess(np.exp(Lmap), 1.5 / f / fs / thop) 

    # Apply Viterbi filtering
    LF0 = np.tile(np.arange(1, Lmap.shape[1] + 1), (Lmap.shape[0], 1))
    ptrans = norm.pdf(np.arange(Lmap.shape[1]), 0, 2)
    print("Applying Viterbi filtering...")
    s, Ltotal = viterbi_filter(LF0, lmap, ptrans / np.sum(ptrans))  # Assume vitfilt1 function exists

    # Compute voicing probability
    print("Computing voicing probability...")
    q, pv = detect_voicing_status(np.log(lmap), s)
    v = 2 - q  # Convert q into voicing decision

    # Refine F0 estimates
    print("Refining F0 estimates...")
    f0 = refine_f0(np.log(lmap), s, f)
    if len(f0) == 1:
        f0 *= v

    return f0, v, pv, lmap


def make_likelihood_map(models, SNR1, IF1, SNR2, IF2, SNR0):
    """
    Computes a likelihood map for F0 estimation using trained GMMs.

    Args:
        model (dict): Contains 'G' (list of GMMs) and 'Lcal' (likelihood calibration).
        SNR1 (np.ndarray): (num_frames, nch) SNR at the center frequency.
        IF1 (np.ndarray): (num_frames, nch) Instantaneous frequency at center frequency.
        SNR2 (np.ndarray): (num_frames, nch) SNR at double frequency.
        IF2 (np.ndarray): (num_frames, nch) Instantaneous frequency at double frequency.
        SNR0 (np.ndarray): (num_frames, nch) SNR at half frequency.

    Returns:
        Lmap (np.ndarray): (num_frames, nf) Log-likelihood map.
        f (np.ndarray): (nf,) Candidate F0 values.
    """
    # load Lcal data
    # shape = (nch, nf)
    Lcal = np.load("model/Lcal.npy")
    Lcal_mean = np.mean(Lcal, axis=0)
    
    nch = len(models)  # Number of bands (GMMs per band)
    nf = Lcal.shape[1]  # Number of candidate F0 values
    
    # Generate logarithmically spaced F0 candidates
    f0_candidates = np.logspace(np.log10(40), np.log10(1000), nf)
    
    # Initialize likelihood map
    num_frames = SNR1.shape[0]
    Lmap = np.full((num_frames, nf), -100.0)  # Default low log-likelihood values

    for i in range(num_frames):
        Li = np.zeros((nch, nf))

        for j in range(nch):
            # Feature vector for GMM
            x = np.array([SNR1[i, j], SNR0[i, j], SNR2[i, j], IF1[i, j], IF2[i, j]])
            
            # Compute log-likelihood using conditional GMM
            L = conditional_gmm(models[j], f0_candidates, x)

            # Normalize by subtracting mean
            L -= np.mean(L)

            # Store likelihoods
            Li[j, :] = L

        # Aggregate across bands & apply calibration
        Lmap[i, :] = np.mean(Li, axis=0) - Lcal_mean
        
        # Apply log-sum-exp normalization for numerical stability
        Lmap[i, :] -= np.log(np.sum(np.exp(Lmap[i, :])))
    
    return Lmap, f0_candidates


def detect_voicing_status(Lmap, s):
    """
    Detects voicing status based on likelihood maps.

    Args:
        Lmap (np.ndarray): Log-likelihood map.
        s (np.ndarray): Smoothed pitch candidates.

    Returns:
        q (np.ndarray): Binary voicing decisions (1=voiced, 0=unvoiced).
        pvoiced (np.ndarray): Probability of being voiced.
    """
    # Extract log-likelihood values for the selected states
    f0p = np.array([Lmap[i, s[i]] for i in range(Lmap.shape[0])])

    # Constants for unvoiced distribution
    umean = -4.78
    ustd = 0.12

    # Negative log-likelihood function for optimization
    def Lfunc(params):
        return -np.mean(np.log(
            norm.pdf(f0p, params[0], params[1]) + 
            norm.pdf(f0p, umean, ustd)
        ))

    # Find optimal parameters for voiced distribution
    # Starting with mean = umean + 4, std = 1
    result = minimize(Lfunc, [umean + 4, 1], method="L-BFGS-B")
    param = result.x
    
    # Calculate voicing probability using Bayes rule
    pvoiced = norm.pdf(f0p, param[0], param[1]) / (
        norm.pdf(f0p, param[0], param[1]) + 
        norm.pdf(f0p, umean, ustd)
    )
    
    # Apply binary Viterbi search to get final voicing decisions
    q, _ = binvitsearch(pvoiced, 0.01)

    return q, pvoiced


def refine_f0(Lmap, s, f):
    """
    Refines F0 estimates using interpolation.

    Args:
        Lmap (np.ndarray): Log-likelihood map.
        s (np.ndarray): Smoothed F0 candidates.
        f (np.ndarray): Candidate F0 values.

    Returns:
        f0 (np.ndarray): Refined F0 contour in Hz.
    """
    nf = len(f)
    # Generate interpolation indices from 1 to nf with step 0.05
    Lidx = np.arange(1, nf+0.001, 0.05)  # Adding small value to include nf
    
    # Transpose for compatibility with MATLAB's interp1 behavior
    # In MATLAB: interp1(1:nf, Lmap', Lidx, 'spline')'
    Linterp = interp1d(np.arange(1, nf+1), Lmap.T, kind="cubic", axis=0, bounds_error=False)(Lidx).T
    
    # Map smoothed indices to interpolated space
    sinterp = interp1d(Lidx, np.arange(len(Lidx)), bounds_error=False, fill_value="extrapolate")(s)
    
    f0 = np.zeros(len(s))
    for i in range(len(s)):
        # Create index range centered around predicted position
        center = int(round(sinterp[i]))
        iidx = np.arange(max(0, center-10), min(len(Linterp[i]), center+11))
        
        if len(iidx) > 0:
            # Find index with maximum likelihood
            is_max = np.argmax(Linterp[i, iidx])
            # Store the index itself (not the value from f array yet)
            f0[i] = iidx[is_max]
    
    # Convert indices to F0 values by interpolation and scaling
    # In MATLAB: f0 = interp1(1:nf, f, f0 / 20) * 8000
    f0 = interp1d(np.arange(1, nf+1), f, bounds_error=False, fill_value="extrapolate")(f0 / 20)
    
    return f0


if __name__ == "__main__":
    import os
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    # Test the nebula_est function with a sample audio file
    print("Testing nebula_est function...")
    
    # Load models
    model_dir = './model'
    models = {}
    for i in range(NUM_BANDS): 
        model_file = os.path.join(model_dir, f"gmm_band_{i}.pkl")
        if not os.path.isfile(model_file):
            print(f"Error: Model file '{model_file}' not found.")
            print("Please run train_gmm.py first to generate the models.")
            exit(1)
        with open(model_file, 'rb') as fd:
            models[i] = pickle.load(fd)
    
    # Check if Lcal.npy exists
    lcal_file = os.path.join(model_dir, "Lcal.npy")
    if not os.path.isfile(lcal_file):
        print(f"Error: Calibration file '{lcal_file}' not found.")
        print("Please run train_gmm.py first to generate the calibration data.")
        exit(1)
    
    # Generate a test signal (or load a sample audio file if available)
    print("Generating a test signal...")
    fs = 8000  # Sample rate in Hz
    duration = 1.0  # Duration in seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Create a signal with a mixture of frequencies
    f0 = 170  # Fundamental frequency in Hz
    signal = np.sin(2 * np.pi * f0 * t)  # Fundamental
    # Add harmonics
    signal += 0.5 * np.sin(2 * np.pi * (2 * f0) * t)  # 2nd harmonic
    signal += 0.3 * np.sin(2 * np.pi * (3 * f0) * t)  # 3rd harmonic
    signal += 0.2 * np.sin(2 * np.pi * (4 * f0) * t)  # 4th harmonic
    
    # Add some noise
    signal += 0.1 * np.random.randn(len(signal))
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # save test signal as audio
    sf.write('test_100.wav', signal, fs)
    
    # 计算 FFT
    from scipy.fftpack import fft
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1/fs)  # 频率轴
    fft_spectrum = np.abs(fft(signal))[:N // 2]  # 计算振幅谱
    
    
    # Run F0 estimation
    print("Estimating F0...")
    f0_est, v, pv, lmap = nebula_est(models, signal, fs)
    
    # Estimate F0 using PyWorld for comparison
    print("Estimating F0 with PyWorld for comparison...")
    import pyworld as pw
    signal_double = signal.astype(np.float64)  # Convert to float64 for PyWorld
    thop = 0.005  # 5ms hop time for analysis
    frame_period = thop * 1000  # Convert to ms for PyWorld
    _f0_pw, t_pw = pw.dio(signal_double, fs, frame_period=frame_period)  # Raw pitch extraction
    f0_pw = pw.stonemask(signal_double, _f0_pw, t_pw, fs)  # Pitch refinement
    
    # Calculate time axes for plots
    t_f0 = np.linspace(0, duration, len(f0_est))
    t_pw_full = np.linspace(0, duration, len(f0_pw))
    
    # Plot results
    print("Plotting results...")
    plt.figure(figsize=(12, 15))  # Make figure taller to accommodate extra plot
    
    # Plot signal
    plt.subplot(5, 1, 1)
    plt.plot(t, signal)
    plt.title("Test Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Plot FFT spectrum for reference
    plt.subplot(5, 1, 2)
    plt.plot(freqs[:N//2], fft_spectrum)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    # Mark fundamental and harmonics
    plt.axvline(x=f0, color='r', linestyle='--', label=f"F0: {f0} Hz")
    plt.axvline(x=2*f0, color='g', linestyle='--', label=f"2F0: {2*f0} Hz")
    plt.axvline(x=3*f0, color='b', linestyle='--', label=f"3F0: {3*f0} Hz")
    plt.legend()
    
    # Plot likelihood map (lmap)
    plt.subplot(5, 1, 3)
    t_lmap = np.linspace(0, duration, lmap.shape[0])
    
    # Generate frequency axis for lmap
    freq_min = 40  # Minimum F0 in Hz
    freq_max = 800  # Maximum F0 in Hz
    n_freq = lmap.shape[1]
    freq_axis = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freq)
    
    # Plot the likelihood map as a heatmap
    plt.pcolormesh(t_lmap, freq_axis, lmap.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Likelihood')
    
    # Overlay F0 estimates on the likelihood map
    plt.plot(t_f0, f0_est * (v > 1), 'r.', markersize=3, label='Nebula F0')
    plt.plot(t_pw_full, f0_pw * (f0_pw > 0), 'w.', markersize=3, label='PyWorld F0')
    
    # Mark true F0 and harmonics
    plt.axhline(y=f0, color='r', linestyle='--', label=f"True F0: {f0} Hz")
    plt.axhline(y=2*f0, color='g', linestyle='--', label=f"2F0: {2*f0} Hz")
    plt.axhline(y=3*f0, color='b', linestyle='--', label=f"3F0: {3*f0} Hz")
    
    plt.yscale('log')  # Use log scale for frequency axis
    plt.ylim([freq_min, freq_max])
    plt.title("Likelihood Map with F0 Estimates")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend(loc='upper right')
    
    # Plot F0 estimates
    plt.subplot(5, 1, 4)
    plt.plot(t_f0, f0_est, 'b-', label='Nebula F0')
    
    # Plot PyWorld F0
    plt.plot(t_pw_full, f0_pw, 'g-', label='PyWorld F0')
    
    plt.title("Estimated F0 Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.axhline(y=f0, color='r', linestyle='--', label=f"True F0: {f0} Hz")
    plt.axhline(y=2*f0, color='g', linestyle='--', label=f"2F0: {2*f0} Hz")
    plt.axhline(y=3*f0, color='b', linestyle='--', label=f"3F0: {3*f0} Hz")
    plt.legend()
    plt.grid(True)
    
    # Plot voicing decision
    plt.subplot(5, 1, 5)
    plt.plot(t_f0, v, 'b-', label='Nebula Voicing')
    # Create voicing decision for PyWorld (0 = unvoiced, 2 = voiced)
    v_pw = (f0_pw > 0).astype(int) * 2
    plt.plot(t_pw_full, v_pw, 'g-', label='PyWorld Voicing')
    plt.title("Voicing Decision (2=voiced, 1=unvoiced)")
    plt.xlabel("Time (s)")
    plt.ylabel("Decision")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("nebula_est_test.png")
    plt.show()
    
    # Calculate metrics
    min_len = min(len(f0_est), len(f0_pw))
    f0_est_trunc = f0_est[:min_len]
    f0_pw_trunc = f0_pw[:min_len]
    v_trunc = v[:min_len]
    v_pw_trunc = v_pw[:min_len]
    
    # Only compare voiced frames
    voiced_mask = (v_trunc > 1) & (v_pw_trunc > 0)
    if np.sum(voiced_mask) > 0:
        mae = np.mean(np.abs(f0_est_trunc[voiced_mask] - f0_pw_trunc[voiced_mask]))
        correlation = np.corrcoef(f0_est_trunc[voiced_mask], f0_pw_trunc[voiced_mask])[0, 1]
        print(f"Mean Absolute Error between Nebula and PyWorld: {mae:.2f} Hz")
        print(f"Correlation between Nebula and PyWorld: {correlation:.4f}")
        
        # Compute octave error rate
        octave_diff = np.abs(np.log2(f0_est_trunc[voiced_mask] / f0_pw_trunc[voiced_mask]))
        octave_error_rate = np.mean(octave_diff > 0.4)  # Threshold for detecting octave errors
        print(f"Octave Error Rate: {octave_error_rate:.4f}")
    
    voicing_agreement = np.mean((v_trunc > 1) == (v_pw_trunc > 0))
    print(f"Voicing Agreement between Nebula and PyWorld: {voicing_agreement:.4f}")
    
    print(f"Test completed. Average estimated F0:")
    print(f"  Nebula: {np.mean(f0_est[v > 1]):.2f} Hz")
    print(f"  PyWorld: {np.mean(f0_pw[f0_pw > 0]):.2f} Hz")
    print(f"  True F0: {f0} Hz")
    print(f"Results saved to nebula_est_test.png")