# PyNebula

A Python implementation of the Nebula fundamental frequency (F0) estimation algorithm, which uses Gaussian Mixture Models (GMMs) to estimate pitch in speech and musical signals.

original implementation: https://github.com/Sleepwalking/nebula

## Project Overview

PyNebula implements a GMM-based likelihood mapping approach to pitch tracking. This method creates a likelihood map of potential fundamental frequencies and selects the most likely F0 trajectory using dynamic programming. The implementation includes:

- Signal preprocessing with SNR and instantaneous frequency features
- GMM-based likelihood calculation for potential F0 candidates
- Viterbi filtering for optimal F0 trajectory extraction
- Voicing probability estimation

PyNebula can be compared with other F0 estimation methods like PyWorld to evaluate its performance.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pynebula.git
cd pynebula

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- NumPy
- SciPy
- Matplotlib
- scikit-learn (for GMM training)
- PyWorld (for comparison)
- Librosa (for audio loading and processing)

## Usage

### Basic F0 Estimation

```python
import numpy as np
import librosa
from nebula_est import nebula_est
import pickle
import os

# Load audio file
audio_file = 'test.wav'
x, fs = librosa.load(audio_file, sr=None)

# Load pre-trained GMM models
model_dir = './model'
models = []
for i in range(36):  # NUM_BANDS = 36
    model_path = os.path.join(model_dir, f"gmm_band_{i}.pkl")
    with open(model_path, 'rb') as f:
        models.append(pickle.load(f))

# Estimate F0 with 5ms frame intervals
f0, voicing, voicing_prob, lmap = nebula_est(models, x, fs, thop=0.005)

# f0: Estimated fundamental frequency for each frame
# voicing: Voicing decision (0=initialization, 1=unvoiced, 2=voiced)
# voicing_prob: Probability of voicing for each frame
# lmap: Likelihood map of frequency candidates
```

### Comparing with PyWorld

```python
import pyworld as pw
import numpy as np

# Convert the audio to float64 for PyWorld
x_double = x.astype(np.float64)
frame_period = 5.0  # 5ms frame period
_f0_pw, t_pw = pw.dio(x_double, fs, frame_period=frame_period)  # Raw pitch extraction
f0_pw = pw.stonemask(x_double, _f0_pw, t_pw, fs)  # Pitch refinement

# Compare the results
import matplotlib.pyplot as plt

# Generate time axis
t = np.arange(len(f0)) * 0.005  # 5ms intervals
t_pw = np.arange(len(f0_pw)) * 0.005

# Plot F0 contours
plt.figure(figsize=(10, 6))
plt.plot(t, f0 * (voicing == 2), 'b-', label='Nebula F0')
plt.plot(t_pw, f0_pw, 'r-', label='PyWorld F0')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('F0 Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

## Testing and Evaluation

The repository includes test scripts to evaluate the performance of the F0 estimation:

1. `test_nebula.py`: Compares nebula_est with PyWorld on real audio files
2. The `nebula_est.py` main function: Creates a synthetic test signal and compares F0 estimation

Run the tests with:

```bash
# Test with real audio
python test_nebula.py

# Test with synthetic signal
python nebula_est.py
```

Test metrics include:
- Mean Absolute Error (MAE)
- Correlation between F0 estimates
- Voicing agreement
- Octave error rate

## Model Training

You can train new GMM models using:

```bash
python train_gmm.py [num_gaussians]
```

The default number of Gaussians is 3, but you can specify a different value for more complex modeling.

## Project Structure

- `nebula_est.py`: Main F0 estimation algorithm implementation
- `preprocess.py`: Signal preprocessing functions
- `estimate_ifsnr.py`: Instantaneous frequency and SNR estimation
- `train_gmm.py`: GMM training code
- `postprocess.py`: F0 post-processing functions
- `viterbi_filter.py`: Viterbi algorithm for optimal path finding
- `binvitsearch.py`: Binary Viterbi search for voicing probability
- `test_nebula.py`: Test and evaluation scripts
- `model/`: Directory containing pre-trained GMM models

## Known Issues and Limitations

- The algorithm may sometimes track the second harmonic instead of the fundamental frequency (octave error)
- Performance may vary depending on the audio characteristics and recording quality
- The algorithm performs best on clean, monophonic audio

## License

[License information]
