# PyNebula

A Python implementation of the Nebula fundamental frequency (F0) estimation algorithm, which uses Gaussian Mixture Models (GMMs) to estimate pitch in speech and musical signals.

original implementation: https://github.com/Sleepwalking/nebula

This implementation is fully based on the original implementation, but will be optimized and extended.

## Project Overview

PyNebula implements a GMM-based likelihood mapping approach to pitch tracking. This method creates a likelihood map of potential fundamental frequencies and selects the most likely F0 trajectory using dynamic programming. The implementation includes:

- Signal preprocessing with SNR and instantaneous frequency features
- GMM-based likelihood calculation for potential F0 candidates
- Viterbi filtering for optimal F0 trajectory extraction
- Voicing probability estimation

PyNebula can be compared with other F0 estimation methods like PyWorld to evaluate its performance.

## TODO

- [ ] optimize the execution speed(too slow now)
- [ ] train on real dataset
- ...

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

### Test F0 estimation
```bash
python test_nebula.py
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


