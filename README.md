# PyNebula

Python implementation of the Nebula fundamental frequency (F0) estimation algorithm, originally written in MATLAB/Octave.

## Project Overview

PyNebula provides a robust method for estimating the fundamental frequency (F0) of speech or musical signals using Gaussian Mixture Models (GMMs). This project is a Python rewrite of the original MATLAB/Octave Nebula project.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
import librosa
from nebula_est import nebula_est
from load_model import load_model

# Load audio file
audio_file = 'example.wav'
x, fs = librosa.load(audio_file, sr=None)

# Load pre-trained models
model = load_model('./model')

# Estimate F0 with 5ms frame intervals
f0 = nebula_est(model, x, fs, dt=0.005)

# Now f0 contains the estimated fundamental frequency for each frame
```

## Project Structure

- `main.py`: Demo script showing how to perform F0 estimation
- `nebula_est.py`: Core F0 estimation interface
- `preprocess.py`: Signal preprocessing functions
- `postprocess.py`: F0 post-processing code
- `load_model.py`: Functions to load pre-trained models
- `train_gmm.py`: GMM training code
- `make_random_dataset.py`: Code to generate training data
- `model/`: Directory containing pre-trained models

## Dependencies

- NumPy
- SciPy
- Librosa
- scikit-learn
- Matplotlib

## License

[License information]
# pynebula
