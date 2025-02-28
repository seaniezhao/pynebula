"""
Functions for loading pre-trained GMM models for Nebula F0 estimation.
"""
import os
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture

def load_model(model_dir):
    """
    Load pre-trained models from the specified directory.
    
    Expected to find:
    - GMM model files (pickle format) for each frequency band
    - Calibration data file (Lcal)
    
    Parameters:
        model_dir: Directory containing model files
    
    Returns:
        model_dict: Dictionary containing loaded models and calibration data
    """
    models = {}
    
    # Check if the model directory exists
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found")
    
    # Try to load GMM models for all frequency bands
    for b in range(36):  # Assume 36 frequency bands as in the original implementation
        model_file = os.path.join(model_dir, f"gmm_band_{b}.pkl")
        
        if os.path.isfile(model_file):
            try:
                with open(model_file, "rb") as f:
                    models[b] = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load model for band {b}: {e}")
    
    if not models:
        print("Warning: No GMM models found in the specified directory")
    
    # Try to load calibration data
    Lcal = None
    Lcal_file = os.path.join(model_dir, "Lcal.txt")
    
    if os.path.isfile(Lcal_file):
        try:
            Lcal = np.loadtxt(Lcal_file)
        except Exception as e:
            print(f"Warning: Could not load calibration data: {e}")
    
    return {"models": models, "Lcal": Lcal}

def check_model_compatibility(model):
    """
    Check if the loaded model is compatible with the current implementation.
    
    Parameters:
        model: Model dictionary from load_model()
    
    Returns:
        is_compatible: True if the model is compatible, False otherwise
    """
    # Check if models key exists and is a dictionary
    if not isinstance(model.get("models", None), dict):
        return False
    
    # Check if at least one model is a GaussianMixture
    for band, gmm in model["models"].items():
        if isinstance(gmm, GaussianMixture):
            return True
    
    return False
