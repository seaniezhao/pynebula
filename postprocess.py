"""
Post-processing functions for smoothing and refining F0 estimates.
"""
import numpy as np
from scipy.signal import medfilt

def postprocess_results(f0_raw):
    """
    Post-process raw F0 estimates to produce a smoother and more accurate F0 contour.
    
    This function applies:
    1. Median filtering to remove outliers
    2. Interpolation to fill in unvoiced regions (gaps in the F0 contour)
    
    Parameters:
        f0_raw: Raw F0 estimates (numpy array)
    
    Returns:
        f0_processed: Processed F0 contour
    """
    # Apply median filtering to remove outliers
    f0_filtered = medfilt(f0_raw, kernel_size=5)
    
    # Simple gap filling through linear interpolation
    # In a real implementation, we would want to identify voiced/unvoiced regions
    # and only interpolate within reasonable boundaries
    f0_processed = fill_gaps(f0_filtered)
    
    return f0_processed

def fill_gaps(f0):
    """
    Fill gaps in the F0 contour using linear interpolation.
    
    In a more advanced implementation, this would consider:
    - Voiced/unvoiced detection
    - Maximum gap duration for interpolation
    - Boundary conditions
    
    Parameters:
        f0: F0 contour with potential gaps (numpy array)
    
    Returns:
        f0_filled: F0 contour with gaps filled by interpolation
    """
    # Make a copy to avoid modifying the input
    f0_filled = f0.copy()
    
    # Find indices where F0 is zero or NaN (indicating unvoiced regions)
    unvoiced = np.where(np.logical_or(f0_filled == 0, np.isnan(f0_filled)))[0]
    
    # If there are no gaps, return the original
    if len(unvoiced) == 0:
        return f0_filled
    
    # Find continuous segments of unvoiced frames
    gaps = []
    gap_start = unvoiced[0]
    
    for i in range(1, len(unvoiced)):
        if unvoiced[i] > unvoiced[i-1] + 1:
            # Gap ends at previous index
            gaps.append((gap_start, unvoiced[i-1]))
            # New gap starts
            gap_start = unvoiced[i]
    
    # Add the last gap
    gaps.append((gap_start, unvoiced[-1]))
    
    # Interpolate across each gap
    for start, end in gaps:
        # Only interpolate if the gap is surrounded by voiced frames
        if start > 0 and end < len(f0_filled) - 1:
            # Linear interpolation
            left_value = f0_filled[start - 1]
            right_value = f0_filled[end + 1]
            
            # Create interpolation values
            gap_length = end - start + 1
            interp_values = np.linspace(left_value, right_value, gap_length + 2)[1:-1]
            
            # Fill the gap
            f0_filled[start:end+1] = interp_values
    
    return f0_filled
