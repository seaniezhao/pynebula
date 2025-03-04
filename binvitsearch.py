import numpy as np
import matplotlib.pyplot as plt

def binvitsearch(pob, ptrans):
    """
    Python implementation of the MATLAB binvitsearch binary state Viterbi filtering algorithm.
    
    Parameters:
    - pob: (N,) Observation probabilities, values between (0,1)
    - ptrans: (float) State transition probability
    
    Returns:
    - q: (N,) Optimal state sequence (0=unvoiced, 1=voiced)
    - ptotal: Log-probability of the optimal path
    """
    length_pob = len(pob)
    a = np.zeros((length_pob, 2))  # Viterbi scores matrix
    bt = np.zeros((length_pob, 2), dtype=int)  # Backtracking pointers
    
    # Compute log probabilities
    pstay = np.log(1 - ptrans)  # Log probability of staying in current state
    ptrans = np.log(ptrans)     # Log probability of transitioning to another state
    pob1 = np.log(pob)          # Log probability of observation being in state 1 (voiced)
    pob2 = np.log(1 - pob)      # Log probability of observation being in state 0 (unvoiced)
    
    # Initialize first frame
    a[0, 0] = pob1[0]  # First state is 1 (voiced) in MATLAB indexing
    a[0, 1] = pob2[0]  # Second state is 2 (unvoiced) in MATLAB indexing
    
    # Forward recursion - compute Viterbi scores
    for i in range(1, length_pob):
        # For state 1 (voiced)
        if a[i-1, 1] + ptrans > a[i-1, 0] + pstay:
            a[i, 0] = a[i-1, 1] + ptrans + pob1[i]
            bt[i, 0] = 2  # Came from state 2 (matching MATLAB 1-based indexing)
        else:
            a[i, 0] = a[i-1, 0] + pstay + pob1[i]
            bt[i, 0] = 1  # Came from state 1 (matching MATLAB 1-based indexing)
        
        # For state 2 (unvoiced)
        if a[i-1, 0] + ptrans > a[i-1, 1] + pstay:
            a[i, 1] = a[i-1, 0] + ptrans + pob2[i]
            bt[i, 1] = 1  # Came from state 1 (matching MATLAB 1-based indexing)
        else:
            a[i, 1] = a[i-1, 1] + pstay + pob2[i]
            bt[i, 1] = 2  # Came from state 2 (matching MATLAB 1-based indexing)
    
    # Find the best end state
    ptotal = np.max(a[-1, :])
    last = np.argmax(a[-1, :]) + 1  # +1 to match MATLAB 1-based indexing
    
    # Backward pass - trace the optimal path
    q = np.zeros(length_pob, dtype=int)
    q[-1] = last
    
    for i in range(length_pob-2, -1, -1):
        q[i] = bt[i+1, q[i+1]-1]  # -1 to adjust for 0-based array indexing in Python
    
    # Convert to 0-based indexing for Python (0=unvoiced, 1=voiced)
    q = q - 1
    
    return q, ptotal


if __name__ == "__main__":
    # Test case similar to the original MATLAB implementation
    print("Testing binvitsearch with example data\n")
    
    # Example 1: Simple alternating pattern
    pob = np.array([0.8, 0.2, 0.9, 0.1, 0.7])  # Observation probabilities
    ptrans = 0.1  # State transition probability
    
    q, ptotal = binvitsearch(pob, ptrans)
    
    print("Example 1: Simple alternating probability pattern")
    print(f"Observation probabilities: {pob}")
    print(f"Transition probability: {ptrans}")
    print(f"Optimal state sequence: {q}")
    print(f"Total log-probability: {ptotal:.4f}\n")
    
    # Example 2: Longer sequence with some ambiguity
    np.random.seed(42)  # For reproducible results
    length = 30
    # Generate a sequence with clear voiced regions (high prob) and unvoiced regions (low prob)
    pob2 = np.zeros(length)
    pob2[0:10] = 0.9 + 0.05 * np.random.randn(10)  # Voiced region
    pob2[10:15] = 0.4 + 0.1 * np.random.randn(5)   # Ambiguous region
    pob2[15:25] = 0.1 + 0.05 * np.random.randn(10) # Unvoiced region
    pob2[25:30] = 0.7 + 0.1 * np.random.randn(5)   # Back to voiced
    
    # Clip probabilities to be between 0 and 1
    pob2 = np.clip(pob2, 0.01, 0.99)
    
    # Two different transition probabilities to compare
    ptrans_low = 0.05   # Less likely to change state (smoother)
    ptrans_high = 0.3   # More likely to change state (more responsive)
    
    q_low, ptotal_low = binvitsearch(pob2, ptrans_low)
    q_high, ptotal_high = binvitsearch(pob2, ptrans_high)
    
    print("Example 2: Longer sequence with voiced, unvoiced, and ambiguous regions")
    print(f"Low transition probability: {ptrans_low}, Log-probability: {ptotal_low:.4f}")
    print(f"High transition probability: {ptrans_high}, Log-probability: {ptotal_high:.4f}")
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(pob2, 'b-', label='Voice probability')
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    plt.title('Input Voice Probability')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.step(range(length), q_low, 'r-', where='mid', label=f'ptrans={ptrans_low}')
    plt.title('Voicing Decision (Low Transition Probability)')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Unvoiced', 'Voiced'])
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.step(range(length), q_high, 'g-', where='mid', label=f'ptrans={ptrans_high}')
    plt.title('Voicing Decision (High Transition Probability)')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Unvoiced', 'Voiced'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('binvitsearch_test.png')
    plt.show()
    
    print("\nVisualization saved as 'binvitsearch_test.png'")