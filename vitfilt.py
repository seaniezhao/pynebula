import numpy as np
from scipy.sparse import dok_matrix

def viterbi_filter(smap, pout, ptran):
    """
    Python implementation of the vitfilt1 MEX function.
    
    Parameters:
    - smap: (nfrm, nstate) State mapping matrix
    - pout: (nfrm, nstate) Observation probabilities
    - ptran: (ntrans,) State transition probabilities by distance
    
    Returns:
    - s: (nfrm,) Optimal state sequence
    - L: Log-likelihood of the optimal path
    """
    nfrm, nstate = pout.shape
    ntrans = len(ptran)
    
    # Initialize score and backpointer matrices
    V = np.full((nfrm, nstate), -np.inf)  # -infinity initialization
    backpointer = np.zeros((nfrm, nstate), dtype=int)
    
    # Initialize first frame
    V[0, :] = np.log(np.maximum(pout[0, :], 1e-10))
    
    # Forward pass - compute Viterbi scores
    for t in range(1, nfrm):
        for j in range(nstate):
            state_j = int(smap[t, j])
            
            max_prob = -np.inf
            max_state = 0
            
            # Check all possible previous states
            for i in range(nstate):
                state_i = int(smap[t-1, i])
                
                # Compute state distance - ensure it's an integer
                dist = int(min(abs(state_i - state_j), abs(state_j - state_i)))
                
                # Get transition probability based on distance
                if dist < ntrans:
                    trans_prob = ptran[dist]
                else:
                    trans_prob = 1e-10
                
                # Compute score
                prob = V[t-1, i] + np.log(max(trans_prob, 1e-10)) + np.log(max(pout[t, j], 1e-10))
                
                # Update if better
                if prob > max_prob:
                    max_prob = prob
                    max_state = i
            
            V[t, j] = max_prob
            backpointer[t, j] = max_state
    
    # Backward pass - find optimal path
    s = np.zeros(nfrm, dtype=int)
    s[-1] = np.argmax(V[-1, :])
    
    # Trace back through the backpointer matrix
    for t in range(nfrm-2, -1, -1):
        s[t] = backpointer[t+1, s[t+1]]
    
    # Log-likelihood of the optimal path
    L = np.max(V[-1, :])
    
    return s, L


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # ========================= TEST CASE 1: ZIGZAG PATTERN =========================
    print("="*80)
    print("TEST CASE 1: PREDEFINED ZIGZAG PATTERN")
    print("This test shows a clear zigzag pattern with obvious optimal path")
    print("="*80)
    
    # Create a test dataset
    nfrm, nstate = 15, 6
    
    # State map: each column represents a possible state (0-5)
    smap = np.ones((nfrm, nstate)) * np.arange(nstate)
    
    # Create a zigzag pattern for the optimal path
    optimal_path = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4]
    
    # Create observation probabilities
    # We'll make a clear distinction: 0.9 probability for the optimal path, 0.02 for others
    pout = np.ones((nfrm, nstate)) * 0.02
    for i in range(nfrm):
        pout[i, optimal_path[i]] = 0.9
    
    # ------------------- Run with different transition probabilities -------------------
    
    # 1. Equal transition probabilities (state distance doesn't matter)
    equal_ptran = np.ones(nstate)
    s_equal, L_equal = viterbi_filter(smap, pout, equal_ptran)
    
    # 2. Strict transition probabilities (strongly prefer staying in same state or adjacent states)
    strict_ptran = np.array([0.95, 0.04, 0.01, 0.0, 0.0, 0.0])
    s_strict, L_strict = viterbi_filter(smap, pout, strict_ptran)
    
    # Print results
    print(f"Expected optimal path: {optimal_path}")
    print("\nResults with equal transition probabilities:")
    print(f"Viterbi path: {s_equal}")
    print(f"Log likelihood: {L_equal}")
    print(f"Matches expected: {np.array_equal(s_equal, optimal_path)}")
    
    print("\nResults with strict transition probabilities:")
    print(f"Viterbi path: {s_strict}")
    print(f"Log likelihood: {L_strict}")
    print(f"Matches expected: {np.array_equal(s_strict, optimal_path)}")
    
    # Visualize results
    plt.figure(figsize=(14, 10))
    plt.suptitle("Test Case 1: Zigzag Pattern", fontsize=16)
    
    # Plot observation probabilities
    plt.subplot(3, 1, 1)
    plt.imshow(pout.T, aspect='auto', cmap='viridis', interpolation='none')
    plt.colorbar(label='Observation Probability')
    plt.title('Observation Probabilities')
    plt.xlabel('Frame')
    plt.ylabel('State')
    plt.yticks(range(nstate))
    
    # Add markers for the optimal path
    for i, opt_state in enumerate(optimal_path):
        plt.plot(i, opt_state, 'ro', markersize=8)
    
    # Plot results with equal transition probabilities
    plt.subplot(3, 1, 2)
    plt.imshow(pout.T, aspect='auto', cmap='Blues', alpha=0.5, interpolation='none')
    plt.plot(range(nfrm), s_equal, 'r-o', linewidth=2, markersize=10, label='Viterbi Path')
    plt.title('Equal Transition Probabilities')
    plt.xlabel('Frame')
    plt.ylabel('State')
    plt.yticks(range(nstate))
    plt.legend()
    
    # Plot results with strict transition probabilities
    plt.subplot(3, 1, 3)
    plt.imshow(pout.T, aspect='auto', cmap='Blues', alpha=0.5, interpolation='none')
    plt.plot(range(nfrm), s_strict, 'r-o', linewidth=2, markersize=10, label='Viterbi Path')
    plt.title('Strict Transition Probabilities')
    plt.xlabel('Frame')
    plt.ylabel('State')
    plt.yticks(range(nstate))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('viterbi_test_zigzag.png')
    plt.show()
    
    # ========================= TEST CASE 2: STAIR PATTERN WITH NOISE =========================
    print("\n"+"="*80)
    print("TEST CASE 2: STAIR PATTERN WITH NOISE")
    print("This test shows a stair pattern with noise to demonstrate path smoothing")
    print("="*80)
    
    # Create a test dataset
    nfrm, nstate = 30, 10
    
    # State map: each column represents a possible state (0-9)
    smap = np.ones((nfrm, nstate)) * np.arange(nstate)
    
    # Create a base stair pattern
    base_pattern = []
    for i in range(0, 10, 2):
        # Each step is 3 frames wide
        base_pattern.extend([i, i, i])  
    
    # Ensure the pattern is at least 30 frames long by repeating
    while len(base_pattern) < nfrm:
        base_pattern.append(base_pattern[-2])  # Continue with recent values
    
    # Trim if too long
    base_pattern = base_pattern[:nfrm]
    
    # Add noise (create occasional outliers)
    noisy_path = base_pattern.copy()
    noisy_path[5] = 8    # Big jump up
    noisy_path[12] = 1   # Big jump down
    noisy_path[20] = 9   # Big jump up
    
    # Create observation probabilities
    # We'll make strong but not overwhelming preference: 0.7 for optimal states
    pout = np.ones((nfrm, nstate)) * 0.03
    for i in range(nfrm):
        pout[i, noisy_path[i]] = 0.7
    
    # ------------------- Run with different transition probabilities -------------------
    
    # 1. Equal transition probabilities (state distance doesn't matter)
    equal_ptran = np.ones(nstate)
    s_equal, L_equal = viterbi_filter(smap, pout, equal_ptran)
    
    # 2. Moderate transition probabilities (prefers smaller jumps but allows bigger ones)
    mod_ptran = np.array([0.5, 0.25, 0.15, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001, 0.001])
    s_mod, L_mod = viterbi_filter(smap, pout, mod_ptran)
    
    # 3. Strict transition probabilities (strongly prefer staying in same state or adjacent states)
    strict_ptran = np.array([0.8, 0.15, 0.04, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_strict, L_strict = viterbi_filter(smap, pout, strict_ptran)
    
    # Print results
    print(f"\nResults with equal transition probabilities:")
    print(f"Log likelihood: {L_equal}")
    
    print(f"\nResults with moderate transition probabilities:")
    print(f"Log likelihood: {L_mod}")
    
    print(f"\nResults with strict transition probabilities:")
    print(f"Log likelihood: {L_strict}")
    
    # Visualize results
    plt.figure(figsize=(14, 12))
    plt.suptitle("Test Case 2: Stair Pattern with Noise", fontsize=16)
    
    # Plot observation probabilities
    plt.subplot(4, 1, 1)
    plt.imshow(pout.T, aspect='auto', cmap='viridis', interpolation='none')
    plt.colorbar(label='Observation Probability')
    plt.title('Observation Probabilities (Noise Added)')
    plt.xlabel('Frame')
    plt.ylabel('State')
    plt.yticks(range(nstate))
    
    # Plot noisy path
    for i, opt_state in enumerate(noisy_path):
        plt.plot(i, opt_state, 'ro', markersize=5)
    
    # Plot results with equal transition probabilities
    plt.subplot(4, 1, 2)
    plt.imshow(pout.T, aspect='auto', cmap='Blues', alpha=0.3, interpolation='none')
    plt.plot(range(nfrm), s_equal, 'r-o', linewidth=2, markersize=8, label='Viterbi Path')
    plt.title('Equal Transition Probabilities (No Smoothing)')
    plt.xlabel('Frame')
    plt.ylabel('State')
    plt.yticks(range(nstate))
    plt.legend()
    
    # Plot results with moderate transition probabilities
    plt.subplot(4, 1, 3)
    plt.imshow(pout.T, aspect='auto', cmap='Blues', alpha=0.3, interpolation='none')
    plt.plot(range(nfrm), s_mod, 'r-o', linewidth=2, markersize=8, label='Viterbi Path')
    plt.title('Moderate Transition Probabilities (Medium Smoothing)')
    plt.xlabel('Frame')
    plt.ylabel('State')
    plt.yticks(range(nstate))
    plt.legend()
    
    # Plot results with strict transition probabilities
    plt.subplot(4, 1, 4)
    plt.imshow(pout.T, aspect='auto', cmap='Blues', alpha=0.3, interpolation='none')
    plt.plot(range(nfrm), s_strict, 'r-o', linewidth=2, markersize=8, label='Viterbi Path')
    plt.title('Strict Transition Probabilities (Strong Smoothing)')
    plt.xlabel('Frame')
    plt.ylabel('State')
    plt.yticks(range(nstate))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('viterbi_test_stair.png')
    plt.show()
    
    # ========================= TEST CASE 3: F0 TRACKING SIMULATION =========================
    print("\n"+"="*80)
    print("TEST CASE 3: F0 TRACKING SIMULATION")
    print("This test simulates F0 tracking with an ABABC pattern and a dropout section")
    print("="*80)
    
    # Create a test dataset
    nfrm, nstate = 40, 8
    
    # State map: each column represents a possible state (0-7)
    smap = np.ones((nfrm, nstate)) * np.arange(nstate)
    
    # Create a pitch contour that mimics F0 tracking: pattern ABABC with a dropout section
    # A: stable note, B: higher note, C: dropout and transition
    f0_pattern = [
        1, 1, 1, 1, 1,                # A: stable note
        4, 4, 4, 4, 4,                # B: higher note
        1, 1, 1, 1, 1,                # A: back to stable note
        4, 4, 4, 4, 4,                # B: higher note again
        0, 0, 0, 0, 0,                # C: silence/dropout (low confidence)
        2, 2, 3, 3, 4, 4, 5, 5, 6, 6  # Rising transition
    ]
    
    # Create observation probabilities
    pout = np.ones((nfrm, nstate)) * 0.05  # Base low probability for all states
    
    # High probability for target sequence
    for i in range(nfrm):
        pout[i, f0_pattern[i]] = 0.6
        
    # Lower probabilities for dropout section (frames 20-24)
    for i in range(20, 25):
        pout[i, :] = 0.125  # Equal probabilities for all states (complete uncertainty)
    
    # ------------------- Run with different transition probabilities -------------------
    
    # 1. Permissive transition probabilities (allows jumps)
    permissive_ptran = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
    s_permissive, L_permissive = viterbi_filter(smap, pout, permissive_ptran)
    
    # 2. Strict transition probabilities (strongly prefer staying in same state or adjacent states)
    strict_ptran = np.array([0.7, 0.25, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_strict, L_strict = viterbi_filter(smap, pout, strict_ptran)
    
    # Visualize results
    plt.figure(figsize=(14, 10))
    plt.suptitle("Test Case 3: F0 Tracking Simulation", fontsize=16)
    
    # Plot observation probabilities
    plt.subplot(3, 1, 1)
    plt.imshow(pout.T, aspect='auto', cmap='viridis', interpolation='none')
    plt.colorbar(label='Observation Probability')
    plt.title('Observation Probabilities with Dropout Region (frames 20-24)')
    plt.xlabel('Frame')
    plt.ylabel('F0 State')
    plt.yticks(range(nstate))
    
    # Add rectangle to highlight dropout region
    rect = patches.Rectangle((20, -0.5), 5, nstate, linewidth=2, 
                            edgecolor='red', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect)
    plt.text(22.5, -0.9, "Dropout", color='red', 
             horizontalalignment='center', fontsize=10)
    
    # Plot with permissive transition probabilities
    plt.subplot(3, 1, 2)
    plt.imshow(pout.T, aspect='auto', cmap='Blues', alpha=0.3, interpolation='none')
    plt.plot(range(nfrm), s_permissive, 'r-o', linewidth=2, markersize=8)
    plt.title('Permissive Transition Probabilities')
    plt.xlabel('Frame')
    plt.ylabel('F0 State')
    plt.yticks(range(nstate))
    
    # Add rectangle to highlight dropout region
    rect = patches.Rectangle((20, -0.5), 5, nstate, linewidth=2, 
                            edgecolor='red', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect)
    
    # Plot with strict transition probabilities
    plt.subplot(3, 1, 3)
    plt.imshow(pout.T, aspect='auto', cmap='Blues', alpha=0.3, interpolation='none')
    plt.plot(range(nfrm), s_strict, 'r-o', linewidth=2, markersize=8)
    plt.title('Strict Transition Probabilities (Smoother Path)')
    plt.xlabel('Frame')
    plt.ylabel('F0 State')
    plt.yticks(range(nstate))
    
    # Add rectangle to highlight dropout region
    rect = patches.Rectangle((20, -0.5), 5, nstate, linewidth=2, 
                            edgecolor='red', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect)
    
    plt.tight_layout()
    plt.savefig('viterbi_test_f0.png')
    plt.show()
    
    print("\nAll visualizations saved to:")
    print("1. viterbi_test_zigzag.png")
    print("2. viterbi_test_stair.png")
    print("3. viterbi_test_f0.png")