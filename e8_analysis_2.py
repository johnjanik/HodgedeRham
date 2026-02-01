#!/usr/bin/env python3
"""
E8 Prime Decoder - Analysis and Visualization

This companion script analyzes the output of e8_prime_decoder.py and generates
visualizations and additional statistical tests.

Usage: python e8_analysis.py
"""

import numpy as np
from pathlib import Path
import struct

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("/home/john/Writing/exterior_derivative/riemann_hypothesis/step3")
ERRORS_FILE = OUTPUT_DIR / "decoding_errors.npy"
BITS_FILE = OUTPUT_DIR / "decoded_bits.bin"
LATTICE_POINTS_FILE = OUTPUT_DIR / "lattice_points.npy"

E8_PACKING_RADIUS = 1.0 / np.sqrt(2)

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    """Load the experiment output files."""
    print("Loading experiment data...")
    
    errors = np.load(ERRORS_FILE)
    print(f"  Errors: {len(errors):,} values")
    
    lattice_points = np.load(LATTICE_POINTS_FILE)
    print(f"  Lattice points: {lattice_points.shape}")
    
    # Load bits from binary file
    with open(BITS_FILE, 'rb') as f:
        data = f.read()
    
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    bits = np.array(bits, dtype=np.uint8)
    print(f"  Bits: {len(bits):,}")
    
    return errors, lattice_points, bits

# =============================================================================
# ADDITIONAL STATISTICAL TESTS
# =============================================================================

def monobit_test(bits: np.ndarray) -> dict:
    """
    NIST Monobit (Frequency) Test.
    Tests if the proportion of 1s is approximately 0.5.
    """
    n = len(bits)
    s = np.sum(bits)
    s_obs = np.abs(s - n/2) / np.sqrt(n/4)
    
    # erfc approximation
    p_value = np.exp(-s_obs**2 / 2) * np.sqrt(2 / np.pi) / s_obs if s_obs > 0 else 1.0
    
    return {
        'test': 'Monobit',
        'n': n,
        'ones': int(s),
        'zeros': int(n - s),
        'statistic': s_obs,
        'p_value': min(p_value, 1.0),
        'pass': p_value > 0.01
    }

def block_frequency_test(bits: np.ndarray, block_size: int = 128) -> dict:
    """
    NIST Block Frequency Test.
    Tests if the proportion of 1s in M-bit blocks is approximately 0.5.
    """
    n = len(bits)
    n_blocks = n // block_size
    
    if n_blocks == 0:
        return {'test': 'Block Frequency', 'error': 'Not enough data'}
    
    chi2 = 0
    for i in range(n_blocks):
        block = bits[i*block_size:(i+1)*block_size]
        pi = np.mean(block)
        chi2 += (pi - 0.5) ** 2
    
    chi2 *= 4 * block_size
    
    # Approximate p-value
    df = n_blocks
    z = (chi2 - df) / np.sqrt(2 * df)
    p_value = 0.5 * (1 - np.tanh(z * 0.7978845608))
    
    return {
        'test': 'Block Frequency',
        'block_size': block_size,
        'n_blocks': n_blocks,
        'chi2': chi2,
        'p_value': p_value,
        'pass': p_value > 0.01
    }

def longest_run_test(bits: np.ndarray) -> dict:
    """
    Test for longest run of ones in a block.
    """
    n = len(bits)
    
    # Find all runs of ones
    runs = []
    current_run = 0
    for bit in bits:
        if bit == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)
    
    if len(runs) == 0:
        return {'test': 'Longest Run', 'longest': 0, 'expected': 0, 'pass': True}
    
    longest = max(runs)
    
    # Expected longest run for n random bits is approximately log2(n)
    expected = np.log2(n) if n > 0 else 0
    
    return {
        'test': 'Longest Run of Ones',
        'longest_run': longest,
        'expected': expected,
        'ratio': longest / expected if expected > 0 else 0,
        'pass': longest < 2 * expected  # Heuristic threshold
    }

def serial_test(bits: np.ndarray, m: int = 4) -> dict:
    """
    Serial test: check uniformity of m-bit patterns.
    """
    n = len(bits)
    n_patterns = n - m + 1
    
    # Count m-bit patterns
    counts = {}
    for i in range(n_patterns):
        pattern = tuple(bits[i:i+m])
        counts[pattern] = counts.get(pattern, 0) + 1
    
    # Chi-square against uniform
    expected = n_patterns / (2 ** m)
    chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
    
    # Add missing patterns
    n_missing = 2**m - len(counts)
    chi2 += n_missing * expected
    
    df = 2**m - 1
    z = (chi2 - df) / np.sqrt(2 * df)
    p_value = 0.5 * (1 - np.tanh(z * 0.7978845608))
    
    return {
        'test': f'Serial (m={m})',
        'n_patterns': len(counts),
        'chi2': chi2,
        'df': df,
        'p_value': p_value,
        'pass': p_value > 0.01
    }

def approximate_entropy_test(bits: np.ndarray, m: int = 4) -> dict:
    """
    Approximate entropy test.
    """
    n = len(bits)
    
    def phi(m_val):
        if m_val == 0:
            return 0
        counts = {}
        for i in range(n):
            pattern = tuple(bits[i:i+m_val] if i+m_val <= n else 
                          np.concatenate([bits[i:], bits[:m_val-(n-i)]]))
            counts[pattern] = counts.get(pattern, 0) + 1
        
        result = 0
        for c in counts.values():
            pi = c / n
            if pi > 0:
                result += pi * np.log(pi)
        return result
    
    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    ap_en = phi_m - phi_m1
    
    chi2 = 2 * n * (np.log(2) - ap_en)
    df = 2 ** m
    
    z = (chi2 - df) / np.sqrt(2 * df)
    p_value = 0.5 * (1 - np.tanh(z * 0.7978845608))
    
    return {
        'test': f'Approximate Entropy (m={m})',
        'ap_en': ap_en,
        'chi2': chi2,
        'p_value': p_value,
        'pass': p_value > 0.01
    }

# =============================================================================
# PATTERN SEARCH
# =============================================================================

def search_for_pi(bits: np.ndarray, max_bits: int = 10000) -> dict:
    """
    Search for the binary expansion of pi in the decoded bits.
    """
    # Binary expansion of pi (first 100 bits after the point)
    pi_bits_str = "11001001000011111101101010100010001000010110100011000010001101001100010011000110011000101000101110"
    pi_bits = np.array([int(b) for b in pi_bits_str], dtype=np.uint8)
    
    search_bits = bits[:max_bits]
    
    # Search for matches of various lengths
    results = {}
    for match_len in [8, 16, 24, 32]:
        if match_len > len(pi_bits):
            continue
        pi_pattern = pi_bits[:match_len]
        
        matches = []
        for i in range(len(search_bits) - match_len + 1):
            if np.array_equal(search_bits[i:i+match_len], pi_pattern):
                matches.append(i)
        
        # Expected matches for random data
        expected = (len(search_bits) - match_len + 1) / (2 ** match_len)
        
        results[match_len] = {
            'matches': len(matches),
            'positions': matches[:10],  # First 10 positions
            'expected_random': expected
        }
    
    return results

def search_for_ascii(bits: np.ndarray, min_run: int = 4) -> list:
    """
    Search for runs of printable ASCII characters.
    """
    n_bytes = len(bits) // 8
    runs = []
    current_run = []
    current_start = 0
    
    for i in range(n_bytes):
        byte_bits = bits[i*8:(i+1)*8]
        byte_val = sum(int(b) << (7-j) for j, b in enumerate(byte_bits))
        
        if 32 <= byte_val < 127:  # Printable ASCII
            if len(current_run) == 0:
                current_start = i
            current_run.append(chr(byte_val))
        else:
            if len(current_run) >= min_run:
                runs.append({
                    'start': current_start,
                    'length': len(current_run),
                    'text': ''.join(current_run)
                })
            current_run = []
    
    if len(current_run) >= min_run:
        runs.append({
            'start': current_start,
            'length': len(current_run),
            'text': ''.join(current_run)
        })
    
    return runs

# =============================================================================
# ERROR DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_error_distribution(errors: np.ndarray) -> dict:
    """
    Analyze the distribution of decoding errors.
    """
    n = len(errors)
    
    # Basic stats
    results = {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'median': np.median(errors),
        'min': np.min(errors),
        'max': np.max(errors),
    }
    
    # Fraction exceeding threshold
    threshold = E8_PACKING_RADIUS
    results['above_threshold'] = np.sum(errors >= threshold) / n
    results['below_threshold'] = np.sum(errors < threshold) / n
    
    # Histogram bins
    bins = np.linspace(0, 1.5, 31)
    hist, _ = np.histogram(errors, bins=bins)
    results['histogram'] = hist
    results['bin_edges'] = bins
    
    # Fit to Rayleigh distribution (expected for Gaussian noise in R^8)
    # Rayleigh: f(x) = x/σ² exp(-x²/2σ²), mean = σ√(π/2)
    rayleigh_sigma = results['mean'] / np.sqrt(np.pi / 2)
    results['rayleigh_sigma'] = rayleigh_sigma
    
    # Expected histogram under Rayleigh
    bin_centers = (bins[:-1] + bins[1:]) / 2
    rayleigh_pdf = (bin_centers / rayleigh_sigma**2) * np.exp(-bin_centers**2 / (2 * rayleigh_sigma**2))
    rayleigh_expected = rayleigh_pdf * n * (bins[1] - bins[0])
    results['rayleigh_expected'] = rayleigh_expected
    
    # Chi-square test against Rayleigh
    mask = rayleigh_expected > 5  # Only use bins with sufficient expected count
    if np.sum(mask) > 1:
        chi2 = np.sum((hist[mask] - rayleigh_expected[mask])**2 / rayleigh_expected[mask])
        df = np.sum(mask) - 1
        z = (chi2 - df) / np.sqrt(2 * df)
        p_value = 0.5 * (1 - np.tanh(z * 0.7978845608))
        results['rayleigh_chi2'] = chi2
        results['rayleigh_p_value'] = p_value
    
    return results

# =============================================================================
# LATTICE POINT ANALYSIS
# =============================================================================

def analyze_lattice_points(lattice_points: np.ndarray) -> dict:
    """
    Analyze the distribution of decoded E8 lattice points.
    """
    n = len(lattice_points)
    
    # Count unique lattice points
    unique_points = set(tuple(lp) for lp in lattice_points)
    results = {
        'total_points': n,
        'unique_points': len(unique_points),
        'repetition_rate': 1 - len(unique_points) / n
    }
    
    # Most common lattice points
    from collections import Counter
    point_counts = Counter(tuple(lp) for lp in lattice_points)
    results['most_common'] = point_counts.most_common(10)
    
    # Norms of lattice points
    norms = np.linalg.norm(lattice_points, axis=1)
    results['mean_norm'] = np.mean(norms)
    results['std_norm'] = np.std(norms)
    
    # Distribution of norms (should cluster at sqrt(2), 2, sqrt(6), etc. for E8)
    norm_bins = np.arange(0, 5.5, 0.1)
    norm_hist, _ = np.histogram(norms, bins=norm_bins)
    results['norm_histogram'] = norm_hist
    results['norm_bin_edges'] = norm_bins
    
    # Count points at each shell
    shells = {
        'origin': np.sum(norms < 0.1),
        'sqrt2': np.sum(np.abs(norms - np.sqrt(2)) < 0.1),
        '2': np.sum(np.abs(norms - 2) < 0.1),
        'sqrt6': np.sum(np.abs(norms - np.sqrt(6)) < 0.1),
    }
    results['shell_counts'] = shells
    
    return results

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis():
    """Run the complete analysis suite."""
    
    # Load data
    errors, lattice_points, bits = load_data()
    
    print("\n" + "=" * 70)
    print("EXTENDED STATISTICAL ANALYSIS")
    print("=" * 70)
    
    # NIST-style tests
    print("\n--- NIST Statistical Tests ---")
    tests = [
        monobit_test(bits),
        block_frequency_test(bits, block_size=128),
        longest_run_test(bits),
        serial_test(bits, m=4),
        serial_test(bits, m=8),
        approximate_entropy_test(bits, m=4),
    ]
    
    for test in tests:
        if 'error' in test:
            print(f"{test['test']}: {test['error']}")
        else:
            status = "PASS" if test['pass'] else "FAIL"
            p_val = test.get('p_value', None)
            if p_val is not None:
                print(f"{test['test']}: p={p_val:.4f} [{status}]")
            else:
                print(f"{test['test']}: [{status}]")
    
    # Pattern search
    print("\n--- Pattern Search ---")
    
    pi_results = search_for_pi(bits)
    print("Search for binary expansion of π:")
    for length, data in pi_results.items():
        print(f"  {length}-bit pattern: {data['matches']} matches "
              f"(expected random: {data['expected_random']:.2f})")
    
    ascii_runs = search_for_ascii(bits)
    print(f"\nASCII runs (length ≥ 4): {len(ascii_runs)} found")
    for run in ascii_runs[:5]:
        print(f"  Position {run['start']}: \"{run['text']}\" (length {run['length']})")
    if len(ascii_runs) > 5:
        print(f"  ... and {len(ascii_runs) - 5} more")
    
    # Error distribution
    print("\n--- Error Distribution Analysis ---")
    error_analysis = analyze_error_distribution(errors)
    print(f"Mean error: {error_analysis['mean']:.4f}")
    print(f"Std error: {error_analysis['std']:.4f}")
    print(f"Below threshold: {100*error_analysis['below_threshold']:.2f}%")
    print(f"Above threshold: {100*error_analysis['above_threshold']:.2f}%")
    print(f"Rayleigh fit σ: {error_analysis['rayleigh_sigma']:.4f}")
    if 'rayleigh_p_value' in error_analysis:
        print(f"Rayleigh chi² p-value: {error_analysis['rayleigh_p_value']:.4f}")
    
    # Lattice point analysis
    print("\n--- Lattice Point Analysis ---")
    lp_analysis = analyze_lattice_points(lattice_points)
    print(f"Total points: {lp_analysis['total_points']:,}")
    print(f"Unique points: {lp_analysis['unique_points']:,}")
    print(f"Repetition rate: {100*lp_analysis['repetition_rate']:.2f}%")
    print(f"Mean norm: {lp_analysis['mean_norm']:.4f}")
    print("\nShell distribution:")
    for shell, count in lp_analysis['shell_counts'].items():
        print(f"  {shell}: {count:,}")
    print("\nMost common lattice points:")
    for point, count in lp_analysis['most_common'][:5]:
        print(f"  {point}: {count:,} times")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    n_pass = sum(1 for t in tests if t.get('pass', False))
    n_fail = len(tests) - n_pass
    print(f"NIST tests: {n_pass} pass, {n_fail} fail")
    
    if n_fail == 0 and error_analysis['above_threshold'] < 0.01:
        print("\nCONCLUSION: Data is consistent with random bits and RH-compatible errors.")
    elif n_fail > 2:
        print("\nCONCLUSION: Significant non-random structure detected!")
    else:
        print("\nCONCLUSION: Marginal evidence of structure; further investigation needed.")
    
    return {
        'tests': tests,
        'pi_search': pi_results,
        'ascii_runs': ascii_runs,
        'error_analysis': error_analysis,
        'lattice_analysis': lp_analysis
    }

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("E8 Prime Decoder - Extended Analysis")
    print("=" * 70)
    
    try:
        results = run_analysis()
        print("\nAnalysis completed successfully.")
    except FileNotFoundError as e:
        print(f"\nError: Could not find experiment output files.")
        print(f"Please run e8_prime_decoder.py first.")
        raise
    except Exception as e:
        print(f"\nError during analysis: {e}")
        raise
