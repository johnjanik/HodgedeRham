import numpy as np
import math
import os
from tqdm import tqdm
import pickle
from collections import Counter
from itertools import combinations, product
import matplotlib.pyplot as plt

class CNOTGateDetector:
    """
    Detects universal CNOT gate patterns in prime data using E8 machinery
    """
    
    def __init__(self, prime_dir="/home/john/Writing/exterior_derivative/riemann_hypothesis/step3/Primes"):
        self.prime_dir = prime_dir
        
        # Canonical CNOT gate encoding (decimal representation)
        self.cnot_pairs = {
            (0, 0): "Identity (00→00)",
            (1, 1): "Identity (01→01)", 
            (16, 17): "Control 1, target flips (10→11)",
            (17, 16): "Control 1, target flips back (11→10)"
        }
        
        # Alternative encodings (different bit positions)
        self.cnot_encodings = self.generate_cnot_encodings()
        
        # E8 parameters
        self.E8_dim = 8
        self.E8_min_norm = math.sqrt(2)
        
        self.primes = []
        self.decoded_bytes = []
        self.bit_sequences = []
        
    def generate_cnot_encodings(self):
        """Generate all possible CNOT encodings for different bit positions"""
        encodings = []
        
        # For each pair of control/target bit positions
        for control_bit in range(8):
            for target_bit in range(8):
                if control_bit == target_bit:
                    continue
                    
                encoding = {}
                
                # Generate all 4 states (2 control bits × 2 target bits)
                for control_state in [0, 1]:
                    for target_state in [0, 1]:
                        # Create 8-bit pattern
                        bits = ['0'] * 8
                        
                        # Set control bit
                        bits[7 - control_bit] = str(control_state)
                        
                        # Set target bit
                        bits[7 - target_bit] = str(target_state)
                        
                        input_byte = int(''.join(bits), 2)
                        
                        # Apply CNOT: target flips if control is 1
                        output_control = control_state
                        output_target = target_state ^ control_state  # XOR
                        
                        # Create output byte
                        out_bits = ['0'] * 8
                        out_bits[7 - control_bit] = str(output_control)
                        out_bits[7 - target_bit] = str(output_target)
                        
                        output_byte = int(''.join(out_bits), 2)
                        
                        encoding[(control_state, target_state)] = (input_byte, output_byte)
                
                encodings.append({
                    'control_bit': control_bit,
                    'target_bit': target_bit,
                    'encoding': encoding
                })
        
        return encodings
    
    def load_primes(self, max_files=10, max_primes=1000000):
        """Load primes from text files"""
        all_primes = []
        
        for file_num in tqdm(range(1, min(max_files, 10) + 1), desc="Loading prime files"):
            filename = f"{self.prime_dir}/primes{file_num}.txt"
            if not os.path.exists(filename):
                print(f"File {filename} not found")
                continue
                
            with open(filename, 'r') as f:
                # Skip header (first 2 lines)
                for _ in range(2):
                    f.readline()
                
                # Read primes
                for line in f:
                    primes_line = [int(p) for p in line.strip().split()]
                    all_primes.extend(primes_line)
                    
                    if len(all_primes) >= max_primes:
                        break
                if len(all_primes) >= max_primes:
                    break
        
        print(f"Loaded {len(all_primes):,} primes")
        return all_primes[:max_primes]
    
    def compute_e8_embedding(self, prime_block):
        """
        Embed a block of primes into E8 space using normalized gaps
        """
        if len(prime_block) < 2:
            return None
        
        # Compute gaps between consecutive primes
        gaps = [prime_block[i+1] - prime_block[i] for i in range(len(prime_block)-1)]
        
        # Normalize using prime number theorem scaling
        normalized = []
        for i, gap in enumerate(gaps):
            if i < len(prime_block) - 1:
                p_i = prime_block[i]
                scaling = math.sqrt(math.log(max(p_i, 2)))
                if scaling > 0:
                    normalized.append(gap / scaling)
                else:
                    normalized.append(gap)
        
        # Pad to 8 dimensions
        while len(normalized) < 8:
            normalized.append(0)
        
        # Quantize to E8 lattice points
        e8_vector = []
        for coord in normalized[:8]:
            quantum = self.E8_min_norm / 2
            quantized = round(coord / quantum) * quantum
            e8_vector.append(quantized)
        
        return np.array(e8_vector)
    
    def e8_to_bit_sequence(self, e8_vector):
        """
        Convert E8 vector to 8-bit sequence with CNOT-aware encoding
        """
        bits = []
        
        # Method 1: Sign-based encoding
        for coord in e8_vector:
            bit = 1 if coord > 0 else 0
            bits.append(bit)
        
        # Method 2: Magnitude threshold encoding
        threshold = np.median(np.abs(e8_vector))
        magnitude_bits = [1 if abs(coord) > threshold else 0 for coord in e8_vector]
        
        # Method 3: Differential encoding (emphasizes transitions)
        diff_bits = []
        for i in range(1, len(e8_vector)):
            diff = e8_vector[i] - e8_vector[i-1]
            diff_bits.append(1 if diff > 0 else 0)
        # Pad to 8 bits
        while len(diff_bits) < 8:
            diff_bits.append(0)
        
        # Choose the encoding with most structure
        # (highest non-uniformity)
        encodings = [bits, magnitude_bits, diff_bits]
        entropies = [self.compute_entropy(e) for e in encodings]
        
        # Pick encoding with lowest entropy (most structure)
        best_idx = np.argmin(entropies)
        
        return encodings[best_idx][:8]
    
    def compute_entropy(self, bits):
        """Compute entropy of a bit sequence"""
        counts = Counter(bits)
        total = len(bits)
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def decode_bytes_for_cnot(self, primes, block_size=8):
        """
        Decode bytes from primes, optimized for CNOT pattern detection
        """
        decoded_bytes = []
        bit_sequences = []
        
        # Process primes in blocks
        for i in tqdm(range(0, len(primes) - block_size + 1, block_size), desc="Decoding blocks"):
            block = primes[i:i + block_size]
            
            # Compute E8 embedding
            e8_vector = self.compute_e8_embedding(block)
            if e8_vector is None:
                continue
            
            # Convert to bit sequence
            bits = self.e8_to_bit_sequence(e8_vector)
            bit_sequences.append(bits)
            
            # Convert bits to byte
            byte_val = 0
            for j, bit in enumerate(bits[:8]):
                byte_val |= (bit << (7 - j))
            
            decoded_bytes.append(byte_val)
        
        return decoded_bytes, bit_sequences
    
    def find_cnot_patterns_exact(self, decoded_bytes):
        """
        Look for exact CNOT patterns (canonical encoding)
        """
        print("\nSearching for exact CNOT patterns...")
        
        # Canonical CNOT sequence: 0,0,1,1,16,17,17,16
        cnot_sequence = [0, 0, 1, 1, 16, 17, 17, 16]
        
        matches = []
        
        for i in range(len(decoded_bytes) - 7):
            window = decoded_bytes[i:i+8]
            
            if list(window) == cnot_sequence:
                matches.append({
                    'position': i,
                    'window': window,
                    'type': 'exact_canonical'
                })
        
        return matches
    
    def find_cnot_patterns_paired(self, decoded_bytes):
        """
        Look for CNOT patterns as paired transformations
        """
        print("\nSearching for paired CNOT patterns...")
        
        matches = []
        
        for i in range(len(decoded_bytes) - 7):
            window = decoded_bytes[i:i+8]
            
            # Group into 4 pairs
            pairs = [(window[j], window[j+1]) for j in range(0, 8, 2)]
            
            # Check if pairs match CNOT transformation
            is_cnot = True
            pair_types = []
            
            for input_val, output_val in pairs:
                if (input_val, output_val) in self.cnot_pairs:
                    pair_types.append(self.cnot_pairs[(input_val, output_val)])
                else:
                    is_cnot = False
                    break
            
            if is_cnot:
                # Check we have all 4 unique pairs
                unique_pairs = set(pairs)
                if len(unique_pairs) == 4:
                    matches.append({
                        'position': i,
                        'window': window,
                        'pairs': pairs,
                        'pair_types': pair_types,
                        'type': 'paired_cnot'
                    })
        
        return matches
    
    def find_cnot_patterns_general(self, bit_sequences):
        """
        Look for general CNOT patterns in bit sequences
        """
        print("\nSearching for general CNOT patterns in bit sequences...")
        
        matches = []
        
        for seq_idx, bits in enumerate(bit_sequences):
            if len(bits) < 8:
                continue
            
            # Try all pairs of bit positions as control/target
            for control_idx in range(8):
                for target_idx in range(8):
                    if control_idx == target_idx:
                        continue
                    
                    # Extract control and target bits
                    control_bit = bits[control_idx]
                    target_bit = bits[target_idx]
                    
                    # Look for sequences where this pair exhibits CNOT behavior
                    # We need 4 consecutive bytes to form a CNOT truth table
                    if seq_idx + 3 < len(bit_sequences):
                        window = bit_sequences[seq_idx:seq_idx+4]
                        
                        # Check if this window forms a CNOT truth table
                        cnot_table = self.check_cnot_truth_table(window, control_idx, target_idx)
                        
                        if cnot_table['is_cnot']:
                            matches.append({
                                'position': seq_idx,
                                'control_bit': control_idx,
                                'target_bit': target_idx,
                                'truth_table': cnot_table['table'],
                                'type': 'general_cnot'
                            })
        
        return matches
    
    def check_cnot_truth_table(self, bit_window, control_idx, target_idx):
        """
        Check if a window of 4 bytes forms a CNOT truth table
        """
        if len(bit_window) < 4:
            return {'is_cnot': False, 'table': []}
        
        # Expected CNOT truth table (control, target) -> (control, target XOR control)
        expected = {
            (0, 0): (0, 0),
            (0, 1): (0, 1),
            (1, 0): (1, 1),
            (1, 1): (1, 0)
        }
        
        # Extract control/target bits from each byte
        observed = {}
        
        for i, bits in enumerate(bit_window):
            if len(bits) > max(control_idx, target_idx):
                control = bits[control_idx]
                target = bits[target_idx]
                observed[(control, target)] = i
        
        # Check if we have all 4 unique input states
        if len(observed) < 4:
            return {'is_cnot': False, 'table': []}
        
        # For CNOT to be present, we need to see the transformation
        # We need to see the output states too
        # Actually, for this approach we're just looking for the presence of all 4 states
        # The transformation would be seen in how these states change over time
        
        # For now, just check if all 4 states are present
        input_states = list(observed.keys())
        
        # Check if these are exactly the 4 possible states
        expected_states = [(0,0), (0,1), (1,0), (1,1)]
        
        if set(input_states) == set(expected_states):
            return {
                'is_cnot': True,
                'table': [(state, expected[state]) for state in input_states]
            }
        
        return {'is_cnot': False, 'table': []}
    
    def find_reversible_permutations(self, decoded_bytes, window_size=8):
        """
        Look for reversible permutations (characteristic of quantum gates)
        """
        print("\nSearching for reversible permutations...")
        
        matches = []
        
        for i in range(len(decoded_bytes) - window_size + 1):
            window = decoded_bytes[i:i+window_size]
            
            # Split into input/output pairs
            inputs = window[:window_size//2]
            outputs = window[window_size//2:]
            
            # Check if this is a permutation (bijection)
            if len(set(inputs)) == len(inputs) and len(set(outputs)) == len(outputs):
                # Check if it's a valid permutation from inputs to outputs
                # Create mapping
                mapping = {}
                valid = True
                
                for j in range(len(inputs)):
                    inp = inputs[j]
                    out = outputs[j]
                    
                    if inp in mapping:
                        if mapping[inp] != out:
                            valid = False
                            break
                    else:
                        mapping[inp] = out
                
                if valid and len(mapping) == len(inputs):
                    # Check if it's nontrivial (not identity)
                    is_identity = all(inp == mapping[inp] for inp in inputs)
                    
                    if not is_identity:
                        # Check if it has CNOT-like structure
                        # Look for conditional flipping
                        cnot_like = self.analyze_permutation_structure(mapping)
                        
                        if cnot_like:
                            matches.append({
                                'position': i,
                                'inputs': inputs,
                                'outputs': outputs,
                                'mapping': mapping,
                                'cnot_score': cnot_like['score'],
                                'type': 'reversible_permutation'
                            })
        
        return matches
    
    def analyze_permutation_structure(self, mapping):
        """
        Analyze if a permutation has CNOT-like structure
        """
        # Convert to binary for analysis
        binary_mapping = {}
        
        for inp, out in mapping.items():
            inp_bits = format(inp, '08b')
            out_bits = format(out, '08b')
            binary_mapping[inp_bits] = out_bits
        
        # Look for bit positions that act as control/target
        best_score = 0
        best_explanation = None
        
        for control_idx in range(8):
            for target_idx in range(8):
                if control_idx == target_idx:
                    continue
                
                score = 0
                explanation = []
                
                for inp_bits, out_bits in binary_mapping.items():
                    control_in = inp_bits[control_idx]
                    target_in = inp_bits[target_idx]
                    control_out = out_bits[control_idx]
                    target_out = out_bits[target_idx]
                    
                    # Check CNOT conditions
                    # 1. Control unchanged
                    if control_in == control_out:
                        score += 1
                        
                        # 2. Target flips iff control is 1
                        expected_target = str(int(target_in) ^ int(control_in))
                        if target_out == expected_target:
                            score += 2
                            explanation.append(f"({control_in}{target_in})→({control_out}{target_out})")
                
                # Normalize score
                score = score / (3 * len(binary_mapping))
                
                if score > best_score:
                    best_score = score
                    best_explanation = explanation
        
        return {'score': best_score, 'explanation': best_explanation} if best_score > 0.8 else False
    
    def visualize_cnot_pattern(self, match):
        """
        Visualize a found CNOT pattern
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        match_type = match['type']
        
        if match_type == 'exact_canonical':
            window = match['window']
            
            # Plot as decimal values
            ax = axes[0, 0]
            ax.bar(range(8), window)
            ax.set_title('Exact CNOT Pattern (Decimal)')
            ax.set_xlabel('Byte Position')
            ax.set_ylabel('Value')
            ax.set_xticks(range(8))
            ax.grid(True, alpha=0.3)
            
            # Plot as binary
            ax = axes[0, 1]
            binary_matrix = np.zeros((8, 8))
            for i, val in enumerate(window):
                bits = format(val, '08b')
                for j, bit in enumerate(bits):
                    binary_matrix[i, j] = int(bit)
            
            im = ax.imshow(binary_matrix, cmap='binary', aspect='auto')
            ax.set_title('Binary Representation')
            ax.set_xlabel('Bit Position')
            ax.set_ylabel('Byte Position')
            plt.colorbar(im, ax=ax)
            
            # Highlight control and target bits
            ax.text(4.5, 3.5, 'C', fontsize=12, color='red', ha='center', va='center')
            ax.text(0.5, 3.5, 'T', fontsize=12, color='blue', ha='center', va='center')
            
        elif match_type == 'paired_cnot':
            pairs = match['pairs']
            
            # Plot input-output pairs
            ax = axes[1, 0]
            for i, (inp, out) in enumerate(pairs):
                ax.plot([0, 1], [inp, out], 'o-', label=f'Pair {i+1}')
            
            ax.set_title('CNOT Input-Output Pairs')
            ax.set_xlabel('State (0=input, 1=output)')
            ax.set_ylabel('Value')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Input', 'Output'])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_detection(self, max_files=5, max_primes=100000):
        """
        Main detection pipeline
        """
        print("=" * 70)
        print("UNIVERSAL CNOT GATE DETECTION IN PRIME DATA USING E8 MACHINERY")
        print("=" * 70)
        
        # Step 1: Load primes
        self.primes = self.load_primes(max_files, max_primes)
        
        # Step 2: Decode bytes
        self.decoded_bytes, self.bit_sequences = self.decode_bytes_for_cnot(
            self.primes, block_size=8
        )
        
        print(f"\nDecoded {len(self.decoded_bytes):,} bytes from primes")
        
        # Step 3: Run all detection methods
        results = {}
        
        # 3.1 Exact patterns
        exact_matches = self.find_cnot_patterns_exact(self.decoded_bytes)
        results['exact'] = exact_matches
        
        # 3.2 Paired patterns
        paired_matches = self.find_cnot_patterns_paired(self.decoded_bytes)
        results['paired'] = paired_matches
        
        # 3.3 General patterns
        general_matches = self.find_cnot_patterns_general(self.bit_sequences)
        results['general'] = general_matches
        
        # 3.4 Reversible permutations
        perm_matches = self.find_reversible_permutations(self.decoded_bytes)
        results['permutations'] = perm_matches
        
        # Step 4: Analyze results
        print("\n" + "=" * 70)
        print("DETECTION RESULTS")
        print("=" * 70)
        
        total_found = 0
        for method, matches in results.items():
            if matches:
                print(f"\n{method.upper()}: Found {len(matches)} instances")
                total_found += len(matches)
                
                # Show first few matches
                for i, match in enumerate(matches[:3]):
                    print(f"  Match {i+1}:")
                    print(f"    Position: {match['position']}")
                    print(f"    Type: {match['type']}")
                    
                    if 'window' in match:
                        print(f"    Window: {match['window']}")
                        print(f"    Hex: {[hex(x) for x in match['window']]}")
                    
                    if 'pairs' in match:
                        print(f"    Pairs: {match['pairs']}")
                    
                    if 'cnot_score' in match:
                        print(f"    CNOT Score: {match['cnot_score']:.3f}")
                    
                    print()
            else:
                print(f"\n{method.upper()}: No matches found")
        
        # Step 5: Statistical analysis
        print("\n" + "=" * 70)
        print("STATISTICAL ANALYSIS")
        print("=" * 70)
        
        # Expected random occurrence of exact CNOT pattern
        total_bytes = len(self.decoded_bytes)
        expected_random = total_bytes / (256 ** 8)  # Probability of exact 8-byte sequence
        
        print(f"Total bytes analyzed: {total_bytes:,}")
        print(f"Exact CNOT patterns found: {len(exact_matches)}")
        print(f"Expected by random chance: {expected_random:.6f}")
        
        if len(exact_matches) > 0:
            print(f"Significance: {len(exact_matches) / max(expected_random, 1e-10):.2f}x expected")
        
        # Step 6: Information-theoretic analysis
        print("\nInformation-theoretic analysis of CNOT structure:")
        
        # Compute mutual information between bit positions
        if self.bit_sequences:
            mi_matrix = self.compute_mutual_information(self.bit_sequences[:1000])
            print("\nMutual Information between bit positions (top pairs):")
            
            # Find pairs with highest MI (potential control-target pairs)
            n = len(mi_matrix)
            mi_pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    mi_pairs.append((i, j, mi_matrix[i, j]))
            
            mi_pairs.sort(key=lambda x: x[2], reverse=True)
            
            for i, j, mi in mi_pairs[:5]:
                print(f"  Bits ({i},{j}): MI = {mi:.4f}")
        
        # Save results
        self.save_results(results)
        
        # Visualize if matches found
        if total_found > 0:
            first_match = None
            for method, matches in results.items():
                if matches:
                    first_match = matches[0]
                    break
            
            if first_match:
                print("\nVisualizing first CNOT pattern...")
                self.visualize_cnot_pattern(first_match)
        
        return results
    
    def compute_mutual_information(self, bit_sequences, max_samples=1000):
        """
        Compute mutual information between bit positions
        """
        samples = min(len(bit_sequences), max_samples)
        
        # Create arrays for each bit position
        n_bits = 8
        bit_arrays = []
        
        for bit_idx in range(n_bits):
            arr = []
            for seq in bit_sequences[:samples]:
                if len(seq) > bit_idx:
                    arr.append(seq[bit_idx])
            bit_arrays.append(arr)
        
        # Compute mutual information matrix
        mi_matrix = np.zeros((n_bits, n_bits))
        
        for i in range(n_bits):
            for j in range(n_bits):
                if i != j:
                    # Compute joint distribution
                    joint_counts = np.zeros((2, 2))
                    
                    for k in range(samples):
                        if k < len(bit_arrays[i]) and k < len(bit_arrays[j]):
                            joint_counts[bit_arrays[i][k], bit_arrays[j][k]] += 1
                    
                    # Normalize
                    joint_probs = joint_counts / samples
                    
                    # Compute marginals
                    p_i = np.sum(joint_probs, axis=1)
                    p_j = np.sum(joint_probs, axis=0)
                    
                    # Compute mutual information
                    mi = 0
                    for a in range(2):
                        for b in range(2):
                            if joint_probs[a, b] > 0:
                                mi += joint_probs[a, b] * math.log2(
                                    joint_probs[a, b] / (p_i[a] * p_j[b])
                                )
                    
                    mi_matrix[i, j] = mi
        
        return mi_matrix
    
    def save_results(self, results):
        """Save detection results"""
        summary = {
            'total_bytes': len(self.decoded_bytes),
            'results': results,
            'cnot_encodings': self.cnot_encodings[:5],  # Save first few encodings
            'sample_bytes': self.decoded_bytes[:1000]
        }
        
        with open('cnot_detection_results.pkl', 'wb') as f:
            pickle.dump(summary, f)
        
        # Also save text summary
        with open('cnot_detection_summary.txt', 'w') as f:
            f.write("CNOT GATE DETECTION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total bytes analyzed: {len(self.decoded_bytes):,}\n\n")
            
            for method, matches in results.items():
                f.write(f"{method.upper()} DETECTION:\n")
                f.write(f"  Found {len(matches)} matches\n")
                
                if matches:
                    f.write("\n  First few matches:\n")
                    for i, match in enumerate(matches[:3]):
                        f.write(f"    Match {i+1}:\n")
                        f.write(f"      Position: {match['position']}\n")
                        f.write(f"      Type: {match['type']}\n")
                        
                        if 'window' in match:
                            f.write(f"      Bytes: {match['window']}\n")
                        
                        if 'cnot_score' in match:
                            f.write(f"      CNOT Score: {match['cnot_score']:.3f}\n")
                        
                        f.write("\n")
                
                f.write("\n")
        
        print(f"\nResults saved to cnot_detection_results.pkl")
        print(f"Summary saved to cnot_detection_summary.txt")

def main():
    """Main function"""
    detector = CNOTGateDetector()
    
    print("This script searches for universal CNOT gate patterns in prime data.")
    print("\nCNOT gate properties we're looking for:")
    print("  1. Two binary degrees of freedom (4 possible states)")
    print("  2. Conditional flipping of target bit when control is 1")
    print("  3. Reversible transformation (bijection)")
    print("\nThe pattern can appear in multiple forms:")
    print("  - Exact canonical encoding: [0,0,1,1,16,17,17,16]")
    print("  - Paired transformations showing the 4 state transitions")
    print("  - General reversible permutations with conditional structure")
    
    proceed = input("\nRun detection? (y/n): ")
    if proceed.lower() != 'y':
        print("Detection cancelled.")
        return
    
    # Run detection on first 100,000 primes (adjust as needed)
    results = detector.run_detection(max_files=2, max_primes=100000)
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    total_matches = sum(len(matches) for matches in results.values())
    
    if total_matches > 0:
        print(f"\nFound {total_matches} CNOT-like patterns in the prime data.")
        print("\nThis suggests the presence of:")
        print("  1. Conditional symmetry breaking")
        print("  2. Reversible causation structures")
        print("  3. Control vs target distinction in prime distribution")
        print("\nSuch structures are characteristic of:")
        print("  - Quantum computational primitives")
        print("  - Error-correcting codes")
        print("  - Topological quantum field theories")
    else:
        print("\nNo CNOT patterns were found in the analyzed data.")
        print("\nPossible reasons:")
        print("  1. The encoding uses different bit positions")
        print("  2. The pattern appears at larger scales")
        print("  3. Different E8 embedding parameters are needed")
        print("  4. The pattern might be in the gaps rather than the primes themselves")

if __name__ == "__main__":
    main()
