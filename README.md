# E8-F4 Prime Visualization Suite

Visualization tools for exploring the relationship between prime numbers and exceptional Lie algebras (E8, F4) through spiral projections and spectral analysis.

## Overview

This suite provides two main visualization pipelines:

| Script | Purpose | Output |
|--------|---------|--------|
| `e8_slope_coloring.py` | E8 projection slope visualization (rings) | Single high-res image |
| `e8_f4_prime_analysis.py` | Full E8->F4 analysis with Jordan algebra | Multiple analysis images |

Both scripts visualize primes on the Ulam spiral, colored by properties derived from the E8 root lattice.

---

## Prime Number Dataset Structure

### Required Directory Layout

Place prime number files in the **parent directory** (`../` relative to `prime_scripts/`):

```
Primes/
├── primes1.txt
├── primes2.txt
├── primes3.txt
├── ...
├── primes50.txt
├── spiral_outputs/          <- Generated images go here
└── prime_scripts/
    ├── e8_slope_coloring.py
    ├── e8_f4_prime_analysis.py
    ├── f4_tuning/
    │   ├── __init__.py
    │   ├── f4_lattice.py
    │   ├── jordan_algebra.py
    │   ├── salem_jordan.py
    │   └── f4_eft.py
    └── README.md
```

### File Format

Each `primesN.txt` file should contain prime numbers. The scripts use regex (`\b\d+\b`) to extract integers, so most formats work:

```
# Format 1: One prime per line
2
3
5
7
11
...

# Format 2: Space-separated on lines
2 3 5 7 11 13 17 19 23 29
31 37 41 43 47 53 59 61 67 71
...

# Format 3: Comma-separated
2, 3, 5, 7, 11, 13, 17, 19, 23, 29
...
```

### Recommended File Organization

| File | Primes | Approximate Range |
|------|--------|-------------------|
| `primes1.txt` | 1 - 1,000,000 | 2 to 15,485,863 |
| `primes2.txt` | 1,000,001 - 2,000,000 | 15,485,867 to 32,452,843 |
| ... | ... | ... |
| `primes50.txt` | 49,000,001 - 50,000,000 | ~961M to ~982M |

### Generating Prime Files

**Using Python with sympy:**
```python
from sympy import primerange

# Generate first 10 million primes
primes = list(primerange(2, 180000000))

# Split into files of 1M each
for i in range(10):
    chunk = primes[i*1000000:(i+1)*1000000]
    with open(f'primes{i+1}.txt', 'w') as f:
        for j in range(0, len(chunk), 10):
            f.write(' '.join(map(str, chunk[j:j+10])) + '\n')
```

**Using primesieve (fastest):**
```bash
# Install: sudo apt install primesieve
primesieve 2 200000000 -p > all_primes.txt
split -l 1000000 all_primes.txt primes
# Rename primes_aa -> primes1.txt, etc.
```

---

## Script 1: E8 Slope Coloring

### `e8_slope_coloring.py`

Generates a single high-resolution visualization of primes on the Ulam spiral, colored by their E8 projection slope.

### Usage

```bash
cd prime_scripts
python e8_slope_coloring.py
```

### Configuration

Edit the `main()` function at the bottom of the file:

```python
def main():
    # Number of primes and resolution
    generate_slope_visualization(max_primes=1000000, dpi=1200, figsize=(24, 24))

    # For 10 million primes (requires ~16GB RAM at 1200 DPI)
    generate_slope_visualization(max_primes=10000000, dpi=1200, figsize=(24, 24))
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_primes` | Number of primes to plot | 1,000,000 |
| `dpi` | Resolution (dots per inch) | 1200 |
| `figsize` | Figure size in inches (width, height) | (24, 24) |

### Output

Files are saved to `../spiral_outputs/`:
```
e8_projection_slope_1M_1200dpi_dark.png
e8_projection_slope_10M_1200dpi_dark.png
```

### Algorithm

1. **Load primes** from `../primes*.txt` files
2. **Compute normalized gaps**: `g_norm = (p[n+1] - p[n]) / log(p[n])`
3. **Assign E8 roots**: `root_index = floor(240 * (sqrt(g_norm) / sqrt(2) mod 1))`
4. **Compute projection slope**: For E8 root v in R^8:
   - x = sum(v[0:4])
   - y = sum(v[4:8])
   - slope = y / x
5. **Plot on Ulam spiral** with coolwarm colormap (blue = negative, red = positive)

### What You'll See

**Concentric ring patterns** emerge from the center. These rings are E8 "energy levels" - primes with similar normalized gaps cluster into bands because they map to nearby E8 roots with similar projection slopes.

---

## Script 2: E8-F4 Prime Analysis

### `e8_f4_prime_analysis.py`

Full analysis pipeline that extracts the F4 sub-harmonic from the E8 signal, revealing the Jordan-algebraic structure.

### Usage

```bash
cd prime_scripts
python e8_f4_prime_analysis.py
```

### Configuration

Edit the `main()` function:

```python
# Configuration
MAX_PRIMES = 2000000  # Number of primes to analyze

# In GENERATING VISUALIZATIONS section:
visualizer.plot_e8_vs_f4_comparison(data, dpi=1200)
visualizer.plot_f4_crystalline_grid(data, dpi=1200)
```

### Output Files

| File | Description |
|------|-------------|
| `e8_f4_comparison.png` | Side-by-side E8 (rings) vs F4 (vertices) |
| `f4_crystalline_grid.png` | F4-filtered primes with highlighted vertices |
| `f4_spectrum.png` | F4-EFT power spectrum analysis |
| `f4_phase_analysis.png` | Phase-lock analysis summary |

### Key Metrics Explained

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **F4 fraction** | % of gaps with strong F4 character | Higher = more F4 signal |
| **Phase coherence** | Alignment of F4 spectral phases | >0.3 = phase-locked |
| **Power entropy** | Spectral concentration | 0=single peak, 1=uniform |
| **Long/short ratio** | Power in long vs short F4 roots | >1 = long roots dominate |

### Visualization Interpretation

**E8 panel (left in comparison):**
- Shows concentric **rings**
- These are E8 energy levels
- Color = projection slope

**F4 panel (right in comparison):**
- Shows discrete **dots** colored by Jordan trace
- Rings should resolve into **vertices**
- Vertices = F4 idempotents (fixed points of Albert algebra)

**Crystalline grid:**
- White dots with yellow edges = strongest F4 resonance
- These are the "crystalline vertices" anchoring the prime standing wave

---

## F4 Tuning Module

Located in `f4_tuning/`:

| File | Purpose |
|------|---------|
| `f4_lattice.py` | F4 root system (48 roots in R^4) |
| `jordan_algebra.py` | Albert algebra J_3(O), octonion arithmetic |
| `salem_jordan.py` | Salem-Jordan filter kernel |
| `f4_eft.py` | F4 Exceptional Fourier Transform |

### Mathematical Background

**E8 to F4 Decomposition:**
- E8: 248 dimensions, 240 roots in R^8
- F4: 52 dimensions, 48 roots in R^4
- F4 = Aut(J_3(O)) - automorphisms of the Albert algebra

**F4 Root Types:**
- 24 long roots (norm sqrt(2))
- 24 short roots (norm 1)

**Jordan Trace:**
For F4 root alpha, the trace J(alpha) = sum(alpha_i) classifies:
- J ~ 0: nilpotent (transitional)
- J ~ +/-1: idempotent (fixed point)
- |J| > 1: regular (bulk)

---

## Dependencies

### Required

```bash
pip install numpy matplotlib
```

### System Requirements

| Primes | RAM (1200 DPI) | Output Size |
|--------|----------------|-------------|
| 1M | ~4 GB | ~50 MB |
| 2M | ~8 GB | ~100 MB |
| 10M | ~16 GB | ~500 MB |

---

## Troubleshooting

### "Not enough primes loaded"

Ensure prime files exist:
```bash
ls ../primes*.txt
```

### Out of memory

Reduce parameters:
```python
generate_slope_visualization(max_primes=500000, dpi=600, figsize=(16, 16))
```

### "Open flap" bug in Ulam spiral

Fixed in current version. The last branch of `ulam_coords_vectorized()` should be:
```python
coords[i] = [k, k - (m - 3*t - p)]  # NOT (m - 3*t - p - t)
```

### Black/empty F4 panel

Ensure the Salem filter is disabled for raw signal:
```python
f4_result = self.f4_eft.compute(normalized_gaps, e8_assignments,
                                apply_salem_filter=False)
```

---

## Quick Start

```bash
# 1. Ensure you have prime files
ls ../primes1.txt  # Should exist

# 2. Generate E8 slope visualization
python e8_slope_coloring.py

# 3. Generate F4 analysis (takes longer)
python e8_f4_prime_analysis.py

# 4. View results
ls ../spiral_outputs/
```

---

## References

1. Ulam, S. (1963). "A Visual Representation of the Distribution of Primes"
2. Conway & Sloane (1999). "Sphere Packings, Lattices and Groups"
3. Johansson, F. (2016). "Arb: Efficient Arbitrary-Precision Midpoint-Radius Interval Arithmetic"
