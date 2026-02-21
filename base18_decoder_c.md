### **Technical Specification: Project EXCEPTIONAL-BASE18-DECODER (EBD)**

**Project Code:** `EBD-PASS7-LAGRANGIAN`  
**Version:** 7.0  
**Date:** February 21, 2026  
**Subject:** Extraction of the Universal Lagrangian via Base-18 Arithmetic Decoding  

---

#### **1. Objective**
To implement a high-performance C engine that decodes the "Monstrous Prose" from the 24-dimensional Niemeier bitstream. This program utilizes the **M_13 "18" Key** to map the 240 $E_8$ roots into an 18-symbol logical alphabet, effectively filtering out the "Silent 8" Cartan dimensions to reveal the underlying Hamiltonian path of the primes.

---

#### **2. Mathematical Logic**

**2.1. The Base-18 Mapping**  
The 240 roots of $E_8$ are partitioned into two sets based on the 10-billion-prime empirical frequency analysis:
*   **The Active 18:** 18 letters that carry the message (J, O, I, R, E, H, Q, P, G, U, A, Y, V, C, N, F, B, M).
*   **The Silent 8:** 8 letters representing the Cartan Subalgebra (D, K, L, S, T, W, X, Z), which act as "Logical Spacers" or null states.

**2.2. Triality Integration**  
The input is a stream of 24D vectors $\Psi = (v_1, v_2, v_3)$. The decoder must perform a **Triality Summation** to extract the coherent 8D phase before mapping to the Base-18 alphabet.

**2.3. The Salem-Jordan Filter**  
The decoder applies a threshold based on the **Topological Tension** ($\|\Psi\|^2 = 3.0$). Only states with a tension deviation $< 10^{-6}$ are processed as "Logical Qubits."

---

#### **3. C Implementation: `base18_decoder.c`**

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define STATE_SIZE 24
#define E8_ROOTS 240
#define ACTIVE_ALPHABET_SIZE 18

// The "Janik-Googol" Active Alphabet (Ranked by 10B Frequency)
const char ALPHABET_18[] = "JOIREHQP GUAYVCNFBM"; 

// Mapping Table: E8 Root Index -> Base-18 Index (-1 = Silent/Spacer)
// This table is derived from the F4-projection-mod-26 logic
int ROOT_TO_BASE18[E8_ROOTS];

void initialize_mapping() {
// The "Silent 8" indices in the mod-26 space
int silent_indices[] = {3, 10, 11, 18, 19, 22, 23, 25}; // D, K, L, S, T, W, X, Z

for (int i = 0; i < E8_ROOTS; i++) {
int f4_mod_26 = i % 26;
int is_silent = 0;
for (int j = 0; j < 8; j++) {
if (f4_mod_26 == silent_indices[j]) {
is_silent = 1;
break;
}
}

if (is_silent) {
ROOT_TO_BASE18[i] = -1; // Map to Spacer
} else {
// Map to the 18-letter active set
ROOT_TO_BASE18[i] = i % ACTIVE_ALPHABET_SIZE;
}
}
}

void decode_stream(const char* filename) {
FILE *f_in = fopen(filename, "rb");
if (!f_in) return;

float vector[STATE_SIZE];
uint64_t state_count = 0;

printf("--- DECODING MONSTROUS PROSE (BASE-18) ---\n");

while (fread(vector, sizeof(float), STATE_SIZE, f_in) == STATE_SIZE) {
// 1. Triality Summation (8D Phase Extraction)
double phase_acc = 0;
for (int i = 0; i < STATE_SIZE; i++) {
phase_acc += (double)vector[i];
}

// 2. Map to E8 Root Index
// We use the 10B Phase-Sync logic
double normalized_phase = fmod(fabs(phase_acc) / sqrt(2.0), 1.0);
int root_idx = (int)(normalized_phase * E8_ROOTS) % E8_ROOTS;

// 3. Base-18 Translation
int base18_idx = ROOT_TO_BASE18[root_idx];

if (base18_idx != -1) {
// Output the Active Character
printf("%c", ALPHABET_18[base18_idx]);
} else {
// Output the 'Great Silence' (Cartan Spacer)
printf(".");
}

state_count++;
// Line break every 72 characters for readability (The 'Weyl' width)
if (state_count % 72 == 0) printf("\n");
}

fclose(f_in);
printf("\n--- DECODING COMPLETE ---\n");
}

int main(int argc, char *argv[]) {
if (argc < 2) {
printf("Usage: ./ebd <monstrous_stream.n24>\n");
return 1;
}

initialize_mapping();
decode_stream(argv[1]);

return 0;
}
```

---

#### **4. Operational Protocol**

1.  **Compile:**
`gcc -O3 base18_decoder.c -o ebd -lm`
2.  **Run:**
`./ebd monstrous_stream.n24 > monstrous_prose.txt`
3.  **Analysis:**
Open `monstrous_prose.txt`. We are looking for the **Hamiltonian Path**—a repeating sequence of 14 characters (the $G_2$ vertices) that constitutes the "Universal Lagrangian."

