# Decoded Quantum Interferometry (DQI) -- Comprehensive Learning Guide

**Project Context:** Yale Hackathon 2026 -- P&C Insurance Product Bundling Optimization via DQI

**Primary Reference:** Sabater et al., "Towards Solving Industrial Integer Linear Programs with Decoded Quantum Interferometry," arXiv:2509.08328 (Dec 2025)

**Original DQI Paper:** Jordan, Shutty, Wootters, Zalcman, Schmidhuber, King, Isakov, Babbush, "Optimization by Decoded Quantum Interferometry," 2024

**BCG-X Implementation:** https://bcg-x-official.github.io/dqi/

---

## Table of Contents

1. [DQI Algorithm Overview](#1-dqi-algorithm-overview)
2. [The 6 DQI Steps](#2-the-6-dqi-steps)
3. [ILP to max-XORSAT Pipeline](#3-ilp-to-max-xorsat-pipeline)
4. [BP1 Quantum Circuit Details](#4-bp1-quantum-circuit-details)
5. [BCG-X Library Structure](#5-bcg-x-library-structure)
6. [Implementation Notes for Qiskit](#6-implementation-notes-for-qiskit)
7. [Adapting DQI for P&C Insurance Bundling](#7-adapting-dqi-for-pc-insurance-bundling)

---

## 1. DQI Algorithm Overview

### 1.1 What Is DQI?

Decoded Quantum Interferometry (DQI) is a quantum algorithm for solving combinatorial optimization problems that fundamentally differs from variational approaches like QAOA. Instead of using parameterized circuits that are tuned through classical optimization loops, DQI converts an optimization problem into a **decoding problem** and uses quantum interference (via the Quantum Fourier Transform) to amplify the probability of measuring high-quality solutions.

The core insight: DQI reframes optimization as **syndrome decoding** over sparse parity-check codes (analogous to LDPC codes in classical error correction). The algorithm:

1. Encodes the optimization problem as a **max-XORSAT** instance, defined by a parity-check matrix B and constraint vector v.
2. Prepares a quantum state that encodes problem-specific information in its phases.
3. Uses a **coherent quantum decoder** to uncompute the "error pattern" (the message register).
4. Applies Hadamard transforms to convert phase information into amplitude information via interference.
5. Measures to obtain candidate solutions whose quality is enhanced by the decoding step.

### 1.2 The max-XORSAT Formulation

A general 0-1 Integer Linear Program (ILP) is:

```
min   c^T z
s.t.  C z >= a
      z in {0,1}^{n'}
```

where z is the binary decision variable of dimension n', c is the cost vector, C is the constraint matrix, and a is the right-hand side vector.

DQI operates on the **max-XORSAT** reformulation of this ILP. A max-XORSAT instance is defined by:

```
B x = v  (mod 2)
```

where x in {0,1}^n is the decision variable, v in {0,1}^m is the constraint vector, and B^T in {0,1}^{n x m} encodes the constraints as a parity-check matrix. The objective becomes:

```
max_{x in {0,1}^n}  sum_{i=1}^{m} (-1)^{v_i + b_i . x}       (Eq. 4)
```

where b_i is the i-th row of B. This computes the maximum difference between satisfied and unsatisfied XOR constraints. The transformation from ILP to max-XORSAT (detailed in Section 3 below) incurs overhead: n > n' and m > m'.

### 1.3 Why DQI Is Different from QAOA

| Aspect | QAOA | DQI |
|--------|------|-----|
| **Approach** | Variational: parameterized circuits optimized classically | Interference-based: fixed circuit, no classical optimization loop |
| **Circuit depth** | Grows with number of rounds p; limited by noise | Determined by decoder iterations T and problem structure |
| **Guarantees** | Known limitations: symmetry/locality barriers on high-girth graphs, concentration phenomena | Instance-wise, decoder-controlled guarantees; imports coding-theory machinery |
| **Scaling evidence** | Worst-case separations limit constant-depth; empirical evidence on specific families | Provably effective in Fourier space; sidesteps known variational barriers |
| **Problem encoding** | Ising Hamiltonian (QUBO) | max-XORSAT via parity-check matrix (LDPC-like structure) |
| **Key mechanism** | Alternating problem/mixer unitaries | Quantum Fourier transform converts phase -> amplitude; decoder focuses probability mass |
| **Classical analog** | Simulated annealing variants | LDPC decoding (belief propagation) |

**Key conceptual difference:** QAOA relies on low-depth variational expressivity, which faces structural obstacles. DQI leverages the Quantum Fourier Transform where it is provably effective, and offloads the algorithmic difficulty to the decoder. Algorithmic progress thus depends on the strength of the decoder rather than on variational circuit depth.

---

## 2. The 6 DQI Steps

The DQI algorithm for a max-XORSAT problem defined by parity-check matrix B in {0,1}^{m x n} and constraint vector v in {0,1}^m proceeds through six sequential stages. We use error-correction terminology: the **message register** has m qubits (corresponding to codeword bits) and the **syndrome register** has n qubits.

### 2.1 Step 1: Dicke-State Preparation

**Goal:** Prepare the message register in a weighted superposition of Dicke states representing corrupted codewords with varying numbers of bit flips.

The initial state of the message register (m qubits) is:

```
|Psi_0> = sum_{k=0}^{ell} w_k * (1 / sqrt(C(m,k))) * sum_{|y|=k} |y>       (Eq. 5)
```

where:
- The sum over |y| = k denotes all computational basis states of Hamming weight k
- ell is the maximum number of errors the decoder is required to correct
- w_k are real coefficients (weights for each Hamming-weight sector)
- The normalized superposition over weight-k states is the **Dicke state**: `|D_{m,k}> = (1/sqrt(C(m,k))) sum_{|y|=k} |y>`  (Eq. 6)

**Computing the weights w_k:** The coefficients w_k in R^{ell+1} are the components of the principal eigenvector of an (ell+1) x (ell+1) tridiagonal matrix A:

```
        | 0    a_1   0    ...  0   |
        | a_1  d     a_2  ...  0   |
A   =   | 0    a_2   2d   ...  0   |       (Eq. 7)
        | :    :     :    '.   a_l |
        | 0    0     0    a_l  l*d |
```

with parameters:

```
a_k = sqrt(k(m - k + 1)),     d = (p - 2r) / sqrt(r(p - r))       (Eq. 8)
```

For max-XORSAT problems, **p = 2 and r = 1**, yielding:

```
d = (2 - 2) / sqrt(1 * 1) = 0
```

So the diagonal entries are all zero, and A is purely off-diagonal (a symmetric tridiagonal matrix with zeros on the diagonal).

**Circuit implementation:** The Dicke state preparation uses a two-step process (from Ref. [45]):
1. **Unary Amplitude Encoding:** Prepare the state `sum_{k=0}^{ell} w_k |k>` in a unary-encoded register (Eq. 14)
2. **Dicke state conversion:** Convert from the unary-encoded state into the desired superposition of Dicke states over m qubits (Ref. [39])

### 2.2 Step 2: Encoding Problem-Specific Phases

**Goal:** Imprint the constraint vector v into the phases of the message register.

Apply a Z gate to each message qubit i for which v_i = 1:

```
|Psi_1> = prod_{i=1}^{m} Z_i^{v_i} |Psi_0>
        = sum_{k=0}^{ell} w_k * (1/sqrt(C(m,k))) * sum_{|y|=k} (-1)^{v.y} |y>       (Eq. 9)
```

This is described in Algorithm 2 of the paper:

```
Algorithm 2: Apply Z gates based on vector v
for i = 0 to m-1 do:
    if v[i] = 1 then:
        Apply Z gate on qubit y[i]
```

The Z gate maps |0> -> |0> and |1> -> -|1>, so it introduces a phase of (-1)^{v_i} when the qubit is in state |1>. The product over all qubits gives the phase factor (-1)^{v.y} (dot product mod 2 in the exponent).

**Complexity:** At most m single-qubit Z gates (trivially parallelizable).

### 2.3 Step 3: Encoding the Syndrome

**Goal:** Create the syndrome register by computing B^T y (mod 2) into n ancilla qubits initialized to |0>.

Starting from |Psi_1> tensor |0^{n}>, apply an entangling unitary U to get:

```
|Psi_2> = U |Psi_1> |0^n>
        = sum_{k=0}^{ell} w_k * (1/sqrt(C(m,k))) * sum_{|y|=k} (-1)^{v.y} |y> |B^T y>       (Eq. 10)
```

The syndrome encoding is a matrix-vector multiplication mod 2, implemented via CNOT gates as described in Algorithm 3:

```
Algorithm 3: Syndrome encoding
for i = 0 to n-1 do:
    for j = 0 to m-1 do:
        if B^T[i,j] = 1 then:
            Apply CNOT with control qubit y[j] and target qubit syndrome[i]
```

Each CNOT computes the XOR of the target with the control. Iterating over all entries of B^T that are 1, the target qubit syndrome[i] accumulates the XOR of all y[j] connected to it, yielding the i-th component of B^T y mod 2.

**Complexity:** The number of CNOT gates equals the number of 1s in B^T (i.e., the number of nonzero entries, which for sparse/LDPC-like matrices is O(n) or O(m)).

### 2.4 Step 4: Decoding (The Core Challenge)

**Goal:** Uncompute the message register -- transform y back to the all-zero state while preserving the syndrome B^T y.

```
U_D |y> |B^T y>  ->  |0^m> |B^T y>       (Eq. 11)
```

If the decoder succeeds:

```
|Psi_3> = sum_{k=0}^{ell} w_k * (1/sqrt(C(m,k))) * sum_{|y|=k} (-1)^{v.y} |B^T y>       (Eq. 12)
```

This is identified as the well-known problem of **syndrome decoding** in classical error correction. The matrix B^T defines a binary linear code (e.g., an LDPC code), and transforming y -> 0^m based solely on knowledge of B^T y is equivalent to decoding the error pattern from a given syndrome.

The paper implements this step using **Binary Belief Propagation (BP1)** as a quantum circuit. This is the most complex and novel part of the implementation, detailed fully in Section 4 below.

**Critical property:** The decoder does not need to succeed on every branch of the superposition. Imperfect decoding still provides useful results (see Section 6.3 of the paper, Eqs. 65-66). The decoder's success rate directly influences the sampling efficiency: the probability of measuring the message register in the all-zero state equals R = sum_{k=0}^{ell} w_k^2 (1 - epsilon_k), where epsilon_k is the failure rate for weight-k errors.

### 2.5 Step 5: Hadamard Transform

**Goal:** Convert phase information into amplitude information via quantum interference.

Discard (trace out) the message register and apply a Hadamard gate to each qubit of the syndrome register:

```
|DQI> = prod_{i=1}^{n} H_i |Psi_3>       (Eq. 13)
```

This is the step where quantum interference does its work. The Hadamard transform maps computational basis states to superpositions with phase factors that depend on the bit values. Applied to each syndrome qubit, it converts the (-1)^{v.y} phase factors accumulated in the previous steps into **amplitude differences** between different solution candidates.

The result is a quantum state where the probability of measuring a particular bit string x is enhanced if x corresponds to a high-quality solution to the max-XORSAT instance. Specifically, the interference pattern constructively reinforces solutions that satisfy more XOR constraints and destructively cancels solutions that satisfy fewer.

**Complexity:** Exactly n single-qubit Hadamard gates (trivially parallelizable, depth 1).

### 2.6 Step 6: Measurement and Postprocessing

**Goal:** Extract candidate solutions from the quantum state.

Measure all qubits in both the syndrome and message registers in the computational basis.

**Postprocessing filter:** Since the decoder may not succeed with 100% probability, post-process the measurement outcomes by **keeping only those instances where the message register is measured in the all-zero state** |0^m>. This ensures the message register was properly uncomputed, meaning the interference step operated correctly.

The measurement results from the **syndrome register** encode candidate solutions to the max-XORSAT problem. Each accepted measurement outcome is a bit string in {0,1}^n that represents a candidate assignment for the decision variables.

**Sampling overhead:** The probability of passing the post-selection filter is R (defined above). Therefore, if we need N valid samples, we must run the circuit approximately N/R times. The decoder's success rate thus directly determines the sampling complexity.

---

## 3. ILP to max-XORSAT Pipeline

This section details the transformation from a general 0-1 Integer Linear Program to a max-XORSAT instance suitable for DQI (Section 5 of the paper). This is the critical bridge between real-world optimization problems and DQI's native input format.

### 3.1 Overview: Two-Step Transformation

Mapping a 0-1 ILP to max-XORSAT proceeds in two steps:

**Step 1 -- Constraint Feasibility:** Map the constraints into max-XORSAT form. An assignment x is feasible in the ILP (satisfies Ax <= b, A_= x = b_=, x in {0,1}^n) if and only if its binary representation s maximally satisfies the equation M s = y mod 2 for some matrix M and vector y of size polynomial in n.

**Step 2 -- Objective Function:** Reduce the objective function maximization to a max-XORSAT feasibility problem. Use binary search: for a lower bound beta on the objective, formulate ILP-c (the ILP with constraint c^T x >= beta, Ax < b+1, A_= x = b_=). The optimization problem is solved by finding the critical value beta* such that ILP-c is feasible when beta = beta* but not when beta = beta* + 1.

The total number of equations in the max-XORSAT instance scales as (Eq. 58):

```
xi^{ILP-c} in O(n(gamma + m*alpha + p*alpha^=)) subset O~(mn)
```

where alpha = max_{i,j}{alpha_{ij}}, alpha^= = max_{i,j}{alpha^=_{ij}}, and gamma = max_i{gamma_i} represent the maximum bit-widths of constraint matrix entries and cost coefficients respectively.

### 3.2 Encoding Pseudo-Boolean Constraints

The fundamental building block is embedding a pseudo-Boolean constraint of the form:

```
a^(1) x_1 + a^(2) x_2 + ... + a^(n) x_n  [bowtie]  b       (Eq. 34)
```

where each a^(k) is a positive integer, x_k in {0,1}, bowtie is either = or <=, and b is a positive integer. The key question is representing integer addition in binary operations XOR and AND.

For two integers u = u_0 u_1 ... u_{ell-1} and v = v_0 v_1 ... v_{ell-1}, schoolbook binary addition gives at each bit position k (Eq. 35):

```
s_k = v_k + u_k + c_{k-1}  mod 2       (sum bit)
c_k = (v_k . u_k) + (v_k . c_{k-1}) + (u_k . c_{k-1})  mod 2       (carry bit)
```

The sum s_k = v_k + u_k + c_{k-1} mod 2 is already in XOR (max-XORSAT) form. The carry requires the AND operation, which is handled by dedicated gadgets.

### 3.3 The AND Gadget (Eq. 36)

The AND gadget encodes the relation z = x . y (logical AND) as a max-XORSAT instance. For x, y, z in {0,1}:

```
x + y + z = 1  mod 2
    x + z = 0  mod 2
    y + z = 0  mod 2
        z = 0  mod 2
```

An assignment **maximally satisfies** this system (3 out of 4 equations) **if and only if** z = x . y.

**Proof sketch:** Enumerate all 8 possible (x,y,z) assignments and verify that only when z = x.y are exactly 3 equations satisfied, while any other z value satisfies at most 2.

### 3.4 The CARRY Gadget (Eq. 37)

The carry bit c_k from Eq. 35 is computed by introducing auxiliary variables p_k = v_k . u_k, q_k = v_k . c_{k-1}, and r_k = u_k . c_{k-1}, leading to c_k = p_k + q_k + r_k mod 2.

The full max-XORSAT instance embedding both the sum and carry at position k > 0 consists of **14 equations** (all mod 2):

```
    c_k + p_k + q_k + r_k = 0
              p_k + v_k + u_k = 0
                    p_k + v_k = 0
                    v_k + u_k = 0
                          p_k = 0
            q_k + v_k + c_{k-1} = 0       (Eq. 37)
                    q_k + v_k = 0
              v_k + c_{k-1} = 0
                          q_k = 0
      r_k + u_k + c_{k-1} = 0
                    r_k + u_k = 0
              u_k + c_{k-1} = 0
                          r_k = 0
s_k + u_k + v_k + c_{k-1} = 0
```

An assignment maximally satisfies this instance (11 out of 14 equations) if and only if Eq. 35 holds for the bits s_k, v_k, u_k, c_k, and c_{k-1}.

### 3.5 The CARRY1 Gadget (Eqs. 38-39)

A special case for the first step of addition (k = 0, no incoming carry):

```
s_0 = u_0 + v_0  mod 2
c_0 = u_0 . v_0
```

The max-XORSAT instance encoding this has **5 equations** (4 of which are satisfiable at maximum):

```
c_0 + u_0 + v_0 = 1
      c_0 + v_0 = 0       (Eq. 39)
      u_0 + v_0 = 0
            v_0 = 0
s_0 + v_0 + u_0 = 0
```

This is essentially the AND gadget (Eq. 36) applied with x = c_0, y = u_0, z = v_0, plus the sum equation.

### 3.6 Integer Adder (IA) Circuit

Adding two ell-bit integers u and v to produce sum s = s_0 s_1 ... s_ell uses:
- 1 CARRY1 gadget (for position k=0)
- (ell - 1) CARRY gadgets (for positions k=1 to ell-1)
- 1 final equation: s_ell + c_{ell-1} = 0 mod 2 (to capture the overflow bit)

**Total equations** (Eq. 40):

```
xi_ell^{IA} = 14(ell - 1) + 5 + 1 = 14*ell - 8
```

**Maximum satisfiable** (Eq. 40):

```
eta_ell^{IA} = 11(ell - 1) + 4 + 1 = 11*ell - 6
```

### 3.7 Weighted Integer Adder (WIA)

Embeds weighted addition of two bits:

```
a^(1) x_1 + a^(2) x_2 = b       (Eq. 41)
```

where a^(1), a^(2) are ell-bit integers, x_1, x_2 in {0,1}, and b is an (ell+1)-bit integer.

The construction depends on the bit values of a^(1) and a^(2) at each position k. For each of the 2 x 4 = 8 combinations of (a_k^(1), a_k^(2)) and range of k (k=0 vs k>0), Table 1 in the paper specifies which gadget to apply and how many equations are generated.

The total equations (Eq. 42):

```
xi_ell^{WIA} = 1 + sum_{k=0}^{ell-1} f_k(a_k^(1), a_k^(2))
```

and maximum satisfiable:

```
eta_ell^{WIA} = 1 + sum_{k=0}^{ell-1} g_k(a_k^(1), a_k^(2))
```

where f_k and g_k are tabulated in Table 1 based on the bit patterns.

### 3.8 Half Weighted Adder (HWA)

For constraints of the form a * x = y where a is a known ell-bit integer and x in {0,1} (Eq. 43):

```
x_k + y_k = 0  mod 2,    if a_k = 1
       y_k = 0  mod 2,    if a_k = 0       (Eq. 44)
```

Total: xi_ell^{HWA} = ell equations, all satisfiable: eta_ell^{HWA} = ell.

### 3.9 Multiple Integer Adder (MIA)

For n-term pseudo-Boolean constraints (Eq. 45):

```
a^(1) x_1 + a^(2) x_2 + ... + a^(n) x_n = y
```

The construction uses a **binary tree** of adders:
1. **First layer:** Pair up terms using WIAs (and one HWA if n is odd), producing ceil(n/2) intermediate results w_{1,k}.
2. **Subsequent layers:** Pair up intermediate results using Integer Adders (IAs), propagating up the tree until a single output y remains.

The tree has J(n) layers where J(n) is defined by m_1 = ceil(n/2), m_{j+1} = ceil(m_j/2), with m_J = 2. Figure 8 in the paper illustrates this for n=7.

The total and maximum satisfiable equations are given by Eqs. 46-47, with the bit-width at each tree node determined recursively by Eq. 48.

### 3.10 Integer Comparator

For inequality constraints x < b (Eq. 49), the comparison is converted to addition:

```
z = x + b_bar + 1,    where b_bar = 2^ell - b - 1    (2's complement)
```

Then x < b iff z_ell = 0 (the overflow bit). The embedding uses CARRY1 and CARRY2 gadgets (a variant of CARRY for when one operand bit is known). For embedding x < b, add z_ell = 0; for x >= b, add z_ell = 1.

Total equations (Eq. 54): xi_ell^{<} = 5*ell - 2, max satisfiable (Eq. 55): eta_ell^{<} = 4*ell - 1.

### 3.11 Equality Constraints

For x = b where b = b_0 b_1 ... b_{ell-1} is known: simply add equations x_k + b_k = 0 mod 2 for each k. Total: xi_ell^{=} = ell, all satisfiable.

### 3.12 Full ILP Embedding (Eq. 56-57)

The complete ILP-c (with objective lower bound beta) becomes:

```
c^T x >= beta       -->  MIA for objective + integer comparator (>=)
A x < b + 1         -->  MIA for each constraint row + integer comparator (<)
A_= x = b_=         -->  MIA for each equality constraint row + equality
```

The total equation count (Eq. 57):

```
xi^{ILP-c} = xi^{MIA}_{gamma_1,...,gamma_n} + xi^{>=}_{...}    (objective bound)
           + sum_{i=1}^{m} (xi^{MIA}_{...} + xi^{<}_{...})     (inequality constraints)
           + sum_{i=1}^{p} (xi^{MIA}_{...} + xi^{=}_{...})     (equality constraints)
```

### 3.13 From max-XORSAT to LDPC Code (Parity Check Matrix B)

Let B y = s be the max-XORSAT instance from ILP-c. Then B is a matrix of size xi^{ILP-c} x n' where n' in O~(mn) is the total number of bits in the max-XORSAT instance.

**Code distance:** The quality of the LDPC code C^{perp} = {x' in {0,1}^{n'} | B^T x' = 0} is characterized by its code distance d (the minimum number of linearly dependent rows in B). From the CARRY gadget construction (Eq. 63), three equations are always linearly dependent:

```
v_k + u_k = 0
v_k + c_{k-1} = 0
u_k + c_{k-1} = 0
```

This gives code distance d = 3, which barely qualifies as an error-correcting code. The paper notes that an alternative CARRY encoding using majority logic (Eq. 64) could improve d >= 4, but this is left for future work.

---

## 4. BP1 Quantum Circuit Details

This section details the quantum circuit implementation of the Binary Belief Propagation (BP1) decoder, which is the central novel contribution of the paper and the most complex part of the DQI circuit.

### 4.1 Classical BP1 Algorithm (Algorithm 1)

The classical BP1 algorithm takes as input:
- Parity-check matrix B^T in {0,1}^{n x m}
- Code word y in {0,1}^m (the "damaged" all-zero codeword)
- Maximum iterations T

The algorithm tries to decode y back to the all-zero codeword c = [0,...,0]^T:

```
Algorithm 1: Binary Belief Propagation (BP1)
Input: B^T, y, T
Output: (is_feasible, bits)

Initialize bits <- y
for t = 0 to T-1 do:
    syndrome <- (B^T . bits) mod 2
    if all entries of syndrome are zero then:
        return (True, bits)          // Decoding successful

    Initialize flip_candidates as zero vector (same shape as bits)
    for j = 0 to n-1 do:            // for each variable node
        connected_cnodes <- {i | B^T_{i,j} = 1}
        threshold <- |connected_cnodes|
        unsatisfied <- number of i in connected_cnodes with syndrome[i] = 1
        if unsatisfied >= threshold then:
            flip_candidates[j] <- 1

    bits <- (bits + flip_candidates) mod 2

if all entries of bits are zero then:
    return (True, bits)
else:
    return (False, bits)
```

**Key design choice:** The threshold for flipping is set to the **full degree** of the variable node (number of connected check nodes). This means a bit is flipped only if **all** its connected check nodes report unsatisfied syndromes. This is more stringent than typical BP thresholds but simplifies the quantum implementation.

**Why the stringent requirement:** In DQI, we need to map code words containing bit flips |y> to the all-zero state |0^m>, not just to any valid codeword. This is a stronger requirement than classical decoding (which only needs B^T b mod 2 = 0).

### 4.2 Quantum Registers

The quantum circuit extends the basic message (m qubits, denoted y) and syndrome (n qubits, denoted s0) registers with ancillary registers:

| Register | Size | Purpose |
|----------|------|---------|
| **y** (message) | m qubits | Stores the codeword/bit-flip pattern |
| **s0** (syndrome) | n qubits | Original syndrome B^T y |
| **h** (hamming) | ceil(log_2(t+1)) qubits | Stores Hamming weight of connected syndrome bits |
| **c** (comparator) | ceil(log_2(t+1)) qubits | Stores comparison result |
| **f_i** (flip) | m qubits each, T copies | Stores flip candidates for each iteration |
| **s_i** (syndrome updates) | n qubits each, T-1 copies | Updated syndrome after each iteration (except last) |

where t is the maximum number of variables in any single constraint (maximum row weight of B).

**Total qubit count** (Eq. 71):

```
N_q = m + n + 2*ceil(log_2(t+1)) + T*m + (T-1)*n
    = (T+1)*m + T*n + 2*ceil(log_2(t+1))
```

Since t <= n, the total number of qubits grows linearly with both the number of constraints (m), the number of variables (n), and the number of decoder iterations (T).

### 4.3 The Hamming Weight Subroutine (Fig. 2, Algorithm 4)

**Purpose:** Coherently compute the Hamming weight H(x_1, ..., x_t) of a subset of syndrome qubits and store it in the hamming register.

The circuit uses successive controlled U_{+1} gates, where U_{+1} performs modular binary increment:

```
U_{+1} |bin(a)> = |bin(a+1)>       (Eq. 17)
```

**Circuit structure (Fig. 2):** For each input qubit x_i (i = 1 to t), apply a controlled-U_{+1} gate on the hamming register, controlled by x_i. The hamming register is initialized to |0^r> where r = ceil(log_2(t+1)).

**Algorithm 4: Circuit construction for U_{+1}**

```
Initialize quantum circuit with r qubits
for i = 0 to r-1 do:
    if i = 0 then:
        Apply X gate to qubit 0     // Flip least significant bit
    else:
        Set controls = {0, 1, ..., i-1}
        for each qubit index c in controls do:
            Apply X gate to qubit c          // Prepare for multi-controlled Toffoli
        Apply multi-controlled X (Toffoli) gate with controls targeting qubit i
        for each qubit index c in controls do:
            Apply X gate to qubit c          // Uncompute preparation
```

The U_{+1} gate implements binary increment by:
1. Flipping bit 0 (LSB) unconditionally
2. For each higher bit i: flipping bit i only when all lower bits are 1 (about to overflow)

The X gates before and after the multi-controlled Toffoli transform the "all ones" condition into the proper carry propagation.

### 4.4 IntegerComparator Usage

After computing the Hamming weight H, compare it against the threshold (the degree of the current variable node = number of connected check nodes):

- Use Qiskit's `IntegerComparator` gate to check if H >= threshold
- The result is stored in the first qubit of the comparator register
- If the comparison is true (all connected syndromes are unsatisfied), this qubit is set to |1>

### 4.5 Flip Register Logic

For each variable node j in each iteration:

1. Compute Hamming weight of connected syndrome qubits into the hamming register
2. Compare against threshold using IntegerComparator
3. Apply a **CNOT gate** controlled on the first qubit of the comparator register, targeting the j-th qubit of the flip register f_i
4. **Uncompute** the hamming and comparator registers (apply inverse operations) to reset them for the next variable node

This means the flip register f_i accumulates all the flip decisions for iteration i.

### 4.6 Syndrome Update (Eq. 18)

At the end of each iteration i (except the last), create a new syndrome register s_i representing:

```
|B^T y_i> = |B^T(y_{i-1} xor flip_i)> = |B^T y_{i-1} xor B^T flip_i>       (Eq. 18)
```

The update is done in two steps:
1. **Copy previous syndrome:** Apply CNOT gates from each qubit of s_{i-1} to the corresponding qubit of s_i (mod 2 addition)
2. **Add flip contribution:** Apply the syndrome encoding subroutine (Algorithm 3) **controlled on the flip register f_i** to update s_i according to Eq. 18

### 4.7 Final Uncomputation and Inverse (Fig. 5)

After completing all T iterations:

1. **Sum all flips into message register:** Apply CNOT from each qubit of each flip register f_i to the corresponding qubit of the message register y:
   ```
   |y> --> |y xor flip_1 xor flip_2 xor ... xor flip_T>       (Eq. 19)
   ```

2. **Uncompute all ancilla registers:** Apply the **inverse** of all operations carried out before the final modulo-2 addition step. This reverses all computations on the ancilla registers (hamming, comparator, flip registers, intermediate syndrome registers), effectively cleaning up everything while leaving only:
   - The original syndrome register s0 (used only as a control, never modified)
   - The modified message register y

The inverse is applied as a single block (labeled "Inv" in Fig. 5). Because the original syndrome and message registers are only ever used as controls in the intermediate operations, the reversal cleanly uncomputes all ancillary state.

### 4.8 Key Insight: Extra Iterations Are Harmless (Eq. 20)

A crucial property of the quantum BP1 circuit: **if the decoder succeeds after d < T iterations, the remaining T - d iterations do not disturb the result.**

Specifically, if after d iterations:

```
|y xor flip_1 xor ... xor flip_d> = |0^m>       (Eq. 20)
```

then the corresponding syndrome becomes |B^T y_d> = |0^n>, and:
- All Hamming weight computations in subsequent iterations yield zero
- No CNOT gates from the comparator activate (threshold is never met when syndrome is all-zero)
- Each subsequent flip register f_i (for i > d) remains in the all-zero state
- The state is unchanged by iterations d+1 through T

This is critical because quantum circuits cannot perform mid-circuit conditional termination (without mid-circuit measurements). The algorithm must run all T iterations regardless, but the result is correct as long as decoding succeeded at any point during the T iterations.

### 4.9 Circuit Walkthrough (Figs. 3-5)

For the worked example with m=3 constraints, n=2 variables, T=2 iterations, and:

```
B = [[1, 0],
     [1, 1],
     [0, 1]]
```

**Fig. 3 (First iteration):** Shows registers s0 (2 qubits), h (2 qubits), c (2 qubits), f1 (3 qubits), s1 (2 qubits). The circuit iterates over the 3 rows of B (constraints), computing Hamming weights, comparing, and setting flip bits. Then the syndrome update constructs s1.

**Fig. 4 (Second iteration):** Same structure but reads from s1 instead of s0, writes to f2 (3 qubits) and would write to s2 (but this is the last iteration, so s2 is not needed).

**Fig. 5 (Final step):** CNOT gates from f1 and f2 into y (the message register), followed by the large "Inv" block that uncomputes everything except y and s0.

---

## 5. BCG-X Library Structure

The BCG-X DQI implementation (https://bcg-x-official.github.io/dqi/, Apache-2.0 license) provides an end-to-end Qiskit-based pipeline. Below is an overview of its architecture and how to use it.

### 5.1 Repository Organization

```
dqi/
|-- pipelines/                         # Experiment scripts and data pipeline
|   |-- experiment_empirical_vs_analytic.py
|   |-- experiment_histogram_small_max_xorsat.py
|   |-- experiment_performance_resources.py
|   |-- experiment_success_rates_2d_plots.py
|   |-- compute_milp_formulation.py    # ILP instance creation
|   |-- data/                          # Generated data (ILP, take rates, etc.)
|
|-- synthetic_data_generation/         # Mock data generation for automotive bundling
|   |-- generate_vehicles.py
|   |-- build_package_take_rate_data.py
|   |-- data/                          # Families, options CSVs
|   |-- package_templates.yaml         # Template for package configurations
|
|-- images/                            # Documentation images
|-- requirements.txt                   # Python dependencies
```

### 5.2 Key Modules and Their Roles

Based on the documentation and paper, the implementation contains these logical modules:

1. **ILP Formulation (`compute_milp_formulation.py`):** Takes synthetic automotive data (options, families, take rates) and constructs the ILP instance (Eq. 33) with objective (Eq. 21) and constraints (Eqs. 27-32). Outputs a JSON file.

2. **ILP-to-max-XORSAT Transformation:** Implements the full Section 5 pipeline -- encoding pseudo-Boolean constraints via AND, CARRY, CARRY1 gadgets, building the MIA tree, integer comparators, and assembling the parity-check matrix B.

3. **DQI Quantum Circuit Builder:** Constructs the full Qiskit circuit with:
   - Dicke state preparation (unary amplitude encoding + Dicke conversion)
   - Phase encoding (Z gates from v)
   - Syndrome encoding (CNOT matrix-vector multiply)
   - BP1 decoder circuit (Hamming weight, IntegerComparator, flip logic, syndrome update, inverse)
   - Hadamard transform on syndrome register

4. **Belief Propagation Decoders:**
   - BP1 (hard-decision, Gallager-type) -- implemented as both classical and quantum circuit
   - BP2 (soft-decision, state-of-the-art) -- classical only, used for comparison benchmarks

5. **Performance Estimation:** Classical computation of expected satisfied constraints via Eqs. 65-70 (avoids full quantum simulation for large instances).

6. **Resource Estimation:** Computes qubit counts (Eq. 71) and gate counts by block-wise transpilation into the gate set {Z, CNOT, RX, RY, RZ, SWAP}.

### 5.3 Pipeline Usage

**Environment setup:**

```bash
conda create -n dqi_env python=3.11
conda activate dqi_env
pip install -r requirements.txt
# Optional GPU support:
pip install --upgrade "jax[cuda12]"
```

**End-to-end workflow:**

```bash
# 1. Generate synthetic data
python synthetic_data_generation/generate_vehicles.py \
    -N 1000 \
    --input_dir synthetic_data_generation/data \
    --output_file synthetic_data_generation/data/test_vehicles.csv

# 2. Build take-rate data
python synthetic_data_generation/build_package_take_rate_data.py \
    --families_file synthetic_data_generation/data/families.csv \
    --options_file synthetic_data_generation/data/options.csv \
    --vehicles_file synthetic_data_generation/data/test_vehicles.csv \
    --template_file synthetic_data_generation/package_templates.yaml \
    --output_file pipelines/data/take_rates.parquet \
    --output_format parquet

# 3. Create ILP instance
python pipelines/compute_milp_formulation.py

# 4. Run experiments
python -m pipelines.experiment_performance_resources
python -m pipelines.experiment_histogram_small_max_xorsat
python -m pipelines.experiment_empirical_vs_analytic
python -m pipelines.experiment_success_rates_2d_plots
```

### 5.4 Key Experiments

| Experiment | What It Does | Key Output |
|-----------|-------------|-----------|
| `experiment_empirical_vs_analytic` | Compares quantum simulation results with analytical predictions on small random matrices | Scatter plot (Fig. 16) showing agreement |
| `experiment_histogram_small_max_xorsat` | Analyzes constraint satisfaction histograms for a small fixed B matrix at different ell values | Bar charts (Fig. 15) comparing DQI vs random |
| `experiment_performance_resources` | Full-scale: transforms ILP -> max-XORSAT, benchmarks against Gurobi, estimates resources | Performance curves (Fig. 11), qubit scaling (Fig. 12), gate scaling (Figs. 13-14) |
| `experiment_success_rates_2d_plots` | Benchmarks decoder success rates across problem sizes | 2D heatmaps (Figs. 1, 10) |

---

## 6. Implementation Notes for Qiskit

Practical guidance for implementing DQI in Qiskit >= 1.0, based on the paper's approach and the BCG-X codebase.

### 6.1 Qiskit Version and Dependencies

The BCG-X implementation targets Qiskit >= 1.0 (the 2024 rewrite). Key dependencies:
- **qiskit** >= 1.0 (core quantum circuits)
- **jax** (for efficient numerical computation, optional GPU acceleration)
- Python 3.11 recommended

Note: Qiskit 1.0 introduced breaking changes from 0.x. The `IntegerComparator` gate is from `qiskit.circuit.library`.

### 6.2 Circuit Construction Strategy

**Modular block-wise construction:** Due to the high depth of DQI circuits for realistic problem sizes, construct the circuit in modular blocks:

1. **Dicke state preparation block** -- Use existing implementations (Ref. [39], [45])
2. **Phase encoding block** -- Trivial: conditional Z gates
3. **Syndrome encoding block** -- CNOT cascade from Algorithm 3
4. **BP1 iteration blocks** (one per iteration T):
   - For each constraint (row of B): Hamming weight + comparator + flip + uncompute
   - Syndrome update at end of iteration
5. **Final flip accumulation** -- CNOT from all flip registers to message
6. **Inverse block** -- Inverse of steps 4-5 (excluding final flip CNOTs)
7. **Hadamard block** -- H gates on syndrome register

### 6.3 Transpilation Approach

**Block-wise transpilation** is essential for feasibility. The paper transpiles each block independently into the gate set {Z, CNOT, RX, RY, RZ, SWAP} using `qiskit.transpile()`. This yields an upper bound on total gate count (cross-block cancellations are not exploited).

Key blocks to transpile:
- The controlled U_{+1} gate (Algorithm 4) -- pre-transpile as a reusable module
- The `IntegerComparator` -- already available in Qiskit's circuit library, insert in transpiled form
- Their inverses -- use the transpiled versions directly

### 6.4 Register Management

```
# Pseudocode for register allocation
from qiskit import QuantumCircuit, QuantumRegister

m = num_constraints   # rows of B (message register size)
n = num_variables     # columns of B (syndrome register size)
t = max_row_weight    # max number of 1s in any row of B
T = num_iterations    # decoder iterations
r = ceil(log2(t + 1))  # hamming/comparator register size

y_reg = QuantumRegister(m, 'y')          # message
s0_reg = QuantumRegister(n, 's0')        # original syndrome
h_reg = QuantumRegister(r, 'h')          # hamming weight (reused)
c_reg = QuantumRegister(r, 'c')          # comparator (reused)

flip_regs = [QuantumRegister(m, f'f{i}') for i in range(1, T+1)]
synd_regs = [QuantumRegister(n, f's{i}') for i in range(1, T)]

# Total qubits: m + n + 2r + T*m + (T-1)*n
```

### 6.5 Key Qiskit Components

**IntegerComparator:** Available in `qiskit.circuit.library`. Compares an integer (stored in a quantum register) against a classical value. Used to check if the Hamming weight >= threshold.

```python
from qiskit.circuit.library import IntegerComparator
# Compare r-qubit register against value 'threshold'
comparator = IntegerComparator(num_state_qubits=r, value=threshold, geq=True)
```

**Multi-controlled X (Toffoli) gates:** Used in U_{+1} construction. In Qiskit >= 1.0:

```python
from qiskit.circuit.library import MCXGate
mcx = MCXGate(num_ctrl_qubits=i)  # i control qubits
```

**Circuit inversion:** To create the "Inv" block (Fig. 5):

```python
# Build the forward circuit for all BP1 iterations (without final flip CNOTs)
forward_circuit = build_bp1_forward(...)
inverse_circuit = forward_circuit.inverse()
```

### 6.6 Simulation Considerations

- For circuits with fewer than ~30 qubits, use Qiskit's `AerSimulator` (statevector or qasm)
- The smallest industrially relevant instance (m=827, n=345) requires ~1000+ qubits and is far beyond simulation
- For testing, use small random max-XORSAT instances (e.g., m=8, n=6, ell=2, T=1 yields ~26 qubits)
- The BCG-X code uses JAX for the classical performance estimation (Eq. 69), which can leverage GPUs

### 6.7 Practical Parameter Choices

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| ell (max errors) | 1-3 | Higher ell improves solution quality but increases Dicke state prep cost |
| T (decoder iterations) | 1-5 | More iterations help but add qubits linearly; extra iterations are harmless if decoding succeeds early |
| t (max row weight) | Problem-dependent | Determines hamming/comparator register size; sparse B matrices preferred |

### 6.8 Common Pitfalls

1. **Register reuse:** The hamming and comparator registers are reused across constraints within each iteration. You must uncompute them after each constraint processing step.

2. **Syndrome register proliferation:** Each iteration (except the last) creates a new syndrome register. This is the main driver of qubit count growth with T.

3. **Inverse circuit correctness:** The inverse block must exclude the final flip-to-message CNOTs. Only the intermediate computations are inverted.

4. **Dicke state weights:** For max-XORSAT, p=2 and r=1 give d=0, making the tridiagonal matrix A have zero diagonal. Verify eigenvector computation numerically.

5. **Parity-check matrix orientation:** The paper uses B^T (transpose) as the parity-check matrix in Algorithm 1 and 3. Be careful about which orientation is used where: B is m x n (equations x variables), B^T is n x m.

---

## 7. Adapting DQI for P&C Insurance Bundling

Our hackathon project applies DQI to P&C (Property & Casualty) insurance product bundling optimization, which is structurally analogous to the automotive option-packaging problem in the reference paper.

### 7.1 Problem Mapping

| Automotive (Paper) | Insurance (Our Project) |
|-------------------|----------------------|
| Vehicle options (paint, features, etc.) | Insurance coverages/riders (liability, collision, etc.) |
| Option packages (bundles of features) | Product bundles (packages of coverages) |
| Take rates (fraction of vehicles with option) | Attachment rates (fraction of policies with coverage) |
| Contribution margin per package | Premium margin per bundle |
| Compatibility restrictions | Regulatory/underwriting constraints |
| Dependency requirements | Coverage prerequisites (e.g., collision requires comprehensive) |
| Maximum options per package | Maximum coverages per bundle |
| Mandatory/optional family partitions | Required/optional coverage categories |

### 7.2 ILP Formulation Analogy

The objective function (Eq. 21 from the paper) maps directly:

```
max sum_m sum_i [T_m * C_i((1-d) * P_i) * x_{im}] + sum_i [T'_{only_i} * C_i(P_i)]
```

becomes:

```
max sum_b sum_c [A_b * M_c((1-d) * R_c) * x_{cb}] + sum_c [A'_{only_c} * M_c(R_c)]
```

where:
- b indexes bundles (packages), c indexes coverages (options)
- A_b = attachment rate of bundle b (take rate T_m)
- M_c = margin function for coverage c (contribution margin C_i)
- R_c = standalone rate/premium for coverage c (price P_i)
- d = discount factor for bundled pricing
- x_{cb} = binary decision variable (coverage c in bundle b)

The constraints (Eqs. 27-32) have direct analogs in insurance:
- Exactly one from mandatory coverage categories (Eq. 27)
- At most one from optional categories (Eq. 28)
- Maximum coverages per bundle (Eq. 29)
- Regulatory compatibility restrictions (Eq. 30)
- Coverage prerequisite/dependency requirements (Eq. 31)
- Avoid duplicating existing bundles (Eq. 32)

### 7.3 Workflow for Our Implementation

1. **Define insurance data:** Coverage options, families/categories, prices, attachment rates, compatibility/dependency rules
2. **Formulate as ILP:** Using the constraint structure from Section 4 of the paper
3. **Transform to max-XORSAT:** Apply the Section 5 pipeline (AND/CARRY/WIA/MIA gadgets)
4. **Build DQI circuit:** Following Section 2 (6 steps) with BP1 decoder from Section 3
5. **Simulate/run and extract solutions:** Measure, post-select, decode solution bit strings back to bundle assignments

### 7.4 Scaling Expectations

From the paper's results:
- The smallest industrially relevant automotive instance yields m=827 constraints, n=345 variables (m*n = 285,315)
- This requires ~1000+ logical qubits (Fig. 12) and ~10^6 gates (Fig. 13)
- For a hackathon demonstration, target small instances (m*n < 100) that can be simulated classically
- DQI consistently outperforms random sampling even at small scales (Fig. 15)

---

## Appendix A: Notation Quick Reference

| Symbol | Meaning |
|--------|---------|
| n | Number of decision variables (columns of B) |
| m | Number of constraints/equations (rows of B) |
| n' | Number of original ILP variables |
| m' | Number of original ILP constraints |
| B | Parity-check matrix, B in {0,1}^{m x n} |
| B^T | Transpose, B^T in {0,1}^{n x m} |
| v | Constraint vector, v in {0,1}^m |
| x | Decision variable, x in {0,1}^n |
| y | Message register state (codeword/error pattern) |
| ell | Maximum number of bit-flip errors to correct |
| T | Number of BP1 decoder iterations |
| t | Maximum row weight of B (max variables per constraint) |
| w_k | Dicke state weights (eigenvector of tridiagonal A) |
| d | Code distance of the LDPC code defined by B^T |
| S | Number of satisfied constraints |
| R | Post-selection probability (decoder success rate) |

## Appendix B: Key Equations Summary

| Eq. | Formula | Description |
|-----|---------|-------------|
| (1)-(2) | min c^T z, s.t. Cz >= a | General 0-1 ILP |
| (3)-(4) | Bx = v mod 2; max sum (-1)^{v_i + b_i.x} | max-XORSAT formulation |
| (5) | Psi_0 = sum w_k (1/sqrt(C(m,k))) sum_{abs(y)=k} abs(y) | Dicke state preparation |
| (7)-(8) | Tridiagonal A; a_k = sqrt(k(m-k+1)), d = (p-2r)/sqrt(r(p-r)) | Weight matrix |
| (9) | Psi_1 = prod Z_i^{v_i} Psi_0 | Phase encoding |
| (10) | Psi_2 = sum w_k ... (-1)^{v.y} abs(y) abs(B^T y) | Syndrome encoding |
| (11)-(12) | U_D abs(y) abs(B^T y) -> abs(0^m) abs(B^T y) | Decoding goal |
| (13) | abs(DQI) = prod H_i abs(Psi_3) | Hadamard transform |
| (17) | U_{+1} abs(bin(a)) = abs(bin(a+1)) | Binary increment gate |
| (18) | abs(B^T y_i) = abs(B^T y_{i-1} xor B^T flip_i) | Syndrome update |
| (19)-(20) | abs(y xor flip_1 xor ... xor flip_T) | Final flip accumulation |
| (36) | AND gadget: 4 equations, max 3 satisfied iff z = x.y | AND encoding |
| (37) | CARRY gadget: 14 equations, max 11 satisfied | Full carry encoding |
| (38)-(39) | CARRY1 gadget: 5 equations, max 4 satisfied | Initial carry encoding |
| (71) | N_q = (T+1)m + Tn + 2 ceil(log_2(t+1)) | Total qubit count |

## Appendix C: References

1. Jordan, Shutty, Wootters, Zalcman, Schmidhuber, King, Isakov, Babbush. "Optimization by decoded quantum interferometry." 2024. (The original DQI paper)
2. Sabater, El Harzli, Besjes, Erdmann, Klepsch, Hiltrop, Bobier, Cao, Riofrio. "Towards solving industrial integer linear programs with decoded quantum interferometry." arXiv:2509.08328, 2025. (Primary reference paper)
3. Patamawisut, Benchasattabuse, Hajdusek, Van Meter. "Quantum circuit design for decoded quantum interferometry." arXiv:2504.18334, 2025. (Gauss-Jordan decoder alternative)
4. Bu, Gu, Koh, Li. "Decoded quantum interferometry under noise." arXiv:2508.10725, 2025. (Noise analysis)
5. Chen, Liu, Zhandry. "Quantum algorithms for variants of average-case lattice problems via filtering." arXiv, 2021. (Filtering framework underlying DQI)
6. Gallager. "Low-density parity-check codes." IRE Trans. Information Theory, 1962. (Original LDPC codes)
7. Richardson, Urbanke. "Modern Coding Theory." Cambridge University Press, 2008. (BP decoder theory)
8. Bartschi, Eidenbenz. "Short-depth circuits for Dicke state preparation." IEEE QCE, 2022. (Dicke state circuits)
9. Javadi-Abhari et al. "Quantum computing with Qiskit." 2024. (Qiskit framework)
