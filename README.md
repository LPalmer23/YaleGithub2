# DQI vs QAOA for P&C Insurance Optimization

> **|Y>Quantum 2026 Hackathon Challenge**
> Sponsored by **Travelers Insurance + Quantinuum + LTM**

Comparing two quantum optimization algorithms -- Decoded Quantum Interferometry (DQI) and the Quantum Approximate Optimization Algorithm (QAOA) -- on a P&C insurance product bundling problem, targeting Quantinuum's trapped-ion H-series platform.

## The Problem

A P&C insurer sells individual coverage options (auto liability, collision, comprehensive, roadside, home dwelling, etc.). The goal: decide **which coverages to bundle into discount packages** to **maximize total contribution margin**.

This is a 0-1 Integer Linear Program (ILP) with constraints on coverage families, compatibility, dependencies, and package size. As the number of coverages and packages grows, the problem becomes intractable for classical solvers -- and a candidate for quantum speedup.

## Two Quantum Approaches

| | DQI | QAOA |
|---|---|---|
| **Approach** | Interference-based (no variational loop) | Variational (classical optimizer in the loop) |
| **Circuit** | Dicke state + syndrome encoding + BP1 decoder | Alternating cost/mixer layers |
| **Key mechanism** | Quantum Fourier transform on syndrome register | Parameter optimization to maximize expectation |
| **Constraint handling** | Parity check matrix B (max-XORSAT) | Penalty terms in cost Hamiltonian |
| **Hardware fit** | All-to-all connectivity avoids SWAP overhead in QFT | Native ZZPhase gate = one gate per cost interaction |

## Hackathon Challenge

Teams benchmark DQI vs QAOA on insurance bundling instances of increasing size:

See [challenge.docx](challenge.docx) for the full challenge specification.



## License

All solutions and source code are shared with challenge sponsors per |Y>Quantum 2026 rules.
