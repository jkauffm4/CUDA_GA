# CUDA_GA — GPU-Accelerated Genetic Algorithm for RSSI Localization

CUDA kernel implementation of the Genetic Algorithm (GA) described in [Shue et al. (SoutheastCon 2024)](https://ieeexplore.ieee.org/abstract/document/10500295) for simultaneous RSSI path loss model calibration and wireless sensor node localization.

Two implementations are included and benchmarked against each other:
- **CPU baseline** — single-threaded C++ GA
- **CUDA GPU** — fully parallelized GA with one thread per chromosome

---

## Requirements

- CUDA Toolkit 12.6
- NVCC 12.6

Built on Windows in VS Code.

---

## Building and Running

```powershell
# Standard
build.ps1

# If execution policy blocks the script
powershell -ExecutionPolicy Bypass -File build.ps1
```

Then run:

```
cuda_rssi_ga.exe
```

---

## Benchmark Results

Results below were collected on an **NVIDIA GeForce RTX 3090** and an **Intel Core i7-8700**. Your results may vary depending on GPU VRAM, cache size, clock speed, and other system factors.

### Configuration

| Parameter | Value |
|-----------|-------|
| GENE_COUNT | 4 |
| GENERATIONS | 300 |
| MUTATION_RATE | 0.05 |
| NUM_TRIALS | 10 |
| BASE_POP | 65,536 |

---

### Table I — Wall-Clock Execution Time

*Baseline: 65,536 individuals, 300 generations, 10 trials (mean ± std dev)*

| Implementation | Wall Time | Kernel Only | Speedup (W / K) |
|----------------|-----------|-------------|-----------------|
| CPU (single thread) | 6,129 ± 135 ms | N/A | 1.0× |
| GPU (CUDA) | 194 ± 84 ms | 124 ± 60 ms | **31.64× / 49.30×** |

#### Per-Trial Breakdown

| Trial | GPU Wall (ms) | CPU Wall (ms) | Speedup |
|-------|--------------|---------------|---------|
| 1  | 50.80  | 5,909.05 | 116.32× |
| 2  | 56.57  | 6,152.30 | 108.75× |
| 3  | 165.90 | 6,225.63 | 37.53× |
| 4  | 172.63 | 6,176.95 | 35.78× |
| 5  | 184.11 | 5,996.53 | 32.57× |
| 6  | 197.93 | 6,287.64 | 31.77× |
| 7  | 266.88 | 6,184.90 | 23.17× |
| 8  | 308.60 | 6,262.84 | 20.29× |
| 9  | 252.40 | 6,197.08 | 24.55× |
| 10 | 281.15 | 5,900.76 | 20.99× |

> The high variance in GPU wall time across trials is a known characteristic of GPU clock boosting behavior. Trials 1–2 benefit from a cold boost state, while later trials reflect sustained thermal steady-state performance. This is why results are averaged over 10 trials.

---

### Table II — GPU Kernel Breakdown

*65,536 individuals, 300 generations, 10 trials (mean ± std dev)*

| Stage | Mean (ms) | SD (ms) | % of GPU Total |
|-------|-----------|---------|----------------|
| `initRand` kernel | 22.196 | 38.564 | 11.5% |
| `initPopulation` kernel | 0.229 | 0.123 | 0.1% |
| `evaluateFitness` (all generations) | 4.440 | 1.381 | 2.3% |
| `crossoverKernel` (all generations) | 63.327 | 35.460 | 32.9% |
| `mutationKernel` (all generations) | 34.126 | 14.735 | 17.7% |
| **Kernel subtotal** | **124.318** | **60.063** | **64.6%** |
| H→D Transfer (target vector) | 0.165 | 0.252 | 0.1% |
| D→H Transfer (population + fitness) | 0.805 | 0.309 | 0.4% |
| **Transfer subtotal** | **0.970** | **0.329** | **0.5%** |
| **GPU Total** | **192.555** | **83.640** | **100%** |

**Transfer overhead: 0.5% of GPU total**

---

### Table III — Scalability Sweep

*10 trials per population size, mean values. W = wall-clock speedup, K = kernel-only speedup.*

| Population Size | CPU (ms) | GPU (ms) | Kernel (ms) | Speedup (W) | Speedup (K) | Transfer % |
|----------------|----------|----------|-------------|-------------|-------------|------------|
| 512 | 44.26 | 88.77 | 22.58 | 0.50× | 1.96× | 0.3% |
| 1,024 | 90.81 | 81.43 | 20.01 | 1.12× | 4.54× | 0.4% |
| 2,048 | 170.44 | 90.69 | 24.65 | 1.88× | 6.91× | 0.3% |
| 4,096 | 363.95 | 83.10 | 28.27 | 4.38× | 12.87× | 0.4% |
| 8,192 | 738.73 | 95.69 | 29.49 | 7.72× | 25.05× | 0.3% |
| 16,384 | 1,524.98 | 86.77 | 28.64 | **17.57×** | **53.25×** | 0.4% |
| 32,768 | 3,104.70 | 120.79 | 58.68 | 25.70× | 52.91× | 0.5% |
| 65,536 | 6,129.37 | 193.70 | 124.32 | 31.64× | 49.30× | 0.5% |

> Below ~1,024 individuals the GPU is slower than the CPU due to fixed kernel launch and cuRAND initialization overhead. Kernel-only speedup peaks at **53.25×** near 16,384 individuals and then declines — a signature of global memory bandwidth saturation as the population buffer outgrows the device L2 cache.

---

### Speedup vs. Population Size

  <img width="1332" height="728" alt="speedup_vs_population" src="https://github.com/user-attachments/assets/563554fc-87ef-4223-b39e-ed0f399ad7d5" />


---

## Kernel Overview

| Kernel | Description |
|--------|-------------|
| `initRand` | Initializes per-thread cuRAND states for independent random streams |
| `initPopulation` | Fills population array with uniform random values in [0, 10] |
| `evaluateFitness` | Computes sum-of-squared-differences fitness score per individual |
| `crossoverKernel` | Tournament selection + single-point crossover to produce offspring |
| `mutationKernel` | Applies Gaussian perturbations per gene at MUTATION_RATE probability |

