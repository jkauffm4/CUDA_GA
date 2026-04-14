# CUDA_GA
Cuda kernel for Genetic Algorithm

Built on windows in VS code. To run you need the following installed:
- CUDA Toolkit 12.6
- NVCC 12.6

Run build.ps1 to compile. Depending on security you may need to run powershell -ExecutionPolicy Bypass -File build.ps1 for compilation.

Afterwords run the executation cuda_rssi_ga.exe to run the code.

The code compiles and runs two different Genetic Algorithm (GA) implementations of the RSSI Path Loss Model Genetic Algorithm described in this paper by Shue et al. https://ieeexplore.ieee.org/abstract/document/10500295
One of those implementations is a CPU only GA implementation that produces a benchmark to compare to. The other implementation is the CUDA enabled GPU Accelerated GA implementation that parallelizes all functions of the GA. 
Below is the printout and report for the code run and benchmarked on an NVIDIA GeForce RTX 3090 GPU and an Intel Core i7-8700 CPU. Your results may differ depending on various factors like GPU VRAM, Catch size, Clock Speed, etc. 
If you are only interested in the CUDA code, below the report will be a breakdown of each of the kernels. 

+==========================================================+
|      CUDA GA BENCHMARK -- PAPER METRICS REPORT           |
+==========================================================+

  GENE_COUNT   = 4
  GENERATIONS  = 300
  MUTATION_RATE= 0.05
  NUM_TRIALS   = 10
  BASE_POP     = 65536

+==========================================================+
|  BASELINE BENCHMARK  --  POP=65536  GEN=300  GENES=4  TRIALS=10|
+==========================================================+
  Trial 1 /10  GPU wall: 50.80     ms  CPU wall: 5909.05   ms  Speedup: 116.32x
  Trial 2 /10  GPU wall: 56.57     ms  CPU wall: 6152.30   ms  Speedup: 108.75x
  Trial 3 /10  GPU wall: 165.90    ms  CPU wall: 6225.63   ms  Speedup: 37.53x
  Trial 4 /10  GPU wall: 172.63    ms  CPU wall: 6176.95   ms  Speedup: 35.78x
  Trial 5 /10  GPU wall: 184.11    ms  CPU wall: 5996.53   ms  Speedup: 32.57x
  Trial 6 /10  GPU wall: 197.93    ms  CPU wall: 6287.64   ms  Speedup: 31.77x
  Trial 7 /10  GPU wall: 266.88    ms  CPU wall: 6184.90   ms  Speedup: 23.17x
  Trial 8 /10  GPU wall: 308.60    ms  CPU wall: 6262.84   ms  Speedup: 20.29x
  Trial 9 /10  GPU wall: 252.40    ms  CPU wall: 6197.08   ms  Speedup: 24.55x
  Trial 10/10  GPU wall: 281.15    ms  CPU wall: 5900.76   ms  Speedup: 20.99x

  TABLE I -- Wall-Clock Execution Time (mean +/- std dev)
  ----------------------------------------------------------
  Metric                           Mean (ms)  Std Dev (ms)
  ----------------------------------------------------------
  CPU wall time                     6129.368       134.552
  GPU wall time                      193.696        83.786
  ----------------------------------------------------------
  Speedup (wall)                      31.64x
  Speedup (kernel only)               49.30x

  TABLE II -- GPU Kernel Breakdown (mean +/- std dev, 10 trials)
  ----------------------------------------------------------
  Stage                                    Mean (ms)    Std (ms)  % of GPU
  ----------------------------------------------------------
  initRand kernel                             22.196      38.564     11.5%
  initPopulation kernel                        0.229       0.123      0.1%
  evaluateFitness (all generations)            4.440       1.381      2.3%
  crossoverKernel (all generations)           63.327      35.460     32.9%
  mutationKernel (all generations)            34.126      14.735     17.7%
  ----------------------------------------------------------
    Kernel subtotal                          124.318      60.063     64.6%
  Transfer H->D (target vector)                0.165       0.252      0.1%
  Transfer D->H (population + fitness)         0.805       0.309      0.4%
    Transfer subtotal                          0.970       0.329      0.5%
  ----------------------------------------------------------
  GPU TOTAL (event timer)                    192.555      83.640    100.0%

  Transfer overhead as % of GPU total : 0.5%

  Solution Quality
  ----------------------------------------------------------
  GPU best fitness (mean)           0.000000
  CPU best fitness (mean)           0.000000

+==========================================================+
|  SCALABILITY SWEEP  --  10 trials per population size     |
+==========================================================+

  Pop Size        CPU (ms)      GPU (ms)   Kernel (ms)  Speedup(W)  Speedup(K)     Trans %
  ----------------------------------------------------------
  512                44.26         88.77         22.58        0.50        1.96        0.3%
  1024               90.81         81.43         20.01        1.12        4.54        0.4%
  2048              170.44         90.69         24.65        1.88        6.91        0.3%
  4096              363.95         83.10         28.27        4.38       12.87        0.4%
  8192              738.73         95.69         29.49        7.72       25.05        0.3%
  16384            1524.98         86.77         28.64       17.57       53.25        0.4%
  32768            3104.70        120.79         58.68       25.70       52.91        0.5%
  65536            6129.37        193.70        124.32       31.64       49.30        0.5%

  <img width="1332" height="728" alt="speedup_vs_population" src="https://github.com/user-attachments/assets/563554fc-87ef-4223-b39e-ed0f399ad7d5" />

