#pragma once

#include <vector>

// Plain C++ interface - no CUDA types exposed here
// so main.cpp can include this without needing nvcc

// Timing breakdown returned from the GPU run
struct GpuMetrics
{
    float transferToDevice_ms   = 0.0f;  // cudaMemcpy host -> device
    float transferFromDevice_ms = 0.0f;  // cudaMemcpy device -> host
    float kernel_initRand_ms    = 0.0f;
    float kernel_initPop_ms     = 0.0f;
    float kernel_evaluate_ms    = 0.0f;  // accumulated across all generations
    float kernel_crossover_ms   = 0.0f;  // accumulated across all generations
    float kernel_mutation_ms    = 0.0f;  // accumulated across all generations
    float totalGpu_ms           = 0.0f;  // everything between first alloc and last free
};

void runGeneticAlgorithm(
    const std::vector<float>& h_target,
    std::vector<float>&       h_population_out,
    std::vector<float>&       h_fitness_out,
    GpuMetrics&               metrics
);
