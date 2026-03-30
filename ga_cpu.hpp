#pragma once

#include <vector>

// CPU-only genetic algorithm — mirrors the CUDA version exactly
// so results and speedup are directly comparable

void runGeneticAlgorithmCPU(
    const std::vector<float>& h_target,
    std::vector<float>&       h_population_out,
    std::vector<float>&       h_fitness_out
);
