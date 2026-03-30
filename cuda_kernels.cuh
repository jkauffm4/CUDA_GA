#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define POP_SIZE 8192
#define GENE_COUNT 64
#define GENERATIONS 300
#define MUTATION_RATE 0.05f
#define ELITE_COUNT 2

// Kernel declarations
__global__ void initRand(curandState* states, unsigned long seed);
__global__ void initPopulation(float* population, curandState* states);
__global__ void evaluateFitness(float* population, float* fitness, float* target);
__global__ void crossoverKernel(float* population, float* newPopulation, float* fitness, curandState* states);
__global__ void mutationKernel(float* population, curandState* states);
