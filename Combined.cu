#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#define POP_SIZE 512
#define GENE_COUNT 4
#define GENERATIONS 300
#define MUTATION_RATE 0.05f
#define ELITE_COUNT 2

// ============================================================
// Device Fitness Function (Squared Error)
// ============================================================

__device__ float fitnessFunction(float* individual, float* target)
{
    float error = 0.0f;
    for (int i = 0; i < GENE_COUNT; i++)
    {
        float diff = individual[i] - target[i];
        error += diff * diff;
    }
    return error;
}

// ============================================================
// Initialize CURAND
// ============================================================

__global__ void initRand(curandState* states, unsigned long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < POP_SIZE)
        curand_init(seed, id, 0, &states[id]);
}

// ============================================================
// Initialize Population
// ============================================================

__global__ void initPopulation(float* population, curandState* states)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < POP_SIZE)
    {
        for (int g = 0; g < GENE_COUNT; g++)
        {
            population[id * GENE_COUNT + g] =
                curand_uniform(&states[id]) * 10.0f;
        }
    }
}

// ============================================================
// Evaluate Fitness
// ============================================================

__global__ void evaluateFitness(
    float* population,
    float* fitness,
    float* target)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < POP_SIZE)
    {
        fitness[id] = fitnessFunction(
            &population[id * GENE_COUNT],
            target
        );
    }
}

// ============================================================
// Tournament Selection
// ============================================================

__device__ int tournamentSelect(
    float* fitness,
    curandState* state)
{
    int a = curand(state) % POP_SIZE;
    int b = curand(state) % POP_SIZE;
    return (fitness[a] < fitness[b]) ? a : b;
}

// ============================================================
// Crossover
// ============================================================

__global__ void crossoverKernel(
    float* population,
    float* newPopulation,
    float* fitness,
    curandState* states)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < POP_SIZE)
    {
        int parent1 = tournamentSelect(fitness, &states[id]);
        int parent2 = tournamentSelect(fitness, &states[id]);

        int crossPoint = curand(&states[id]) % GENE_COUNT;

        for (int g = 0; g < GENE_COUNT; g++)
        {
            if (g < crossPoint)
                newPopulation[id * GENE_COUNT + g] =
                    population[parent1 * GENE_COUNT + g];
            else
                newPopulation[id * GENE_COUNT + g] =
                    population[parent2 * GENE_COUNT + g];
        }
    }
}

// ============================================================
// Mutation
// ============================================================

__global__ void mutationKernel(
    float* population,
    curandState* states)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < POP_SIZE)
    {
        for (int g = 0; g < GENE_COUNT; g++)
        {
            float r = curand_uniform(&states[id]);

            if (r < MUTATION_RATE)
            {
                population[id * GENE_COUNT + g] +=
                    curand_normal(&states[id]) * 0.1f;
            }
        }
    }
}

// ============================================================
// Host: Generate Noisy Target
// ============================================================

void generateNoisyTarget(std::vector<float>& target,
                         std::vector<float>& trueValues)
{
    trueValues = { 3.0f, 7.0f, 1.5f, 9.0f };

    for (int i = 0; i < GENE_COUNT; i++)
    {
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        target[i] = trueValues[i] + noise;
    }
}

// ============================================================
// MAIN
// ============================================================

int main()
{
    srand(42);

    float* d_population;
    float* d_newPopulation;
    float* d_fitness;
    float* d_target;
    curandState* d_states;

    cudaMalloc(&d_population, POP_SIZE * GENE_COUNT * sizeof(float));
    cudaMalloc(&d_newPopulation, POP_SIZE * GENE_COUNT * sizeof(float));
    cudaMalloc(&d_fitness, POP_SIZE * sizeof(float));
    cudaMalloc(&d_target, GENE_COUNT * sizeof(float));
    cudaMalloc(&d_states, POP_SIZE * sizeof(curandState));

    int threads = 256;
    int blocks = (POP_SIZE + threads - 1) / threads;

    initRand<<<blocks, threads>>>(d_states, 1234);
    initPopulation<<<blocks, threads>>>(d_population, d_states);

    std::vector<float> h_target(GENE_COUNT);
    std::vector<float> trueValues(GENE_COUNT);

    generateNoisyTarget(h_target, trueValues);

    cudaMemcpy(d_target, h_target.data(),
               GENE_COUNT * sizeof(float),
               cudaMemcpyHostToDevice);

    for (int gen = 0; gen < GENERATIONS; gen++)
    {
        evaluateFitness<<<blocks, threads>>>(
            d_population, d_fitness, d_target);

        crossoverKernel<<<blocks, threads>>>(
            d_population, d_newPopulation,
            d_fitness, d_states);

        mutationKernel<<<blocks, threads>>>(
            d_newPopulation, d_states);

        std::swap(d_population, d_newPopulation);
    }

    evaluateFitness<<<blocks, threads>>>(
        d_population, d_fitness, d_target);

    std::vector<float> h_population(POP_SIZE * GENE_COUNT);
    std::vector<float> h_fitness(POP_SIZE);

    cudaMemcpy(h_population.data(), d_population,
               POP_SIZE * GENE_COUNT * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(h_fitness.data(), d_fitness,
               POP_SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);

    int best = 0;
    for (int i = 1; i < POP_SIZE; i++)
        if (h_fitness[i] < h_fitness[best])
            best = i;

    std::cout << "\nRecovered parameters:\n";
    for (int g = 0; g < GENE_COUNT; g++)
        std::cout << h_population[best * GENE_COUNT + g] << " ";

    std::cout << "\n\nTrue parameters (noisy target):\n";
    for (int g = 0; g < GENE_COUNT; g++)
        std::cout << h_target[g] << " ";

    std::cout << "\n\nOriginal ground truth:\n";
    for (int g = 0; g < GENE_COUNT; g++)
        std::cout << trueValues[g] << " ";

    std::cout << "\n\nFinal fitness: "
              << h_fitness[best] << "\n";

    if (h_fitness[best] < 0.05f)
        std::cout << "\nRecovery successful.\n";
    else
        std::cout << "\nRecovery not perfect but converging.\n";

    cudaFree(d_population);
    cudaFree(d_newPopulation);
    cudaFree(d_fitness);
    cudaFree(d_target);
    cudaFree(d_states);

    return 0;
}
