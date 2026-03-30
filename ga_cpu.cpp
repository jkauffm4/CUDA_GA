#include "ga_cpu.hpp"
#include "cuda_kernels.cuh"  // for POP_SIZE, GENE_COUNT, GENERATIONS, MUTATION_RATE

#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// ============================================================
// Fitness (squared error) — mirrors fitnessFunction() device code
// ============================================================

static float fitnessCPU(const float* individual, const float* target)
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
// Tournament selection — mirrors tournamentSelect() device code
// ============================================================

static int tournamentSelectCPU(const std::vector<float>& fitness)
{
    int a = rand() % POP_SIZE;
    int b = rand() % POP_SIZE;
    return (fitness[a] < fitness[b]) ? a : b;
}

// ============================================================
// Main CPU GA runner
// ============================================================

void runGeneticAlgorithmCPU(
    const std::vector<float>& h_target,
    std::vector<float>&       h_population_out,
    std::vector<float>&       h_fitness_out)
{
    // Use a separate RNG seed so CPU and GPU start from the same
    // logical initial conditions (different streams, same intent)
    srand(1234);

    std::vector<float> population(POP_SIZE * GENE_COUNT);
    std::vector<float> newPopulation(POP_SIZE * GENE_COUNT);
    std::vector<float> fitness(POP_SIZE);

    // --------------------------------------------------------
    // Initialize population with uniform random [0, 10]
    // mirrors initPopulation kernel
    // --------------------------------------------------------
    for (int id = 0; id < POP_SIZE; id++)
        for (int g = 0; g < GENE_COUNT; g++)
            population[id * GENE_COUNT + g] =
                ((float)rand() / RAND_MAX) * 10.0f;

    // --------------------------------------------------------
    // Evolution loop
    // --------------------------------------------------------
    for (int gen = 0; gen < GENERATIONS; gen++)
    {
        // Evaluate fitness — mirrors evaluateFitness kernel
        for (int id = 0; id < POP_SIZE; id++)
            fitness[id] = fitnessCPU(
                &population[id * GENE_COUNT],
                h_target.data());

        // Crossover — mirrors crossoverKernel
        for (int id = 0; id < POP_SIZE; id++)
        {
            int parent1   = tournamentSelectCPU(fitness);
            int parent2   = tournamentSelectCPU(fitness);
            int crossPoint = rand() % GENE_COUNT;

            for (int g = 0; g < GENE_COUNT; g++)
            {
                newPopulation[id * GENE_COUNT + g] =
                    (g < crossPoint)
                    ? population[parent1 * GENE_COUNT + g]
                    : population[parent2 * GENE_COUNT + g];
            }
        }

        // Mutation — mirrors mutationKernel
        for (int id = 0; id < POP_SIZE; id++)
        {
            for (int g = 0; g < GENE_COUNT; g++)
            {
                float r = (float)rand() / RAND_MAX;
                if (r < MUTATION_RATE)
                {
                    // Box-Muller approximation for normal-ish noise
                    float u1 = ((float)rand() / RAND_MAX) + 1e-6f;
                    float u2 = (float)rand() / RAND_MAX;
                    float normal = sqrtf(-2.0f * logf(u1)) *
                                   cosf(2.0f * 3.14159265f * u2);
                    newPopulation[id * GENE_COUNT + g] += normal * 0.1f;
                }
            }
        }

        std::swap(population, newPopulation);
    }

    // Final fitness evaluation
    for (int id = 0; id < POP_SIZE; id++)
        fitness[id] = fitnessCPU(
            &population[id * GENE_COUNT],
            h_target.data());

    h_population_out = population;
    h_fitness_out    = fitness;
}
