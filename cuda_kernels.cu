#include "cuda_kernels.cuh"

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
