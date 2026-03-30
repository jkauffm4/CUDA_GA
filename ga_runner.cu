#include "ga_runner.cuh"
#include "cuda_kernels.cuh"

#include <vector>

// Helper macro: creates a start/stop event pair, records start
#define EVENT_CREATE(start, stop)       \
    cudaEvent_t start, stop;            \
    cudaEventCreate(&start);            \
    cudaEventCreate(&stop);

#define EVENT_START(start)  cudaEventRecord(start)
#define EVENT_STOP(stop)    cudaEventRecord(stop); cudaEventSynchronize(stop)
#define EVENT_MS(start, stop, out)  cudaEventElapsedTime(&out, start, stop)
#define EVENT_DESTROY(start, stop)  cudaEventDestroy(start); cudaEventDestroy(stop)

void runGeneticAlgorithm(
    const std::vector<float>& h_target,
    std::vector<float>&       h_population_out,
    std::vector<float>&       h_fitness_out,
    GpuMetrics&               metrics)
{
    // --------------------------------------------------------
    // Total GPU timer (covers everything)
    // --------------------------------------------------------
    EVENT_CREATE(totalStart, totalStop);
    EVENT_START(totalStart);

    // --------------------------------------------------------
    // Allocate device memory
    // --------------------------------------------------------
    float*       d_population;
    float*       d_newPopulation;
    float*       d_fitness;
    float*       d_target;
    curandState* d_states;

    cudaMalloc(&d_population,    POP_SIZE * GENE_COUNT * sizeof(float));
    cudaMalloc(&d_newPopulation, POP_SIZE * GENE_COUNT * sizeof(float));
    cudaMalloc(&d_fitness,       POP_SIZE * sizeof(float));
    cudaMalloc(&d_target,        GENE_COUNT * sizeof(float));
    cudaMalloc(&d_states,        POP_SIZE * sizeof(curandState));

    int threads = 256;
    int blocks  = (POP_SIZE + threads - 1) / threads;

    // --------------------------------------------------------
    // Init kernels
    // --------------------------------------------------------
    {
        EVENT_CREATE(s, e);
        EVENT_START(s);
        initRand<<<blocks, threads>>>(d_states, 1234);
        EVENT_STOP(e);
        EVENT_MS(s, e, metrics.kernel_initRand_ms);
        EVENT_DESTROY(s, e);
    }

    {
        EVENT_CREATE(s, e);
        EVENT_START(s);
        initPopulation<<<blocks, threads>>>(d_population, d_states);
        EVENT_STOP(e);
        EVENT_MS(s, e, metrics.kernel_initPop_ms);
        EVENT_DESTROY(s, e);
    }

    // --------------------------------------------------------
    // Host -> Device transfer
    // --------------------------------------------------------
    {
        EVENT_CREATE(s, e);
        EVENT_START(s);
        cudaMemcpy(d_target, h_target.data(),
                   GENE_COUNT * sizeof(float),
                   cudaMemcpyHostToDevice);
        EVENT_STOP(e);
        EVENT_MS(s, e, metrics.transferToDevice_ms);
        EVENT_DESTROY(s, e);
    }

    // --------------------------------------------------------
    // Evolution loop
    // --------------------------------------------------------
    EVENT_CREATE(evS, evE);
    EVENT_CREATE(coS, coE);
    EVENT_CREATE(muS, muE);

    for (int gen = 0; gen < GENERATIONS; gen++)
    {
        float ms = 0.0f;

        EVENT_START(evS);
        evaluateFitness<<<blocks, threads>>>(d_population, d_fitness, d_target);
        EVENT_STOP(evE);
        EVENT_MS(evS, evE, ms);
        metrics.kernel_evaluate_ms += ms;

        EVENT_START(coS);
        crossoverKernel<<<blocks, threads>>>(
            d_population, d_newPopulation, d_fitness, d_states);
        EVENT_STOP(coE);
        EVENT_MS(coS, coE, ms);
        metrics.kernel_crossover_ms += ms;

        EVENT_START(muS);
        mutationKernel<<<blocks, threads>>>(d_newPopulation, d_states);
        EVENT_STOP(muE);
        EVENT_MS(muS, muE, ms);
        metrics.kernel_mutation_ms += ms;

        std::swap(d_population, d_newPopulation);
    }

    EVENT_DESTROY(evS, evE);
    EVENT_DESTROY(coS, coE);
    EVENT_DESTROY(muS, muE);

    // Final evaluation
    {
        EVENT_CREATE(s, e);
        float ms = 0.0f;
        EVENT_START(s);
        evaluateFitness<<<blocks, threads>>>(d_population, d_fitness, d_target);
        EVENT_STOP(e);
        EVENT_MS(s, e, ms);
        metrics.kernel_evaluate_ms += ms;
        EVENT_DESTROY(s, e);
    }

    // --------------------------------------------------------
    // Device -> Host transfer
    // --------------------------------------------------------
    h_population_out.resize(POP_SIZE * GENE_COUNT);
    h_fitness_out.resize(POP_SIZE);

    {
        EVENT_CREATE(s, e);
        EVENT_START(s);
        cudaMemcpy(h_population_out.data(), d_population,
                   POP_SIZE * GENE_COUNT * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fitness_out.data(), d_fitness,
                   POP_SIZE * sizeof(float),
                   cudaMemcpyDeviceToHost);
        EVENT_STOP(e);
        EVENT_MS(s, e, metrics.transferFromDevice_ms);
        EVENT_DESTROY(s, e);
    }

    cudaFree(d_population);
    cudaFree(d_newPopulation);
    cudaFree(d_fitness);
    cudaFree(d_target);
    cudaFree(d_states);

    // --------------------------------------------------------
    // Total GPU time
    // --------------------------------------------------------
    EVENT_STOP(totalStop);
    EVENT_MS(totalStart, totalStop, metrics.totalGpu_ms);
    EVENT_DESTROY(totalStart, totalStop);
}

