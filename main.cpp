#include "main.h"

int main() {
    std::cout << "+==========================================================+\n";
    std::cout << "|      CUDA GA BENCHMARK -- PAPER METRICS REPORT           |\n";
    std::cout << "+==========================================================+\n";
    std::cout << "\n  GENE_COUNT   = " << GENE_COUNT
              << "\n  GENERATIONS  = " << GENERATIONS
              << "\n  MUTATION_RATE= " << MUTATION_RATE
              << "\n  NUM_TRIALS   = " << NUM_TRIALS
              << "\n  BASE_POP     = " << BASE_POP << "\n";

    // Warm up the GPU before timed runs to avoid first-launch overhead.
    std::cout << "\n  [Warming up GPU...]\n";
    {
        std::vector<float> t(GENE_COUNT, 1.0f), pop, fit;
        GpuMetrics m;
        runGeneticAlgorithm(t, pop, fit, m, 256);
    }
    std::cout << "  [Warm-up complete.]\n";

    BaselineMeans baseline = runBaselineBenchmark();
    runScalabilityBenchmark(baseline);

    return 0;
}