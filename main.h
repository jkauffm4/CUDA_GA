#pragma once

#include "ga_runner.cuh"
#include "ga_cpu.hpp"
#include "cuda_kernels.cuh"   // for GENE_COUNT, GENERATIONS, MUTATION_RATE

#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <string>
#include <fstream>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ============================================================
// Configuration
// ============================================================

// Number of repeated trials at the baseline population size.
static const int NUM_TRIALS   = 10;

// Baseline population size (used for Table I and Table II).
static const int BASE_POP     = 65536;

// Population sizes to sweep for the scalability study (Table III).
static const int SCALE_POPS[] = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
static const int N_SCALE      = sizeof(SCALE_POPS) / sizeof(SCALE_POPS[0]);

// ============================================================
// Structs
// ============================================================

struct TrialResult {
    double     gpuWall_ms;
    double     cpuWall_ms;
    GpuMetrics gpu;
    double     gpuBestFitness;
    double     cpuBestFitness;
};

// Returned by runBaselineBenchmark() and passed into runScalabilityBenchmark()
// so the BASE_POP row in Table III is identical to Table I.
struct BaselineMeans {
    double cpuWall_ms;
    double gpuWall_ms;
    double kernelTotal_ms;
    double transTotal_ms;
    double totalGpu_ms;
};

// ============================================================
// Utility helpers
// ============================================================

static double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double stddev(const std::vector<double>& v) {
    double m = mean(v);
    double sq = 0.0;
    for (double x : v) sq += (x - m) * (x - m);
    return std::sqrt(sq / v.size());
}

// Generate a noisy target vector. Values cycle through {3, 7, 1.5, 9}
// with small additive noise seeded by 'seed'.
static void generateTarget(std::vector<float>& target, unsigned seed) {
    srand(seed);
    const float base[] = { 3.0f, 7.0f, 1.5f, 9.0f };
    for (int i = 0; i < GENE_COUNT; i++) {
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        target[i] = base[i % 4] + noise;
    }
}

static double bestFitness(const std::vector<float>& fitness) {
    return *std::min_element(fitness.begin(), fitness.end());
}

static void printHeader(const std::string& title) {
    std::cout << "\n";
    std::cout << "+==========================================================+\n";
    std::cout << "|  " << std::left << std::setw(57) << title << "|\n";
    std::cout << "+==========================================================+\n";
}

static void printDivider() {
    std::cout << "  ----------------------------------------------------------\n";
}

// ============================================================
// Single-trial runner
// ============================================================

static TrialResult runTrial(int popSize, unsigned targetSeed) {
    std::vector<float> h_target(GENE_COUNT);
    generateTarget(h_target, targetSeed);

    TrialResult r;

    // GPU run
    {
        std::vector<float> pop, fit;
        GpuMetrics m;
        auto t0 = Clock::now();
        runGeneticAlgorithm(h_target, pop, fit, m, popSize);
        auto t1 = Clock::now();
        r.gpuWall_ms     = Ms(t1 - t0).count();
        r.gpu            = m;
        r.gpuBestFitness = bestFitness(fit);
    }

    // CPU run
    {
        std::vector<float> pop, fit;
        auto t0 = Clock::now();
        runGeneticAlgorithmCPU(h_target, pop, fit, popSize);
        auto t1 = Clock::now();
        r.cpuWall_ms     = Ms(t1 - t0).count();
        r.cpuBestFitness = bestFitness(fit);
    }

    return r;
}

// ============================================================
// CSV export for pgfplots figure
// ============================================================

static void writeScalabilityCSV(
    const std::string& filename,
    const std::vector<int>& popSizes,
    const std::vector<double>& speedupWall,
    const std::vector<double>& speedupKernel) {

    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "  [Warning] Could not open " << filename << " for writing.\n";
        return;
    }

    f << "pop_size,speedup_wall,speedup_kernel\n";
    for (int i = 0; i < (int)popSizes.size(); i++) {
        f << std::fixed << std::setprecision(4)
          << popSizes[i] << ","
          << speedupWall[i] << ","
          << speedupKernel[i] << "\n";
    }

    f.close();
    std::cout << "  [CSV written to " << filename << "]\n";
}

// ============================================================
// TABLE I + TABLE II  --  Multi-trial baseline benchmark
// ============================================================

static BaselineMeans runBaselineBenchmark() {
    printHeader("BASELINE BENCHMARK  --  POP=" + std::to_string(BASE_POP)
                + "  GEN=" + std::to_string(GENERATIONS)
                + "  GENES=" + std::to_string(GENE_COUNT)
                + "  TRIALS=" + std::to_string(NUM_TRIALS));

    std::vector<double> gpuWall, cpuWall, kernelTotal,
                        initRand, initPop, evaluate, crossover, mutation,
                        transH2D, transD2H, transTotal, totalGpu,
                        gpuFit, cpuFit;

    for (int t = 0; t < NUM_TRIALS; t++) {
        TrialResult r = runTrial(BASE_POP, 42 + t);

        gpuWall.push_back(r.gpuWall_ms);
        cpuWall.push_back(r.cpuWall_ms);

        double kt = r.gpu.kernel_initRand_ms
                  + r.gpu.kernel_initPop_ms
                  + r.gpu.kernel_evaluate_ms
                  + r.gpu.kernel_crossover_ms
                  + r.gpu.kernel_mutation_ms;
        kernelTotal.push_back(kt);
        initRand   .push_back(r.gpu.kernel_initRand_ms);
        initPop    .push_back(r.gpu.kernel_initPop_ms);
        evaluate   .push_back(r.gpu.kernel_evaluate_ms);
        crossover  .push_back(r.gpu.kernel_crossover_ms);
        mutation   .push_back(r.gpu.kernel_mutation_ms);
        transH2D   .push_back(r.gpu.transferToDevice_ms);
        transD2H   .push_back(r.gpu.transferFromDevice_ms);
        transTotal .push_back(r.gpu.transferToDevice_ms + r.gpu.transferFromDevice_ms);
        totalGpu   .push_back(r.gpu.totalGpu_ms);
        gpuFit     .push_back(r.gpuBestFitness);
        cpuFit     .push_back(r.cpuBestFitness);

        std::cout << "  Trial " << std::setw(2) << (t + 1) << "/" << NUM_TRIALS
                  << "  GPU wall: " << std::fixed << std::setprecision(2)
                  << std::setw(9) << r.gpuWall_ms << " ms"
                  << "  CPU wall: " << std::setw(9) << r.cpuWall_ms << " ms"
                  << "  Speedup: " << std::setprecision(2)
                  << (r.cpuWall_ms / r.gpuWall_ms) << "x\n";
    }

    std::cout << "\n";

    // -- TABLE I: Wall-clock comparison ---------------------------------------
    std::cout << "  TABLE I -- Wall-Clock Execution Time (mean +/- std dev)\n";
    printDivider();
    std::cout << "  " << std::left  << std::setw(28) << "Metric"
              << std::right << std::setw(14) << "Mean (ms)"
              << std::setw(14) << "Std Dev (ms)" << "\n";
    printDivider();

    auto row = [&](const std::string& label, const std::vector<double>& data) {
        std::cout << "  " << std::left  << std::setw(28) << label
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(14) << mean(data)
                  << std::setw(14) << stddev(data) << "\n";
    };

    row("CPU wall time", cpuWall);
    row("GPU wall time", gpuWall);
    printDivider();

    double meanSpeedupWall   = mean(cpuWall) / mean(gpuWall);
    double meanSpeedupKernel = mean(cpuWall) / mean(kernelTotal);
    std::cout << "  " << std::left << std::setw(28) << "Speedup (wall)"
              << std::right << std::setw(13) << std::fixed << std::setprecision(2)
              << meanSpeedupWall << "x\n";
    std::cout << "  " << std::left << std::setw(28) << "Speedup (kernel only)"
              << std::right << std::setw(13) << meanSpeedupKernel << "x\n";

    // -- TABLE II: Per-kernel breakdown ---------------------------------------
    std::cout << "\n  TABLE II -- GPU Kernel Breakdown (mean +/- std dev, "
              << NUM_TRIALS << " trials)\n";
    printDivider();
    std::cout << "  " << std::left  << std::setw(38) << "Stage"
              << std::right << std::setw(12) << "Mean (ms)"
              << std::setw(12) << "Std (ms)"
              << std::setw(10) << "% of GPU" << "\n";
    printDivider();

    double gpuTotalMean = mean(totalGpu);
    auto krow = [&](const std::string& label, const std::vector<double>& data) {
        double m   = mean(data);
        double s   = stddev(data);
        double pct = (m / gpuTotalMean) * 100.0;
        std::cout << "  " << std::left  << std::setw(38) << label
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << m
                  << std::setw(12) << s
                  << std::setprecision(1) << std::setw(9) << pct << "%\n";
    };

    krow("initRand kernel",                     initRand);
    krow("initPopulation kernel",               initPop);
    krow("evaluateFitness (all generations)",   evaluate);
    krow("crossoverKernel (all generations)",   crossover);
    krow("mutationKernel (all generations)",    mutation);
    printDivider();
    krow("  Kernel subtotal",                   kernelTotal);
    krow("Transfer H->D (target vector)",        transH2D);
    krow("Transfer D->H (population + fitness)", transD2H);
    krow("  Transfer subtotal",                  transTotal);
    printDivider();
    krow("GPU TOTAL (event timer)",              totalGpu);

    double transOverheadPct = (mean(transTotal) / gpuTotalMean) * 100.0;
    std::cout << "\n  Transfer overhead as % of GPU total : "
              << std::fixed << std::setprecision(1) << transOverheadPct << "%\n";

    std::cout << "\n  Solution Quality\n";
    printDivider();
    std::cout << "  " << std::left << std::setw(28) << "GPU best fitness (mean)"
              << std::right << std::fixed << std::setprecision(6)
              << std::setw(14) << mean(gpuFit) << "\n";
    std::cout << "  " << std::left << std::setw(28) << "CPU best fitness (mean)"
              << std::right << std::setw(14) << mean(cpuFit) << "\n";

    // Return means so the scalability sweep can reuse the BASE_POP row,
    // guaranteeing Table I and Table III are always consistent.
    BaselineMeans bm;
    bm.cpuWall_ms     = mean(cpuWall);
    bm.gpuWall_ms     = mean(gpuWall);
    bm.kernelTotal_ms = mean(kernelTotal);
    bm.transTotal_ms  = mean(transTotal);
    bm.totalGpu_ms    = mean(totalGpu);
    return bm;
}

// ============================================================
// TABLE III  --  Scalability sweep
// ============================================================

static void runScalabilityBenchmark(const BaselineMeans& baseline) {
    printHeader("SCALABILITY SWEEP  --  " + std::to_string(NUM_TRIALS)
                + " trials per population size");

    std::cout << "\n  " << std::left
              << std::setw(10) << "Pop Size"
              << std::right
              << std::setw(14) << "CPU (ms)"
              << std::setw(14) << "GPU (ms)"
              << std::setw(14) << "Kernel (ms)"
              << std::setw(12) << "Speedup(W)"
              << std::setw(12) << "Speedup(K)"
              << std::setw(12) << "Trans %" << "\n";
    printDivider();

    std::vector<int>    allPopSizes;
    std::vector<double> allSpeedupWall;
    std::vector<double> allSpeedupKernel;

    for (int pi = 0; pi < N_SCALE; pi++) {
        int pop = SCALE_POPS[pi];
        double mCPU, mGPU, mKernel, mTrans, mTotal;

        // Reuse baseline means for BASE_POP so Table I and Table III match.
        if (pop == BASE_POP) {
            mCPU    = baseline.cpuWall_ms;
            mGPU    = baseline.gpuWall_ms;
            mKernel = baseline.kernelTotal_ms;
            mTrans  = baseline.transTotal_ms;
            mTotal  = baseline.totalGpu_ms;
        } else {
            std::vector<double> gpuWall, cpuWall, kernelTotal, transTotal, totalGpu;

            for (int t = 0; t < NUM_TRIALS; t++) {
                TrialResult r = runTrial(pop, 42 + t);

                gpuWall   .push_back(r.gpuWall_ms);
                cpuWall   .push_back(r.cpuWall_ms);
                double kt = r.gpu.kernel_initRand_ms
                          + r.gpu.kernel_initPop_ms
                          + r.gpu.kernel_evaluate_ms
                          + r.gpu.kernel_crossover_ms
                          + r.gpu.kernel_mutation_ms;
                kernelTotal.push_back(kt);
                transTotal .push_back(r.gpu.transferToDevice_ms
                                    + r.gpu.transferFromDevice_ms);
                totalGpu   .push_back(r.gpu.totalGpu_ms);
            }

            mCPU    = mean(cpuWall);
            mGPU    = mean(gpuWall);
            mKernel = mean(kernelTotal);
            mTrans  = mean(transTotal);
            mTotal  = mean(totalGpu);
        }

        double sW = mCPU / mGPU;
        double sK = mCPU / mKernel;

        allPopSizes   .push_back(pop);
        allSpeedupWall  .push_back(sW);
        allSpeedupKernel.push_back(sK);

        std::cout << "  " << std::left
                  << std::setw(10) << pop
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(14) << mCPU
                  << std::setw(14) << mGPU
                  << std::setw(14) << mKernel
                  << std::setprecision(2)
                  << std::setw(12) << sW
                  << std::setw(12) << sK
                  << std::setprecision(1)
                  << std::setw(11) << ((mTrans / mTotal) * 100.0) << "%\n";
    }

    std::cout << "\n  (W) = wall-clock speedup   (K) = kernel-only speedup\n";
    std::cout << "  Trans % = PCIe transfer as percentage of GPU total time\n";

    writeScalabilityCSV("scalability.csv", allPopSizes, allSpeedupWall, allSpeedupKernel);
}