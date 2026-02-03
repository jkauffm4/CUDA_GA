#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <ctime>

extern "C" {
void initRand(curandState*, unsigned long, int);
void fitnessfunc(float*, float*, float*, float*, float*, int, int);
void tournamentSelect(float*, float*, float*, curandState*, int, int);
void crossover(float*, float*, curandState*, int);
void mutation(float*, curandState*, int, float, float);
}

int replacement(float* h_fit, int P) {
    int best = 0;
    for (int i = 1; i < P; i++)
        if (h_fit[i] < h_fit[best]) best = i;
    return best;
}
