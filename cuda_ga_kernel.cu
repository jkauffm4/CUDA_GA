#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void fitnessfunc(float* pop, float* fit, float* ax, float* ay, float* rssi, int P, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < P) {
        float A = pop[i*3 + 0];
        float x = pop[i*3 + 1];
        float y = pop[i*3 + 2];

        float err = 0.0f;
        for (int k = 0; k < m; k++) {
            float dx = x - ax[k];
            float dy = y - ay[k];
            float d  = sqrtf(dx*dx + dy*dy) + 1e-6f;
            float pred = A + 10.0f * log10f(d);
            float diff = pred - rssi[k];
            err += diff * diff;
        }
        fit[i] = err;
        //fit[i] = 1.0f / (1.0f + err);
    }
}

__global__ void tournamentSelect(float* pop, float* fit, float* parents, curandState* states, int P, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < P) {
        int best = -1;
        float bestFit = 1e30f;

        for (int j = 0; j < k; j++) {
            int r = curand(&states[i]) % P;
            if (fit[r] < bestFit) {
                bestFit = fit[r];
                best = r;
            }
        }
        parents[i*3+0] = pop[best*3+0];
        parents[i*3+1] = pop[best*3+1];
        parents[i*3+2] = pop[best*3+2];
    }
}

__global__ void crossover(float* parents, float* children, curandState* states, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < P) {

        // pick two random parents
        int p1 = curand(&states[i]) % P;
        int p2 = curand(&states[i]) % P;

        float alpha = curand_uniform(&states[i]); // in (0,1]

        // genes: A, x, y
        for (int g = 0; g < 3; g++) {
            float a = parents[p1*3 + g];
            float b = parents[p2*3 + g];
            children[i*3 + g] = alpha * a + (1.0f - alpha) * b;
        }
    }
}
