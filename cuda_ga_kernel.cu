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
    }
}