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

void swap(float*& a, float*& b) {
    float* tmp = a;
    a = b;
    b = tmp;
}

int replacement(float* h_fit, int P) {
    int best = 0;
    for (int i = 1; i < P; i++)
        if (h_fit[i] < h_fit[best]) best = i;
    return best;
}

int main() {

    const int P = 1024;
    const int m = 6;
    const int G = 200;

    // Host data
    std::vector<float> h_pop(P*3);
    std::vector<float> h_fit(P);

    // Random init
    for (int i = 0; i < P; i++) {
        h_pop[i*3+0] = -40;          // A
        h_pop[i*3+1] = rand()%10;   // x
        h_pop[i*3+2] = rand()%10;   // y
    }

    // Device buffers
    float *d_pop, *d_children, *d_parents, *d_fit;
    float *d_ax, *d_ay, *d_rssi;
    curandState* d_states;

    cudaMalloc(&d_pop, P*3*sizeof(float));
    cudaMalloc(&d_children, P*3*sizeof(float));
    cudaMalloc(&d_parents, P*3*sizeof(float));
    cudaMalloc(&d_fit, P*sizeof(float));
    cudaMalloc(&d_states, P*sizeof(curandState));

    cudaMemcpy(d_pop, h_pop.data(), P*3*sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (P + threads - 1) / threads;

    initRand<<<blocks,threads>>>(d_states, time(NULL), P);

    for (int gen = 0; gen < G; gen++) {

        fitnessfunc<<<blocks,threads>>>(d_pop, d_fit, d_ax, d_ay, d_rssi, P, m);

        cudaMemcpy(h_fit.data(), d_fit, P*sizeof(float), cudaMemcpyDeviceToHost);
        int best = replacement(h_fit.data(), P);

        tournamentSelect<<<blocks,threads>>>(d_pop, d_fit, d_parents, d_states, P, 3);
        crossover<<<blocks,threads>>>(d_parents, d_children, d_states, P);
        mutation<<<blocks,threads>>>(d_children, d_states, P, 0.02f, 0.5f);

        // Elitism
        cudaMemcpy(
            d_children,
            &d_pop[best*3],
            3*sizeof(float),
            cudaMemcpyDeviceToDevice
        );

        swap(d_pop, d_children);

        if (gen % 20 == 0)
            std::cout << "Gen " << gen << " best error = " << h_fit[best] << "\n";
    }

    cudaMemcpy(h_pop.data(), d_pop, P*3*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nBest solution:\n";
    std::cout << "A = " << h_pop[0] << "\n";
    std::cout << "x = " << h_pop[1] << "\n";
    std::cout << "y = " << h_pop[2] << "\n";
}
