/**
 * hipcc -std=c++20 ./vectorAdd.cpp  && ./a.out
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "HipContainer.hpp"

// Computes ceil(numerator/divisor) for integer types.
template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
intT1 ceildiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}

__global__ void vecAdd(float *a, const float *b)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] += b[idx];
}

// Print the array to stdout
template <typename T>
void printVec(const T v)
{
    for (int i = 0; i < v.size(); ++i)
    {
        std::cout << v[i] << " ";
    }
    std::cout << "\n";
}

// Fill the array with some values
template <typename T>
void fillArray(T &v)
{
    for (int i = 0; i < v.size(); ++i)
    {
        v[i] = i; // sin(i);
    }
}

int main()
{
    hipEvent_t start, stop;
    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&stop));
    float ms;

    std::cout << "HIP vector addition example\n";

    const int N = 1 << 28;

    std::vector<float> vala(N);
    fillArray(vala);

    std::vector<float> valb(N);
    fillArray(valb);

    auto a = SmartArray(std::move(vala));
    auto b = SmartArray(std::move(valb));

    // Run Kernel
    int blockSize = N;
    int blocks = 1;

    HIP_CALL(hipEventRecord(start));

    vecAdd<<<blocks, blockSize>>>(a.toDevice(), b.toDevice());

    HIP_CALL(hipEventRecord(stop));
    HIP_CALL(hipEventSynchronize(stop));
    HIP_CALL(hipEventElapsedTime(&ms, start, stop));
    std::cout << "Elapsed time = " << ms << std::endl;

    assert(hipGetLastError() == hipSuccess);

    // printVec(a.getHost());

    return 0;
}