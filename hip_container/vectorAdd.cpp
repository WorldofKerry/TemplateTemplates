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

__global__ void vecAdd(size_t *a, const size_t *b)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
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
    for (size_t i = 0; i < v.size(); ++i)
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

    const size_t N = 1 << 10; // 28 for benchmark

    std::vector<size_t> vala(N);
    fillArray(vala);

    std::vector<size_t> valb(N);
    fillArray(valb);

    auto a = SmartArray(std::move(vala));
    auto b = SmartArray(std::move(valb));

    // Run Kernel
    const size_t blockSize = N;
    const size_t blocks = 1;

    HIP_CALL(hipEventRecord(start));

    // vecAdd<<<blocks, blockSize>>>(a.toDevice(), b.toDevice());
    vecAdd<<<blocks, blockSize>>>(a, b);

    HIP_CALL(hipEventRecord(stop));
    HIP_CALL(hipEventSynchronize(stop));
    HIP_CALL(hipEventElapsedTime(&ms, start, stop));
    std::cout << "Elapsed time = " << ms << std::endl;

    assert(hipGetLastError() == hipSuccess);

    auto container = a.getHost();
    size_t error = 0;
    for (size_t i = 0; i < container.size(); ++i) {
        error += container[i] - 2LL * i;
    }
    std::cout << "ERROR: " << error << "\n";

    // printVec(a.getHost());

    return 0;
}