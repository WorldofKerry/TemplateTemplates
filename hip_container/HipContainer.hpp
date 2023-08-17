#pragma once

#include <hip/hip_runtime_api.h>
#include <iostream>
#include <cstdlib>

#ifndef HIP_CALL
#define HIP_CALL(cmd)                                                                      \
    do                                                                                     \
    {                                                                                      \
        hipError_t error = (cmd);                                                          \
        if (error != hipSuccess)                                                           \
        {                                                                                  \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error)             \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (0)
#endif

template <typename T>
class SmartArray
{
    static_assert(std::is_arithmetic<T>::value, "");
    T *host = nullptr;
    T *device = nullptr;
    size_t bytes;
    size_t size;

public:
    SmartArray(std::vector<T> const &vec)
    {
        size = vec.size();
        bytes = size * sizeof(T);
        HIP_CALL(hipHostMalloc(&host, bytes));
        for (int i = 0; i < size; ++i)
        {
            host[i] = vec[i];
        }
    }
    ~SmartArray()
    {
        if (device)
        {
            HIP_CALL(hipFree(device));
        }
    }
    [[nodiscard]] T *toDevice()
    {
        if (!device)
        {
            HIP_CALL(hipMalloc(&device, bytes));
            HIP_CALL(hipMemcpy(device, host, bytes, hipMemcpyHostToDevice));
            return device;
        }
        assert(false);
    }
    [[nodiscard]] operator T *()
    {
        return toDevice();
    }

    [[nodiscard]] std::vector<T> getHost()
    {
        HIP_CALL(hipMemcpy(host, device, bytes, hipMemcpyDeviceToHost));
        std::vector<T> vec;
        vec.assign(host, host + bytes / sizeof(T));
        return vec;
    }
};