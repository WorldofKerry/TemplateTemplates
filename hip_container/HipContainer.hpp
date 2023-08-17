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
    requires std::is_arithmetic_v<typename T::value_type>
class SmartArray
{
    T::value_type *host = nullptr;
    T::value_type *device = nullptr;
    size_t bytes;
    size_t size;

public:
    SmartArray(T &&vec)
    {
        size = vec.size();
        bytes = size * sizeof(typename T::value_type);
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
    [[nodiscard]] T::value_type *toDevice()
    {
        if (!device)
        {
            HIP_CALL(hipMalloc(&device, bytes));
            HIP_CALL(hipMemcpy(device, host, bytes, hipMemcpyHostToDevice));
            return device;
        }
        assert(false);
    }

    [[nodiscard]] operator typename T::value_type *()
    {
        return toDevice();
    }

    [[nodiscard]] T getHost()
    {
        HIP_CALL(hipMemcpy(host, device, bytes, hipMemcpyDeviceToHost));
        T vec;
        vec.assign(host, host + bytes / sizeof(typename T::value_type));
        return vec;
    }
};