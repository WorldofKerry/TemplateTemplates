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
    T::value_type *device = nullptr;
    T container; 
    size_t bytes;
    size_t size;

public:
    SmartArray(T &&vec)
    {
        size = vec.size();
        bytes = size * sizeof(typename T::value_type);

        HIP_CALL(hipHostRegister(vec.data(), bytes, hipHostRegisterDefault));

        container = std::move(vec);
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
            HIP_CALL(hipMemcpy(device, container.data(), bytes, hipMemcpyHostToDevice));
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
        HIP_CALL(hipMemcpy(container.data(), device, bytes, hipMemcpyDeviceToHost));
        return container;
    }
};