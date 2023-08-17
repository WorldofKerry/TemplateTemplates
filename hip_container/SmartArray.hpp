template <typename T, size_t N>
class SmartArray
{
    static_assert(std::is_arithmetic<T>::value, "");
    std::array<T, N> host;
    size_t bytes;
    T *device = nullptr;

public:
    SmartArray(std::array<T, N> &&host) : bytes(N * sizeof(T))
    {
        this->host = std::move(host);
    }
    ~SmartArray()
    {
        if (device)
        {
            assert(hipFree(device) == hipSuccess);
        }
    }
    [[nodiscard]] T *toDevice()
    {
        if (device)
            return device;
        assert(hipMalloc(&device, bytes) == hipSuccess);
        assert(hipMemcpy(device, host.data(), bytes, hipMemcpyHostToDevice) == hipSuccess);
        return device;
    }
    [[nodiscard]] operator T *()
    {
        return this->toDevice();
    }

    [[nodiscard]] std::array<T, N> getHost()
    {
        assert(hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost) == hipSuccess);
        return host;
    }
};

// No C++17 type deduction
template <typename T, size_t N>
SmartArray<T, N> makeSmartArray(std::array<T, N> &&host)
{
    return SmartArray<T, N>(std::move(host));
}