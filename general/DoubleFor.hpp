template <template <size_t, size_t> class VecAddImpl, int batchSize, int blockSize, int batchSizeMin, int blockSizeMin, int batchSizeMax, int blockSizeMax>
void batchSizeBlockSizeBenchmark(const size_t N, const size_t benchmarkIterations, std::vector<std::tuple<int, int, int>> &bandWidths)
{
    if constexpr (batchSize <= batchSizeMax)
    {
        if constexpr (blockSize <= blockSizeMax)
        {
            VecAddImpl<batchSize, blockSize> impl;
            int bandwidth = benchmarkVecAdd(N, impl, benchmarkIterations);
            bandWidths.push_back({batchSize, blockSize, bandwidth});

            batchSizeBlockSizeBenchmark<VecAddImpl, batchSize, blockSize * 2, batchSizeMin, blockSizeMin, batchSizeMax, blockSizeMax>(N, benchmarkIterations, bandWidths);
        }
        else
        {
            batchSizeBlockSizeBenchmark<VecAddImpl, batchSize * 2, blockSizeMin, batchSizeMin, blockSizeMin, batchSizeMax, blockSizeMax>(N, benchmarkIterations, bandWidths);
        }
    }
}