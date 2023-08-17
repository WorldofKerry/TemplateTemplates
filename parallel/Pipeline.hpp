constexpr auto getPipelineIndexes = []<size_t N, size_t StepCount>()
{
    std::vector<std::pair<int, int>> arr;
    for (int iCar = 0; iCar < N + StepCount; ++iCar)
    {
        for (int iFunc = std::max(0, iCar - static_cast<int>(N) + 1); iFunc < std::min(static_cast<int>(StepCount), iCar + 1); ++iFunc)
        {
            arr.emplace_back(iFunc, iCar - iFunc);
        }
    }
    return arr;
};

const std::array steps{
    AssembleFrame, InsertEngine, PaintAndInstallBody, InstallWheelsAndTires, ShipToCustomer};

for (const auto &index : getPipelineIndexes.template operator()<cars.size(), steps.size()>())
{
    steps[index.first](&cars[index.second]);
}