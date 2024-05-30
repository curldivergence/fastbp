#include "utils.h"

namespace debug
{
    void CheckCUDAErrors()
    {
#ifdef DEBUG_CUDA
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cerr << "CUDA error after sync: " << cudaGetErrorString(error) << std::endl;
            assert(false);
        }
#endif // DEBUG_CUDA
    }
} // namespace debug

namespace logging
{
    std::ostream& GetBriefLogStream()
    {
        return std::cout;
    }

    std::ostream& GetVerboseLogStream()
    {
        // return std::cout;
        static std::ofstream logFile{k_LogFileName};
        return logFile;
    }
} // namespace logging

std::ostream& operator<<(std::ostream& os, const Point& p)
{
    return os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
}

std::ostream& operator<<(std::ostream& os, const uint2& p)
{
    return os << "(" << p.x << ", " << p.y << ")";
}

std::ostream& operator<<(std::ostream& os, const uint3& p)
{
    return os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
}

std::ostream& operator<<(std::ostream& os, const ComplexIntensity& v)
{
    return os << "(" << v.real() << (v.imag() >= 0. ? "+" : "") << v.imag() << "i" << ")";
}

std::ostream& operator<<(std::ostream& os, const AABB& aabb)
{
    return os << "AABB(Min: " << aabb.Min << ", Max: " << aabb.Max << ")";
}

std::ostream& operator<<(std::ostream& os, const Settings& settings)
{
    os << "Settings(" << "UseComplexIntensity: " << (settings.UseComplexIntensity ? "true" : "false") << ", "
       << "UseFineVoxels: " << (settings.UseFineVoxels ? "true" : "false") << ", "
       << "VoxelEdge: " << settings.VoxelEdge << ", " << "CoarseToFineFactor: " << settings.CoarseToFineFactor << ", "
       << "BatchSize: " << settings.BatchSize << ", " << "SensorGridStride: " << settings.SensorGridStride << ", "
       << "IntensityEpsilon: " << settings.IntensityEpsilon << ", " << "LogInterval: " << settings.LogInterval << ", "
       << "TimeRangeStart: " << settings.TimeRangeStart << ", " << "TimeRangeEnd: " << settings.TimeRangeEnd << ")";
    return os;
}
