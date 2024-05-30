#pragma once

#include <assert.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <nvtx3/nvtx3.hpp>

#include "types.h"

namespace debug
{
    void CheckCUDAErrors();
}

namespace logging
{
// `LOG()` works always, `LOG_DEBUG` only in `DEBUG` config, otherwise is a noop
#ifdef DEBUG_CUDA

#define LOG_DEBUG(x) logging::GetVerboseLogStream() << x << std::endl
#define LOG(x)    \
    LOG_DEBUG(x); \
    logging::GetBriefLogStream() << x << std::endl

#else // DEBUG_CUDA

#define LOG_DEBUG(x)
#define LOG(x) logging::GetBriefLogStream() << x << std::endl

#endif // DEBUG_CUDA

    constexpr const char* k_LogFileName = "rasterizer.log";

    std::ostream& GetBriefLogStream();
    std::ostream& GetVerboseLogStream();

    template<typename T>
    void PrintHostVector(std::string_view message, const thrust::host_vector<T>& values, size_t maxValues = 10)
    {
#ifdef DEBUG_CUDA
        LOG_DEBUG(message);
        size_t elementsToPrintCount = std::min(values.size(), maxValues);

        LOG_DEBUG("Vector size: " << values.size() << ", printing first " << elementsToPrintCount << " elements");
        for (size_t idx = 0; idx < elementsToPrintCount; idx++)
        {
            LOG_DEBUG(values[idx]);
        }
#endif // DEBUG_CUDA
    }

    template<typename T>
    void PrintDeviceVector(std::string_view message, const thrust::device_vector<T>& values, size_t maxValues = 10)
    {
#ifdef DEBUG_CUDA
        thrust::host_vector<T> hostValues = values;
        debug::CheckCUDAErrors();
        PrintHostVector(message, hostValues, maxValues);
#endif // DEBUG_CUDA
    }

    template<typename I>
    inline void DumpIntensities(const std::vector<I>& intensities)
    {
        std::ofstream fout{"intensities.txt"};
        for (const I& value : intensities)
        {
            fout << value << ", ";
        }
    }
} // namespace logging

std::ostream& operator<<(std::ostream& os, const Point& p);
std::ostream& operator<<(std::ostream& os, const uint2& p);
std::ostream& operator<<(std::ostream& os, const uint3& p);
std::ostream& operator<<(std::ostream& os, const ComplexIntensity& v);
std::ostream& operator<<(std::ostream& os, const AABB& aabb);

template<typename Intensity>
inline std::ostream& operator<<(std::ostream& os, const Voxel<Intensity>& v)
{
    return os << "Voxel(Center: " << v.Center << ", EllipsoidIndex: " << v.EllipsoidIndex << ")";
}

template<typename Intensity>
inline std::ostream& operator<<(std::ostream& os, const Ellipsoid<Intensity>& ellipsoid)
{
    os << "Ellipsoid(F1: " << ellipsoid.F1 << ", F2: " << ellipsoid.F2
       << ", SemiMajorAxisLength: " << ellipsoid.SemiMajorAxisLength << ", IntensityValue: " << ellipsoid.IntensityValue
       << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Settings& settings);

// There is no `ceil` in `helper_math.h`, only `floor`
inline __host__ __device__ float3 ceilf(float3 v)
{
    float3 temp = floorf(-v);
    return -temp;
}

// Given two floats, rounds the first one down to the next integer multiple of
// the second one
inline __device__ float3 RoundDown(float3 a, float b)
{
    return floorf(a / b) * b;
}

inline __device__ float3 RoundUp(float3 a, float b)
{
    return ceilf(a / b) * b;
}

class Timer
{
public:
    void Start() { m_Start = std::chrono::high_resolution_clock::now(); }

    double Finish()
    {
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - m_Start).count();

        return duration;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
};
