#pragma once

#include <cuda/std/complex>

#include "helper_math.h"

using ScalarIntensity = float;
using ComplexIntensity = cuda::std::complex<ScalarIntensity>;

using Point = float3;

struct Settings
{
    bool UseComplexIntensity; // Whether input is pre-filtered (and thus complex) or not (and thus real-valued)
    bool UseFineVoxels;       // Whether to use two-stage rasterization (coarse->fine) or single-stage
    float VoxelEdge; // If two-stage rasterization is used, this is the edge of a fine voxel, otherwise - of the coarse
                     // one
    size_t CoarseToFineFactor; // If two-stage rasterization is used, this is the ratio between edge lengths of coarse
                               // and fine voxels
    size_t BatchSize;          // How many ellipsoids to process at once
    size_t SensorGridStride;   // Used for debugging to make computations faster
    float IntensityEpsilon;    // Ellipsoids that would have added intensity lesser than this will be culled
    size_t LogInterval;        // Log every N time samples
    size_t TimeRangeStart;     // Allows to override the time range and rasterize only part of it (useful for debugging)
    size_t TimeRangeEnd;       // Log every N time samples
};

inline __host__ __device__ bool operator==(const Point& lhs, const Point& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

inline __host__ __device__ bool operator<(const Point& lhs, const Point& rhs)
{
    if (lhs.x != rhs.x)
        return lhs.x < rhs.x;
    if (lhs.y != rhs.y)
        return lhs.y < rhs.y;
    return lhs.z < rhs.z;
}

template<typename I>
struct Ellipsoid
{
    Point F1;
    Point F2;
    float SemiMajorAxisLength;
    I IntensityValue;
};

struct AABB
{
    Point Min;
    Point Max;

    __device__ bool Contains(const Point& p) const
    {
        return (p.x >= Min.x && p.x <= Max.x && p.y >= Min.y && p.y <= Max.y && p.z >= Min.z && p.z <= Max.z);
    }
};

template<typename I>
struct Voxel
{
    Point Center;
    uint EllipsoidIndex;
    I IntensityValue;
};

template<typename Intensity>
inline __device__ bool operator<(const Voxel<Intensity>& lhs, const Voxel<Intensity>& rhs)
{
    return lhs.Center < rhs.Center;
}

template<typename Intensity>
inline __device__ bool operator==(const Voxel<Intensity>& lhs, const Voxel<Intensity>& rhs)
{
    return lhs.Center == rhs.Center;
}

template<typename I>
struct SparseVoxelSet
{
    thrust::device_vector<Point> Centers;
    thrust::device_vector<I> Values;
};

struct LaserGrid
{
    size_t RowCount;
    size_t ColCount;
    Point* Data;

    Point operator()(size_t row, size_t col) const
    {
        size_t index = row * ColCount + col;
        return Data[index];
    }
};

struct SensorGrid
{
    size_t RowCount;
    size_t ColCount;
    Point* Data;

    Point operator()(size_t row, size_t col) const
    {
        size_t index = row * ColCount + col;
        return Data[index];
    }
};

template<typename I>
struct Histogram
{
    float DeltaT;
    size_t TimeSampleCount;
    size_t LaserRowCount;
    size_t LaserColCount;
    size_t SensorRowCount;
    size_t SensorColCount;

    I* Data;

    I operator()(size_t timeIdx, size_t laserRow, size_t laserCol, size_t sensorRow, size_t sensorCol) const
    {
        size_t index = ((timeIdx * LaserRowCount * LaserColCount * SensorRowCount * SensorColCount)
                        + (laserRow * LaserColCount * SensorRowCount * SensorColCount)
                        + (laserCol * SensorRowCount * SensorColCount) + (sensorRow * SensorColCount) + sensorCol);
        return Data[index];
    }
};

// A CPU-stored volume that we receive from a client and fill
template<typename I>
struct ReconstructedVolume
{
    uint3 EdgeVoxelCount;
    AABB AABB;
    I* Values;
};

// A GPU-stored volume that we use as a temporary storage for accumulation to avoid frequent CPU-GPU copying
template<typename I>
struct VolumeBuffer;

template<>
struct VolumeBuffer<ComplexIntensity>
{
    thrust::device_vector<ScalarIntensity> VolumeRe;
    thrust::device_vector<ScalarIntensity> VolumeIm;

    VolumeBuffer(size_t voxelCount)
        : VolumeRe(voxelCount, 0.f)
        , VolumeIm(voxelCount, 0.f)
    {}

    size_t GetVoxelCount() const { return VolumeRe.size(); }
};

template<>
struct VolumeBuffer<ScalarIntensity>
{
    thrust::device_vector<ScalarIntensity> Volume;

    VolumeBuffer(size_t voxelCount)
        : Volume(voxelCount)
    {}

    size_t GetVoxelCount() const { return Volume.size(); }
};