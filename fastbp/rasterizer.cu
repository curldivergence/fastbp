#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/merge.h>

#include "types.h"
#include "utils.h"

struct ComputeCoarseVoxelCount
{
    __device__ uint3 operator()(const AABB& aabb) const
    {
        float3 floatVoxelCount = ceilf((aabb.Max - aabb.Min) / VoxelEdge);

        // printf("floatVoxelCount: %f/%f/%f\n", floatVoxelCount.x, floatVoxelCount.y, floatVoxelCount.z);

        uint3 intVoxelCount{};
        intVoxelCount.x = static_cast<uint>(floatVoxelCount.x);
        intVoxelCount.y = static_cast<uint>(floatVoxelCount.y);
        intVoxelCount.z = static_cast<uint>(floatVoxelCount.z);

        return intVoxelCount;
    }

    float VoxelEdge{};
};

template<typename I>
struct ComputeEllipsoidAABB
{
    float VoxelEdge{};

    __device__ AABB operator()(const Ellipsoid<I>& ellipsoid) const
    {
        Point center = (ellipsoid.F1 + ellipsoid.F2) / 2.f;

        float focalDistance = length(ellipsoid.F1 - ellipsoid.F2) / 2.f;

        // We don't assert that ellipsoid is a prolate spheroid because it is expected that sometimes the parameters are
        // not compatible, i.e. the major axis length, which is defined by time sample, can be less than the distance
        // between foci which does not depend on time. We return a degenerate AABB in such a case.
        if (ellipsoid.SemiMajorAxisLength < focalDistance)
        {
            // printf("SemiMajorAxisLength: %f, focalDistance: %f, F1: %f/%f/%f, F2: %f/%f/%f\n",
            //     ellipsoid.SemiMajorAxisLength, focalDistance,
            //     ellipsoid.F1.x, ellipsoid.F1.y, ellipsoid.F1.z,
            //     ellipsoid.F2.x, ellipsoid.F2.y, ellipsoid.F2.z);
            return AABB{};
        }

        float semiMinorAxisLength = sqrt(pow(ellipsoid.SemiMajorAxisLength, 2.f) - pow(focalDistance, 2.f));

        // Important: this uses the fact that it is a prolate spheroid oriented along Z axis. For a more general case,
        // this calculation must be rewritten
        auto aabb = AABB{
            Point{center.x - semiMinorAxisLength,
                  center.y - semiMinorAxisLength,
                  center.z - ellipsoid.SemiMajorAxisLength},
            Point{center.x + semiMinorAxisLength,
                  center.y + semiMinorAxisLength,
                  center.z + ellipsoid.SemiMajorAxisLength}
        };

        // Extend AABB so that it consists of an integer number of voxels
        auto extendedAABB = AABB{RoundDown(aabb.Min, VoxelEdge), RoundUp(aabb.Max, VoxelEdge)};

        return extendedAABB;
    }
};

template<typename I>
thrust::device_vector<AABB> ComputeEllipsoidAABBs(const thrust::device_vector<Ellipsoid<I>>& ellipsoids,
                                                  float coarseVoxelEdge)
{
    nvtx3::scoped_range profilingRange{"ComputeEllipsoidAABBs"};

    thrust::device_vector<AABB> aabbs(ellipsoids.size());
    debug::CheckCUDAErrors();

    // Given a batch of ellipsoids, find their AABBs, extend those AABBs in-place to the integer coarse voxel borders,
    // and store them in the device buffer
    thrust::transform(ellipsoids.begin(), ellipsoids.end(), aabbs.begin(), ComputeEllipsoidAABB<I>{coarseVoxelEdge});
    debug::CheckCUDAErrors();
    logging::PrintDeviceVector("Ellipsoid bounds:", aabbs);

    return aabbs;
}

template<typename I>
__device__ bool IsPointInside(const Ellipsoid<I>& ellipsoid, Point point)
{
    float root_1 = sqrt((point.x - ellipsoid.F1.x) * (point.x - ellipsoid.F1.x)
                        + (point.y - ellipsoid.F1.y) * (point.y - ellipsoid.F1.y)
                        + (point.z - ellipsoid.F1.z) * (point.z - ellipsoid.F1.z));

    float root_2 = sqrt((point.x - ellipsoid.F2.x) * (point.x - ellipsoid.F2.x)
                        + (point.y - ellipsoid.F2.y) * (point.y - ellipsoid.F2.y)
                        + (point.z - ellipsoid.F2.z) * (point.z - ellipsoid.F2.z));

    float value = root_1 + root_2 - 2 * ellipsoid.SemiMajorAxisLength;

    return value < 0;
}

__device__ void GetVoxelVertices(Point voxelCenter, float voxelEdge, Point vertices[8])
{
    float halfEdge = voxelEdge / 2.0f;

    // Get the center coordinates
    float xCenter = voxelCenter.x;
    float yCenter = voxelCenter.y;
    float zCenter = voxelCenter.z;

    // Calculate the vertices
    vertices[0] = make_float3(xCenter - halfEdge, yCenter - halfEdge, zCenter - halfEdge);
    vertices[1] = make_float3(xCenter + halfEdge, yCenter - halfEdge, zCenter - halfEdge);
    vertices[2] = make_float3(xCenter - halfEdge, yCenter + halfEdge, zCenter - halfEdge);
    vertices[3] = make_float3(xCenter + halfEdge, yCenter + halfEdge, zCenter - halfEdge);
    vertices[4] = make_float3(xCenter - halfEdge, yCenter - halfEdge, zCenter + halfEdge);
    vertices[5] = make_float3(xCenter + halfEdge, yCenter - halfEdge, zCenter + halfEdge);
    vertices[6] = make_float3(xCenter - halfEdge, yCenter + halfEdge, zCenter + halfEdge);
    vertices[7] = make_float3(xCenter + halfEdge, yCenter + halfEdge, zCenter + halfEdge);
}

// Besides filtering border voxels, also culls voxels against global volume AABB
template<typename I>
struct CullVoxel
{
    using Voxel = Voxel<I>;
    using Ellipsoid = Ellipsoid<I>;

    const Voxel* InputVoxels;
    const Ellipsoid* Ellipsoids;
    float VoxelEdge;
    AABB VolumeAABB;

    __device__ bool operator()(uint voxelIndex)
    {
        const Voxel& voxel = InputVoxels[voxelIndex];

        if (!VolumeAABB.Contains(voxel.Center))
        {
            return false;
        }

        const Ellipsoid& ellipsoid = Ellipsoids[voxel.EllipsoidIndex];

        // ToDo: implement non-conservative rasterization
        Point voxelVertices[8];
        GetVoxelVertices(voxel.Center, VoxelEdge, voxelVertices);

        bool hasOutsideVertex = false;
        bool hasInsideVertex = false;

        for (uint vertexIdx = 0; vertexIdx < 8; vertexIdx++)
        {
            if (IsPointInside(ellipsoid, voxelVertices[vertexIdx]))
            {
                hasInsideVertex = true;
            }
            else
            {
                hasOutsideVertex = true;
            }
        }

        if (hasOutsideVertex && hasInsideVertex)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

template<typename I>
thrust::device_vector<Voxel<I>> CullVoxels(const thrust::device_vector<Ellipsoid<I>>& ellipsoids,
                                           thrust::device_vector<Voxel<I>> allVoxels,
                                           float voxelEdge,
                                           const AABB& volumeAABB)
{
    nvtx3::scoped_range profilingRange{"CullVoxels"};

    thrust::device_vector<Voxel<I>> borderVoxels(allVoxels.size());
    CullVoxel cullVoxel{thrust::raw_pointer_cast(allVoxels.data()),
                        thrust::raw_pointer_cast(ellipsoids.data()),
                        voxelEdge,
                        volumeAABB};

    thrust::device_vector<uint> indices(allVoxels.size());
    thrust::sequence(indices.begin(), indices.end());

    thrust::device_vector<uint> filteredIndices(indices.size());

    auto end = thrust::copy_if(indices.begin(), indices.end(), filteredIndices.begin(), cullVoxel);

    filteredIndices.resize(end - filteredIndices.begin());
    borderVoxels.resize(filteredIndices.size());

    // Gather the centers of the voxels that have survived
    thrust::gather(filteredIndices.begin(), filteredIndices.end(), allVoxels.begin(), borderVoxels.begin());
    debug::CheckCUDAErrors();
    logging::PrintDeviceVector("Border voxels:", borderVoxels);

    return borderVoxels;
}

template<typename I>
__global__ void CreateCoarseVoxelsKernel(const Ellipsoid<I>* ellipsoids,
                                         const AABB* aabbs,
                                         const uint3* coarseVoxelCounts,
                                         const uint* outputOffsets,
                                         uint ellipsoidCount,
                                         Voxel<I>* outputVoxels)
{
    using Voxel = Voxel<I>;
    using Ellipsoid = Ellipsoid<I>;

    uint ellipsoidIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (ellipsoidIndex >= ellipsoidCount)
    {
        return;
    }

    const Ellipsoid& ellipsoid = ellipsoids[ellipsoidIndex];
    const AABB& box = aabbs[ellipsoidIndex];

    // Determine the number of outputs for this AABB
    const uint3& voxelsExtent = coarseVoxelCounts[ellipsoidIndex];

    float3 voxelSize;
    voxelSize.x = (box.Max.x - box.Min.x) / voxelsExtent.x;
    voxelSize.y = (box.Max.y - box.Min.y) / voxelsExtent.y;
    voxelSize.z = (box.Max.z - box.Min.z) / voxelsExtent.z;

    // Find the starting position in the output buffer using the prefix sum results
    const uint outputOffset = outputOffsets[ellipsoidIndex];
    uint currentOutputIndex = 0;
    for (uint z = 0; z < voxelsExtent.z; z++)
    {
        for (uint y = 0; y < voxelsExtent.y; y++)
        {
            for (uint x = 0; x < voxelsExtent.x; x++)
            {
                Point voxelCenter = make_float3(box.Min.x + (x + 0.5f) * voxelSize.x,
                                                box.Min.y + (y + 0.5f) * voxelSize.y,
                                                box.Min.z + (z + 0.5f) * voxelSize.z);

                // printf("[ellipsoidIndex: %u] Output offset: %u, current output index: %u\n",
                //        ellipsoidIndex,
                //        outputOffset,
                //        currentOutputIndex);

                outputVoxels[outputOffset + currentOutputIndex] = Voxel{voxelCenter,
                                                                        ellipsoidIndex,
                                                                        ellipsoid.IntensityValue};
                currentOutputIndex++;
            }
        }
    }
}

template<typename I>
thrust::device_vector<Voxel<I>> CreateCoarseVoxels(const thrust::device_vector<Ellipsoid<I>>& ellipsoids,
                                                   const thrust::device_vector<AABB>& aabbs,
                                                   float coarseVoxelEdge)
{
    nvtx3::scoped_range profilingRange{"CreateCoarseVoxels"};

    const uint ellipsoidCount = aabbs.size();

    // Compute the number of coarse voxels for each AABB
    thrust::device_vector<uint3> coarseVoxelCounts{ellipsoidCount};
    thrust::transform(aabbs.begin(), aabbs.end(), coarseVoxelCounts.begin(), ComputeCoarseVoxelCount{coarseVoxelEdge});
    debug::CheckCUDAErrors();
    logging::PrintDeviceVector("Coarse voxel count:", coarseVoxelCounts);

    // Calculate total size for the output buffer
    auto computeVoxelCount = [] __host__ __device__(const uint3& extent) { return extent.x * extent.y * extent.z; };

    // A vector of offsets so that each voxel knows where to write its output in the output buffer
    thrust::device_vector<uint> outputOffsets(ellipsoidCount);
    thrust::exclusive_scan(thrust::make_transform_iterator(coarseVoxelCounts.begin(), computeVoxelCount),
                           thrust::make_transform_iterator(coarseVoxelCounts.end(), computeVoxelCount),
                           outputOffsets.begin());
    logging::PrintDeviceVector("Output offets:", outputOffsets);

    size_t totalVoxels = outputOffsets.back() + computeVoxelCount(coarseVoxelCounts.back());
    LOG_DEBUG("Total size of coarse voxel array: " << totalVoxels);

    thrust::device_vector<Voxel<I>> coarseVoxels(totalVoxels);

    uint blockSize = 256;
    uint numBlocks = (ellipsoidCount + blockSize - 1) / blockSize;

    LOG_DEBUG("Coarse voxels kernel parameters: " << "ellipsoidCount: " << ellipsoidCount
                                                  << ", blockSize: " << blockSize << ", numBlocks: " << numBlocks);

    CreateCoarseVoxelsKernel<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(ellipsoids.data()),
                                                       thrust::raw_pointer_cast(aabbs.data()),
                                                       thrust::raw_pointer_cast(coarseVoxelCounts.data()),
                                                       thrust::raw_pointer_cast(outputOffsets.data()),
                                                       ellipsoidCount,
                                                       thrust::raw_pointer_cast(coarseVoxels.data()));

    cudaDeviceSynchronize();
    debug::CheckCUDAErrors();
    logging::PrintDeviceVector("Coarse voxels:", coarseVoxels);

    return coarseVoxels;
}

template<typename I>
struct RefineVoxels
{
    float FineVoxelEdge;
    uint CoarseToFineFactor;
    uint FineVoxelsPerCoarse;
    const Voxel<I>* CoarseVoxels;
    Voxel<I>* FineVoxels;

    RefineVoxels(float fineVoxelEdge,
                 uint coarseToFineFactor,
                 uint fineVoxelsPerCoarse,
                 const Voxel<I>* coarseVoxels,
                 Voxel<I>* fineVoxels)
        : FineVoxelEdge(fineVoxelEdge)
        , CoarseToFineFactor(coarseToFineFactor)
        , FineVoxelsPerCoarse(fineVoxelsPerCoarse)
        , CoarseVoxels(coarseVoxels)
        , FineVoxels(fineVoxels)
    {}

    __device__ void operator()(uint coarseIdx)
    {
        const Voxel<I>& coarseVoxel = CoarseVoxels[coarseIdx];
        const float coarseVoxelEdge = FineVoxelEdge * CoarseToFineFactor;

        // At which offset start the fine voxels of the current coarse voxel
        const uint globalOffset = coarseIdx * FineVoxelsPerCoarse;
        uint currentOutputIndex = 0;

        const Point coarseVoxelMinPoint = Point{coarseVoxel.Center.x - coarseVoxelEdge / 2.f,
                                                coarseVoxel.Center.y - coarseVoxelEdge / 2.f,
                                                coarseVoxel.Center.z - coarseVoxelEdge / 2.f};

        for (uint z = 0; z < CoarseToFineFactor; z++)
        {
            for (uint y = 0; y < CoarseToFineFactor; y++)
            {
                for (uint x = 0; x < CoarseToFineFactor; x++)
                {
                    Point fineVoxelCenter = coarseVoxelMinPoint
                                            + float3{FineVoxelEdge * x + FineVoxelEdge / 2.f,
                                                     FineVoxelEdge * y + FineVoxelEdge / 2.f,
                                                     FineVoxelEdge * z + FineVoxelEdge / 2.f};

                    // printf("[ellipsoidIndex: %u] Output offset: %u, current output index: %u\n",
                    //        ellipsoidIndex,
                    //        outputOffset,
                    //        currentOutputIndex);

                    FineVoxels[globalOffset + currentOutputIndex] = Voxel{fineVoxelCenter,
                                                                          coarseVoxel.EllipsoidIndex,
                                                                          coarseVoxel.IntensityValue};
                    currentOutputIndex++;
                }
            }
        }
    }
};

template<typename I>
thrust::device_vector<Voxel<I>> CreateFineVoxels(const thrust::device_vector<Voxel<I>>& coarseVoxels,
                                                 float fineVoxelEdge,
                                                 uint32_t coarseToFineFactor)
{
    nvtx3::scoped_range profilingRange{"CreateFineVoxels"};

    const uint fineVoxelsPerCoarse = coarseToFineFactor * coarseToFineFactor * coarseToFineFactor;
    const uint fineVoxelCount = coarseVoxels.size() * fineVoxelsPerCoarse;

    thrust::device_vector<Voxel<I>> fineVoxels(fineVoxelCount);

    thrust::for_each(thrust::counting_iterator<uint>(0),
                     thrust::counting_iterator<uint>(coarseVoxels.size()),
                     RefineVoxels(fineVoxelEdge,
                                  coarseToFineFactor,
                                  fineVoxelsPerCoarse,
                                  thrust::raw_pointer_cast(coarseVoxels.data()),
                                  thrust::raw_pointer_cast(fineVoxels.data())));
    logging::PrintDeviceVector("Fine voxels:", fineVoxels);

    debug::CheckCUDAErrors();

    return fineVoxels;
}

// Important: as a precondition, expects both sets to be sorted
template<typename I>
SparseVoxelSet<I> AccumulateIntensities(thrust::device_vector<Voxel<I>> newVoxels)
{
    nvtx3::scoped_range profilingRange{"AccumulateIntensities"};

    logging::PrintDeviceVector("New voxels:", newVoxels);

    // Note: although it seems surprising, `__host__` is actually needed here to avoid "Attempt to use an extended
    // __device__ lambda in a context that requires querying its return type in host code. Use a named function object,
    // a __host__ __device__ lambda, or cuda::proclaim_return_type instead."
    auto getVoxelPosition = [] __host__ __device__(const Voxel<I>& voxel) { return voxel.Center; };
    auto getVoxelIntensity = [] __host__ __device__(const Voxel<I>& voxel) { return voxel.IntensityValue; };

    thrust::device_vector<Point> uniqueVoxels(newVoxels.size());
    thrust::device_vector<I> aggregatedIntensities(newVoxels.size());

    // Reduce the voxel list to eliminate duplicates, aggregating the intensities
    auto end = thrust::reduce_by_key(
        thrust::make_transform_iterator(newVoxels.begin(), getVoxelPosition),
        thrust::make_transform_iterator(newVoxels.end(), getVoxelPosition),
        thrust::make_transform_iterator(newVoxels.begin(), getVoxelIntensity),
        uniqueVoxels.begin(),
        aggregatedIntensities.begin(),
        // Binary predicate to compare for equality
        [] __device__(const Point& a, const Point& b) { return a == b; },
        // Binary op to use for reduction
        [] __device__(I a, I b) { return a + b; });

    debug::CheckCUDAErrors();

    // Resize the output vectors to the number of unique points
    uniqueVoxels.resize(thrust::distance(uniqueVoxels.begin(), thrust::get<0>(end)));
    aggregatedIntensities.resize(thrust::distance(aggregatedIntensities.begin(), thrust::get<1>(end)));

    LOG_DEBUG("Voxel count after reduce: "
              << uniqueVoxels.size() << ", reduce ratio: " << std::setprecision(3)
              << static_cast<float>(uniqueVoxels.size()) / static_cast<float>(newVoxels.size()));

    return {std::move(uniqueVoxels), std::move(aggregatedIntensities)};
}

template<typename I>
SparseVoxelSet<I> RasterizeEllipsoids(const thrust::host_vector<Ellipsoid<I>>& hostEllipsoids,
                                      const Settings* settings,
                                      const AABB& volumeAABB)
{
    nvtx3::scoped_range profilingRange{"RasterizeEllipsoids"};

    thrust::device_vector<Ellipsoid<I>> devEllipsoids = hostEllipsoids;

    const float coarseVoxelEdge = settings->UseFineVoxels ? (settings->VoxelEdge * settings->CoarseToFineFactor) :
                                                            settings->VoxelEdge;
    const auto aabbs = ComputeEllipsoidAABBs(devEllipsoids, coarseVoxelEdge);

    auto coarseVoxels = CreateCoarseVoxels(devEllipsoids, std::move(aabbs), coarseVoxelEdge);
    auto borderCoarseVoxels = CullVoxels(devEllipsoids, std::move(coarseVoxels), coarseVoxelEdge, volumeAABB);
    if (borderCoarseVoxels.empty())
    {
        return {};
    }

    if (settings->UseFineVoxels)
    {
        nvtx3::scoped_range profilingRange{"ProcessFineVoxels"};

        float fineVoxelEdge = coarseVoxelEdge / settings->CoarseToFineFactor;

        // We've filtered only border coarse voxels, let's refine them
        auto fineVoxels = CreateFineVoxels(borderCoarseVoxels, fineVoxelEdge, settings->CoarseToFineFactor);
        auto borderFineVoxels = CullVoxels(devEllipsoids, std::move(fineVoxels), fineVoxelEdge, volumeAABB);

        // LOG_DEBUG("Sorting " << borderFineVoxels.size() << " elements");

        // Sort by coordinates so that it's possible to reduce over `IntensityValue`s between neighbors
        thrust::sort(borderFineVoxels.begin(), borderFineVoxels.end());

        // assert(thrust::is_sorted(borderFineVoxels.begin(), borderFineVoxels.end()));

        return AccumulateIntensities(std::move(borderFineVoxels));
    }

    else
    {
        nvtx3::scoped_range profilingRange{"ProcessCoarseVoxels"};

        thrust::sort(borderCoarseVoxels.begin(), borderCoarseVoxels.end());

        return AccumulateIntensities(std::move(borderCoarseVoxels));
    }
}

// Sadly, this is needed to pass either one or two device pointers to the kernel while keeping a single code path (we
// can't just pass `const SparseVoxelSet&` due to CUDA limitations)
template<typename I>
struct IntensityGrid;

template<>
struct IntensityGrid<ComplexIntensity>
{
    IntensityGrid(VolumeBuffer<ComplexIntensity>& buffer)
        : Real{thrust::raw_pointer_cast(buffer.VolumeRe.data())}
        , Imag{thrust::raw_pointer_cast(buffer.VolumeIm.data())}
    {}

    __device__ void AtomicAdd(size_t index, ComplexIntensity value)
    {
        atomicAdd(&Real[index], value.real());
        atomicAdd(&Real[index], value.imag());
    }

    ScalarIntensity* Real;
    ScalarIntensity* Imag;
};

template<>
struct IntensityGrid<ScalarIntensity>
{
    IntensityGrid(VolumeBuffer<ScalarIntensity>& buffer)
        : Data{thrust::raw_pointer_cast(buffer.Volume.data())}
    {}

    __device__ void AtomicAdd(size_t index, ScalarIntensity value) { atomicAdd(&Data[index], value); }

    ScalarIntensity* Data;
};

template<typename I>
__global__ void UpdateVolumeKernel(const Point* points,
                                   const I* values,
                                   IntensityGrid<I> intensityGrid,
                                   AABB volumeAABB,
                                   uint3 edgeVoxelCount,
                                   size_t pointCount)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pointCount)
    {
        return;
    }

    const Point& point = points[idx];
    // Clip points outside of AABB requested by user
    if (!volumeAABB.Contains(point))
    {
        return;
    }

    I value = values[idx];

    float3 gridSize = {(volumeAABB.Max.x - volumeAABB.Min.x) / edgeVoxelCount.x,
                       (volumeAABB.Max.y - volumeAABB.Min.y) / edgeVoxelCount.y,
                       (volumeAABB.Max.z - volumeAABB.Min.z) / edgeVoxelCount.z};

    size_t voxelXIdx = std::min(std::max(uint((point.x - volumeAABB.Min.x) / gridSize.x), 0U), edgeVoxelCount.x - 1);
    size_t voxelYIdx = std::min(std::max(uint((point.y - volumeAABB.Min.y) / gridSize.y), 0U), edgeVoxelCount.y - 1);
    size_t voxelZIdx = std::min(std::max(uint((point.z - volumeAABB.Min.z) / gridSize.z), 0U), edgeVoxelCount.z - 1);

    size_t outputIndex = voxelXIdx * edgeVoxelCount.y * edgeVoxelCount.z + voxelYIdx * edgeVoxelCount.z + voxelZIdx;
    intensityGrid.AtomicAdd(outputIndex, value);
}

template<typename I>
void UpdateVolume(const SparseVoxelSet<I>& accumulatedVoxels,
                  const ReconstructedVolume<I>* volume,
                  VolumeBuffer<I>& devVolume)
{
    if (accumulatedVoxels.Centers.empty())
    {
        return;
    }

    nvtx3::scoped_range profilingRange{"UpdateVolume"};

    const auto& aabb = volume->AABB;

    // Launch kernel to accumulate values into the grid
    int numPoints = accumulatedVoxels.Centers.size();
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    LOG_DEBUG("Accumulation kernel parameters: " << "numPoints: " << numPoints << ", blockSize: " << blockSize
                                                 << ", numBlocks: " << numBlocks);

    UpdateVolumeKernel<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(accumulatedVoxels.Centers.data()),
                                                 thrust::raw_pointer_cast(accumulatedVoxels.Values.data()),
                                                 IntensityGrid<I>{devVolume},
                                                 aabb,
                                                 volume->EdgeVoxelCount,
                                                 numPoints);

    debug::CheckCUDAErrors();
}

void ReadbackVolume(VolumeBuffer<ComplexIntensity> devVolume, ReconstructedVolume<ComplexIntensity>* hostVolume)
{
    nvtx3::scoped_range profilingRange{"ReadbackVolume"};

    thrust::host_vector<ScalarIntensity> tempRe(devVolume.VolumeRe.size());
    thrust::host_vector<ScalarIntensity> tempIm(devVolume.VolumeIm.size());

    thrust::copy(devVolume.VolumeRe.begin(), devVolume.VolumeRe.end(), tempRe.begin());
    thrust::copy(devVolume.VolumeIm.begin(), devVolume.VolumeIm.end(), tempIm.begin());

    for (size_t idx = 0; idx < devVolume.VolumeRe.size(); idx++)
    {
        hostVolume->Values[idx] = std::complex<ScalarIntensity>(tempRe[idx], tempIm[idx]);
        // std::cout << hostVolume->Values[idx] << ", ";
    }
    // std::cout << std::endl;
}

void ReadbackVolume(VolumeBuffer<ScalarIntensity> devVolume, ReconstructedVolume<ScalarIntensity>* hostVolume)
{
    nvtx3::scoped_range profilingRange{"ReadbackVolume"};

    thrust::host_vector<ScalarIntensity> temp(devVolume.Volume.begin(), devVolume.Volume.end());
    std::memcpy(hostVolume->Values, temp.data(), sizeof(ScalarIntensity) * temp.size());
}

template<typename I>
void RunReconstructionImpl(const Settings* settings,
                           const LaserGrid* laserGrid,
                           const SensorGrid* sensorGrid,
                           // `void*` is used to avoid the hassle of exposing template instantiations to Python
                           const Histogram<I>* histogram,
                           ReconstructedVolume<I>* hostVolume)
{
    auto globalTimer = Timer{};
    globalTimer.Start();

    LOG("Requested AABB for reconstruced volume: " << hostVolume->AABB);

    LOG("Histogram paramerers: " << "time samples: " << histogram->TimeSampleCount
                                 << ", laser grid size (rows x cols): " << histogram->LaserRowCount << "x"
                                 << histogram->LaserColCount
                                 << ", sensor grid size (rows x cols): " << histogram->SensorRowCount << "x"
                                 << histogram->SensorColCount << ", deltaT: " << histogram->DeltaT);

    thrust::host_vector<Ellipsoid<I>> ellipsoidBatch{};
    VolumeBuffer<I> devVolume{hostVolume->EdgeVoxelCount.x * hostVolume->EdgeVoxelCount.y
                              * hostVolume->EdgeVoxelCount.z};

    size_t totalVoxelCount = 0;
    std::vector<I> intensities{};

    const size_t timeRangeStart = std::max(0UZ, settings->TimeRangeStart);
    const size_t timeRangeEnd = settings->TimeRangeEnd > timeRangeStart
                                        && settings->TimeRangeEnd < histogram->TimeSampleCount ?
                                    settings->TimeRangeEnd :
                                    histogram->TimeSampleCount;
    for (size_t timeSampleIdx = timeRangeStart; timeSampleIdx < timeRangeEnd; timeSampleIdx++)
    {
        size_t currentEllipsoidCount = 0;
        auto localTimer = Timer{};

        if (timeSampleIdx % settings->LogInterval == 0)
        {
            LOG("Processing time sample " << timeSampleIdx + 1 << "/" << histogram->TimeSampleCount);
            localTimer.Start();
        }

        float timeSample = timeSampleIdx * histogram->DeltaT;

        for (size_t lasRowIdx = 0; lasRowIdx < histogram->LaserRowCount; lasRowIdx++)
        {
            for (size_t lasColIdx = 0; lasColIdx < histogram->LaserColCount; lasColIdx++)
            {
                Point laserPos = (*laserGrid)(lasRowIdx, lasColIdx);
                LOG_DEBUG("Laser position: " << laserPos);

                for (size_t sensorRowIdx = 0; sensorRowIdx < histogram->SensorRowCount;
                     sensorRowIdx += settings->SensorGridStride)
                {
                    for (size_t sensorColIdx = 0; sensorColIdx < histogram->SensorColCount;
                         sensorColIdx += settings->SensorGridStride)
                    {
                        Point sensorPos = (*sensorGrid)(sensorRowIdx, sensorColIdx);
                        LOG_DEBUG("Sensor position (row " << sensorRowIdx << ", col " << sensorColIdx
                                                          << "): " << sensorPos);

                        I intensityValue = (*histogram)(timeSampleIdx,
                                                        lasRowIdx,
                                                        lasColIdx,
                                                        sensorRowIdx,
                                                        sensorColIdx);

                        if (cuda::std::abs(intensityValue) < settings->IntensityEpsilon)
                        {
                            // This ellipsoid would not have contributed (almost) anything, so there is no point in
                            // rasterizing it
                            continue;
                        }

                        LOG_DEBUG("Intensity value: " << intensityValue);
                        intensities.push_back(intensityValue);

                        // Constructor accepts focal points, semi-major axis length and intensity value
                        ellipsoidBatch.push_back(Ellipsoid{sensorPos, laserPos, timeSample / 2.f, intensityValue});
                        if (ellipsoidBatch.size() == settings->BatchSize)
                        {
                            logging::PrintHostVector("Ellipsoids:", ellipsoidBatch);
                            SparseVoxelSet newVoxels = RasterizeEllipsoids(ellipsoidBatch, settings, hostVolume->AABB);

                            UpdateVolume(newVoxels, hostVolume, devVolume);
                            totalVoxelCount += newVoxels.Centers.size();

                            currentEllipsoidCount += ellipsoidBatch.size();
                            ellipsoidBatch.clear();
                        }
                    }
                }
            }
        }

        if (timeSampleIdx % settings->LogInterval == 0 && currentEllipsoidCount > 0)
        {
            double iterationDuration = localTimer.Finish();
            LOG("Rasterized " << currentEllipsoidCount << " ellipsoids in " << iterationDuration << " seconds ("
                              << static_cast<double>(currentEllipsoidCount) / iterationDuration << " items/second)");
        }
    }

    ReadbackVolume(std::move(devVolume), hostVolume);

    double totalDuraion = globalTimer.Finish();
    LOG("Rasterized " << totalVoxelCount << " voxels in " << std::setprecision(3) << totalDuraion << " seconds");
}

// Public interface meant for using from Python
extern "C"
{
    void run_reconstruction(const Settings* settings,
                            const LaserGrid* laserGrid,
                            const SensorGrid* sensorGrid,
                            // `void*` is used to avoid the hassle of exposing template instantiations to Python
                            const void* histogram,
                            void* hostVolume)
    {
        LOG("Reconstruction settings: " << *settings);

        if (settings->UseComplexIntensity)
        {
            const auto* typedHistogram = static_cast<const Histogram<ComplexIntensity>*>(histogram);
            auto* typedHostVolume = static_cast<ReconstructedVolume<ComplexIntensity>*>(hostVolume);
            RunReconstructionImpl<ComplexIntensity>(settings, laserGrid, sensorGrid, typedHistogram, typedHostVolume);
        }
        else
        {
            const auto* typedHistogram = static_cast<const Histogram<ScalarIntensity>*>(histogram);
            auto* typedHostVolume = static_cast<ReconstructedVolume<ScalarIntensity>*>(hostVolume);
            RunReconstructionImpl<ScalarIntensity>(settings, laserGrid, sensorGrid, typedHistogram, typedHostVolume);
        }
    }
}