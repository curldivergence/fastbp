import ctypes
import os
from typing import Tuple

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]


class UInt3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]


class AABB(ctypes.Structure):
    _fields_ = [("Min", Point), ("Max", Point)]


# struct Settings
# {
#     bool UseComplexIntensity;  // Whether input is pre-filtered (and thus complex) or not (and thus real-valued)
#     bool UseFineVoxels;        // Whether to use two-stage rasterization (coarse->fine) or single-stage
#     float VoxelEdge;           // If two-stage rasterization is used, this is the edge of a fine voxel, otherwise - of the coarse one
#     size_t CoarseToFineFactor; // If two-stage rasterization is used, this is the ratio between edge lengths of coarse
#                                // and fine voxels
#     size_t BatchSize;          // How many ellipsoids to process at once
#     size_t SensorGridStride;   // Used for debugging to make computations faster
#     float IntensityEpsilon;    // Ellipsoids that would have added intensity lesser than this will be culled
#     size_t LogInterval;        // How often to log progress to stdout
#     size_t TimeRangeStart;     // Allows to override the time range and rasterize only part of it (useful for debugging)
#     size_t TimeRangeEnd;       // See `TimeRangeStart`
# };


class Settings(ctypes.Structure):
    _fields_ = [
        ("UseComplexIntensity", ctypes.c_bool),
        ("UseFineVoxels", ctypes.c_bool),
        ("VoxelEdge", ctypes.c_float),
        ("CoarseToFineFactor", ctypes.c_size_t),
        ("BatchSize", ctypes.c_size_t),
        ("SensorGridStride", ctypes.c_size_t),
        ("IntensityEpsilon", ctypes.c_float),
        ("LogInterval", ctypes.c_size_t),
        ("TimeRangeStart", ctypes.c_size_t),
        ("TimeRangeEnd", ctypes.c_size_t),
    ]


class LaserGrid(ctypes.Structure):
    _fields_ = [
        ("RowCount", ctypes.c_size_t),
        ("ColCount", ctypes.c_size_t),
        ("Data", ctypes.POINTER(Point)),
    ]

    def __init__(self, np_array):
        np_array = np.ascontiguousarray(np_array, dtype=np.float32)
        self.RowCount, self.ColCount, _ = np_array.shape
        self.Data = np_array.ctypes.data_as(ctypes.POINTER(Point))


class SensorGrid(ctypes.Structure):
    _fields_ = [
        ("RowCount", ctypes.c_size_t),
        ("ColCount", ctypes.c_size_t),
        ("Data", ctypes.POINTER(Point)),
    ]

    def __init__(self, np_array):
        np_array = np.ascontiguousarray(np_array, dtype=np.float32)
        self.RowCount, self.ColCount, _ = np_array.shape
        self.Data = np_array.ctypes.data_as(ctypes.POINTER(Point))


class ComplexFloat(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class Histogram(ctypes.Structure):
    _fields_ = [
        ("DeltaT", ctypes.c_float),
        ("TimeSampleCount", ctypes.c_size_t),
        ("LaserRowCount", ctypes.c_size_t),
        ("LaserColCount", ctypes.c_size_t),
        ("SensorRowCount", ctypes.c_size_t),
        ("SensorColCount", ctypes.c_size_t),
        ("Data", ctypes.c_void_p),
    ]

    def __init__(self, delta_t, H):
        assert (
            H.dtype.name == "float32" or H.dtype.name == "complex64"
        ), f"Intensity type {H.dtype.name} is not supported"

        self.DeltaT = delta_t

        H = np.ascontiguousarray(H)
        self.TimeSampleCount = H.shape[0]
        if len(H.shape) == 3:  # Single laser point
            self.LaserRowCount, self.LaserColCount = 1, 1
            self.SensorRowCount, self.SensorColCount = H.shape[1], H.shape[2]
            # Beware, we're passing a `void*` because there is no `c_complex` in ctypes
            self.Data = H.ctypes.data_as(ctypes.c_void_p)
        else:
            assert len(H.shape) == 5, "Unexpected H shape"

            self.LaserRowCount, self.LaserColCount = H.shape[1], H.shape[2]
            self.SensorRowCount, self.SensorColCount = H.shape[3], H.shape[4]
            # Same here, passing `c_void`
            self.Data = H.ctypes.data_as(ctypes.c_void_p)


# A non-owning view over an np.array
class ReconstructedVolume(ctypes.Structure):
    _fields_ = [
        ("EdgeVoxelCount", UInt3),
        ("AABB", AABB),
        # Same here, passing `c_void`
        ("Intensity", ctypes.c_void_p),
    ]

    def __init__(self, np_array, aabb):
        assert len(np_array.shape) == 3, "Volume array must be 3-dimensional"
        assert (
            np_array.dtype.name == "float32" or np_array.dtype.name == "complex64"
        ), f"Intensity type {np_array.dtype.name} is not supported"

        # Row major order: depth slice, row, col
        self.EdgeVoxelCount = UInt3(*np_array.shape)
        self.AABB = aabb
        np_array = np.ascontiguousarray(np_array)
        self.Intensity = np_array.ctypes.data_as(ctypes.c_void_p)


def plot_2d(image, filtered, caption=""):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    cax1 = ax[0].imshow(image, cmap="viridis")
    ax[0].set_title(f"{caption} (Original)")
    ax[0].axis("off")
    fig.colorbar(cax1, ax=ax[0], orientation="vertical")

    cax2 = ax[1].imshow(filtered, cmap="viridis")
    ax[1].set_title(f"{caption} (Laplacian Filtered)")
    ax[1].axis("off")
    fig.colorbar(cax2, ax=ax[1], orientation="vertical")

    plt.show()


def plot_3d(image, axis="all", slice_step=8):
    filtered = scipy.ndimage.laplace(image)
    image_shape = image.shape

    plot_2d(np.sum(image, axis=0), np.sum(filtered, axis=0), "Sum along X axis")
    plot_2d(np.sum(image, axis=1), np.sum(filtered, axis=1), "Sum along Y axis")
    plot_2d(np.sum(image, axis=2), np.sum(filtered, axis=2), "Sum along Z axis")

    if axis == "all" or "x" in axis:
        for x in range(0, image_shape[0], slice_step):
            image_slice = image[x, :, :]
            filtered = scipy.ndimage.laplace(image_slice)
            plot_2d(image_slice, filtered, f"Slice along X axis, x = {x}")

    if axis == "all" or "y" in axis:
        for y in range(0, image_shape[1], slice_step):
            image_slice = image[:, y, :]
            filtered = scipy.ndimage.laplace(image_slice)
            plot_2d(image_slice, filtered, f"Slice along Y axis, y = {y}")

    if axis == "all" or "z" in axis:
        for z in range(0, image_shape[2], slice_step):
            image_slice = image[:, :, z]
            filtered = scipy.ndimage.laplace(image_slice)
            plot_2d(image_slice, filtered, f"Slice along Z axis, z = {z}")


# Public API
def run_reconstruction(
    settings: Settings,
    laser_grid: LaserGrid,
    sensor_grid: SensorGrid,
    histogram: Histogram,
    volumeAABB: Tuple[np.ndarray, np.ndarray],
):
    # Extend AABB to the nearest voxel boundaries
    extentedAABB = (
        np.floor(volumeAABB[0] / settings.VoxelEdge) * settings.VoxelEdge,
        np.ceil(volumeAABB[1] / settings.VoxelEdge) * settings.VoxelEdge,
    )
    print(
        f"Volume AABB used for reconstruction: min {extentedAABB[0]}, max {extentedAABB[1]}"
    )

    # How many voxels has the volume along each axis
    volume_resolution = np.round(
        (extentedAABB[1] - extentedAABB[0]) / settings.VoxelEdge
    ).astype(int)
    print(f"Volume resolution (in voxels): {volume_resolution}")

    if settings.UseComplexIntensity:
        volume_array = np.zeros(volume_resolution, dtype=np.complex64)
    else:
        volume_array = np.zeros(volume_resolution, dtype=np.float32)

    volume_aabb = AABB(Point(*extentedAABB[0]), Point(*extentedAABB[1]))
    volume = ReconstructedVolume(volume_array, volume_aabb)

    _lib.run_reconstruction(
        ctypes.byref(settings),
        ctypes.byref(laser_grid),
        ctypes.byref(sensor_grid),
        ctypes.byref(histogram),
        ctypes.byref(volume),
    )

    return volume_array


_lib_path = os.path.join(os.path.dirname(__file__), "build", "librasterizer.so")
_lib = ctypes.CDLL(_lib_path)
_lib.run_reconstruction.argtypes = [
    ctypes.POINTER(Settings),
    ctypes.POINTER(LaserGrid),
    ctypes.POINTER(SensorGrid),
    ctypes.POINTER(Histogram),
    ctypes.POINTER(ReconstructedVolume),
]
_lib.run_reconstruction.restype = None
