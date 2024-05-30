import numpy as np
import tal

from fastbp import fastbp

FILE_PATH = "/home/andrew/Dev/T.hdf5"

data = tal.io.read_capture(FILE_PATH)
# data.downscale(4)

# Input parameters are grouped here

# Reconstruction volume params
volume_min_point = np.array([-1, -1, 0.9])
volume_max_point = np.array([1, 1, 1.1])

# Reconstruction settings

# Whether input is pre-filtered (and thus complex) or not (and thus real-valued)
use_complex_intensity = False
# Whether to use two-stage rasterization (coarse->fine) or single-stage
use_fine_voxels = True
# If two-stage rasterization is used, this is the ratio between edge lengths of coarse
# and fine voxels. Also note that voxels are always cubic, hence it's a scalar
voxel_edge = 0.05
# If two-stage rasterization How many ellipsoids to process at onceis used,
# this is the ratio between edge lengths of coarse and fine voxel
coarse_to_fine_factor = 2
# How many ellipsoids to process at once
batch_size = 50
# Used for debugging purposes to make computation faster
sensor_grid_stride = 1
# Ellipsoids that would have added intensity lesser than this will be culled (also useful for debugging)
intensity_epsilon = 10
# How often to log progress to stdout
log_interval = 5
# Allows to override the time range and rasterize only part of the data (useful for debugging)
# time_range_start = 0
time_range_start = 1000
# See `time_range_start`
# time_range_end = data.H.shape[0]
time_range_end = 1001

settings = fastbp.Settings(
    use_complex_intensity,
    use_fine_voxels,
    voxel_edge,
    coarse_to_fine_factor,
    batch_size,
    sensor_grid_stride,
    intensity_epsilon,
    log_interval,
    time_range_start,
    time_range_end,
)

# Rows, cols
laser_grid_size = (1, 1)
sensor_grid_size = data.sensor_grid_xyz.shape[:2]

laser_grid_array = np.array([[[0.0, 0.0, 0.0]]])
print(f"Laser grid ({laser_grid_array.shape})")
laser_grid = fastbp.LaserGrid(laser_grid_array)

sensor_grid_array = data.sensor_grid_xyz
print(f"Sensor grid ({sensor_grid_array.shape})")
sensor_grid = fastbp.SensorGrid(sensor_grid_array)

if use_complex_intensity:
    histogram_array = tal.reconstruct.filter_H(
        data, filter_name="pf", wl_mean=0.06, wl_sigma=0.06
    )
else:
    histogram_array = data.H

print(f"Histogram ({histogram_array.shape})")

histogram = fastbp.Histogram(data.delta_t, histogram_array)

volume = fastbp.run_reconstruction(
    settings, laser_grid, sensor_grid, histogram, (volume_min_point, volume_max_point)
)

print("Reconstructed volume:", volume.shape, volume[0][0][0])
# print(result_array)
