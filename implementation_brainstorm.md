# Implementation Brainstorm for Mapillary DTM

This document outlines high-level ideas for generating a Digital Terrain Model (DTM) from Mapillary data. The focus is on using open-source tools and libraries.

## 1. Data Acquisition and Pre-processing

*   **Idea:** Use the Mapillary API to download image metadata and SfM (Structure from Motion) point cloud data for a specific geographic area.
*   **Tools:**
    *   `mapillary-tools` (official Python library) for interacting with the Mapillary API.
    *   `pyproj` or `proj4` for handling coordinate reference system transformations.

## 2. Filtering Non-Street Points

*   **Idea A: Use Mapillary's entity detection.**
    *   Mapillary's data includes semantic segmentation of images. We can filter for points that correspond to "road" or "ground" segments.
*   **Idea B: Geometric filtering based on camera height.**
    *   Assuming a relatively constant camera height above the road surface, we can filter out points that are significantly above or below the camera's estimated ground level.
*   **Idea C: Clustering-based filtering.**
    *   Use a clustering algorithm like DBSCAN on the 3D point cloud. The largest cluster is likely to be the ground plane.

## 3. Terrain Point Extraction

*   **Idea A: Cloth Simulation Filter (CSF).**
    *   This is a common algorithm for DTM generation from LiDAR data that can be adapted for SfM point clouds. It simulates a "cloth" draped over the inverted point cloud to separate ground from non-ground points.
    *   **Libraries:** `CSF.py` (Python implementation of the CSF algorithm).
*   **Idea B: Progressive Morphological Filter.**
    *   Another established algorithm for ground filtering in LiDAR data that uses morphological operations (erosion, dilation) with an increasing window size.
    *   **Libraries:** `PDAL` (Point Data Abstraction Library) has implementations of this.
*   **Idea C: Voxel-based filtering.**
    *   Divide the point cloud into a 3D grid (voxels). For each vertical column of voxels, select the lowest point as a candidate for the terrain.

## 4. Robust Regression for DTM Generation

*   **Idea A: 3D RANSAC for plane/line fitting.**
    *   As mentioned in the `README.md`, use RANSAC (Random Sample Consensus) to fit planes or lines to local neighborhoods of points. This will help identify and remove outliers (e.g., cars, vegetation) that were not caught in the initial filtering steps.
    *   **Libraries:** `scikit-learn` has a `RANSACRegressor`.
*   **Idea B: TIN (Triangulated Irregular Network) generation.**
    *   After filtering, create a TIN from the ground points. This is a vector-based representation of the DTM.
    *   **Libraries:** `scipy.spatial.Delaunay` for Delaunay triangulation.
*   **Idea C: Interpolation to a regular grid.**
    *   Interpolate the filtered ground points onto a regular grid to create a raster DTM.
    *   **Methods:** Inverse Distance Weighting (IDW), Kriging.
    *   **Libraries:** `gdal`, `scipy.interpolate`.

## 5. Integration with OpenStreetMap

*   **Idea:** Once the DTM is generated, create a process to query it. For a given OSM road (represented as a line), sample points along the line, query the DTM for the Z coordinate at each point, and add this information back to the OSM data.
*   **Tools:**
    *   `osmnx` or `overpy` to fetch OSM data.
    *   `rasterio` or `shapely` for querying the DTM.