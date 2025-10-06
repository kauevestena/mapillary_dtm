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

## 3. Projecting points into Terrain

(mapillary's imagery metadata are all representing points that are "floating above the surface", so, for having twrrain points they must be projected, and for that endeavor one might take advantage of the imagery metadata, image semantic segmentation, and image operations using photogrammetry or simplified single vision metrics or even monodepth. The points are not required to be projected orthogonally on the same XY coordinates)

*   **Idea A: Projection using local SfM point cloud vicinity.**
    *   Use the local vicinity of the Structure from Motion (SfM) point cloud to identify ground points. For each camera position, analyze nearby 3D points from the SfM reconstruction to estimate the ground surface below the camera.
    *   Apply statistical methods (e.g., lowest percentile of Z values, plane fitting to lowest points) to identify the terrain elevation in the camera's neighborhood.
    *   Project the camera position vertically downward to the estimated ground surface.
    *   **Libraries:** `numpy`, `scipy.spatial` for k-d tree nearest neighbor searches, `scikit-learn` for plane fitting.
*   **Idea B: Image-based semantic segmentation for ground detection.**
    *   Leverage Mapillary's semantic segmentation data to identify "road" and "ground" pixels in each image.
    *   Use the camera's intrinsic parameters (focal length, principal point) and pose (position, orientation) to ray-trace from ground pixels to 3D space.
    *   Estimate distance to ground using monocular depth estimation or simplified geometric assumptions (e.g., flat ground hypothesis, known camera height).
    *   Project rays from ground pixels to intersect with an estimated ground plane or surface.
    *   **Tools:** Mapillary's API for segmentation masks, `opencv-python` for image operations.
    *   **Libraries:** `torch` or `tensorflow` with pre-trained monocular depth networks (e.g., MiDaS, DPT), `pyproj` for coordinate transformations.
*   **Idea C: Monocular depth estimation for terrain projection.**
    *   Apply monocular depth estimation neural networks to Mapillary images to generate depth maps.
    *   Combine depth maps with semantic segmentation to focus on ground/road areas.
    *   Use camera pose and estimated depth values to back-project ground pixels into 3D world coordinates, creating terrain points.
    *   Filter depth estimates using confidence scores or consistency checks across multiple overlapping images.
    *   **Libraries:** Pre-trained models like MiDaS (`torch`), DPT, or ZoeDepth for depth estimation.
*   **Idea D: Simplified geometric projection based on camera height.**
    *   Assume a constant or smoothly varying camera height above the road surface (e.g., typical vehicle-mounted camera height of 1.5-2.5 meters).
    *   Project each camera position vertically downward by the estimated camera height to obtain terrain points.
    *   Optionally refine camera height estimates using local terrain slope information or by analyzing the bottom portion of images.
    *   This is a fast approximation that works well for relatively flat terrain.
    *   **Libraries:** Basic geometric operations with `numpy` and `shapely`.


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
