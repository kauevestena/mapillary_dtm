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
