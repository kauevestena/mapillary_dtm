# Mapillary DTM Architecture

This document describes the high-level architecture and data flow of the `mapillary_dtm` pipeline. The pipeline focuses on maximizing accuracy via redundancy and cross-validation, using multiple independent reconstruction backends, and strictly requiring ground-only semantic evidence.

## Pipeline Diagram

```mermaid
graph TD
    %% Ingestion
    subgraph Data Ingestion & Preflight
        A[AOI Bounding Box] --> B(Mapillary API Client)
        B --> C{Sequence Scanner}
        C -->|Raw Sequences| D[Car-only Filter]
        D --> E[Filtered Sequences]
        E --> F[Imagery Fetch / Cache]
    end

    %% Semantics
    subgraph Semantics
        F --> G[Ground Masks Segmentation]
        F --> H[Curb / Lane Extraction]
    end

    %% Geometry
    subgraph Geometry Tracks
        F --> I(OpenSfM Reconstruction)
        F --> J(COLMAP / DIM Reconstruction)
        F --> K(Visual Odometry - OpenCV)
    end

    %% Scale & Height Solver
    subgraph Scale & Height
        E --> L[Semantic Anchors Detection]
        L --> M[Height & Scale Solver]
        I --> M
        J --> M
        K --> M
        M -->|Scale & Height| N(Metric Calibrated Trajectories)
    end

    %% Monodepth
    subgraph Monocular Depth
        F --> O[Midas Monodepth Prediction]
    end

    %% Ground Extraction
    subgraph Ground Extraction
        I --> P[Ground Point Extraction]
        J --> P
        K --> P
        G --> P
        N --> P
        O --> P
        P -->|Track A Ground Points| QA[Track A]
        P -->|Track B Ground Points| QB[Track B]
        P -->|Track C Ground Points| QC[Track C]
    end

    %% Consensus & TIN
    subgraph Consensus & Boundary Fill
        QA --> R{Consensus Agree}
        QB --> R
        QC --> R
        R -->|Voxel Agreement| S[Consensus Ground Points]
        
        H --> T[3D Breakline Projection]
        T --> U[Constrained TIN]
        S --> U
        
        A --> V[OSMnx Corridor Polygon]
        V --> W[Corridor Mask & TIN Sample]
        U --> W
        W --> X[Final 3D Points including Corridor Fill]
    end

    %% Fusion
    subgraph Fusion & Smoothing
        X --> Y[Heightmap Lower-Envelope Fusion]
        Y --> Z[Edge-Aware Smoothing]
        Z --> AA[0.5m DTM Raster]
        AA --> AB[Slope Rasters]
    end

    %% QA & Export
    subgraph QA & Output
        AA --> AC[Internal QA - Agreement Maps]
        AA --> AD[External QA vs Geotiff]
        AA --> AE(HTML Report & TIFF/LAZ Export)
    end
```

## Core Principles

1.  **Strict Production Run:** The pipeline strictly runs on actual models (e.g., PyTorch models for monodepth/segmentation, COLMAP, OpenSfM) and fails fast if the necessary infrastructure is unavailable. Mocks and synthetic fallbacks are strictly not used to guarantee data science integrity.
2.  **Redundancy everywhere**: Two independent SfM stacks (OpenSfM, COLMAP) plus VO, resolving consensus.
3.  **Metric scale**: Derived from constant camera height per sequence, GNSS distance consistency, and semantic footpoint anchors.
4.  **Ground-only focus**: 3D semantic voting from per-image ground masks; rejection of dynamic obstacles.
5.  **Slope fidelity**: Edge-aware smoothing and breakline enforcement preserving curbs/crowns.
