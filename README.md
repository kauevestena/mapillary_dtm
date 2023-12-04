# mapillary_dtm

Repository meant to try one idea: to generate a DTM using mapillary's metadata

1) non-street points might be filtered out
2) points shall be reduced to the terrain or by height computation
   A) or by the local vicinity of the SFM point cloud
   B) or by some sort of image technique
3) Robust regression such as 3D-line ransac shall be used to filter out outliers or as an approximate way of modelling

The main motivation is to enable 3D info for OSM roads (obtain a Z coordinate through DTM querying), this may enable, for example, projecting road stretches to terrestrial imagery.
