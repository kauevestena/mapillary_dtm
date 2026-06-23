import argparse
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from pyproj import Transformer
import pathlib
import glob

def create_count_raster(points_xy, out_path, minx, maxy, maxx, miny, resolution=0.5):
    # Determine grid size in meters (approximate locally using pyproj)
    # We will project the points to Web Mercator (EPSG:3857) to use meters
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    
    # Project bounds
    minx_m, maxy_m = transformer.transform(minx, maxy)
    maxx_m, miny_m = transformer.transform(maxx, miny)
    
    # Ensure min < max
    minx_m, maxx_m = min(minx_m, maxx_m), max(minx_m, maxx_m)
    miny_m, maxy_m = min(miny_m, maxy_m), max(miny_m, maxy_m)
    
    width = int(np.ceil((maxx_m - minx_m) / resolution))
    height = int(np.ceil((maxy_m - miny_m) / resolution))
    
    if width <= 0 or height <= 0:
        print(f"Invalid dimensions: width={width}, height={height}")
        return
        
    # Project all points
    pts_x = []
    pts_y = []
    for x, y in points_xy:
        px, py = transformer.transform(x, y)
        pts_x.append(px)
        pts_y.append(py)
        
    # Create 2D histogram
    # histogram2d uses (x, y) bins, but returns shape (nx, ny)
    # We want top-left origin, so y goes from maxy_m to miny_m
    x_edges = np.linspace(minx_m, maxx_m, width + 1)
    y_edges = np.linspace(maxy_m, miny_m, height + 1) # Descending
    
    # histogram2d expects coordinates
    # Note: numpy's histogram2d uses first argument for rows (y), second for cols (x) if we want standard image coordinates
    # Let's just use it properly: x is horizontal, y is vertical
    H, xedges, yedges = np.histogram2d(pts_x, pts_y, bins=[x_edges, np.flip(y_edges)])
    
    # H is shape (width, height) according to bins. We want (height, width) for raster
    img = H.T
    # We flipped y_edges for histogram, so we need to flip the result vertically to match raster (top-left origin)
    img = np.flipud(img).astype(np.float32)
    
    # Add nodata value for 0
    img[img == 0] = np.nan
    
    transform = from_origin(minx_m, maxy_m, resolution, resolution)
    
    with rasterio.open(
        out_path, 'w', driver='GTiff', height=img.shape[0], width=img.shape[1],
        count=1, dtype=img.dtype, crs=CRS.from_epsg(3857), transform=transform, nodata=np.nan
    ) as dst:
        dst.write(img, 1)
    
    print(f"Wrote {out_path} with {len(points_xy)} points (shape: {img.shape})")

def extract_coords(geojson_path):
    coords = []
    if not pathlib.Path(geojson_path).exists():
        return coords
    with open(geojson_path, 'r') as f:
        try:
            data = json.load(f)
            for feature in data.get('features', []):
                geom = feature.get('geometry', {})
                if geom.get('type') == 'Point':
                    coords.append(geom['coordinates'][:2])
        except Exception as e:
            print(f"Failed to read {geojson_path}: {e}")
    return coords

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-dir", default="qa/bbox_test")
    parser.add_argument("--bbox", default="-48.5968279,-27.5915750,-48.5897436,-27.5864953")
    args = parser.parse_args()
    
    minx, miny, maxx, maxy = map(float, args.bbox.split(","))
    qa_path = pathlib.Path(args.qa_dir)
    
    # 1. Total images from frames.geojson
    frames_path = qa_path / "metadata" / "frames.geojson"
    all_points = extract_coords(frames_path)
    if all_points:
        create_count_raster(
            all_points, 
            qa_path / "qa" / "image_count_all.tif", 
            minx, maxy, maxx, miny
        )
    else:
        print("No frames found in frames.geojson")
        
    # 2. Used images from camera_positions_*.geojson
    used_points = []
    for cam_file in qa_path.glob("metadata/camera_positions_*.geojson"):
        used_points.extend(extract_coords(cam_file))
        
    if used_points:
        # Deduplicate points based on coordinates (assuming same image = same coord in output? 
        # Wait, camera_positions might have slightly different coords. 
        # But image_id is not easily deduped here unless we read properties. Let's read properties)
        # Actually, let's just collect all and we can just use the union of image_ids
        pass
    
    # Let's re-read with image_ids to deduplicate
    used_image_ids = set()
    used_unique_points = []
    for cam_file in qa_path.glob("metadata/camera_positions_*.geojson"):
        with open(cam_file, 'r') as f:
            data = json.load(f)
            for feature in data.get('features', []):
                img_id = feature.get('properties', {}).get('image_id')
                if img_id and img_id not in used_image_ids:
                    used_image_ids.add(img_id)
                    used_unique_points.append(feature['geometry']['coordinates'][:2])
                    
    if used_unique_points:
        create_count_raster(
            used_unique_points, 
            qa_path / "qa" / "image_count_used.tif", 
            minx, maxy, maxx, miny
        )
    else:
        print("No used camera positions found")

if __name__ == "__main__":
    main()
