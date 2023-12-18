from lib import *

import requests
import os
import json
import numpy as np
from shapely.geometry import Point, box
from math import cos, pi
import urllib
import wget
from time import sleep
import geopandas as gpd


def get_mapillary_token():
    with open('mapillary_token', 'r') as f:
        return f.readline()

MAPPILARY_TOKEN = get_mapillary_token()


#function to define a random lat, lon in the bounding box:
def random_point_in_bbox(input_bbox):
    """
    Generate a random point within a given bounding box.

    Parameters:
        bbox (list): A list containing the coordinates of the bounding box in the format [min_lon, min_lat, max_lon, max_lat].

    Returns:
        tuple: A tuple containing the latitude and longitude of the randomly generated point.
    """
    min_lon, min_lat, max_lon, max_lat = input_bbox
    lat = min_lat + (max_lat - min_lat) * np.random.random()
    lon = min_lon + (max_lon - min_lon) * np.random.random()
    return lon, lat

def get_mapillary_images_metadata(minLat, minLon, maxLat, maxLon, token,outpath=None,params_dict=None):
    """
    Request images from Mapillary API given a bbox

    Parameters:
        minLat (float): The latitude of the first coordinate.
        minLon (float): The longitude of the first coordinate.
        maxLat (float): The latitude of the second coordinate.
        maxLon (float): The longitude of the second coordinate.
        token (str): The Mapillary API token.

    Returns:
        dict: A dictionary containing the response from the API.
    """
    url = "https://graph.mapillary.com/images"

    if not params_dict:
        params = {
        "bbox": f"{minLon},{minLat},{maxLon},{maxLat}",
        'limit': 5000,
        "access_token": token,
        "fields": ",".join([
            "altitude", 
            "atomic_scale", 
            "camera_parameters", 
            "camera_type", 
            "captured_at",
            "compass_angle", 
            "computed_altitude", 
            "computed_compass_angle", 
            "computed_geometry",
            "computed_rotation", 
            "creator", 
            "exif_orientation", 
            "geometry", 
            "height", 
            # "is_pano",
            "make", 
            "model", 
            # "thumb_256_url", 
            # "thumb_1024_url", 
            # "thumb_2048_url",
            "thumb_original_url", 
            # "merge_cc", 
            # "mesh", 
            "sequence", 
            "sfm_cluster", 
            "width",
            "detections",
        ])
    }
        
    else:
        params = params_dict
        
    response = requests.get(url, params=params)

    as_dict = response.json()

    if outpath:
        dump_json(as_dict, outpath)

    return as_dict



def radius_to_degrees(radius,lat):
    """
    Convert a radius in meters to degrees.  
    """
    return radius / (111320 * cos(lat * pi / 180))

def degrees_to_radius(degrees, lat):
    """
    Convert a radius in degrees to meters.  
    """
    return degrees * 111320 * cos(lat * pi / 180)

def get_bounding_box(lon, lat, radius):
    """
    Return a bounding box tuple as (minLon, minLat, maxLon, maxLat) from a pair of coordinates and a radius, using shapely.

    Parameters:
        lon (float): The longitude of the center of the bounding box.
        lat (float): The latitude of the center of the bounding box.
        radius (float): The radius of the bounding box in meters.

    Returns: 
        tuple: A tuple containing the minimum and maximum longitude and latitude of the bounding box.
    """


    # Convert radius from meters to degrees
    radius_deg = radius_to_degrees(radius, lat)

    point = Point(lon, lat)
    return box(point.x - radius_deg, point.y - radius_deg, point.x + radius_deg, point.y + radius_deg).bounds

# function to download an image from a url:
def download_mapillary_image(url, outfilepath,cooldown=1):
    try:
        wget.download(url, out=outfilepath)

        if cooldown:
            sleep(cooldown)
    except Exception as e:
        print('error:',e)

def mapillary_data_to_gdf(data,outpath=None,filtering_polygon=None):
    
    if isinstance(data,str):
        data = read_json(data)

    if data.get('data'):
        as_df = pd.DataFrame.from_records(data['data'])

        if 'geometry' in as_df.columns:

            as_df.geometry = as_df.geometry.apply(get_coordinates_as_point)

            as_gdf = gpd.GeoDataFrame(as_df,crs='EPSG:4326',geometry='geometry')

            selected_columns_to_str(as_gdf)

            if filtering_polygon:
                as_gdf = as_gdf[as_gdf.intersects(filtering_polygon)]

            if outpath:
                as_gdf.to_file(outpath)

            return as_gdf
        else:
            return gpd.GeoDataFrame()
    else:
        return gpd.GeoDataFrame()

# # # doesn't seems to be working, some kind of weird bug...
# # def filter_metadata_with_polygon(data, polygon,anti_rounding_factor=1000000):

# #     data_list = data['data']

# #     for item in data_list:

# #         point = Point(item['geometry']['coordinates'])


# #         if not polygon.contains(point):
# #             data_list.remove(item)
