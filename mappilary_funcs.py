from lib import *

import numpy as np
from shapely.geometry import Point, box
from math import cos, pi
import sys
import os

# Add the my_mappilary_api directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'my_mappilary_api'))

# Import functions from the external library
from mapillary_api import (
    get_mapillary_token,
    get_mapillary_images_metadata as _get_mapillary_images_metadata,
    mapillary_data_to_gdf,
    radius_to_degrees,
    degrees_to_radius,
    get_bounding_box,
    download_mapillary_image,
    MAPPILARY_TOKEN
)

# Keep the old function for backward compatibility by wrapping the new one
# The old function signature was: (minLat, minLon, maxLat, maxLon, token, ...)
# The new function signature is: (minLon, minLat, maxLon, maxLat, token, ...)
def get_mapillary_images_metadata(minLat, minLon, maxLat, maxLon, token, outpath=None, params_dict=None):
    """
    Request images from Mapillary API given a bbox
    This is a wrapper for backward compatibility with the old parameter order.
    
    Parameters:
        minLat (float): The latitude of the first coordinate.
        minLon (float): The longitude of the first coordinate.
        maxLat (float): The latitude of the second coordinate.
        maxLon (float): The longitude of the second coordinate.
        token (str): The Mapillary API token.
        outpath (str, optional): Path to save the response JSON.
        params_dict (dict, optional): Custom parameters (not supported in the new API).

    Returns:
        dict: A dictionary containing the response from the API.
    """
    # Convert old parameter order (minLat, minLon, maxLat, maxLon) to new (minLon, minLat, maxLon, maxLat)
    return _get_mapillary_images_metadata(minLon, minLat, maxLon, maxLat, token=token, outpath=outpath)


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
