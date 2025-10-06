from config import *
import json, os
import osmnx as ox
from shapely.geometry import Point
import pandas as pd
from tqdm import tqdm
import tempfile, zlib, wget


def dump_json(data, filename,encoding='utf-8'):
    """
    Save data as a JSON file with pretty formatting.

    Parameters:
        data: The data to be serialized to JSON.
        filename (str): Path to the output JSON file.
        encoding (str, optional): Character encoding to use. Default is 'utf-8'.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4,ensure_ascii=False)

def read_json(filename,encoding='utf-8'):
    """
    Read and parse a JSON file.

    Parameters:
        filename (str): Path to the JSON file to read.
        encoding (str, optional): Character encoding to use. Default is 'utf-8'.

    Returns:
        The parsed JSON data (typically dict or list).
    """
    with open(filename, 'r') as f:
        return json.load(f)

def path_on_tests(filename):
    """
    Construct a path to a file within the tests directory.

    Parameters:
        filename (str): The name of the file in the tests directory.

    Returns:
        str: The full path to the file in the tests directory.
    """
    return os.path.join('tests', filename)

def get_coordinates_as_point(inputdict):
    """
    Extract coordinates from a dictionary and create a Shapely Point geometry.

    Parameters:
        inputdict (dict): A dictionary containing a 'coordinates' key with coordinate values.

    Returns:
        Point: A Shapely Point object created from the coordinates.
    """
    return Point(inputdict['coordinates'])

def get_streets(outpath=None):
    """
    Retrieve street network data for a specified place or bounding box using OSMnx.

    Uses the PLACE_NAME or BBOX from the config module to fetch street data.
    The network type is set to 'drive' (drivable streets).

    Parameters:
        outpath (str, optional): Path to save the GeoDataFrame to a file. If None, the data is not saved.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the street network edges.

    Raises:
        ValueError: If neither PLACE_NAME nor BBOX is specified in the config.
    """
    if PLACE_NAME:
        graph = ox.graph_from_place(PLACE_NAME, network_type='drive', simplify=False)
    elif BBOX:
        graph = ox.graph_from_bbox(*BBOX, network_type='drive', simplify=False)
    else:
        raise ValueError('Either PLACE_NAME or BBOX must be specified.')
    
    gdf = ox.graph_to_gdfs(graph)[1].reset_index()

    if outpath:
        gdf.to_file(outpath)
    
    return gdf


def resort_bbox(bbox):
    """
    Reorder bounding box coordinates from (minLon, minLat, maxLon, maxLat) to (minLat, minLon, maxLat, maxLon).

    Parameters:
        bbox (tuple or list): A bounding box with coordinates in the format (minLon, minLat, maxLon, maxLat).

    Returns:
        tuple: A tuple with reordered coordinates (minLat, minLon, maxLat, maxLon).
    """
    return bbox[1],bbox[0],bbox[3],bbox[2]


def check_type_by_first_valid(series):
    """
    Determine the type of the first non-None value in a series.

    Parameters:
        series: An iterable (e.g., pandas Series or list) of values.

    Returns:
        type or None: The type of the first non-None value found, or None if all values are None.
    """
    for value in series:
        if value is not None:
            return type(value)
    return None

def selected_columns_to_str(df,desired_type=list):
    """
    Convert DataFrame columns of a specific type to strings.

    Iterates through all columns in the DataFrame and converts those containing
    values of the desired type to string representation.

    Parameters:
        df (DataFrame): A pandas DataFrame to process.
        desired_type (type, optional): The type to look for in columns. Default is list.
    """
    for column in df.columns:
        c_type = check_type_by_first_valid(df[column])
        
        if c_type == desired_type:
            # print(column)
            df[column] = df[column].apply(lambda x: str(x))

def download_and_decompress(url, output_file=None):
    """
    Download a compressed file from a URL, decompress it, and optionally save or return as JSON.

    Uses wget to download the file, then decompresses it using zlib. The decompressed
    content can be saved to a file or returned as a parsed JSON object.

    Parameters:
        url (str): The URL of the compressed file to download.
        output_file (str, optional): Path to save the decompressed data. If None, returns parsed JSON.

    Returns:
        dict or None: If output_file is None, returns the parsed JSON data. Otherwise returns None.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, 'temp')

        wget.download(url, temp_path)
        with open(temp_path, 'rb') as f_in:
            # decompress
            as_str = zlib.decompress(f_in.read())

            if output_file:
                with open(output_file, 'wb') as f_out:
                    f_out.write(as_str)
            else:
                return json.loads(as_str)
