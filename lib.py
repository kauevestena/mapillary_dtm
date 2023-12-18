from config import *
import json, os
import osmnx as ox
from shapely.geometry import Point
import pandas as pd
from tqdm import tqdm
import tempfile, zlib, wget


def dump_json(data, filename,encoding='utf-8'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4,ensure_ascii=False)

def read_json(filename,encoding='utf-8'):
    with open(filename, 'r') as f:
        return json.load(f)

def path_on_tests(filename):
    return os.path.join('tests', filename)

def get_coordinates_as_point(inputdict):

    return Point(inputdict['coordinates'])

def get_streets(outpath=None):
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
    return bbox[1],bbox[0],bbox[3],bbox[2]


def check_type_by_first_valid(series):
    for value in series:
        if value is not None:
            return type(value)
    return None

def selected_columns_to_str(df,desired_type=list):
    for column in df.columns:
        c_type = check_type_by_first_valid(df[column])
        
        if c_type == desired_type:
            # print(column)
            df[column] = df[column].apply(lambda x: str(x))

def download_and_decompress(url, output_file=None):
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
