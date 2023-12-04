from config import *
import json, os
import osmnx as ox
from shapely.geometry import Point

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


class data_handler:
    def __init__(self,compute_proj_version=False,compute_buffers=False):
        self.data = get_streets()

        self.local_utm = self.data.estimate_utm_crs()

        if compute_proj_version:
            self.get_projected_version()

            if compute_buffers:
                self.compute_buffers()


    def __iter__(self):
        for row in self.data.itertuples():
            yield row

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]
    
    def __repr__(self):
        return self.data.__repr__()
    
    def get_projected_version(self):
        self.proj_version = self.data.to_crs(self.local_utm)

    def compute_buffers(self):
        self.buffered = self.proj_version.copy()
        self.buffered.geometry = self.proj_version.geometry.buffer(BUFFER_LEN)

        self.buffered_wgs84 = self.buffered.to_crs('EPSG:4326')

def format_bbox(input_bbox):
    min_lon, min_lat, max_lon, max_lat = input_bbox
    return min_lon, min_lat, max_lon, max_lat
