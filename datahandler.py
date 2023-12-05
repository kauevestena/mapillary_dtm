from mappilary_funcs import *


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

    def mappilary_data_generator(self):
        for entry in self.buffered_wgs84.itertuples():
            bbox = resort_bbox(entry.geometry.bounds)
            data_dict = get_mapillary_images_metadata(*bbox,MAPPILARY_TOKEN)
            yield mapillary_data_to_gdf(data_dict,filtering_polygon=entry.geometry)
