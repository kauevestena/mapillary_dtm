from mappilary_funcs import *


class data_handler:
    """
    A handler for managing street network data and Mapillary image metadata.

    This class provides functionality to load street networks, perform coordinate
    transformations, compute buffers around streets, and retrieve Mapillary images
    within those buffered areas.

    Attributes:
        data (GeoDataFrame): The street network data loaded from OSMnx.
        local_utm (CRS): The local UTM coordinate reference system estimated from the data.
        proj_version (GeoDataFrame): A projected version of the data in the local UTM CRS.
        buffered (GeoDataFrame): The projected data with buffered geometries.
        buffered_wgs84 (GeoDataFrame): The buffered data transformed back to WGS84.
        gdf_list (list): A list of GeoDataFrames containing Mapillary data.
        gdf (GeoDataFrame): Concatenated GeoDataFrame of all Mapillary data.
    """
    def __init__(self,compute_proj_version=False,compute_buffers=False):
        """
        Initialize the data_handler with street network data.

        Parameters:
            compute_proj_version (bool, optional): If True, computes the projected version of the data.
                Default is False.
            compute_buffers (bool, optional): If True, computes buffered geometries (requires
                compute_proj_version to be True). Default is False.
        """
        self.data = get_streets()

        self.local_utm = self.data.estimate_utm_crs()

        if compute_proj_version:
            self.get_projected_version()

            if compute_buffers:
                self.compute_buffers()


    def __iter__(self):
        """
        Iterate over the rows of the street network data.

        Yields:
            namedtuple: Each row of the data as a named tuple from itertuples().
        """
        for row in self.data.itertuples():
            yield row

    def __len__(self):
        """
        Get the number of rows in the street network data.

        Returns:
            int: The number of street segments in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Access a specific row of the street network data by index.

        Parameters:
            index (int): The index of the row to retrieve.

        Returns:
            Series: The row at the specified index.
        """
        return self.data.iloc[index]
    
    def __repr__(self):
        """
        Return the string representation of the street network data.

        Returns:
            str: String representation of the underlying data GeoDataFrame.
        """
        return self.data.__repr__()
    
  
    def get_projected_version(self):
        """
        Create a projected version of the data in the local UTM coordinate system.

        Transforms the street network data from WGS84 to the estimated local UTM CRS
        and stores it in the proj_version attribute.
        """
        self.proj_version = self.data.to_crs(self.local_utm)

    def compute_buffers(self):
        """
        Compute buffered geometries around street segments.

        Creates buffer zones of BUFFER_LEN meters around each street segment in the
        projected coordinate system, then transforms the buffered geometries back to WGS84.
        Results are stored in buffered and buffered_wgs84 attributes.
        """
        self.buffered = self.proj_version.copy()
        self.buffered.geometry = self.proj_version.geometry.buffer(BUFFER_LEN)

        self.buffered_wgs84 = self.buffered.to_crs('EPSG:4326')

    def mappilary_data_generator(self):
        """
        Generate Mapillary image metadata for each buffered street segment.

        Iterates through buffered street segments and queries the Mapillary API for
        images within each buffer zone. Filters the results to only include images
        within the actual buffer polygon.

        Yields:
            GeoDataFrame: A GeoDataFrame containing Mapillary image metadata for each street segment.
        """
        for entry in self.buffered_wgs84.itertuples():
            bbox = resort_bbox(entry.geometry.bounds)
            data_dict = get_mapillary_images_metadata(*bbox,MAPPILARY_TOKEN)
            yield mapillary_data_to_gdf(data_dict,filtering_polygon=entry.geometry)


    def save_all_data(self,outpath):
        """
        Retrieve all Mapillary data and save it to a file.

        Collects Mapillary image metadata for all street segments using the generator,
        concatenates all results into a single GeoDataFrame, and saves it to the
        specified output path.

        Parameters:
            outpath (str): Path where the combined Mapillary data should be saved.
        """
        self.gdf_list = [data for data in tqdm(self.mappilary_data_generator(),total=len(self))]

        self.gdf = pd.concat(self.gdf_list)
        
        self.gdf.to_file(outpath)
