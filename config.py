# specify a geocodable place, the recommended is a neighborhood or small town:
PLACE_NAME      = 'Agua Verde, Curitiba'

# you can also specify a bounding box as 
# tuple/list (minLon, minLat, maxLon, maxLat):
# then also put a None in the PLACE_NAME
BBOX            = ()

# but always specify a place shortname, this will be used to name output files:
PLACE_SHORTNAME = 'agua_verde'

# specify the output folder:
OUTFOLDERPATH   = '../data/mapillary_dtm/'

# the search buffer length in meters:
BUFFER_LEN = 10