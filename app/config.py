class Config:
    WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    WMS_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"
    DEFAULT_BBOX = (-180, -90, 180, 90)
    DEFAULT_SIZE = (800, 600)
    INTERPOLATION_FRAMES = 7
    VIDEO_FPS = 30 