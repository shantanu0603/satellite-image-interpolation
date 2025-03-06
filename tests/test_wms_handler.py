from datetime import datetime, timedelta
import numpy as np
from app.config import Config
from app.wms_handler import WMSImageFetcher

def test_image_sequence_fetching():
    fetcher = WMSImageFetcher(Config.WMS_URL, Config.WMS_LAYER)
    images = fetcher.get_image_sequence(
        bbox=Config.DEFAULT_BBOX,
        size=Config.DEFAULT_SIZE,
        time_start=datetime.now() - timedelta(hours=24),
        time_end=datetime.now(),
        interval_minutes=60
    )
    assert len(images) > 0
    assert all(isinstance(img, np.ndarray) for img in images) 