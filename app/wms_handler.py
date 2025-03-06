from owslib.wms import WebMapService
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import io
import logging
from functools import lru_cache
from skimage import exposure
import cv2

class WMSImageFetcher:
    def __init__(self, wms_url, layer_name):
        self.wms = WebMapService(wms_url)
        self.layer_name = layer_name
        # Standard bounds for the Earth in EPSG:4326
        self.max_bounds = (-180, -90, 180, 90)
        # Default interval in minutes between satellite images
        self.default_interval = 60  # 1 hour

    def get_image_sequence(self, bbox, size, time_start, time_end, interval_minutes):
        """
        Fetch a sequence of satellite images from WMS service
        
        Args:
            bbox (tuple): (minx, miny, maxx, maxy) in EPSG:4326
            size (tuple): (width, height)
            time_start (datetime): Start time
            time_end (datetime): End time
            interval_minutes (int): Time interval between images
        
        Returns:
            list: List of numpy arrays containing the images
        """
        try:
            # Ensure bbox is within valid range
            minx, miny, maxx, maxy = bbox
            minx = max(-180, min(180, minx))
            miny = max(-90, min(90, miny))
            maxx = max(-180, min(180, maxx))
            maxy = max(-90, min(90, maxy))
            
            # Ensure bbox has some width and height
            if maxx <= minx or maxy <= miny:
                raise ValueError("Invalid bbox dimensions")
            
            # Calculate aspect ratio of bbox
            bbox_aspect = abs((maxx - minx) / (maxy - miny))
            
            # Adjust size to match bbox aspect ratio
            width, height = size
            target_aspect = width / height
            
            if bbox_aspect > target_aspect:
                # bbox is wider than target
                new_height = int(width / bbox_aspect)
                size = (width, new_height)
            else:
                # bbox is taller than target
                new_width = int(height * bbox_aspect)
                size = (new_width, height)
            
            print(f"Adjusted image size to {size} for bbox aspect ratio {bbox_aspect:.2f}")
            
            # Get the images
            images = []
            current_time = time_start
            
            while current_time <= time_end:
                try:
                    img = self.wms.getmap(
                        layers=[self.layer_name],
                        srs='EPSG:4326',
                        bbox=(minx, miny, maxx, maxy),
                        size=size,
                        format='image/jpeg',
                        time=current_time.strftime('%Y-%m-%d'),
                        transparent=True
                    )
                    
                    # Convert to numpy array
                    img_array = np.array(Image.open(io.BytesIO(img.read())))
                    images.append(img_array)
                    
                except Exception as e:
                    print(f"Error fetching image for {current_time}: {str(e)}")
                
                current_time += timedelta(minutes=interval_minutes)
            
            return images
            
        except Exception as e:
            print(f"Error in get_image_sequence: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    def get_cached_image(self, time_str, bbox_str):
        # Implement caching for frequently requested images
        pass 

    def get_daily_video(self, bbox, size, date, output_path=None, fps=10):
        """
        Fetch all available satellite images for a given day and create an interpolated video
        
        Args:
            bbox (tuple): (minx, miny, maxx, maxy)
            size (tuple): (width, height)
            date (datetime.date or str): The date to fetch images for (YYYY-MM-DD)
            output_path (str, optional): Path to save the video file. If None, returns the video frames.
            fps (int): Frames per second for the output video
            
        Returns:
            str or list: Path to the saved video file or list of interpolated frames
        """
        try:
            # Ensure date is a datetime.date object
            if isinstance(date, str):
                date = datetime.strptime(date, '%Y-%m-%d').date()
                
            # Set time range for the entire day
            time_start = datetime.combine(date, datetime.min.time())
            time_end = datetime.combine(date, datetime.max.time())
            
            # Get the raw satellite images for the day
            raw_images = self.get_image_sequence(
                bbox=bbox,
                size=size,
                time_start=time_start,
                time_end=time_end,
                interval_minutes=self.default_interval
            )
            
            if len(raw_images) < 2:
                logging.warning(f"Not enough images found for date {date} to create a video")
                return raw_images
                
            # Interpolate frames between the raw images
            interpolated_frames = self._interpolate_frames(raw_images, fps)
            
            if output_path:
                return self._save_video(interpolated_frames, output_path, fps, size)
            else:
                return interpolated_frames
                
        except Exception as e:
            logging.error(f"Error creating daily video: {str(e)}")
            raise
            
    def _interpolate_frames(self, images, fps):
        """
        Interpolate frames between satellite images to create smooth transitions
        
        Args:
            images (list): List of numpy arrays containing the raw satellite images
            fps (int): Desired frames per second
            
        Returns:
            list: List of interpolated frames
        """
        # Calculate how many frames to generate between each pair of images
        # Assuming images are taken at self.default_interval minutes apart
        frames_between = int((self.default_interval * 60) / fps)
        
        interpolated_frames = []
        
        for i in range(len(images) - 1):
            start_frame = images[i]
            end_frame = images[i + 1]
            
            # Add the start frame
            interpolated_frames.append(start_frame)
            
            # Generate intermediate frames
            for j in range(1, frames_between):
                # Calculate the weight for linear interpolation
                alpha = j / frames_between
                
                # Linear interpolation between frames
                interpolated = cv2.addWeighted(
                    start_frame, 1 - alpha,
                    end_frame, alpha,
                    0
                )
                
                interpolated_frames.append(interpolated)
                
        # Add the last frame
        interpolated_frames.append(images[-1])
        
        return interpolated_frames
        
    def _save_video(self, frames, output_path, fps, size):
        """
        Save frames as a video file
        
        Args:
            frames (list): List of frames to save
            output_path (str): Path to save the video
            fps (int): Frames per second
            size (tuple): (width, height) of the video
            
        Returns:
            str: Path to the saved video file
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        for frame in frames:
            # Convert to BGR if the frame is in RGB format
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            
        out.release()
        logging.info(f"Video saved to {output_path}")
        return output_path 

    def get_multi_day_video(self, bbox, size, start_date, end_date, output_path=None, fps=10):
        """
        Fetch satellite images for a date range and create an interpolated video
        
        Args:
            bbox (tuple): (minx, miny, maxx, maxy)
            size (tuple): (width, height)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            output_path (str): Path to save the video file
            fps (int): Frames per second for the output video
        """
        try:
            # Convert dates to datetime objects
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Get all images for the date range
            current_dt = start_dt
            all_images = []
            
            while current_dt <= end_dt:
                # Get images for current day
                time_start = datetime.combine(current_dt.date(), datetime.min.time())
                time_end = datetime.combine(current_dt.date(), datetime.max.time())
                
                daily_images = self.get_image_sequence(
                    bbox=bbox,
                    size=size,
                    time_start=time_start,
                    time_end=time_end,
                    interval_minutes=self.default_interval
                )
                
                all_images.extend(daily_images)
                current_dt += timedelta(days=1)
                
            if len(all_images) < 2:
                logging.warning(f"Not enough images found between {start_date} and {end_date}")
                return None
                
            # Interpolate frames between all images
            interpolated_frames = self._interpolate_frames(all_images, fps)
            
            if output_path:
                return self._save_video(interpolated_frames, output_path, fps, size)
            else:
                return interpolated_frames
                
        except Exception as e:
            logging.error(f"Error creating multi-day video: {str(e)}")
            raise 