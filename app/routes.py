from flask import Blueprint, render_template, request, jsonify, send_file
from .wms_handler import WMSImageFetcher
from .interpolator import FrameInterpolator, RIFEInterpolator
from datetime import datetime, timedelta
import os
import uuid
import numpy as np
from skimage import exposure
import torch
import concurrent.futures

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/generate-video', methods=['POST'])
def generate_video():
    """
    Generate interpolated video from satellite images.
    
    Request JSON:
    {
        "bbox": [minx, miny, maxx, maxy],
        "time_start": "YYYY-MM-DDThh:mm:ssZ",
        "time_end": "YYYY-MM-DDThh:mm:ssZ",
        "interval_minutes": 60
    }
    
    Returns:
        str: URL path to generated video
    """
    data = request.json
    
    # Initialize WMS fetcher
    wms_fetcher = WMSImageFetcher(
        wms_url='https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi',
        layer_name='MODIS_Terra_CorrectedReflectance_TrueColor'
    )
    
    # Get images
    images = wms_fetcher.get_image_sequence(
        bbox=data['bbox'],
        size=(800, 600),  # Adjust size as needed
        time_start=datetime.fromisoformat(data['time_start'].replace('Z', '+00:00')),
        time_end=datetime.fromisoformat(data['time_end'].replace('Z', '+00:00')),
        interval_minutes=60  # Adjust interval as needed
    )
    
    # Initialize interpolator
    interpolator = FrameInterpolator()
    
    # Generate interpolated frames
    interpolated_frames = interpolator.interpolate_sequence(images)
    
    # Create video
    output_path = os.path.join('app', 'static', 'videos', 'output.mp4')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    interpolator.create_video(interpolated_frames, output_path)
    
    return '/static/videos/output.mp4'

@main_bp.route('/generate-daily-video', methods=['POST'])
def generate_daily_video():
    """
    Generate interpolated video from all satellite images for a specific day.
    
    Request JSON:
    {
        "bbox": [minx, miny, maxx, maxy],
        "date": "YYYY-MM-DD",
        "fps": 10
    }
    
    Returns:
        str: URL path to generated video
    """
    data = request.json
    
    # Initialize WMS fetcher
    wms_fetcher = WMSImageFetcher(
        wms_url='https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi',
        layer_name='MODIS_Terra_CorrectedReflectance_TrueColor'
    )
    
    # Parse the date
    selected_date = data['date']  # Pass the date string directly
    
    # Generate a unique filename for the video
    date_str = selected_date.replace('-', '')
    video_filename = f"daily_video_{date_str}_{uuid.uuid4().hex[:8]}.mp4"
    video_path = os.path.join('app', 'static', 'videos', video_filename)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # Generate the daily video
    wms_fetcher.get_daily_video(
        bbox=data['bbox'],
        size=(800, 600),  # Adjust size as needed
        date=selected_date,
        output_path=video_path,
        fps=data.get('fps', 10)
    )
    
    # Return the URL to the video
    video_url = f"/static/videos/{video_filename}"
    return jsonify({"video_url": video_url})

@main_bp.route('/videos/<filename>')
def serve_video(filename):
    """Serve the generated video file"""
    return send_file(
        os.path.join('app', 'static', 'videos', filename),
        mimetype='video/mp4',
        as_attachment=False
    )

@main_bp.route('/generate-multi-day-video', methods=['POST'])
def generate_multi_day_video():
    """
    Generate interpolated video from satellite images across multiple days.
    
    Request JSON:
    {
        "bbox": [minx, miny, maxx, maxy],
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "fps": 10
    }
    
    Returns:
        str: URL path to generated video
    """
    data = request.json
    
    # Initialize WMS fetcher
    wms_fetcher = WMSImageFetcher(
        wms_url='https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi',
        layer_name='MODIS_Terra_CorrectedReflectance_TrueColor'
    )
    
    # Generate a unique filename for the video
    start_date_str = data['start_date'].replace('-', '')
    end_date_str = data['end_date'].replace('-', '')
    video_filename = f"multi_day_video_{start_date_str}_{end_date_str}_{uuid.uuid4().hex[:8]}.mp4"
    video_path = os.path.join('app', 'static', 'videos', video_filename)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # Generate the multi-day video
    wms_fetcher.get_multi_day_video(
        bbox=data['bbox'],
        size=(800, 600),  # Adjust size as needed
        start_date=data['start_date'],
        end_date=data['end_date'],
        output_path=video_path,
        fps=data.get('fps', 10)
    )
    
    # Return the URL to the video
    video_url = f"/static/videos/{video_filename}"
    return jsonify({"video_url": video_url})

@main_bp.route('/generate-rife-animation', methods=['POST'])
def generate_rife_animation():
    """Generate smooth animation using RIFE interpolation"""
    video_path = None
    try:
        data = request.json
        
        # Get the current view extent
        size = data.get('size', [800, 600])
        
        # Initialize services
        wms_fetcher = WMSImageFetcher(
            wms_url='https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi',
            layer_name='VIIRS_SNPP_CorrectedReflectance_TrueColor'
        )
        interpolator = RIFEInterpolator()
        
        # Get dates
        start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(data['end_date'], '%Y-%m-%d')
        days_difference = (end_date - start_date).days + 1
        
        # Generate unique filename
        video_filename = f"rife_animation_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}.avi"
        video_path = os.path.join('app', 'static', 'videos', video_filename)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # Fetch images
        images = wms_fetcher.get_image_sequence(
            bbox=data['bbox'],
            size=size,
            time_start=start_date,
            time_end=end_date,
            interval_minutes=1440
        )
        
        # Process images
        processed_images = []
        for img in images:
            if img is not None:
                try:
                    if img.dtype == np.float64 or img.dtype == np.float32:
                        img = (img * 255).astype(np.uint8)
                    processed_images.append(img)
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue
        
        if len(processed_images) < 2:
            return jsonify({"error": "Not enough valid images found"}), 400
        
        # Calculate frames for smooth video
        target_duration = max(10, days_difference * 2)  # 2 seconds per day, minimum 10 seconds
        target_fps = 60  # Higher FPS for smoother motion
        frames_between = 45  # Sweet spot for smooth interpolation
        
        print(f"Processing {len(processed_images)} images at resolution {size}")
        print(f"Generating {frames_between} frames between each pair")
        print(f"Target duration: {target_duration}s at {target_fps}fps")
        
        # Initialize frame collection
        all_frames = []
        all_frames.append(processed_images[0])  # Add first frame
        
        # Process consecutive pairs
        for i in range(len(processed_images) - 1):
            print(f"Interpolating between frames {i+1}/{len(processed_images)-1}")
            frames = interpolator.interpolate_frames(
                [processed_images[i], processed_images[i + 1]],
                n_frames=frames_between,
                progress_callback=progress_callback
            )
            if frames:
                all_frames.extend(frames[1:])  # Skip first frame as it's already included
            
            # Clear GPU memory after each pair
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if len(all_frames) < 2:
            return jsonify({"error": "Failed to generate enough frames"}), 400
        
        actual_duration = len(all_frames) / target_fps
        print(f"Generated {len(all_frames)} frames total")
        print(f"Actual video duration will be: {actual_duration:.1f} seconds")
        
        # Create video
        try:
            interpolator.create_video(
                all_frames,
                video_path,
                fps=target_fps,
                quality='high'
            )
            
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise Exception("Video creation failed")
            
            file_size = os.path.getsize(video_path)
            print(f"Video file size: {file_size / (1024*1024):.2f} MB")
            
            return jsonify({
                "video_url": f"/static/videos/{video_filename}",
                "duration": actual_duration,
                "frame_count": len(all_frames),
                "fps": target_fps,
                "file_size": file_size
            })
            
        except Exception as e:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            raise
            
    except Exception as e:
        error_msg = f"Animation generation failed: {str(e)}"
        print(error_msg)
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({"error": error_msg}), 500

def progress_callback(progress):
    """Handle progress updates during frame interpolation"""
    print(f"Interpolation progress: {progress:.2f}%") 