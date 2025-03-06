import torch
import numpy as np
import cv2
import sys
import os
from skimage import exposure

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.RIFE.model.RIFE import Model
from PIL import Image
import io

class FrameInterpolator:
    def __init__(self):
        pass

    def interpolate_sequence(self, images, n_frames=7, progress_callback=None):
        """
        Interpolate between each pair of consecutive images
        
        Args:
            images (list): List of numpy arrays containing the images
            n_frames (int): Number of frames to generate between each pair
            progress_callback (function): Callback function to report progress
        
        Returns:
            list: List of interpolated frames
        """
        total_frames = (len(images) - 1) * (n_frames + 1)
        current_frame = 0
        
        interpolated_sequence = []
        
        for i in range(len(images) - 1):
            img1 = images[i]
            img2 = images[i + 1]
            
            interpolated_sequence.append(img1)
            
            for t in range(1, n_frames + 1):
                progress = t / (n_frames + 1)
                interpolated = cv2.addWeighted(img1, 1 - progress, img2, progress, 0)
                interpolated_sequence.append(interpolated)
            
            if progress_callback:
                progress = (current_frame / total_frames) * 100
                progress_callback(progress)
            current_frame += 1
        
        interpolated_sequence.append(images[-1])
        return interpolated_sequence

    def create_video(self, frames, output_path, fps=30):
        """
        Create video from frames
        
        Args:
            frames (list): List of numpy arrays containing the frames
            output_path (str): Path to save the video
            fps (int): Frames per second
        """
        if not frames:
            raise ValueError("No frames to create video from")
        
        height, width = frames[0].shape[:2]
        print(f"Creating video with dimensions: {width}x{height}, {len(frames)} frames")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Try different codecs in order of preference
            codecs = ['mp4v', 'avc1', 'XVID']
            out = None
            
            for codec in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if out.isOpened():
                        print(f"Successfully opened video writer with codec: {codec}")
                        break
                except Exception as e:
                    print(f"Failed to use codec {codec}: {str(e)}")
            
            if not out or not out.isOpened():
                raise Exception("Failed to open video writer with any codec")
            
            # Verify first frame
            first_frame = frames[0]
            if first_frame.max() <= 1.0:
                first_frame = (first_frame * 255).astype(np.uint8)
            
            # Debug frame info
            print(f"Frame shape: {first_frame.shape}, dtype: {first_frame.dtype}, range: [{first_frame.min()}, {first_frame.max()}]")
            
            for i, frame in enumerate(frames):
                # Ensure frame is in correct format
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Ensure frame is in BGR format for OpenCV
                if frame.shape[-1] == 3:  # If RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                success = out.write(frame)
                if not success:
                    print(f"Failed to write frame {i}")
            
            # Verify video was created
            if os.path.getsize(output_path) == 0:
                raise Exception("Output video file is empty")
            
        finally:
            if out:
                out.release()
            print(f"Video saved to {output_path}")

class RIFEInterpolator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model()
        self.model.eval()
        
        # Move model to device (GPU/CPU)
        if torch.cuda.is_available():
            self.model.flownet.to(self.device)  # Move the flownet to device
            # Enable DataParallel if multiple GPUs are available
            if torch.cuda.device_count() > 1:
                self.model.flownet = torch.nn.DataParallel(self.model.flownet)
        else:
            self.model.flownet.to(self.device)
        
        # Ensure model is in eval mode
        self.model.flownet.eval()
        
        print(f"Initialized RIFE model on device: {self.device}")
        if torch.cuda.is_available():
            print(f"Using {torch.cuda.device_count()} GPU(s)")

    def _preprocess_image(self, img):
        """Convert numpy array to torch tensor"""
        try:
            # Handle different color formats
            if img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.shape[2] != 3:  # Not RGB
                raise ValueError(f"Unexpected number of channels: {img.shape[2]}")
            
            # Convert to float and normalize
            img = img.astype(np.float32) / 255.0
            
            # Convert to torch tensor and move to device
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            img = img.unsqueeze(0).to(self.device)
            
            return img
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            raise

    def _postprocess_image(self, tensor):
        """Convert torch tensor back to numpy array"""
        # Move to CPU if needed
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        
        # Convert to numpy and transpose dimensions
        img = tensor.squeeze(0).permute(1, 2, 0).numpy()
        
        # Scale to [0, 255] and convert to uint8
        img = (img * 255).clip(0, 255).astype(np.uint8)
        
        return img

    def interpolate_frames(self, images, n_frames=7, progress_callback=None):
        """Optimized frame interpolation"""
        n_frames = int(n_frames)
        
        # Calculate optimal padding size
        h, w = images[0].shape[:2]
        pad_h = ((h - 1) // 32 + 1) * 32 - h
        pad_w = ((w - 1) // 32 + 1) * 32 - w
        
        processed_images = []
        for img in images:
            # Pad image instead of resizing
            img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            processed_images.append(img_padded)
        
        try:
            # Process images in larger batches
            img1 = self._preprocess_image(processed_images[0])
            img2 = self._preprocess_image(processed_images[1])
            
            # Generate timesteps
            timesteps = torch.linspace(0, 1, n_frames + 2, device=self.device)[1:-1]
            
            # Increase batch size for better GPU utilization
            batch_size = min(n_frames, 8)  # Process 8 frames at once
            interpolated_sequence = [processed_images[0]]
            
            # Pre-allocate CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.memory.empty_cache()
            
            for i in range(0, len(timesteps), batch_size):
                batch_timesteps = timesteps[i:i + batch_size]
                
                # Process batch of frames
                with torch.no_grad(), torch.cuda.amp.autocast():  # Use automatic mixed precision
                    batch_flows = []
                    batch_frames = []
                    
                    # Generate all flows first
                    for t in batch_timesteps:
                        flow_t_0, flow_t_1 = self.model.inference(img1, img2, timestep=float(t))
                        batch_flows.append((flow_t_0, flow_t_1))
                    
                    # Then generate all frames
                    for flow_t_0, flow_t_1 in batch_flows:
                        warped_img0 = self.model.warp(img1, flow_t_0)
                        warped_img1 = self.model.warp(img2, flow_t_1)
                        interpolated = (1 - t) * warped_img0 + t * warped_img1
                        batch_frames.append(interpolated)
                    
                    # Process batch results
                    for frame in batch_frames:
                        frame = self._postprocess_image(frame)
                        # Crop padding
                        frame = frame[:h, :w]
                        interpolated_sequence.append(frame)
                
                # Clear memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if progress_callback:
                    progress = (i + len(batch_timesteps)) / len(timesteps) * 100
                    progress_callback(progress)
            
            return interpolated_sequence
            
        except Exception as e:
            print(f"Error during frame interpolation: {str(e)}")
            raise

    def _ease_in_out(self, t):
        """Cubic easing function for smoother motion"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2

    def _generate_smooth_frame(self, img1, img2, timestep):
        """Generate intermediate frame with multiple passes for smoother interpolation"""
        try:
            # First pass - standard interpolation
            flow_t_0, flow_t_1 = self.model.inference(img1, img2, timestep=timestep)
            warped_img0 = self.model.warp(img1, flow_t_0)
            warped_img1 = self.model.warp(img2, flow_t_1)
            
            # Blend warped images with temporal weighting
            blend = self._ease_in_out(timestep)  # Use easing function for blending
            interpolated = (1 - blend) * warped_img0 + blend * warped_img1
            
            # Post-process frame
            frame = self._postprocess_image(interpolated)
            
            return frame
        except Exception as e:
            print(f"Error in frame generation: {str(e)}")
            raise

    def create_video(self, frames, output_path, fps=30, quality='high'):
        """Create high quality video from frames"""
        if not frames:
            raise ValueError("No frames to create video from")
        
        height, width = frames[0].shape[:2]
        print(f"Creating {width}x{height} video at {fps}fps with {len(frames)} frames")
        
        try:
            # Try different codecs in order of compatibility and quality
            codecs = [
                ('XVID', 'XVID'),  # Best balance of quality and compatibility
                ('MJPG', 'MJPG'),  # Very compatible
                ('mp4v', 'mp4v'),  # Most compatible
            ]
            
            temp_path = output_path.replace('.avi', '_temp.avi')
            success = False
            
            for codec_name, fourcc_name in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                    out = cv2.VideoWriter(
                        temp_path,
                        fourcc,
                        float(fps),  # Keep 30 FPS
                        (width, height),
                        isColor=True
                    )
                    
                    if out.isOpened():
                        print(f"Using codec: {codec_name}")
                        # Process frames in batches for better performance
                        batch_size = 30
                        for i in range(0, len(frames), batch_size):
                            batch = frames[i:i + batch_size]
                            for frame in batch:
                                if frame is None:
                                    continue
                                if frame.shape[2] == 3:
                                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                out.write(frame)
                        
                        out.release()
                        
                        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                            if os.path.exists(output_path):
                                os.remove(output_path)
                            os.rename(temp_path, output_path)
                            success = True
                            print(f"Successfully created {fps}fps video using {codec_name} codec")
                            print(f"Video file size: {os.path.getsize(output_path)} bytes")
                            break
                except Exception as e:
                    print(f"Error with {codec_name} codec: {str(e)}")
                    if 'out' in locals():
                        out.release()
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            if not success:
                raise Exception("Failed to create video with any codec")
            
        except Exception as e:
            print(f"Error creating video: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        finally:
            if 'out' in locals():
                out.release()
            cv2.destroyAllWindows()

    def interpolate_multi_pass(self, frames, passes=2, n_frames_per_pass=30):
        """Perform multiple passes of interpolation for ultra-smooth results"""
        print(f"Starting multi-pass interpolation with {passes} passes")
        
        current_frames = frames.copy()
        
        for pass_num in range(passes):
            print(f"Starting pass {pass_num + 1}/{passes}")
            interpolated_frames = []
            interpolated_frames.append(current_frames[0])  # Add first frame
            
            # Process each consecutive pair
            for i in range(len(current_frames) - 1):
                try:
                    # Generate intermediate frames
                    frames = self.interpolate_frames(
                        [current_frames[i], current_frames[i + 1]],
                        n_frames=n_frames_per_pass
                    )
                    if frames:
                        interpolated_frames.extend(frames[1:])  # Skip first frame as it's already included
                    
                    # Clear GPU memory after each pair
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error in pass {pass_num + 1}, pair {i}: {str(e)}")
                    continue
            
            # Update current_frames for next pass
            current_frames = interpolated_frames
            print(f"Pass {pass_num + 1} complete. Generated {len(current_frames)} frames")
            
            # Optional: Save intermediate results
            if pass_num < passes - 1:  # Don't need to clear on final pass
                torch.cuda.empty_cache()  # Clear GPU memory between passes
        
        return current_frames 