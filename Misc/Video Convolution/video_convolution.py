"""
Video Convolution Script
Apply different convolution kernels to entire videos frame by frame.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import time


class VideoConvolution:
    """Class to handle video convolution operations."""
    
    # Predefined kernels
    KERNELS = {
        'identity': np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]]),
        
        'sharpen': np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]]),
        
        'box_blur': np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]]) / 9.0,
        
        'gaussian_blur': np.array([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]]) / 16.0,
        
        'edge_detect': np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]]),
        
        'sobel_x': np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]),
        
        'sobel_y': np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]]),
        
        'laplacian': np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]]),
        
        'emboss': np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]]),
        
        'outline': np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]]),
    }
    
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize the VideoConvolution object.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def apply_convolution(self, kernel: np.ndarray, normalize: bool = False,
                         progress_callback: Optional[callable] = None) -> bool:
        """
        Apply convolution kernel to entire video.
        
        Args:
            kernel: Convolution kernel as numpy array
            normalize: Whether to normalize the output to [0, 255] range
            progress_callback: Optional callback function for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        # Open input video
        cap = cv2.VideoCapture(str(self.input_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {self.input_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video {self.output_path}")
            cap.release()
            return False
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Apply convolution to each color channel
            convolved_frame = np.zeros_like(frame, dtype=np.float32)
            
            for i in range(3):  # BGR channels
                convolved_frame[:, :, i] = cv2.filter2D(
                    frame[:, :, i].astype(np.float32), -1, kernel
                )
            
            # Normalize if requested
            if normalize:
                convolved_frame = cv2.normalize(convolved_frame, None, 0, 255, cv2.NORM_MINMAX)
            else:
                convolved_frame = np.clip(convolved_frame, 0, 255)
            
            # Convert back to uint8
            convolved_frame = convolved_frame.astype(np.uint8)
            
            # Write frame
            out.write(convolved_frame)
            
            frame_count += 1
            
            # Progress update
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count, total_frames)
            elif frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count}/{total_frames} frames ({fps_processing:.2f} FPS)")
        
        # Release resources
        cap.release()
        out.release()
        
        elapsed = time.time() - start_time
        print(f"\nCompleted! Processed {frame_count} frames in {elapsed:.2f} seconds")
        print(f"Output saved to: {self.output_path}")
        
        return True
    
    def apply_predefined_kernel(self, kernel_name: str, normalize: bool = False) -> bool:
        """
        Apply a predefined kernel to the video.
        
        Args:
            kernel_name: Name of the predefined kernel
            normalize: Whether to normalize the output
            
        Returns:
            True if successful, False otherwise
        """
        if kernel_name not in self.KERNELS:
            print(f"Error: Unknown kernel '{kernel_name}'")
            print(f"Available kernels: {list(self.KERNELS.keys())}")
            return False
        
        kernel = self.KERNELS[kernel_name]
        print(f"Applying '{kernel_name}' kernel:")
        print(kernel)
        
        return self.apply_convolution(kernel, normalize)
    
    def invert_video(self, progress_callback: Optional[callable] = None) -> bool:
        """
        Invert all pixel values in the video.
        Pixels with value 0 become 255, pixels with value 255 become 0.
        Formula: new_value = 255 - old_value
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        # Open input video
        cap = cv2.VideoCapture(str(self.input_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {self.input_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        print("Inverting video: pixel_value -> 255 - pixel_value")
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video {self.output_path}")
            cap.release()
            return False
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Invert the frame: new_value = 255 - old_value
            inverted_frame = 255 - frame
            
            # Write frame
            out.write(inverted_frame)
            
            frame_count += 1
            
            # Progress update
            if progress_callback and frame_count % 10 == 0:
                progress_callback(frame_count, total_frames)
            elif frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count}/{total_frames} frames ({fps_processing:.2f} FPS)")
        
        # Release resources
        cap.release()
        out.release()
        
        elapsed = time.time() - start_time
        print(f"\nCompleted! Processed {frame_count} frames in {elapsed:.2f} seconds")
        print(f"Output saved to: {self.output_path}")
        
        return True
    
    @staticmethod
    def list_available_kernels():
        """Print all available predefined kernels."""
        print("Available kernels:")
        for name, kernel in VideoConvolution.KERNELS.items():
            print(f"\n{name}:")
            print(kernel)


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python video_convolution.py <input_video> <output_video> <kernel_name> [normalize]")
        print("\nAvailable kernels:")
        for name in VideoConvolution.KERNELS.keys():
            print(f"  - {name}")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    kernel_name = sys.argv[3]
    normalize = len(sys.argv) > 4 and sys.argv[4].lower() in ['true', '1', 'yes']
    
    # Create VideoConvolution object
    vc = VideoConvolution(input_video, output_video)
    
    # Apply convolution
    success = vc.apply_predefined_kernel(kernel_name, normalize)
    
    if success:
        print("Video processing completed successfully!")
    else:
        print("Video processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
