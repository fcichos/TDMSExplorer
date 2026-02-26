"""
TDMS Explorer Core Module

Core functionality for working with TDMS (Technical Data Management Streaming) files.
This module provides the main TDMSFileExplorer class and utility functions.
"""

import os
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

try:
    from nptdms import TdmsFile
except ImportError:
    print("Error: nptdms library not found. Please install it with:")
    print("pip install nptdms")
    raise

try:
    from tdms_explorer.image_analysis import ImageAnalyzer
except ImportError:
    print("Warning: Image analysis module not available. Some features will be disabled.")
    ImageAnalyzer = None

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


class TDMSFileExplorer:
    """
    Main class for exploring and working with TDMS files.
    
    Args:
        filename (str): Path to the TDMS file
    """
    
    def __init__(self, filename: str):
        """Initialize TDMS file explorer."""
        self.filename = filename
        self.tdms_file = TdmsFile.read(filename)
        self.properties = dict(self.tdms_file.properties)
        
        # Extract basic metadata
        self._extract_metadata()
        
        # Analyze file structure
        self._analyze_structure()
    
    def _extract_metadata(self):
        """Extract metadata from TDMS file properties."""
        self.metadata = {}
        
        # Common metadata fields
        metadata_fields = ['dimx', 'dimy', 'frames', 'description', 
                          'author', 'date', 'instrument', 'sampling_rate']
        
        for field in metadata_fields:
            if field in self.properties:
                self.metadata[field] = self.properties[field]
        
        # Add file size
        self.metadata['file_size'] = os.path.getsize(self.filename)
    
    def _analyze_structure(self):
        """Analyze the structure of the TDMS file."""
        self.groups = []
        self.channels = {}
        
        for group in self.tdms_file.groups():
            group_name = group.name
            self.groups.append(group_name)
            
            channel_list = []
            for channel in group.channels():
                channel_name = channel.name
                channel_list.append(channel_name)
                
                # Store channel info
                if group_name not in self.channels:
                    self.channels[group_name] = {}
                
                self.channels[group_name][channel_name] = {
                    'data_type': str(channel.data.dtype),
                    'shape': channel.data.shape,
                    'size': channel.data.size
                }
            
            self.channels[group_name]['_channels'] = channel_list
    
    def list_contents(self) -> Dict:
        """
        List the contents of the TDMS file.
        
        Returns:
            Dictionary containing file structure information
        """
        return {
            'filename': self.filename,
            'metadata': self.metadata,
            'groups': self.groups,
            'channels': self.channels
        }
    
    def print_contents(self):
        """Print the contents of the TDMS file in a readable format."""
        print(f"TDMS File: {self.filename}")
        print(f"File Size: {self.metadata.get('file_size', 'Unknown')} bytes")
        print("\nMetadata:")
        for key, value in self.metadata.items():
            if key != 'file_size':
                print(f"  {key}: {value}")
        
        print(f"\nGroups ({len(self.groups)}):")
        for group_name in self.groups:
            print(f"  - {group_name}")
            if group_name in self.channels:
                channels = self.channels[group_name].get('_channels', [])
                print(f"    Channels ({len(channels)}):")
                for channel_name in channels:
                    if channel_name in self.channels[group_name]:
                        info = self.channels[group_name][channel_name]
                        print(f"      - {channel_name}: {info['data_type']}, shape={info['shape']}, size={info['size']}")
    
    def has_image_data(self) -> bool:
        """
        Check if the TDMS file contains image data.
        
        Returns:
            True if image data is found, False otherwise
        """
        # Look for common image group/channel names
        image_indicators = ['image', 'Image', 'frame', 'Frame', 'video', 'Video']
        
        for group_name in self.groups:
            if any(indicator in group_name.lower() for indicator in image_indicators):
                channels = self.channels[group_name].get('_channels', [])
                for channel_name in channels:
                    if any(indicator in channel_name.lower() for indicator in image_indicators):
                        return True
        
        return False
    
    def extract_images(self) -> Optional[np.ndarray]:
        """
        Extract image data from TDMS file.
        
        Returns:
            Numpy array containing images, or None if no image data found
        """
        if not self.has_image_data():
            print("No image data found in TDMS file.")
            return None
        
        # Try to find image data
        for group_name in self.groups:
            if 'image' in group_name.lower():
                group = self.tdms_file[group_name]
                for channel_name in self.channels[group_name].get('_channels', []):
                    if 'image' in channel_name.lower():
                        channel = group[channel_name]
                        
                        # Try to determine dimensions
                        dimx = int(self.metadata.get('dimx', 0))
                        dimy = int(self.metadata.get('dimy', 0))
                        
                        if dimx > 0 and dimy > 0:
                            total_size = channel.data.size
                            frames = total_size // (dimx * dimy)
                            if frames > 0:
                                return channel.data.reshape([frames, dimx, dimy])
                        else:
                            # Try to guess dimensions from common patterns
                            return self._guess_image_dimensions(channel.data)
        
        return None
    
    def _guess_image_dimensions(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Try to guess image dimensions from raw data.
        
        Args:
            data: Raw data array
            
        Returns:
            Reshaped image array or None if guessing fails
        """
        total_size = data.size
        
        # Common image dimensions to try
        common_dims = [
            (512, 512), (256, 256), (128, 128), (64, 64),
            (1024, 1024), (768, 768), (640, 480), (320, 240),
            (1280, 720), (1920, 1080)
        ]
        
        for dimy, dimx in common_dims:
            if total_size % (dimx * dimy) == 0:
                frames = total_size // (dimx * dimy)
                print(f"Guessed dimensions: {frames} frames of {dimx}x{dimy}")
                return data.reshape([frames, dimx, dimy])
        
        print("Could not guess image dimensions.")
        return None
    
    def display_image(self, image_num: int = 0, cmap: str = 'gray'):
        """
        Display a single image from the TDMS file.
        
        Args:
            image_num: Image number/index to display
            cmap: Matplotlib colormap to use
        """
        images = self.extract_images()
        if images is None:
            return
        
        if image_num >= images.shape[0]:
            print(f"Image number {image_num} is out of range. Max: {images.shape[0]-1}")
            return
        
        plt.figure(figsize=(10, 8))
        plt.imshow(images[image_num, :, :], cmap=cmap)
        plt.title(f"Image {image_num} from {self.filename}")
        plt.colorbar()
        plt.show()
    
    def display_animation(self, start_frame: int = 0, end_frame: Optional[int] = None,
                         fps: int = 10, cmap: str = 'gray'):
        """
        Display animation of images from TDMS file.
        
        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all frames)
            fps: Frames per second for animation
            cmap: Matplotlib colormap to use
        """
        images = self.extract_images()
        if images is None:
            return
        
        if end_frame is None:
            end_frame = images.shape[0] - 1
        
        if start_frame < 0 or end_frame >= images.shape[0] or start_frame > end_frame:
            print(f"Invalid frame range. Available: 0 to {images.shape[0]-1}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        img = ax.imshow(images[start_frame, :, :], cmap=cmap)
        plt.title(f"Animation from {self.filename}")
        plt.colorbar(img)
        
        def update(frame):
            img.set_array(images[frame, :, :])
            ax.set_title(f"Frame {frame}")
            return img,
        
        ani = FuncAnimation(fig, update, frames=range(start_frame, end_frame + 1),
                          interval=1000/fps, blit=True)
        plt.show()
        return ani
    
    def write_image(self, image_num: int, output_path: str,
                    dtype: Optional[np.dtype] = None, force: bool = False,
                    normed: bool = True) -> bool:
        images = self.extract_images()
        if images is None:
            return False
        if image_num < 0 or image_num >= images.shape[0]:
            raise IndexError(f"Image {image_num} out of range (0..{images.shape[0]-1})")
        out = Path(output_path)
        if not force and out.exists():
            return False
        if out.parent.as_posix() != '.':
            out.parent.mkdir(parents=True, exist_ok=True)
        target = dtype if dtype is not None else images.dtype
        converted = _convert_dtype(images[image_num], target, normed)
        if target == np.uint8:
            img = Image.fromarray(converted, mode='L')
        elif target == np.uint16:
            img = Image.fromarray(converted, mode='I;16')
        else:
            img = Image.fromarray(converted)
        img.save(out)
        return True

    def write_images(self, output_dir: str, base_name: Optional[str] = None,
                     start_frame: int = 0, end_frame: Optional[int] = None,
                     dtype: Optional[np.dtype] = None, force: bool = False,
                     normed: bool = True) -> int:
        images = self.extract_images()
        if images is None:
            return 0
        if end_frame is None:
            end_frame = images.shape[0]
        subset = images[start_frame:end_frame]
        name = base_name or Path(self.filename).stem.replace('_video', '')
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        target = dtype if dtype is not None else images.dtype
        saved = 0
        for idx, frame in enumerate(subset):
            fp = out / f"{name}_{start_frame + idx + 1:03d}.png"
            if not force and fp.exists():
                continue
            converted = _convert_dtype(frame, target, normed)
            if target == np.uint8:
                img = Image.fromarray(converted, mode='L')
            elif target == np.uint16:
                img = Image.fromarray(converted, mode='I;16')
            else:
                img = Image.fromarray(converted)
            img.save(fp)
            saved += 1
        return saved

    def write_video(self, output_path: str, start_frame: int = 0,
                    end_frame: Optional[int] = None, fps: float = 30.0,
                    dtype: Optional[np.dtype] = None, force: bool = False,
                    normed: bool = True) -> bool:
        if not HAS_IMAGEIO:
            raise ImportError("imageio is required for video. Install with: pip install tdms_explorer[video]")
        out = Path(output_path)
        if not force and out.exists():
            return False
        images = self.extract_images()
        if images is None:
            return False
        if end_frame is None:
            end_frame = images.shape[0]
        subset = images[start_frame:end_frame]
        target = dtype if dtype is not None else images.dtype
        frames: List[np.ndarray] = []
        for frame in subset:
            converted = _convert_dtype(frame, target, normed)
            if converted.dtype != np.uint8:
                converted = _convert_dtype(converted, np.uint8, normed)
            frames.append(converted)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(out), frames, fps=fps, codec='libx264', quality=8)
        return True
    
    def get_image_data(self, image_num: int) -> Optional[np.ndarray]:
        """
        Get raw image data as numpy array.
        
        Args:
            image_num: Image number to get
            
        Returns:
            Numpy array containing image data, or None if not found
        """
        images = self.extract_images()
        if images is None:
            return None
        
        if image_num >= images.shape[0]:
            print(f"Image number {image_num} is out of range. Max: {images.shape[0]-1}")
            return None
        
        return images[image_num, :, :]
    
    def get_raw_channel_data(self, group_name: str, channel_name: str) -> Optional[np.ndarray]:
        """
        Get raw data from a specific channel.
        
        Args:
            group_name: Group name
            channel_name: Channel name
            
        Returns:
            Numpy array containing channel data, or None if not found
        """
        if group_name not in self.channels:
            print(f"Group {group_name} not found.")
            return None
        
        if channel_name not in self.channels[group_name]:
            print(f"Channel {channel_name} not found in group {group_name}.")
            return None
        
        try:
            group = self.tdms_file[group_name]
            channel = group[channel_name]
            return channel.data
        except Exception as e:
            print(f"Error reading channel data: {e}")
            return None
    
    def create_image_analyzer(self, image_num: int = 0) -> Optional['ImageAnalyzer']:
        """
        Create an ImageAnalyzer instance for a specific image.
        
        Args:
            image_num: Image number to analyze
            
        Returns:
            ImageAnalyzer instance or None if image analysis is not available
        """
        if ImageAnalyzer is None:
            print("Image analysis features are not available.")
            return None
            
        image_data = self.get_image_data(image_num)
        if image_data is None:
            return None
            
        return ImageAnalyzer(image_data)
    
    def analyze_image(self, image_num: int = 0) -> Optional[Dict]:
        """
        Perform comprehensive analysis on an image.
        
        Args:
            image_num: Image number to analyze
            
        Returns:
            Dictionary with image analysis results, or None if analysis failed
        """
        analyzer = self.create_image_analyzer(image_num)
        if analyzer is None:
            return None
            
        return analyzer.analyze_image()
    
    def apply_image_filter(self, image_num: int = 0, filter_type: str = 'gaussian', **kwargs) -> Optional[np.ndarray]:
        """
        Apply a filter to an image.
        
        Args:
            image_num: Image number to process
            filter_type: Type of filter to apply
            **kwargs: Filter-specific parameters
            
        Returns:
            Processed image data or None if filtering failed
        """
        analyzer = self.create_image_analyzer(image_num)
        if analyzer is None:
            return None
            
        return analyzer.apply_filter(filter_type, **kwargs)
    
    def detect_edges(self, image_num: int = 0, method: str = 'canny', **kwargs) -> Optional[np.ndarray]:
        """
        Detect edges in an image.
        
        Args:
            image_num: Image number to process
            method: Edge detection method
            **kwargs: Method-specific parameters
            
        Returns:
            Edge-detected image or None if detection failed
        """
        analyzer = self.create_image_analyzer(image_num)
        if analyzer is None:
            return None
            
        return analyzer.detect_edges(method, **kwargs)
    
    def threshold_image(self, image_num: int = 0, method: str = 'otsu', **kwargs) -> Optional[np.ndarray]:
        """
        Apply thresholding to an image.
        
        Args:
            image_num: Image number to process
            method: Thresholding method
            **kwargs: Method-specific parameters
            
        Returns:
            Thresholded image or None if thresholding failed
        """
        analyzer = self.create_image_analyzer(image_num)
        if analyzer is None:
            return None
            
        return analyzer.threshold_image(method, **kwargs)
    
    def set_roi(self, image_num: int = 0, x: int = 0, y: int = 0, width: int = 100, height: int = 100) -> Optional[Dict]:
        """
        Set a region of interest on an image.
        
        Args:
            image_num: Image number
            x, y: Top-left corner coordinates
            width, height: ROI dimensions
            
        Returns:
            ROI coordinates dictionary or None if ROI setting failed
        """
        analyzer = self.create_image_analyzer(image_num)
        if analyzer is None:
            return None
            
        roi_mask = analyzer.set_roi_rectangle(x, y, width, height)
        return analyzer.roi_coords
    
    def get_roi_analysis(self, image_num: int = 0) -> Optional[Dict]:
        """
        Get analysis results for the current ROI.
        
        Args:
            image_num: Image number
            
        Returns:
            Dictionary with ROI analysis results or None if no ROI is set
        """
        analyzer = self.create_image_analyzer(image_num)
        if analyzer is None:
            return None
            
        if analyzer.roi_mask is None:
            print("No ROI is currently set.")
            return None
            
        return analyzer.analyze_image()
    
    def interactive_roi_selection(self, image_num: int = 0, cmap: str = 'gray') -> Optional[Dict]:
        """
        Interactive ROI selection for an image.
        
        Args:
            image_num: Image number
            cmap: Matplotlib colormap
            
        Returns:
            ROI coordinates dictionary or None if selection failed
        """
        analyzer = self.create_image_analyzer(image_num)
        if analyzer is None:
            return None
            
        return analyzer.interactive_roi_selection(cmap)
    
    def compare_images(self, image_num1: int = 0, image_num2: int = 1, method: str = 'difference') -> Optional[np.ndarray]:
        """
        Compare two images from the TDMS file.
        
        Args:
            image_num1: First image number
            image_num2: Second image number
            method: Comparison method
            
        Returns:
            Comparison result as numpy array or None if comparison failed
        """
        analyzer1 = self.create_image_analyzer(image_num1)
        analyzer2 = self.create_image_analyzer(image_num2)
        
        if analyzer1 is None or analyzer2 is None:
            return None
            
        return analyzer1.compare_images(analyzer2.current_image, method)
    
    def get_image_histogram(self, image_num: int = 0, bins: int = 256) -> Optional[Dict]:
        """
        Get histogram data for an image.
        
        Args:
            image_num: Image number
            bins: Number of histogram bins
            
        Returns:
            Dictionary with histogram data or None if histogram creation failed
        """
        analyzer = self.create_image_analyzer(image_num)
        if analyzer is None:
            return None
            
        return analyzer.create_histogram(bins)
    
    def detect_features(self, image_num: int = 0, method: str = 'blob', **kwargs) -> Optional[Dict]:
        """
        Detect features in an image.
        
        Args:
            image_num: Image number
            method: Feature detection method
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary with detected features or None if detection failed
        """
        analyzer = self.create_image_analyzer(image_num)
        if analyzer is None:
            return None
            
        return analyzer.detect_features(method, **kwargs)
    
    def get_image_profile(self, image_num: int = 0, direction: str = 'horizontal', position: Optional[int] = None) -> Optional[Dict]:
        """
        Get intensity profile from an image.
        
        Args:
            image_num: Image number
            direction: Profile direction
            position: Position for line profile
            
        Returns:
            Dictionary with profile data or None if profile extraction failed
        """
        analyzer = self.create_image_analyzer(image_num)
        if analyzer is None:
            return None
            
        return analyzer.get_image_profile(direction, position)


def list_tdms_files(directory: str = ".") -> List[str]:
    """
    List all TDMS files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of TDMS file paths
    """
    tdms_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.tdms'):
            tdms_files.append(os.path.join(directory, file))
    return tdms_files


def create_animation_from_tdms(filename: str, output_path: str,
                               fps: int = 10, cmap: str = 'gray'):
    """
    Create and save an animation from a TDMS file.
    
    Args:
        filename: TDMS file path
        output_path: Output animation file path (e.g., 'animation.mp4')
        fps: Frames per second
        cmap: Matplotlib colormap to use
    """
    explorer = TDMSFileExplorer(filename)
    images = explorer.extract_images()
    
    if images is None:
        print("No image data found.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(images[0, :, :], cmap=cmap)
    plt.title(f"Animation from {filename}")
    plt.colorbar(img)
    
    def update(frame):
        img.set_array(images[frame, :, :])
        ax.set_title(f"Frame {frame}")
        return img,
    
    ani = FuncAnimation(fig, update, frames=range(images.shape[0]),
                      interval=1000/fps, blit=True)
    
    # Save animation
    ani.save(output_path, writer='ffmpeg', fps=fps)
    print(f"Animation saved to {output_path}")
    
    plt.close()


def _convert_dtype(image: np.ndarray, target_dtype: np.dtype, normed: bool = True) -> np.ndarray:
    src = image.dtype
    if target_dtype == src:
        if src == np.uint16 and normed:
            lo, hi = image.min(), image.max()
            if hi > lo:
                return ((image.astype(np.float32) - lo) / (hi - lo) * 65535.0).astype(np.uint16)
        return image
    if target_dtype == np.uint8:
        if src == np.uint16:
            if normed:
                lo, hi = image.min(), image.max()
                if hi > lo:
                    return ((image.astype(np.float32) - lo) / (hi - lo) * 255.0).astype(np.uint8)
                return np.zeros_like(image, dtype=np.uint8)
            return (image.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
        if src in (np.float32, np.float64):
            lo, hi = image.min(), image.max()
            if hi > lo:
                return ((image - lo) / (hi - lo) * 255.0).astype(np.uint8)
            return np.zeros_like(image, dtype=np.uint8)
        return np.clip(image, 0, 255).astype(np.uint8)
    if target_dtype == np.uint16:
        if src == np.uint8:
            return (image.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
        if src in (np.float32, np.float64):
            lo, hi = image.min(), image.max()
            if hi > lo:
                return ((image - lo) / (hi - lo) * 65535.0).astype(np.uint16)
            return np.zeros_like(image, dtype=np.uint16)
        return np.clip(image, 0, 65535).astype(np.uint16)
    return image.astype(target_dtype)


def _process_single_tdms(
    args: Tuple
) -> int:
    tdms_file, output_dir, base_name, dtype, to_video, fps, force, normed = args
    explorer = TDMSFileExplorer(str(tdms_file))
    name = base_name or Path(tdms_file).stem.replace('_video', '')
    saved = explorer.write_images(str(output_dir), base_name=name, dtype=dtype, force=force, normed=normed)
    if to_video:
        explorer.write_video(str(Path(output_dir) / f"{name}.mp4"), fps=fps, dtype=dtype, force=force, normed=normed)
    return saved


def process_tdms_files(
    input_pattern: str,
    output_dir: Union[str, Path],
    start_index: int = 1,
    num_files: Optional[int] = None,
    base_name: Optional[str] = None,
    dtype: Optional[np.dtype] = None,
    num_workers: Optional[int] = None,
    to_video: bool = False,
    fps: float = 30.0,
    force: bool = False,
    normed: bool = True
) -> int:
    input_path = Path(input_pattern)
    if input_path.is_file():
        tdms_files = [input_path]
    elif input_path.is_dir():
        tdms_files = sorted(input_path.glob("*.tdms"))
        if num_files:
            tdms_files = tdms_files[:num_files]
    elif '{' in input_pattern:
        tdms_files = []
        file_idx = start_index
        while True:
            fp = Path(input_pattern.format(file_idx))
            if not fp.exists():
                break
            tdms_files.append(fp)
            file_idx += 1
            if num_files and len(tdms_files) >= num_files:
                break
    else:
        parent = input_path.parent if input_path.parent != Path('.') else Path('.')
        if '*' in input_pattern or '?' in input_pattern:
            tdms_files = sorted(parent.glob(input_path.name))
        else:
            tdms_files = sorted(parent.glob(f"{input_path.name}.tdms"))
        if num_files:
            tdms_files = tdms_files[:num_files]
    if not tdms_files:
        raise FileNotFoundError(f"No TDMS files found matching pattern: {input_pattern}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    workers = num_workers if num_workers is not None else cpu_count()
    tasks = [
        (f, out, base_name, dtype, to_video, fps, force, normed)
        for f in tdms_files
    ]
    if workers <= 1 or len(tasks) == 1:
        return sum(_process_single_tdms(t) for t in tasks)
    with Pool(processes=workers) as pool:
        return sum(pool.map(_process_single_tdms, tasks))