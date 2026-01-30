"""
TDMS Explorer Image Analysis Module

Advanced image analysis functionality for TDMS Explorer.
This module provides image processing, analysis, and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List, Union
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

try:
    import skimage
    from skimage import filters, exposure, feature, measure, morphology, segmentation
    from skimage.draw import polygon2mask
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Some image analysis features will be disabled.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available. Some image analysis features will be disabled.")

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Some image analysis features will be disabled.")


class ImageAnalyzer:
    """
    Image analysis class for processing and analyzing images from TDMS files.
    
    Args:
        image_data (np.ndarray): Image data as numpy array
    """
    
    def __init__(self, image_data: np.ndarray):
        """Initialize image analyzer with image data."""
        self.original_image = image_data.copy()
        self.current_image = image_data.copy()
        self.roi_mask = None
        self.roi_coords = None
        
        # Validate image data
        if len(image_data.shape) not in [2, 3]:
            raise ValueError("Image data must be 2D (grayscale) or 3D (color)")
            
        if len(image_data.shape) == 3 and image_data.shape[2] not in [3, 4]:
            raise ValueError("Color images must have 3 (RGB) or 4 (RGBA) channels")
    
    def reset(self):
        """Reset to original image data."""
        self.current_image = self.original_image.copy()
        self.roi_mask = None
        self.roi_coords = None
    
    def apply_filter(self, filter_type: str, **kwargs) -> Optional[np.ndarray]:
        """
        Apply various filters to the image.
        
        Args:
            filter_type: Type of filter ('gaussian', 'median', 'bilateral', 'sobel', 'prewitt', 'laplace')
            **kwargs: Additional filter parameters
            
        Returns:
            Filtered image or None if filter not available
        """
        if not SKIMAGE_AVAILABLE and filter_type in ['sobel', 'prewitt']:
            print("scikit-image required for this filter")
            return None
            
        if not OPENCV_AVAILABLE and filter_type in ['bilateral']:
            print("OpenCV required for this filter")
            return None
            
        if filter_type == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            self.current_image = filters.gaussian(self.current_image, sigma=sigma)
        
        elif filter_type == 'median':
            size = kwargs.get('size', 3)
            self.current_image = filters.median(self.current_image, np.ones((size, size)))
            
        elif filter_type == 'bilateral' and OPENCV_AVAILABLE:
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            self.current_image = cv2.bilateralFilter(
                self.current_image.astype(np.float32), 
                d, sigma_color, sigma_space
            )
        
        elif filter_type == 'sobel' and SKIMAGE_AVAILABLE:
            self.current_image = filters.sobel(self.current_image)
            
        elif filter_type == 'prewitt' and SKIMAGE_AVAILABLE:
            self.current_image = filters.prewitt(self.current_image)
            
        elif filter_type == 'laplace' and SKIMAGE_AVAILABLE:
            self.current_image = filters.laplace(self.current_image)
        
        else:
            print(f"Unknown filter type: {filter_type}")
            return None
            
        return self.current_image.copy()
    
    def adjust_contrast(self, method: str = 'histogram', **kwargs) -> Optional[np.ndarray]:
        """
        Adjust image contrast.
        
        Args:
            method: Contrast adjustment method ('histogram', 'adaptive', 'gamma')
            **kwargs: Method-specific parameters
            
        Returns:
            Contrast-adjusted image or None if method not available
        """
        if not SKIMAGE_AVAILABLE:
            print("scikit-image required for contrast adjustment")
            return None
            
        if method == 'histogram':
            self.current_image = exposure.equalize_hist(self.current_image)
            
        elif method == 'adaptive':
            clip_limit = kwargs.get('clip_limit', 0.03)
            self.current_image = exposure.equalize_adapthist(
                self.current_image, clip_limit=clip_limit
            )
            
        elif method == 'gamma':
            gamma = kwargs.get('gamma', 1.0)
            self.current_image = exposure.adjust_gamma(self.current_image, gamma)
        
        else:
            print(f"Unknown contrast method: {method}")
            return None
            
        return self.current_image.copy()
    
    def detect_edges(self, method: str = 'canny', **kwargs) -> Optional[np.ndarray]:
        """
        Detect edges in the image.
        
        Args:
            method: Edge detection method ('canny', 'sobel', 'prewitt', 'laplace')
            **kwargs: Method-specific parameters
            
        Returns:
            Edge-detected image or None if method not available
        """
        if not SKIMAGE_AVAILABLE and method != 'canny':
            print("scikit-image required for this edge detection method")
            return None
            
        if not OPENCV_AVAILABLE and method == 'canny':
            print("OpenCV required for Canny edge detection")
            return None
            
        if method == 'canny' and OPENCV_AVAILABLE:
            threshold1 = kwargs.get('threshold1', 100)
            threshold2 = kwargs.get('threshold2', 200)
            edges = cv2.Canny(
                (self.current_image * 255).astype(np.uint8), 
                threshold1, threshold2
            )
            self.current_image = edges.astype(float) / 255.0
            
        elif method == 'sobel':
            self.current_image = filters.sobel(self.current_image)
            
        elif method == 'prewitt':
            self.current_image = filters.prewitt(self.current_image)
            
        elif method == 'laplace':
            self.current_image = filters.laplace(self.current_image)
        
        else:
            print(f"Unknown edge detection method: {method}")
            return None
            
        return self.current_image.copy()
    
    def threshold_image(self, method: str = 'otsu', **kwargs) -> Optional[np.ndarray]:
        """
        Apply thresholding to the image.
        
        Args:
            method: Thresholding method ('otsu', 'adaptive', 'manual')
            **kwargs: Method-specific parameters
            
        Returns:
            Thresholded image or None if method not available
        """
        if not SKIMAGE_AVAILABLE:
            print("scikit-image required for thresholding")
            return None
            
        if method == 'otsu':
            thresh_val = filters.threshold_otsu(self.current_image)
            self.current_image = self.current_image > thresh_val
            
        elif method == 'adaptive':
            block_size = kwargs.get('block_size', 51)
            self.current_image = filters.threshold_adaptive(
                self.current_image, block_size, method='gaussian'
            )
            
        elif method == 'manual':
            threshold = kwargs.get('threshold', 0.5)
            self.current_image = self.current_image > threshold
        
        else:
            print(f"Unknown thresholding method: {method}")
            return None
            
        return self.current_image.copy()
    
    def analyze_image(self) -> Dict:
        """
        Perform comprehensive image analysis.
        
        Returns:
            Dictionary containing various image statistics and metrics
        """
        stats = {}
        
        # Basic statistics
        stats['min'] = float(self.current_image.min())
        stats['max'] = float(self.current_image.max())
        stats['mean'] = float(self.current_image.mean())
        stats['std'] = float(self.current_image.std())
        stats['median'] = float(np.median(self.current_image))
        
        # Histogram
        hist, bins = np.histogram(self.current_image, bins=256)
        stats['histogram'] = {
            'bins': bins.tolist(),
            'counts': hist.tolist()
        }
        
        # ROI statistics if ROI is defined
        if self.roi_mask is not None:
            roi_data = self.current_image[self.roi_mask]
            stats['roi'] = {
                'min': float(roi_data.min()),
                'max': float(roi_data.max()),
                'mean': float(roi_data.mean()),
                'std': float(roi_data.std()),
                'median': float(np.median(roi_data)),
                'area': int(roi_data.size),
                'pixel_count': int(roi_data.size)
            }
        
        # Advanced metrics if scikit-image is available
        if SKIMAGE_AVAILABLE:
            stats['entropy'] = float(measure.shannon_entropy(self.current_image))
            
            # Try to detect features
            try:
                edges = feature.canny(self.current_image)
                stats['edge_density'] = float(edges.mean())
            except:
                stats['edge_density'] = None
        
        return stats
    
    def set_roi_rectangle(self, x: int, y: int, width: int, height: int):
        """
        Set a rectangular region of interest.
        
        Args:
            x, y: Top-left corner coordinates
            width, height: Dimensions of the rectangle
        """
        if len(self.current_image.shape) == 2:
            # Grayscale image
            rows, cols = self.current_image.shape
        else:
            # Color image
            rows, cols, _ = self.current_image.shape
            
        # Ensure coordinates are within bounds
        x = max(0, min(x, cols - 1))
        y = max(0, min(y, rows - 1))
        width = max(1, min(width, cols - x))
        height = max(1, min(height, rows - y))
        
        # Create mask
        self.roi_mask = np.zeros_like(self.current_image, dtype=bool)
        self.roi_mask[y:y+height, x:x+width] = True
        
        self.roi_coords = {
            'x': x, 'y': y, 'width': width, 'height': height,
            'type': 'rectangle'
        }
        
        return self.roi_mask.copy()
    
    def clear_roi(self):
        """Clear the current region of interest."""
        self.roi_mask = None
        self.roi_coords = None
    
    def get_roi_data(self) -> Optional[np.ndarray]:
        """
        Get the data within the current ROI.
        
        Returns:
            ROI data as numpy array, or None if no ROI is defined
        """
        if self.roi_mask is None:
            return None
            
        return self.current_image[self.roi_mask]
    
    def get_roi_image(self) -> Optional[np.ndarray]:
        """
        Get the image data within the current ROI as a 2D array.
        
        Returns:
            ROI image data, or None if no ROI is defined
        """
        if self.roi_mask is None:
            return None
            
        if self.roi_coords['type'] == 'rectangle':
            x, y, width, height = self.roi_coords['x'], self.roi_coords['y'], \
                                 self.roi_coords['width'], self.roi_coords['height']
            return self.current_image[y:y+height, x:x+width]
        
        return None
    
    def show_with_roi(self, cmap: str = 'gray', title: str = "Image with ROI"):
        """
        Display the image with ROI overlay.
        
        Args:
            cmap: Matplotlib colormap
            title: Plot title
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.current_image, cmap=cmap)
        
        if self.roi_coords is not None and self.roi_coords['type'] == 'rectangle':
            x, y, width, height = self.roi_coords['x'], self.roi_coords['y'], \
                                 self.roi_coords['width'], self.roi_coords['height']
            
            # Draw rectangle
            rect = Rectangle((x, y), width, height, 
                           linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            
        plt.title(title)
        plt.colorbar()
        plt.show()
    
    def interactive_roi_selection(self, cmap: str = 'gray'):
        """
        Interactive ROI selection using matplotlib.
        
        Args:
            cmap: Matplotlib colormap
            
        Returns:
            ROI coordinates dictionary
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.current_image, cmap=cmap)
        plt.title("Select ROI: Click and drag to draw rectangle")
        
        # ROI selection callback
        def onselect(eclick, erelease):
            """Callback for rectangle selection."""
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            # Ensure coordinates are ordered correctly
            x = min(x1, x2)
            y = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Set ROI
            self.set_roi_rectangle(x, y, width, height)
            
            # Close the figure
            plt.close(fig)
            
        # Create rectangle selector
        rs = RectangleSelector(ax, onselect, 
                              useblit=True, 
                              button=[1],  # Left mouse button
                              minspanx=5, minspany=5,
                              spancoords='pixels',
                              interactive=True)
        
        plt.show()
        
        return self.roi_coords
    
    def compare_images(self, other_image: np.ndarray, method: str = 'difference') -> np.ndarray:
        """
        Compare current image with another image.
        
        Args:
            other_image: Image to compare with
            method: Comparison method ('difference', 'absolute', 'relative')
            
        Returns:
            Comparison result as numpy array
        """
        if self.current_image.shape != other_image.shape:
            raise ValueError("Images must have the same shape for comparison")
            
        if method == 'difference':
            return self.current_image - other_image
            
        elif method == 'absolute':
            return np.abs(self.current_image - other_image)
            
        elif method == 'relative':
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                relative_diff = np.abs((self.current_image - other_image) / 
                                      (other_image + 1e-10))
            return relative_diff
        
        else:
            raise ValueError(f"Unknown comparison method: {method}")
    
    def create_histogram(self, bins: int = 256, log_scale: bool = False) -> Dict:
        """
        Create image histogram.
        
        Args:
            bins: Number of histogram bins
            log_scale: Whether to use logarithmic scale
            
        Returns:
            Dictionary with histogram data
        """
        hist, bin_edges = np.histogram(self.current_image, bins=bins)
        
        return {
            'hist': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'log_scale': log_scale
        }
    
    def apply_morphology(self, operation: str = 'opening', kernel_size: int = 3) -> Optional[np.ndarray]:
        """
        Apply morphological operations.
        
        Args:
            operation: Morphological operation ('opening', 'closing', 'erosion', 'dilation')
            kernel_size: Size of the morphological kernel
            
        Returns:
            Processed image or None if operation not available
        """
        if not SKIMAGE_AVAILABLE:
            print("scikit-image required for morphological operations")
            return None
            
        # Create kernel
        footprint = morphology.disk(kernel_size)
        
        if operation == 'opening':
            self.current_image = morphology.opening(self.current_image, footprint)
            
        elif operation == 'closing':
            self.current_image = morphology.closing(self.current_image, footprint)
            
        elif operation == 'erosion':
            self.current_image = morphology.erosion(self.current_image, footprint)
            
        elif operation == 'dilation':
            self.current_image = morphology.dilation(self.current_image, footprint)
        
        else:
            print(f"Unknown morphological operation: {operation}")
            return None
            
        return self.current_image.copy()
    
    def segment_image(self, method: str = 'threshold', **kwargs) -> Optional[np.ndarray]:
        """
        Segment the image using various methods.
        
        Args:
            method: Segmentation method ('threshold', 'watershed', 'felzenszwalb')
            **kwargs: Method-specific parameters
            
        Returns:
            Segmented image or None if method not available
        """
        if not SKIMAGE_AVAILABLE:
            print("scikit-image required for image segmentation")
            return None
            
        if method == 'threshold':
            thresh_method = kwargs.get('thresh_method', 'otsu')
            if thresh_method == 'otsu':
                thresh_val = filters.threshold_otsu(self.current_image)
            else:
                thresh_val = kwargs.get('threshold', 0.5)
            
            self.current_image = self.current_image > thresh_val
            
        elif method == 'watershed':
            # Simple watershed implementation
            distance = ndimage.distance_transform_edt(self.current_image)
            self.current_image = segmentation.watershed(-distance, 
                                                       markers=None, 
                                                       mask=self.current_image)
        
        elif method == 'felzenszwalb':
            self.current_image = segmentation.felzenszwalb(self.current_image, 
                                                          scale=kwargs.get('scale', 1.0),
                                                          sigma=kwargs.get('sigma', 0.5),
                                                          min_size=kwargs.get('min_size', 20))
        
        else:
            print(f"Unknown segmentation method: {method}")
            return None
            
        return self.current_image.copy()
    
    def detect_features(self, method: str = 'blob', **kwargs) -> Optional[Dict]:
        """
        Detect features in the image.
        
        Args:
            method: Feature detection method ('blob', 'corner', 'edge')
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary with detected features or None if method not available
        """
        if not SKIMAGE_AVAILABLE:
            print("scikit-image required for feature detection")
            return None
            
        features = {}
        
        if method == 'blob':
            # Detect blobs
            blobs = feature.blob_log(self.current_image, 
                                    min_sigma=kwargs.get('min_sigma', 1),
                                    max_sigma=kwargs.get('max_sigma', 50),
                                    num_sigma=kwargs.get('num_sigma', 10),
                                    threshold=kwargs.get('threshold', 0.1))
            
            features['blobs'] = [
                {'y': int(blob[0]), 'x': int(blob[1]), 'sigma': blob[2]}
                for blob in blobs
            ]
            features['count'] = len(blobs)
            
        elif method == 'corner':
            # Detect corners
            corners = feature.corner_harris(self.current_image)
            coords = feature.corner_peaks(corners, min_distance=kwargs.get('min_distance', 5))
            
            features['corners'] = [
                {'y': int(coord[0]), 'x': int(coord[1])}
                for coord in coords
            ]
            features['count'] = len(coords)
            
        elif method == 'edge':
            # Detect edges
            edges = feature.canny(self.current_image, 
                                 sigma=kwargs.get('sigma', 1.0))
            
            features['edge_pixels'] = np.where(edges)
            features['edge_density'] = edges.mean()
            
        else:
            print(f"Unknown feature detection method: {method}")
            return None
            
        return features
    
    def get_image_profile(self, direction: str = 'horizontal', position: Optional[int] = None) -> Dict:
        """
        Get intensity profile along a line or axis.
        
        Args:
            direction: Profile direction ('horizontal', 'vertical', 'diagonal')
            position: Position for line profile (None for axis average)
            
        Returns:
            Dictionary with profile data
        """
        profile_data = {}
        
        if direction == 'horizontal':
            if position is None:
                # Average along vertical axis
                profile = np.mean(self.current_image, axis=0)
                x_coords = np.arange(self.current_image.shape[1])
            else:
                # Single row
                profile = self.current_image[position, :]
                x_coords = np.arange(self.current_image.shape[1])
                
        elif direction == 'vertical':
            if position is None:
                # Average along horizontal axis
                profile = np.mean(self.current_image, axis=1)
                x_coords = np.arange(self.current_image.shape[0])
            else:
                # Single column
                profile = self.current_image[:, position]
                x_coords = np.arange(self.current_image.shape[0])
                
        elif direction == 'diagonal':
            # Diagonal profile
            min_dim = min(self.current_image.shape[0], self.current_image.shape[1])
            profile = np.array([self.current_image[i, i] for i in range(min_dim)])
            x_coords = np.arange(min_dim)
        
        else:
            raise ValueError(f"Unknown profile direction: {direction}")
            
        profile_data['x'] = x_coords.tolist()
        profile_data['y'] = profile.tolist()
        profile_data['direction'] = direction
        profile_data['position'] = position
        
        return profile_data