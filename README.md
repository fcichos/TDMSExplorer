# TDMS Explorer Package

A comprehensive Python package for working with TDMS (Technical Data Management Streaming) files. This package provides tools to explore, visualize, and export data from TDMS files, particularly those containing image sequences.

## 📦 Installation

### From GitHub

```bash
# Clone the repository
git clone https://github.com/fcichos/TDMSExplorer.git
cd TDMSExplorer

# Install the package
pip install .

# Or install in development mode
pip install -e .
```

### Dependencies

The package requires:

- Python 3.6+
- `nptdms` - For reading TDMS files
- `numpy` - For numerical operations
- `matplotlib` - For visualization
- `PIL` (Pillow) - For image handling
- `ffmpeg` - For animation creation (optional)

Install dependencies:

```bash
pip install nptdms numpy matplotlib pillow
```

For animation support:

```bash
# On macOS
brew install ffmpeg

# On Linux (Debian/Ubuntu)
sudo apt-get install ffmpeg

# On Windows (using conda)
conda install -c conda-forge ffmpeg
```

## 🚀 Quick Start

### Basic Usage

```python
from tdms_explorer import TDMSFileExplorer

# Create explorer for a TDMS file
explorer = TDMSFileExplorer('example.tdms')

# Print file contents
explorer.print_contents()

# Display first image
explorer.display_image(0)

# Display animation
explorer.display_animation(fps=15)

# Write all images to directory
explorer.write_images('output_images')

# Write single image
explorer.write_image(0, 'single_image.png')
```

### Command Line Interface

```bash
# List all TDMS files
python -m tdms_explorer list

# Show information about a specific file
python -m tdms_explorer info "file.tdms"

# Display first image from a file
python -m tdms_explorer show "file.tdms" --image 0

# Save image to file (works in JupyterLab!)
python -m tdms_explorer show "file.tdms" --save output.png

# Show statistics without display (for headless environments)
python -m tdms_explorer show "file.tdms" --no-show

# Export all images to directory
python -m tdms_explorer export "file.tdms" output_dir

# Create animation
python -m tdms_explorer animate "file.tdms" animation.mp4 --fps 10

# Get raw data info
python -m tdms_explorer raw "file.tdms" --group "Image" --channel "Image"
```

## 📚 API Reference

### TDMSFileExplorer Class

#### Constructor

```python
TDMSFileExplorer(filename: str)
```

- `filename`: Path to the TDMS file

#### Methods

**File Information**

- `list_contents() -> Dict`: Return file structure as dictionary
- `print_contents()`: Print file structure in readable format
- `has_image_data() -> bool`: Check if file contains image data

**Image Extraction**

- `extract_images() -> Optional[np.ndarray]`: Extract all images as numpy array
- `get_image_data(image_num: int) -> Optional[np.ndarray]`: Get single image as numpy array

**Visualization**

- `display_image(image_num: int = 0, cmap: str = 'gray')`: Display single image
- `display_animation(start_frame: int = 0, end_frame: Optional[int] = None, fps: int = 10, cmap: str = 'gray')`: Display animation

**Export**

- `write_image(image_num: int, output_path: str, cmap: str = 'gray', overwrite: bool = False)`: Write single image
- `write_images(output_dir: str, start_frame: int = 0, end_frame: Optional[int] = None, cmap: str = 'gray', prefix: str = 'output_', format: str = 'png')`: Write image series

**Raw Data Access**

- `get_raw_channel_data(group_name: str, channel_name: str) -> Optional[np.ndarray]`: Get raw channel data

### Utility Functions

- `list_tdms_files(directory: str = ".") -> List[str]`: List TDMS files in directory
- `create_animation_from_tdms(filename: str, output_path: str, fps: int = 10, cmap: str = 'gray')`: Create and save animation

## 🎯 Features

### File Exploration
- List and analyze the contents of TDMS files
- Extract metadata and properties
- Analyze file structure (groups and channels)

### Image Extraction
- Extract image sequences from TDMS files
- Automatic dimension detection
- Support for various image formats

### Visualization
- Display individual images with custom colormaps
- Create and display animations from image sequences
- Interactive visualization with matplotlib

### Export
- Write individual images to files (PNG, JPG, etc.)
- Export complete image series
- Customizable filename prefixes and formats

### Raw Data Access
- Access raw channel data for custom processing
- Extract specific groups and channels
- Save raw data to numpy files

### Command Line Interface
- Comprehensive CLI with multiple commands
- Easy file listing and analysis
- Image display and export

### Image Analysis (New!)
- Comprehensive image statistics and metrics
- Image filtering (Gaussian, median, bilateral, etc.)
- Edge detection (Canny, Sobel, Prewitt, Laplace)
- Thresholding and segmentation
- Region of Interest (ROI) operations
- Image comparison and difference analysis
- Histogram analysis and visualization
- Feature detection (blobs, corners, edges)
- Intensity profile extraction
- Animation creation
- Raw data access and statistics

## 📁 Package Structure

```
tdms_explorer/
├── __init__.py          # Package initialization
├── __main__.py         # Main entry point for CLI
├── core.py             # Core functionality
└── cli/
    ├── __init__.py      # CLI module initialization
    └── cli.py           # Command line interface
```

## 🔧 Command Line Usage

### Commands

#### 1. List TDMS Files

```bash
# List all TDMS files in current directory
python -m tdms_explorer list

# With details
python -m tdms_explorer list --details

# Specify directory
python -m tdms_explorer list --dir /path/to/files
```

#### 2. File Information

```bash
# Show detailed information about a TDMS file
python -m tdms_explorer info "file.tdms"

# JSON output
python -m tdms_explorer info "file.tdms" --json
```

#### 3. Display Images

```bash
# Display first image from a TDMS file
python -m tdms_explorer show "file.tdms"

# Specific image number
python -m tdms_explorer show "file.tdms" --image 5

# Different colormap
python -m tdms_explorer show "file.tdms" --cmap viridis

# Test without displaying
python -m tdms_explorer show "file.tdms" --no-show
```

#### 4. Create Animations

```bash
# Create animation from TDMS image sequences
python -m tdms_explorer animate "file.tdms" animation.mp4

# Custom frame rate
python -m tdms_explorer animate "file.tdms" animation.mp4 --fps 15

# Specific frame range
python -m tdms_explorer animate "file.tdms" animation.mp4 --start 10 --end 50

# Different colormap
python -m tdms_explorer animate "file.tdms" animation.mp4 --cmap plasma

# Create without displaying
python -m tdms_explorer animate "file.tdms" animation.mp4 --no-display
```

#### 5. Export Images

```bash
# Export all images
python -m tdms_explorer export "file.tdms" output_directory

# Export specific range
python -m tdms_explorer export "file.tdms" output_directory --start 0 --end 100

# Export single image
python -m tdms_explorer export "file.tdms" output_directory --single 42

# Custom prefix and format
python -m tdms_explorer export "file.tdms" output_directory --prefix "frame_" --format jpg

# Overwrite existing files
python -m tdms_explorer export "file.tdms" output_directory --overwrite

# Different colormap
python -m tdms_explorer export "file.tdms" output_directory --cmap inferno
```

#### 6. Raw Data Access

```bash
# Show channel info
python -m tdms_explorer raw "file.tdms" --group "Image" --channel "Image" --info

# Show detailed statistics
python -m tdms_explorer raw "file.tdms" --group "Image" --channel "Image"

# Save raw data to file
python -m tdms_explorer raw "file.tdms" --group "Image" --channel "Image" --save raw_data.npy
```

#### 7. Statistics

```bash
# Basic statistics
python -m tdms_explorer stats "file.tdms"

# Image statistics
python -m tdms_explorer stats "file.tdms" --images

# Channel statistics
python -m tdms_explorer stats "file.tdms" --channels

# All statistics
python -m tdms_explorer stats "file.tdms" --images --channels
```

## 🔄 Migration Guide

If you're migrating from the original standalone scripts:

### Old Way

```python
from tdms_explorer import TDMSFileExplorer, list_tdms_files
```

### New Way

```python
from tdms_explorer import TDMSFileExplorer, list_tdms_files
```

The API remains the same, but now you can also use the package structure:

```python
from tdms_explorer.core import TDMSFileExplorer
from tdms_explorer.cli import main as cli_main
```

## 📋 Examples

### Example 1: Quick File Analysis

```python
from tdms_explorer import TDMSFileExplorer, list_tdms_files

# List files
files = list_tdms_files()
print(f"Found {len(files)} TDMS files")

# Analyze first file
explorer = TDMSFileExplorer(files[0])
explorer.print_contents()

# Check if it has images
if explorer.has_image_data():
    print("File contains image data!")
    images = explorer.extract_images()
    print(f"Found {images.shape[0]} images of size {images.shape[1]}x{images.shape[2]}")
```

### Example 2: Image Export Workflow

```python
from tdms_explorer import TDMSFileExplorer

# Create explorer
explorer = TDMSFileExplorer('experiment.tdms')

# Export first 100 images
explorer.write_images('experiment_images', start_frame=0, end_frame=99)

# Create animation of first 50 images
from tdms_explorer import create_animation_from_tdms
create_animation_from_tdms('experiment.tdms', 'experiment_animation.mp4', fps=10)

# Display a specific image
explorer.display_image(25)
```

### Example 3: Data Analysis

```python
from tdms_explorer import TDMSFileExplorer
import numpy as np

# Load file
explorer = TDMSFileExplorer('data.tdms')

# Get raw image data for custom processing
image_data = explorer.get_image_data(0)
print(f"Image shape: {image_data.shape}")
print(f"Data range: {image_data.min()} to {image_data.max()}")

# Get raw channel data
channel_data = explorer.get_raw_channel_data('Image', 'Power_heating')
print(f"Channel data shape: {channel_data.shape}")
print(f"Mean value: {np.mean(channel_data)}")

# Save raw data for later processing
np.save('raw_images.npy', explorer.extract_images())
```

## 🧪 Testing

The package includes comprehensive error handling:

- Invalid file paths
- Missing dependencies
- Out-of-range image indices
- File write conflicts
- Invalid frame ranges
- Permission issues
- File format problems

## 📝 License

This package is provided under the MIT License. See the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests if applicable
5. Submit a pull request

## 🐛 Issues

If you encounter any issues, please:

1. Check that all dependencies are installed
2. Verify your TDMS files are valid
3. Provide detailed error messages
4. Include sample files if possible
5. Open an issue on GitHub

## 💻 JupyterLab Compatibility

The CLI is designed to work well in JupyterLab terminals where GUI display may not be available:

### JupyterLab-Specific Features

- **Automatic detection of headless environments** - The CLI detects when GUI display isn't available
- **Rich statistics display** - Shows detailed image statistics when display isn't possible
- **Direct file saving** - Use `--save` option to export images directly
- **Helpful error messages** - Clear guidance on how to get results in headless environments

### Example: Using CLI in JupyterLab

```bash
# This will show statistics instead of trying to display
python -m tdms_explorer show "file.tdms" --image 0

# Save image directly to file (recommended for JupyterLab)
python -m tdms_explorer show "file.tdms" --image 0 --save output.png

# Get detailed statistics without display
python -m tdms_explorer show "file.tdms" --image 0 --no-show
```

### Output in JupyterLab

When you run display commands in JupyterLab, you'll see helpful output like:

```
Image 0 from file.tdms
Shape: 256x256
Data range: 442.00 to 750.00
Data type: int32

📊 Image Statistics:
  Min: 442.0000
  Max: 750.0000
  Mean: 503.2990
  Std: 17.4703
  Median: 502.0000

💾 Tip: Use '--save image.png' to save this image to a file
⚠️  Display not available in this environment
   Try running in a terminal with GUI support or use '--save' to export the image
```

### Recommended Workflow for JupyterLab

1. **Explore files**: Use `list` and `info` commands to find and analyze TDMS files
2. **Save images**: Use `--save` option to export specific images
3. **Export series**: Use `export` command for multiple images
4. **Analyze data**: Use `stats` and `raw` commands for numerical analysis
5. **View results**: Use JupyterLab's file browser to view saved images

This makes the TDMS Explorer fully functional in JupyterLab environments!

## 📚 Related Resources

- [nptdms documentation](https://nptdms.readthedocs.io/)
- [National Instruments TDMS format](https://www.ni.com/en-us/support/model.tdms.html)
- [Matplotlib documentation](https://matplotlib.org/)
- [NumPy documentation](https://numpy.org/doc/)

## 🎓 Acknowledgments

This package was developed to facilitate research and analysis of TDMS files containing scientific image data. Special thanks to the developers of the underlying libraries that make this package possible.