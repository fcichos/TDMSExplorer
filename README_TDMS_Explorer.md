# TDMS Explorer Module

A Python module for working with TDMS (Technical Data Management Streaming) files. This module provides comprehensive functionality to explore, visualize, and export data from TDMS files, particularly those containing image sequences.

## Features

- **File Exploration**: List and analyze the contents of TDMS files
- **Image Extraction**: Extract image sequences from TDMS files
- **Visualization**: Display individual images and animations
- **Export**: Write individual images or complete series to files
- **Metadata Access**: Extract and display file metadata and properties
- **Raw Data Access**: Access raw channel data for custom processing

## Requirements

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

## Installation

1. Copy the `tdms_explorer.py` file to your project directory
2. Import it in your Python scripts:

```python
from tdms_explorer import TDMSFileExplorer, list_tdms_files
```

## Usage Examples

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

### List TDMS Files

```python
from tdms_explorer import list_tdms_files

# List all TDMS files in current directory
files = list_tdms_files()
for file in files:
    print(file)
```

### Create Animation

```python
from tdms_explorer import create_animation_from_tdms

# Create and save animation
create_animation_from_tdms('video.tdms', 'animation.mp4', fps=10)
```

### Access Raw Data

```python
# Get raw image data as numpy array
image_data = explorer.get_image_data(0)

# Get raw channel data
channel_data = explorer.get_raw_channel_data('Image', 'Image')
```

## API Reference

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

## Command Line Usage

Run the test script:

```bash
python test_tdms_explorer.py
```

Run the example usage:

```bash
python example_usage.py
```

## Error Handling

The module includes comprehensive error handling:

- Invalid file paths
- Missing dependencies
- Out-of-range image indices
- File write conflicts
- Invalid frame ranges

## Limitations

- Requires `nptdms` library which may have limitations with certain TDMS file formats
- Image dimension guessing may not work for all file formats
- Animation creation requires `ffmpeg` to be installed

## License

This module is provided as-is for research and educational purposes. No warranty is provided.

## Support

For issues or questions, please refer to the existing Jupyter notebooks in this project for additional context and examples.