# TDMS Explorer CLI

A comprehensive command-line interface for working with TDMS files. This CLI provides easy access to all the functionality of the TDMS Explorer module through simple commands.

## Features

- **File Listing**: List and analyze TDMS files in directories
- **File Information**: Detailed analysis of TDMS file structure
- **Image Display**: View individual images from TDMS files
- **Animation Creation**: Create animations from image sequences
- **Image Export**: Export individual images or complete series
- **Raw Data Access**: Extract and analyze raw channel data
- **Statistics**: Generate detailed statistics about TDMS files

## Installation

The CLI is included with the TDMS Explorer module. No additional installation is required.

## Usage

```bash
# Make the script executable (one-time)
chmod +x tdms_cli.py

# Run CLI
python3 tdms_cli.py [command] [options]

# Or directly (if executable)
./tdms_cli.py [command] [options]
```

## Commands

### 1. List TDMS Files

List all TDMS files in a directory:

```bash
python3 tdms_cli.py list

# With details
python3 tdms_cli.py list --details

# Specify directory
python3 tdms_cli.py list --dir /path/to/files
```

### 2. File Information

Show detailed information about a TDMS file:

```bash
python3 tdms_cli.py info "file.tdms"

# JSON output
python3 tdms_cli.py info "file.tdms" --json
```

### 3. Display Images

Display individual images from a TDMS file:

```bash
python3 tdms_cli.py show "file.tdms"

# Specific image number
python3 tdms_cli.py show "file.tdms" --image 5

# Different colormap
python3 tdms_cli.py show "file.tdms" --cmap viridis

# Test without displaying
python3 tdms_cli.py show "file.tdms" --no-show
```

### 4. Create Animations

Create animations from TDMS image sequences:

```bash
python3 tdms_cli.py animate "file.tdms" animation.mp4

# Custom frame rate
python3 tdms_cli.py animate "file.tdms" animation.mp4 --fps 15

# Specific frame range
python3 tdms_cli.py animate "file.tdms" animation.mp4 --start 10 --end 50

# Different colormap
python3 tdms_cli.py animate "file.tdms" animation.mp4 --cmap plasma

# Create without displaying
python3 tdms_cli.py animate "file.tdms" animation.mp4 --no-display
```

### 5. Export Images

Export images from TDMS files:

```bash
# Export all images
python3 tdms_cli.py export "file.tdms" output_directory

# Export specific range
python3 tdms_cli.py export "file.tdms" output_directory --start 0 --end 100

# Export single image
python3 tdms_cli.py export "file.tdms" output_directory --single 42

# Custom prefix and format
python3 tdms_cli.py export "file.tdms" output_directory --prefix "frame_" --format jpg

# Overwrite existing files
python3 tdms_cli.py export "file.tdms" output_directory --overwrite

# Different colormap
python3 tdms_cli.py export "file.tdms" output_directory --cmap inferno
```

### 6. Raw Data Access

Access raw channel data from TDMS files:

```bash
# Show channel info
python3 tdms_cli.py raw "file.tdms" --group "Image" --channel "Image" --info

# Show detailed statistics
python3 tdms_cli.py raw "file.tdms" --group "Image" --channel "Image"

# Save raw data to file
python3 tdms_cli.py raw "file.tdms" --group "Image" --channel "Image" --save raw_data.npy
```

### 7. Statistics

Generate statistics about TDMS files:

```bash
# Basic statistics
python3 tdms_cli.py stats "file.tdms"

# Image statistics
python3 tdms_cli.py stats "file.tdms" --images

# Channel statistics
python3 tdms_cli.py stats "file.tdms" --channels

# All statistics
python3 tdms_cli.py stats "file.tdms" --images --channels
```

## Examples

### Example 1: Quick File Analysis

```bash
# List files
python3 tdms_cli.py list

# Get info about first file
python3 tdms_cli.py info "Manydrops 014 video.tdms"

# Show statistics
python3 tdms_cli.py stats "Manydrops 014 video.tdms" --images
```

### Example 2: Image Export Workflow

```bash
# Create output directory
mkdir experiment_images

# Export first 100 images
python3 tdms_cli.py export "Manydrops 014 video.tdms" experiment_images --end 99

# Create animation of first 50 images
python3 tdms_cli.py animate "Manydrops 014 video.tdms" experiment_animation.mp4 --end 49 --fps 10

# Display a specific image
python3 tdms_cli.py show "Manydrops 014 video.tdms" --image 25
```

### Example 3: Data Analysis

```bash
# Get detailed file info
python3 tdms_cli.py info "2 Fibril_001_video.tdms" --json > file_info.json

# Analyze raw channel data
python3 tdms_cli.py raw "2 Fibril_001_video.tdms" --group "Image" --channel "Power_heating"

# Save raw image data for custom processing
python3 tdms_cli.py raw "2 Fibril_001_video.tdms" --group "Image" --channel "Image" --save raw_images.npy
```

## Command Reference

### Global Options

All commands support standard help:

```bash
python3 tdms_cli.py [command] --help
```

### Common Parameters

- `--help, -h`: Show help for any command
- File paths can be quoted if they contain spaces
- Image numbers are 0-indexed (first image is 0)
- Frame ranges are inclusive

## Error Handling

The CLI includes comprehensive error handling:

- Invalid file paths
- Missing dependencies
- Out-of-range image indices
- Invalid frame ranges
- Permission issues
- File format problems

## Tips

1. **Use tab completion**: Most modern shells support tab completion for file paths
2. **Quote file paths**: Use quotes around file paths that contain spaces
3. **Check file sizes**: Large TDMS files may take time to process
4. **Use `--no-show`**: For testing without displaying images
5. **Start small**: Test with small frame ranges before processing entire files

## Integration

The CLI can be integrated into scripts and workflows:

```bash
# Process multiple files
for file in *.tdms; do
    echo "Processing $file"
    python3 tdms_cli.py info "$file"
    python3 tdms_cli.py export "$file" "${file%.tdms}_images"
done

# Create animations for all files
for file in *.tdms; do
    python3 tdms_cli.py animate "$file" "${file%.tdms}.mp4" --fps 10 --no-display
done
```

## Requirements

- Python 3.6+
- All dependencies of the TDMS Explorer module
- Command line interface (terminal)

## Support

For issues or questions, refer to the main TDMS Explorer documentation or examine the existing Jupyter notebooks in this project.