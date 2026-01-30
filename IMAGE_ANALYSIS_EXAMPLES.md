# TDMS Explorer Image Analysis Features

This document provides comprehensive examples of using the new image analysis features in TDMS Explorer.

## Overview

TDMS Explorer now includes powerful image analysis capabilities that allow you to:

- **Analyze** image statistics and metrics
- **Filter** images with various algorithms
- **Detect edges** using different methods
- **Apply thresholding** for segmentation
- **Work with Regions of Interest (ROI)**
- **Compare images** and find differences
- **Create histograms** and analyze distributions
- **Detect features** like blobs, corners, and edges
- **Extract intensity profiles**

## Installation

To use all image analysis features, install the required dependencies:

```bash
pip install scikit-image opencv-python scipy
```

## Basic Usage

### Analyze Image Statistics

```bash
tdms-explorer analyze file.tdms --image 0
```

This provides comprehensive statistics including:
- Basic statistics (min, max, mean, std, median)
- Entropy and edge density (if scikit-image available)
- Histogram data

**Example Output:**
```
📊 Analysis Results for Image 0:
==================================================
Basic Statistics:
  Min: 100.25, Max: 4095.78
  Mean: 1200.45, Std: 350.12
  Median: 1187.65
  Entropy: 7.23
  Edge Density: 0.08
```

### Apply Filters

```bash
# Gaussian blur
tdms-explorer filter file.tdms --image 0 --type gaussian --sigma 1.5

# Median filter
tdms-explorer filter file.tdms --image 0 --type median --size 5

# Bilateral filter (requires OpenCV)
tdms-explorer filter file.tdms --image 0 --type bilateral
```

**Options:**
- `--type`: Filter type (`gaussian`, `median`, `bilateral`, `sobel`, `prewitt`, `laplace`)
- `--sigma`: Sigma value for Gaussian filter (default: 1.0)
- `--size`: Kernel size for median filter (default: 3)
- `--save`: Save filtered image to file
- `--show`: Display the filtered image

### Detect Edges

```bash
# Canny edge detection (requires OpenCV)
tdms-explorer edges file.tdms --image 0 --method canny

# Sobel edge detection
tdms-explorer edges file.tdms --image 0 --method sobel

# Prewitt edge detection
tdms-explorer edges file.tdms --image 0 --method prewitt

# Laplace edge detection
tdms-explorer edges file.tdms --image 0 --method laplace
```

**Options:**
- `--method`: Edge detection method
- `--save`: Save edge-detected image
- `--show`: Display the result

### Apply Thresholding

```bash
# Otsu's automatic thresholding
tdms-explorer threshold file.tdms --image 0 --method otsu

# Manual thresholding
tdms-explorer threshold file.tdms --image 0 --method manual --threshold 2000

# Adaptive thresholding
tdms-explorer threshold file.tdms --image 0 --method adaptive
```

**Options:**
- `--method`: Thresholding method (`otsu`, `adaptive`, `manual`)
- `--threshold`: Manual threshold value
- `--save`: Save thresholded image
- `--show`: Display the result

## Region of Interest (ROI) Operations

### Set ROI Programmatically

```bash
tdms-explorer roi file.tdms --image 0 --set --x 50 --y 50 --width 200 --height 200
```

### Interactive ROI Selection

```bash
tdms-explorer roi file.tdms --image 0 --interactive
```

This opens an interactive window where you can click and drag to select a rectangular region.

### Analyze ROI

```bash
tdms-explorer roi file.tdms --image 0 --analyze
```

### Clear ROI

```bash
tdms-explorer roi file.tdms --image 0 --clear
```

## Image Comparison

```bash
# Compare two images from the same file
tdms-explorer compare file.tdms --image1 0 --image2 1 --method difference

# Compare with different methods
tdms-explorer compare file.tdms --image1 0 --image2 1 --method absolute
tdms-explorer compare file.tdms --image1 0 --image2 1 --method relative
```

**Options:**
- `--image1`: First image number (default: 0)
- `--image2`: Second image number (default: 1)
- `--method`: Comparison method (`difference`, `absolute`, `relative`)
- `--save`: Save comparison result
- `--show`: Display the comparison

## Histogram Analysis

```bash
# Basic histogram
tdms-explorer histogram file.tdms --image 0

# Histogram with logarithmic scale
tdms-explorer histogram file.tdms --image 0 --log

# Custom number of bins
tdms-explorer histogram file.tdms --image 0 --bins 128

# Save histogram data to JSON
tdms-explorer histogram file.tdms --image 0 --save histogram_data.json
```

## Feature Detection

```bash
# Detect blobs
tdms-explorer features file.tdms --image 0 --method blob

# Detect corners
tdms-explorer features file.tdms --image 0 --method corner

# Detect edges
tdms-explorer features file.tdms --image 0 --method edge

# Output in JSON format
tdms-explorer features file.tdms --image 0 --method blob --json
```

**Example Blob Detection Output:**
```
✅ Feature detection completed
Found 15 features
Blob features:
  Blob 1: x=123, y=456, sigma=2.34
  Blob 2: x=789, y=321, sigma=1.89
  Blob 3: x=456, y=789, sigma=3.12
  ... and 12 more
```

## Intensity Profiles

```bash
# Horizontal profile (average across all rows)
tdms-explorer profile file.tdms --image 0 --direction horizontal

# Vertical profile (average across all columns)
tdms-explorer profile file.tdms --image 0 --direction vertical

# Diagonal profile
tdms-explorer profile file.tdms --image 0 --direction diagonal

# Line profile at specific position
tdms-explorer profile file.tdms --image 0 --direction horizontal --position 64

# Save profile data to JSON
tdms-explorer profile file.tdms --image 0 --save profile_data.json
```

## Advanced Workflows

### Batch Processing Example

```bash
# Process multiple images with a script
for i in {0..10}; do
    echo "Processing image $i"
    tdms-explorer filter input.tdms --image $i --type gaussian --sigma 1.0 --save filtered_$i.png
    tdms-explorer analyze input.tdms --image $i --json > analysis_$i.json
done
```

### ROI Analysis Workflow

```bash
# 1. Set ROI interactively
tdms-explorer roi file.tdms --image 0 --interactive

# 2. Analyze the ROI
tdms-explorer roi file.tdms --image 0 --analyze

# 3. Apply filtering to ROI
# (Note: Current implementation applies to full image, but ROI data can be extracted)
```

### Feature Analysis Pipeline

```bash
# Detect and analyze features across multiple images
for i in {0..5}; do
    echo "Analyzing image $i"
    tdms-explorer features file.tdms --image $i --method blob --json > features_$i.json
    tdms-explorer analyze file.tdms --image $i --json > stats_$i.json
done

# Process the JSON results with Python or other tools
python3 analyze_results.py features_*.json stats_*.json
```

## Python API Usage

You can also use the image analysis features programmatically:

```python
from tdms_explorer.core import TDMSFileExplorer
from tdms_explorer.image_analysis import ImageAnalyzer

# Load TDMS file
explorer = TDMSFileExplorer("file.tdms")

# Create image analyzer for first image
analyzer = explorer.create_image_analyzer(0)

# Perform analysis
analysis = analyzer.analyze_image()
print(f"Mean: {analysis['mean']}, Std: {analysis['std']}")

# Apply Gaussian filter
filtered = analyzer.apply_filter('gaussian', sigma=1.5)

# Detect edges
edges = analyzer.detect_edges('canny')

# Set ROI and analyze
analyzer.set_roi_rectangle(50, 50, 200, 200)
roi_analysis = analyzer.analyze_image()
print(f"ROI mean: {roi_analysis['roi']['mean']}")

# Compare images
analyzer2 = explorer.create_image_analyzer(1)
comparison = analyzer.compare_images(analyzer2.current_image, 'absolute')

# Get histogram
histogram = analyzer.create_histogram(bins=256)

# Detect features
features = analyzer.detect_features('blob')

# Get intensity profile
profile = analyzer.get_image_profile('horizontal')
```

## Performance Tips

1. **For large datasets**: Process images in batches rather than all at once
2. **Memory management**: Use `--save` option to save intermediate results
3. **ROI analysis**: Focus on regions of interest to reduce computation time
4. **Downsampling**: For preview/quick analysis, consider downsampling large images

## Troubleshooting

### Missing Dependencies

If you get warnings about missing dependencies:

```
Warning: scikit-image not available. Some image analysis features will be disabled.
```

Install the required packages:

```bash
pip install scikit-image opencv-python scipy
```

### GUI Display Issues

In headless environments (like JupyterLab terminals), use the `--save` option instead of `--show`:

```bash
tdms-explorer filter file.tdms --image 0 --type gaussian --save output.png
```

### Memory Errors

For very large images, process them individually or use ROI analysis:

```bash
# Process one image at a time
tdms-explorer analyze large_file.tdms --image 0

# Use ROI to focus on specific areas
tdms-explorer roi large_file.tdms --image 0 --set --x 100 --y 100 --width 512 --height 512
tdms-explorer roi large_file.tdms --image 0 --analyze
```

## Available Methods Summary

### Filter Types
- `gaussian`: Gaussian blur (sigma parameter)
- `median`: Median filter (size parameter)
- `bilateral`: Bilateral filter (requires OpenCV)
- `sobel`: Sobel edge enhancement
- `prewitt`: Prewitt edge enhancement
- `laplace`: Laplace edge enhancement

### Edge Detection Methods
- `canny`: Canny edge detection (requires OpenCV)
- `sobel`: Sobel edge detection
- `prewitt`: Prewitt edge detection
- `laplace`: Laplace edge detection

### Thresholding Methods
- `otsu`: Otsu's automatic thresholding
- `adaptive`: Adaptive thresholding
- `manual`: Manual thresholding (specify threshold value)

### Feature Detection Methods
- `blob`: Blob detection (requires scikit-image)
- `corner`: Corner detection (requires scikit-image)
- `edge`: Edge detection (requires scikit-image)

### Profile Directions
- `horizontal`: Average across vertical axis
- `vertical`: Average across horizontal axis
- `diagonal`: Diagonal profile

## Command Reference

| Command | Description | Key Options |
|---------|-------------|-------------|
| `analyze` | Comprehensive image analysis | `--image`, `--roi`, `--json` |
| `filter` | Apply image filters | `--type`, `--sigma`, `--size`, `--save`, `--show` |
| `edges` | Detect edges | `--method`, `--save`, `--show` |
| `threshold` | Apply thresholding | `--method`, `--threshold`, `--save`, `--show` |
| `roi` | ROI operations | `--set`, `--interactive`, `--analyze`, `--clear` |
| `compare` | Compare images | `--image1`, `--image2`, `--method`, `--save`, `--show` |
| `histogram` | Show histogram | `--bins`, `--log`, `--save` |
| `features` | Detect features | `--method`, `--json` |
| `profile` | Get intensity profile | `--direction`, `--position`, `--save` |

## Examples by Use Case

### Quality Control
```bash
# Check image statistics
tdms-explorer analyze production.tdms --image 0

# Detect defects using edge detection
tdms-explorer edges production.tdms --image 0 --method canny --save defects.png

# Compare with reference image
tdms-explorer compare production.tdms --image1 0 --image2 1 --method absolute --save differences.png
```

### Scientific Analysis
```bash
# Analyze particle distribution
tdms-explorer features experiment.tdms --image 0 --method blob --json > particles.json

# Get intensity profiles
tdms-explorer profile experiment.tdms --image 0 --direction horizontal --save profile.json

# ROI analysis of specific regions
tdms-explorer roi experiment.tdms --image 0 --set --x 100 --y 100 --width 300 --height 300
tdms-explorer roi experiment.tdms --image 0 --analyze
```

### Image Processing Pipeline
```bash
# 1. Enhance contrast
tdms-explorer filter raw.tdms --image 0 --type gaussian --sigma 1.0 --save enhanced.tif

# 2. Detect edges
tdms-explorer edges enhanced.tif --method canny --save edges.png

# 3. Analyze results
tdms-explorer analyze edges.png --image 0
```

## Conclusion

The TDMS Explorer image analysis features provide a comprehensive toolkit for analyzing images stored in TDMS files. Whether you need basic statistics, advanced filtering, feature detection, or ROI analysis, these tools can help you extract meaningful information from your scientific and industrial imaging data.

For more advanced usage, consider combining these CLI tools with Python scripting to create custom analysis pipelines tailored to your specific needs.